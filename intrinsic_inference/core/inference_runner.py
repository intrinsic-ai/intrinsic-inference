# Copyright 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import enum
import threading
import time

from absl import logging
import grpc
from tritonclient.grpc import model_config_pb2 as triton_model_pb2
from tritonclient.grpc import service_pb2 as triton_pb2
from tritonclient.grpc import service_pb2_grpc as triton_pb2_grpc

from intrinsic_inference.core import model_controller_base
from intrinsic_inference.core import triton_shm_utils
from intrinsic_inference.core.v1 import ml_model_pb2

_SERVER_READY_TIMEOUT = 30.0
_DEFAULT_SHM_POOL_SIZE = 64 * 1024 * 1024  # 64MB
_DEFAULT_SHM_POOL_NUM = 4


@enum.unique
class ServerState(enum.Enum):
  UNKNOWN = "UNKNOWN"
  LIVE = "LIVE"
  READY = "READY"
  ERROR = "ERROR"


@dataclasses.dataclass
class ServerExtendedState:
  state: ServerState
  message: str


class InferenceRunner:
  """Manages the connection to the Triton server and handles model polling.

  This class is responsible for ensuring the Triton server is ready before
  starting the model synchronization polling loop. It also acts as a proxy,
  forwarding inference and metadata requests directly to the Triton backend.

  Attributes:
    installed_models: A dictionary mapping model names to their loaded MlModel
      instances, representing the current state of models in Triton.
  """

  def __init__(
      self,
      repo_path: str,
      triton_stub: triton_pb2_grpc.GRPCInferenceServiceStub,
      model_controller: model_controller_base.ModelControllerBase,
      poll_models_interval: float = 5.0,
      use_shm: bool = False,
      shm_pool_num: int = _DEFAULT_SHM_POOL_NUM,
      shm_byte_size: int = _DEFAULT_SHM_POOL_SIZE,
  ) -> None:
    """Initializes the InferenceRunner.

    Args:
      repo_path: The local path to the model repository.
      triton_stub: The gRPC stub to communicate with the Triton server.
      model_controller: The manager responsible for reconciling models.
      poll_models_interval: The interval in seconds at which to poll for new
        models.
      use_shm: Whether to use system shared memory for ModelInfer calls.
      shm_pool_num: Number of shared memory region pairs to allocate if use_shm
        is True.
      shm_byte_size: Size in bytes of each shared memory region.
    """
    # Constructor is fast and side-effect free.
    self._state_lock = threading.Lock()
    self._admin_lock = threading.Lock()

    self._repo_path = repo_path
    self._triton_stub = triton_stub
    self._model_controller = model_controller
    self._poll_models_interval = poll_models_interval
    self._use_shm = use_shm
    self._shm_pool_size = shm_pool_num
    self._shm_byte_size = shm_byte_size
    self._shm_pool: triton_shm_utils.RawGrpcSharedMemoryPool | None = None

    self._server_state = ServerExtendedState(
        state=ServerState.UNKNOWN, message="Created"
    )
    self._service_started = threading.Event()
    self._stop_service = threading.Event()
    self._polling_thread: threading.Thread | None = None

  def start(
      self,
  ) -> None:
    """Starts the inference runner.

    This method waits for the Triton server to report ready and then starts the
    background thread for polling installed models.

    Raises:
      Exception: If initialization or connection to Triton fails.
    """
    with self._admin_lock:
      if self._service_started.is_set():
        return
      self._stop_service.clear()
      self._model_controller.start()
      try:
        self._wait_for_server_ready(timeout_seconds=int(_SERVER_READY_TIMEOUT))
        if self._server_state.state != ServerState.READY:
          return
        if self._stop_service.is_set():
          return
        if self._use_shm and self._shm_pool is None:
          logging.info("Initializing Triton shared memory pool...")
          self._shm_pool = triton_shm_utils.RawGrpcSharedMemoryPool(
              stub=self._triton_stub,
              pool_size=self._shm_pool_size,
              byte_size=self._shm_byte_size,
          )
        self._polling_thread = threading.Thread(
            target=self._poll_installed_models_loop,
            kwargs=dict(poll_interval=self._poll_models_interval),
            daemon=True,
        )
        self._polling_thread.start()
        self._service_started.set()
      except Exception as e:
        logging.error("Initialization failed: %s", str(e))
        raise

  def stop(self) -> None:
    """Signals the background polling thread to stop and waits for it to exit."""
    self._stop_service.set()
    polling_thread = None
    with self._admin_lock:
      self._stop_service.set()
      polling_thread = self._polling_thread
      self._polling_thread = None
      self._model_controller.stop(wait=False)
      if self._shm_pool is not None:
        self._shm_pool.cleanup()
        self._shm_pool = None
      self._service_started.clear()

    # Join polling thread outside of lock as it may be blocking for some time.
    if polling_thread:
      polling_thread.join()

  @property
  def server_state(self) -> ServerExtendedState:
    """Returns the current extended state of the Triton server."""
    with self._state_lock:
      return self._server_state

  @property
  def is_started(self) -> bool:
    """Returns True if the runner has finished its startup process."""
    return self._service_started.is_set()

  @property
  def shm_pool(self) -> triton_shm_utils.RawGrpcSharedMemoryPool | None:
    """Returns the shared memory pool if initialized."""
    return self._shm_pool

  @property
  def installed_models(self) -> dict[str, ml_model_pb2.MlModel]:
    """Returns a dictionary mapping model keys to their loaded MlModel proto instances."""
    return self._model_controller.models

  def get_installed_model(self, model_name: str) -> ml_model_pb2.MlModel | None:
    """Retrieves a loaded model by key or model name."""
    return self._model_controller.get_model(model_name)

  @property
  def installed_model_states(
      self,
  ) -> dict[str, model_controller_base.ModelAndState]:
    """Returns a dictionary mapping model keys to their ModelAndState (state & proto)."""
    return self._model_controller.model_states

  def _poll_installed_models_loop(self, poll_interval: float = 5.0) -> None:
    """Background loop that periodically reconciles installed models with Triton.

    Args:
      poll_interval: The time in seconds to wait between reconciliation attempts.
    """
    while not self._stop_service.is_set():
      try:
        self._model_controller.reconcile_models()
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Error during polling installed models: %s", str(e))
      # Wait while keeping the thread responsive to a stop event.
      self._stop_service.wait(timeout=poll_interval)

  def _wait_for_server_ready(self, timeout_seconds: int = 300) -> None:
    """Checks if the remote server is reachable and live.

    This method sends a ServerLive and if successful queries the server with a
    ServerReady request until the server either reports ready or the timeout is
    reached.

    Args:
      timeout_seconds: Timeout in seconds.
    """
    try:
      live_response = self._triton_stub.ServerLive(
          triton_pb2.ServerLiveRequest()
      )
      if not live_response.live:
        message = "Inference server reports not live."
        self._update_state_safe(new_state=ServerState.ERROR, message=message)
        logging.error(message)
        return
      message = "Inference server reports live."
      self._update_state_safe(new_state=ServerState.LIVE, message=message)
      logging.info(
          "Waiting for inference server to report ready... (timeout=%ds)",
          timeout_seconds,
      )
      try_ready_time = time.monotonic()
      while time.monotonic() - try_ready_time < timeout_seconds:
        if self._stop_service.is_set():
          logging.info("Startup aborted by stop signal.")
          return
        ready_response = self._triton_stub.ServerReady(
            triton_pb2.ServerReadyRequest()
        )
        if ready_response.ready:
          message = "Inference server reported ready."
          logging.info(message)
          self._update_state_safe(new_state=ServerState.READY, message=message)
          return
        # Allows the runner to abort the startup sequence immediately if stop()
        # is called.
        if self._stop_service.wait(timeout=1.0):
          logging.info("Startup aborted by stop signal during wait.")
          return
    except grpc.RpcError as e:
      self._update_state_safe(
          new_state=ServerState.ERROR, message=str(e.details())
      )
      logging.error("GRPC Error: %s - %s", e.code(), e.details())
      return

    self._update_state_safe(
        new_state=ServerState.ERROR,
        message="Timed out waiting for inference server to report ready.",
    )

  def _update_state_safe(
      self,
      new_state: ServerState,
      message: str,
  ) -> None:
    """Helper to transition state."""
    with self._state_lock:
      self._server_state = ServerExtendedState(
          state=new_state,
          message=message,
      )

  def _get_triton_model_config(
      self, model_name: str
  ) -> triton_model_pb2.ModelConfig:
    """Retrieves the Triton ModelConfig for the specified model."""
    installed_model = self._model_controller.get_model(model_name)
    if isinstance(installed_model, ml_model_pb2.MlModel):
      triton_config = triton_model_pb2.ModelConfig()
      if installed_model.backend_config.Unpack(triton_config):
        return triton_config

    # Raise error if no config was returned.
    models = self.installed_models
    raise RuntimeError(
        f"Could not obtain model config for model name {model_name} from"
        f" installed models: {list(models.keys())}."
    )

  def ServerLive(
      self,
      request: triton_pb2.ServerLiveRequest,
  ) -> triton_pb2.ServerLiveResponse:
    """Forwards the ServerLive request to the Triton server."""
    return self._triton_stub.ServerLive(request)

  def ServerReady(
      self,
      request: triton_pb2.ServerReadyRequest,
  ) -> triton_pb2.ServerReadyResponse:
    """Forwards the ServerReady request to the Triton server."""
    return self._triton_stub.ServerReady(request)

  def ModelReady(
      self,
      request: triton_pb2.ModelReadyRequest,
  ) -> triton_pb2.ModelReadyResponse:
    """Forwards the ModelReady request to the Triton server."""
    return self._triton_stub.ModelReady(request)

  def ServerMetadata(
      self,
      request: triton_pb2.ServerMetadataRequest,
  ) -> triton_pb2.ServerMetadataResponse:
    """Forwards the ServerMetadata request to the Triton server."""
    return self._triton_stub.ServerMetadata(request)

  def ModelMetadata(
      self,
      request: triton_pb2.ModelMetadataRequest,
  ) -> triton_pb2.ModelMetadataResponse:
    """Forwards the ModelMetadata request to the Triton server."""
    return self._triton_stub.ModelMetadata(request)

  def ModelInfer(
      self,
      request: triton_pb2.ModelInferRequest,
  ) -> triton_pb2.ModelInferResponse:
    """Forwards the inference request and returns the response."""
    if self._shm_pool is not None and request.raw_input_contents:
      try:
        model_config = self._get_triton_model_config(request.model_name)
        return triton_shm_utils.run_inference(
            request=request,
            pool=self._shm_pool,
            stub=self._triton_stub,
            model_config=model_config,
        )
      except ValueError as e:
        # Raise error in case the inputs where already mutated and fallback is
        # not possible.
        if not request.raw_input_contents:
          raise
        logging.warning(
            "SHM inference failed (%s), falling back to standard gRPC"
            " ModelInfer.",
            e,
        )
        triton_shm_utils.clear_shm_parameters(request)
      except Exception as e:  # pylint: disable=broad-except
        # Raise error in case the inputs where already mutated and fallback is
        # not possible.
        if not request.raw_input_contents:
          raise
        logging.warning(
            "Unexpected error in SHM inference (%s), falling back to standard"
            " gRPC ModelInfer.",
            e,
        )
        triton_shm_utils.clear_shm_parameters(request)
    return self._triton_stub.ModelInfer(request)
