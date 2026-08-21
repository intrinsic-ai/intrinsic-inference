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

"""ROS inference node built using the core inference components.

This node embeds InferenceRunner directly and exposes all OpenInferenceProtocol (OIP)
methods (ServerLive, ServerReady, ModelReady, ServerMetadata, ModelMetadata, ModelInfer)
as ROS 2 services using the custom InferenceRPC service interface.
"""

from __future__ import annotations

import dataclasses
import enum
import threading

from absl import logging
from inference_interfaces.srv import InferenceRPC
from rclpy.node import Node
from tritonclient.grpc import service_pb2 as triton_pb2

from intrinsic_inference.core import inference_runner as inference_runner_lib


@enum.unique
class NodeState(enum.Enum):
  UNINITIALIZED = "UNINITIALIZED"
  READY = "READY"
  ERROR = "ERROR"


@dataclasses.dataclass
class RuntimeState:
  """Mutable runtime state of the ROS inference node, protected by _state_lock.

  Attributes:
    status: The current runtime status of the node instance.
    inference_runner: Runner that manages communication with Triton server.
  """

  status: NodeState
  inference_runner: inference_runner_lib.InferenceRunner


class InferenceNode(Node):
  """ROS 2 inference node wrapping InferenceRunner for model inference."""

  def __init__(
      self,
      inference_runner: inference_runner_lib.InferenceRunner,
  ) -> None:
    super().__init__("inference_node")

    # Thread locks for safe concurrent state updates (matching inference_service.py)
    self._state_lock = threading.RLock()
    self._admin_lock = threading.RLock()

    self._runtime_state = RuntimeState(
        status=NodeState.UNINITIALIZED,
        inference_runner=inference_runner,
    )

    # Expose all 6 OpenInferenceProtocol methods using the InferenceRPC service interface
    self._server_live_srv = self.create_service(
        InferenceRPC,
        "ServerLive",
        self.server_live,
    )
    self._server_ready_srv = self.create_service(
        InferenceRPC,
        "ServerReady",
        self.server_ready,
    )
    self._model_ready_srv = self.create_service(
        InferenceRPC,
        "ModelReady",
        self.model_ready,
    )
    self._server_metadata_srv = self.create_service(
        InferenceRPC,
        "ServerMetadata",
        self.server_metadata,
    )
    self._model_metadata_srv = self.create_service(
        InferenceRPC,
        "ModelMetadata",
        self.model_metadata,
    )
    self._model_infer_srv = self.create_service(
        InferenceRPC,
        "ModelInfer",
        self.model_infer,
    )

    self.get_logger().info(
        "InferenceNode created and exposing all OIP ROS 2 services via"
        " InferenceRPC: /ServerLive, /ServerReady, /ModelReady,"
        " /ServerMetadata, /ModelMetadata, /ModelInfer"
    )

  @property
  def inference_runner(self) -> inference_runner_lib.InferenceRunner:
    with self._state_lock:
      return self._runtime_state.inference_runner

  @property
  def status(self) -> NodeState:
    with self._state_lock:
      return self._runtime_state.status

  @status.setter
  def status(self, new_status: NodeState):
    with self._state_lock:
      self._runtime_state.status = new_status

  def start(self) -> None:
    """Starts the InferenceNode and its underlying InferenceRunner.

    Transitions the node to the READY state, mirroring inference_service.start().
    """
    with self._admin_lock:
      try:
        self.inference_runner.start()
        if (
            self.inference_runner.server_state.state
            == inference_runner_lib.ServerState.READY
        ):
          self.status = NodeState.READY
          self.get_logger().info("InferenceNode and InferenceRunner started.")
        else:
          self.status = NodeState.ERROR
          message = self.inference_runner.server_state.message
          err_msg = f"Initialization Error: {message}"
          self.get_logger().error(err_msg)
      except Exception as e:
        logging.error("Initialization failed: %s", str(e))
        self.status = NodeState.ERROR
        raise

  def stop(self) -> None:
    """Stops the InferenceNode and its underlying InferenceRunner.

    Transitions the node to the UNINITIALIZED state, mirroring inference_service.stop().
    """
    with self._admin_lock:
      self.inference_runner.stop()
      self.status = NodeState.UNINITIALIZED
      self.get_logger().info("InferenceNode and InferenceRunner stopped.")

  # -------------------- ROS 2 OIP Service Handlers --------------------

  def server_live(
      self,
      request: InferenceRPC.Request,
      response: InferenceRPC.Response,
  ) -> InferenceRPC.Response:
    """ROS service handler for ServerLive query."""
    try:
      proto_req = triton_pb2.ServerLiveRequest()
      raw_bytes = bytes(request.raw_request)
      if raw_bytes:
        proto_req.ParseFromString(raw_bytes)

      proto_res = self.inference_runner.ServerLive(proto_req)
      response.success = True
      response.error_message = ""
      response.raw_response = bytes(proto_res.SerializeToString())
    except Exception as e:
      response.success = False
      response.error_message = f"ServerLive error: {e}"
      response.raw_response = b""
      self.get_logger().error(response.error_message)
    return response

  def server_ready(
      self,
      request: InferenceRPC.Request,
      response: InferenceRPC.Response,
  ) -> InferenceRPC.Response:
    """ROS service handler for ServerReady query."""
    try:
      proto_req = triton_pb2.ServerReadyRequest()
      raw_bytes = bytes(request.raw_request)
      if raw_bytes:
        proto_req.ParseFromString(raw_bytes)

      proto_res = self.inference_runner.ServerReady(proto_req)
      response.success = True
      response.error_message = ""
      response.raw_response = bytes(proto_res.SerializeToString())
    except Exception as e:
      response.success = False
      response.error_message = f"ServerReady error: {e}"
      response.raw_response = b""
      self.get_logger().error(response.error_message)
    return response

  def model_ready(
      self,
      request: InferenceRPC.Request,
      response: InferenceRPC.Response,
  ) -> InferenceRPC.Response:
    """ROS service handler for ModelReady query."""
    try:
      proto_req = triton_pb2.ModelReadyRequest()
      raw_bytes = bytes(request.raw_request)
      if raw_bytes:
        proto_req.ParseFromString(raw_bytes)

      proto_res = self.inference_runner.ModelReady(proto_req)
      response.success = True
      response.error_message = ""
      response.raw_response = bytes(proto_res.SerializeToString())
    except Exception as e:
      response.success = False
      response.error_message = f"ModelReady error: {e}"
      response.raw_response = b""
      self.get_logger().error(response.error_message)
    return response

  def server_metadata(
      self,
      request: InferenceRPC.Request,
      response: InferenceRPC.Response,
  ) -> InferenceRPC.Response:
    """ROS service handler for ServerMetadata query."""
    try:
      proto_req = triton_pb2.ServerMetadataRequest()
      raw_bytes = bytes(request.raw_request)
      if raw_bytes:
        proto_req.ParseFromString(raw_bytes)

      proto_res = self.inference_runner.ServerMetadata(proto_req)
      response.success = True
      response.error_message = ""
      response.raw_response = bytes(proto_res.SerializeToString())
    except Exception as e:
      response.success = False
      response.error_message = f"ServerMetadata error: {e}"
      response.raw_response = b""
      self.get_logger().error(response.error_message)
    return response

  def model_metadata(
      self,
      request: InferenceRPC.Request,
      response: InferenceRPC.Response,
  ) -> InferenceRPC.Response:
    """ROS service handler for ModelMetadata query."""
    try:
      proto_req = triton_pb2.ModelMetadataRequest()
      raw_bytes = bytes(request.raw_request)
      if raw_bytes:
        proto_req.ParseFromString(raw_bytes)

      proto_res = self.inference_runner.ModelMetadata(proto_req)
      response.success = True
      response.error_message = ""
      response.raw_response = bytes(proto_res.SerializeToString())
    except Exception as e:
      response.success = False
      response.error_message = f"ModelMetadata error: {e}"
      response.raw_response = b""
      self.get_logger().error(response.error_message)
    return response

  def model_infer(
      self,
      request: InferenceRPC.Request,
      response: InferenceRPC.Response,
  ) -> InferenceRPC.Response:
    """ROS service handler for ModelInfer execution."""
    try:
      proto_req = triton_pb2.ModelInferRequest()
      raw_bytes = bytes(request.raw_request)
      if raw_bytes:
        proto_req.ParseFromString(raw_bytes)

      proto_res = self.inference_runner.ModelInfer(proto_req)
      response.success = True
      response.error_message = ""
      response.raw_response = bytes(proto_res.SerializeToString())
    except Exception as e:
      response.success = False
      response.error_message = f"ModelInfer error: {e}"
      response.raw_response = b""
      self.get_logger().error(response.error_message)
    return response
