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

"""Tests for InferenceRunner."""

import sys
import threading
import time
from unittest import mock

mock_shm = mock.MagicMock()
sys.modules["tritonclient.utils.shared_memory"] = mock_shm

from absl.testing import absltest
import grpc
from tritonclient.grpc import model_config_pb2 as triton_model_pb2
from tritonclient.grpc import service_pb2 as triton_pb2
from tritonclient.grpc import service_pb2_grpc as triton_pb2_grpc

from intrinsic_inference.core import inference_runner
from intrinsic_inference.core import model_controller_base
from intrinsic_inference.core import triton_shm_utils
from intrinsic_inference.core.v1 import ml_model_pb2


class MockRpcError(grpc.RpcError):

  def __init__(self, code, details):
    self._code = code
    self._details = details

  def code(self):
    return self._code

  def details(self):
    return self._details

  def __str__(self):
    return f"MockRpcError: code={self._code}, details={self._details}"


class InferenceRunnerTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.mock_triton_stub = mock.MagicMock(
        spec=triton_pb2_grpc.GRPCInferenceServiceServicer
    )
    self.mock_model_controller = mock.MagicMock(
        spec=model_controller_base.ModelControllerBase
    )
    self.mock_model_controller.models = {}

    def mock_get_model(name: str):
      models = self.mock_model_controller.models
      if name in models:
        return models[name]
      for k, v in models.items():
        if (
            v.model_config.name == name
            or k.startswith(f"{name}.")
            or k.endswith(f".{name}")
        ):
          return v
      return None

    self.mock_model_controller.get_model.side_effect = mock_get_model

    # Default mock behavior
    self.mock_triton_stub.ServerLive.return_value = (
        triton_pb2.ServerLiveResponse(live=True)
    )
    self.mock_triton_stub.ServerReady.return_value = (
        triton_pb2.ServerReadyResponse(ready=True)
    )

  def test_init_state(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    self.assertEqual(
        runner._server_state.state, inference_runner.ServerState.UNKNOWN
    )
    self.assertEqual(runner._server_state.message, "Created")

  def test_start_success(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        poll_models_interval=0.1,
    )

    import threading

    reconcile_called = threading.Event()

    def side_effect(*args, **kwargs):
      reconcile_called.set()

    self.mock_model_controller.reconcile_models.side_effect = side_effect

    runner.start()

    self.mock_triton_stub.ServerLive.assert_called_once()
    self.mock_triton_stub.ServerReady.assert_called_once()
    self.assertEqual(
        runner._server_state.state, inference_runner.ServerState.READY
    )
    self.assertTrue(runner._service_started.is_set())

    # Wait for reconcile to be called to prove polling thread is running
    self.assertTrue(reconcile_called.wait(timeout=1.0))

    thread = runner._polling_thread
    runner.stop()
    self.assertIsNotNone(thread)
    thread.join(timeout=1.0)
    self.assertFalse(thread.is_alive())

  def test_start_server_not_live(self):
    self.mock_triton_stub.ServerLive.return_value = (
        triton_pb2.ServerLiveResponse(live=False)
    )
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        poll_models_interval=0.1,
    )

    runner.start()

    self.mock_triton_stub.ServerLive.assert_called_once()
    self.mock_triton_stub.ServerReady.assert_not_called()
    self.assertEqual(
        runner._server_state.state, inference_runner.ServerState.ERROR
    )
    self.assertEqual(
        runner._server_state.message, "Inference server reports not live."
    )

    thread = runner._polling_thread
    runner.stop()
    self.assertIsNone(thread)
    self.assertFalse(runner.is_started)

  @mock.patch("time.sleep", autospec=True)
  @mock.patch("intrinsic_inference.core.inference_runner.time.time")
  def test_start_server_timeout(self, mock_time, mock_sleep):
    # Simulate time passing:
    # 1. start of wait loop: time = 100
    # 2. check elapsed: 100 - 100 = 0 < 2 -> call ServerReady
    # 3. next iteration check elapsed: 103 - 100 = 3 >= 2 -> exit loop
    mock_time.side_effect = [100.0, 100.0, 103.0]
    self.mock_triton_stub.ServerReady.return_value = (
        triton_pb2.ServerReadyResponse(ready=False)
    )

    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        poll_models_interval=0.1,
    )

    # Set timeout to 2 seconds for quick test
    with mock.patch(
        "intrinsic_inference.core.inference_runner._SERVER_READY_TIMEOUT",
        2.0,
    ):
      runner.start()

    self.assertEqual(
        runner._server_state.state, inference_runner.ServerState.ERROR
    )
    self.assertIn(
        "Timed out waiting for inference server", runner._server_state.message
    )

    thread = runner._polling_thread
    runner.stop()
    self.assertIsNone(thread)
    self.assertFalse(runner.is_started)

  def test_start_grpc_error(self):
    self.mock_triton_stub.ServerLive.side_effect = MockRpcError(
        grpc.StatusCode.UNAVAILABLE, "Connection refused"
    )

    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        poll_models_interval=0.1,
    )

    runner.start()

    self.assertEqual(
        runner._server_state.state, inference_runner.ServerState.ERROR
    )
    self.assertEqual(runner._server_state.message, "Connection refused")

    thread = runner._polling_thread
    runner.stop()
    self.assertIsNone(thread)
    self.assertFalse(runner.is_started)

  def test_start_stop_start_lifecycle(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        poll_models_interval=0.1,
    )

    reconcile_called_1 = threading.Event()
    reconcile_called_2 = threading.Event()

    call_count = 0

    def side_effect(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        reconcile_called_1.set()
      else:
        reconcile_called_2.set()

    self.mock_model_controller.reconcile_models.side_effect = side_effect

    # 1. Start first time
    runner.start()
    self.assertTrue(runner.is_started)
    self.assertTrue(reconcile_called_1.wait(timeout=1.0))

    # 2. Stop runner
    runner.stop()
    self.assertFalse(runner.is_started)
    self.assertIsNone(runner._polling_thread)

    # 3. Start runner second time (should cleanly reset _stop_service and start controller)
    runner.start()
    self.assertTrue(runner.is_started)
    self.assertTrue(reconcile_called_2.wait(timeout=1.0))
    runner.stop()
    self.assertFalse(runner.is_started)

  def test_polling_loop_continues_on_error(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        poll_models_interval=0.05,
    )

    import threading

    reconcile_called = threading.Event()
    call_count = 0

    def side_effect(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise RuntimeError("Temporary error")
      elif call_count == 2:
        reconcile_called.set()

    self.mock_model_controller.reconcile_models.side_effect = side_effect

    runner.start()

    # Wait for the second call (after the error in the first call)
    self.assertTrue(reconcile_called.wait(timeout=1.0))

    thread = runner._polling_thread
    runner.stop()
    self.assertIsNotNone(thread)
    thread.join(timeout=1.0)
    self.assertFalse(thread.is_alive())
    self.assertEqual(call_count, 2)

  def test_start_unexpected_exception(self):
    self.mock_triton_stub.ServerLive.side_effect = ValueError("Unexpected")
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )

    with self.assertRaises(ValueError):
      runner.start()

    self.assertIsNone(runner._polling_thread)
    runner.stop()

  def test_polling_loop_one_iteration(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    mock_event = mock.MagicMock(spec=threading.Event)
    mock_event.is_set.side_effect = [False, True]
    runner._stop_service = mock_event

    runner._poll_installed_models_loop(poll_interval=42.0)

    self.mock_model_controller.reconcile_models.assert_called_once()
    self.assertEqual(runner.installed_models, self.mock_model_controller.models)
    mock_event.wait.assert_called_once_with(timeout=42.0)

  def test_stop_before_start(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    # Should not raise any error
    runner.stop()
    self.assertIsNone(runner._polling_thread)

  def test_server_live_forwards_request(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    request = triton_pb2.ServerLiveRequest()
    expected_response = triton_pb2.ServerLiveResponse(live=True)
    self.mock_triton_stub.ServerLive.return_value = expected_response

    response = runner.ServerLive(request)

    self.mock_triton_stub.ServerLive.assert_called_with(request)
    self.assertEqual(response, expected_response)

  def test_server_ready_forwards_request(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    request = triton_pb2.ServerReadyRequest()
    expected_response = triton_pb2.ServerReadyResponse(ready=True)
    self.mock_triton_stub.ServerReady.return_value = expected_response

    response = runner.ServerReady(request)

    self.mock_triton_stub.ServerReady.assert_called_with(request)
    self.assertEqual(response, expected_response)

  def test_model_ready_forwards_request(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    request = triton_pb2.ModelReadyRequest(name="test_model")
    expected_response = triton_pb2.ModelReadyResponse(ready=True)
    self.mock_triton_stub.ModelReady.return_value = expected_response

    response = runner.ModelReady(request)

    self.mock_triton_stub.ModelReady.assert_called_with(request)
    self.assertEqual(response, expected_response)

  def test_server_metadata_forwards_request(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    request = triton_pb2.ServerMetadataRequest()
    expected_response = triton_pb2.ServerMetadataResponse(name="triton")
    self.mock_triton_stub.ServerMetadata.return_value = expected_response

    response = runner.ServerMetadata(request)

    self.mock_triton_stub.ServerMetadata.assert_called_with(request)
    self.assertEqual(response, expected_response)

  def test_model_metadata_forwards_request(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    request = triton_pb2.ModelMetadataRequest(name="test_model")
    expected_response = triton_pb2.ModelMetadataResponse(name="test_model")
    self.mock_triton_stub.ModelMetadata.return_value = expected_response

    response = runner.ModelMetadata(request)

    self.mock_triton_stub.ModelMetadata.assert_called_with(request)
    self.assertEqual(response, expected_response)

  def test_model_infer_forwards_request(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    request = triton_pb2.ModelInferRequest(model_name="test_model")
    expected_response = triton_pb2.ModelInferResponse(model_name="test_model")
    self.mock_triton_stub.ModelInfer.return_value = expected_response

    response = runner.ModelInfer(request)

    self.mock_triton_stub.ModelInfer.assert_called_with(request)
    self.assertEqual(response, expected_response)

  def test_start_and_stop_with_shm_enabled(self):
    mock_shm.reset_mock()
    mock_shm.create_shared_memory_region.side_effect = (
        lambda name, path, size: f"handle_{name}"
    )

    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        poll_models_interval=0.1,
        use_shm=True,
        shm_pool_num=1,
        shm_byte_size=1024,
    )

    runner.start()
    self.assertIsNotNone(runner.shm_pool)
    self.assertEqual(runner.shm_pool.pool_size, 1)
    self.assertEqual(runner.shm_pool.byte_size, 1024)

    runner.stop()
    self.assertIsNone(runner.shm_pool)

  @mock.patch.object(triton_shm_utils, "run_inference", autospec=True)
  def test_model_infer_uses_shm_when_enabled(self, mock_run_shm_inference):
    mock_shm.reset_mock()
    mock_shm.create_shared_memory_region.side_effect = (
        lambda name, path, size: f"handle_{name}"
    )

    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        use_shm=True,
        shm_pool_num=1,
        shm_byte_size=1024,
    )
    runner.start()

    expected_response = triton_pb2.ModelInferResponse(model_name="test_model")
    mock_run_shm_inference.return_value = expected_response

    # Populate installed_models with a model_config
    ml_model = ml_model_pb2.MlModel()
    triton_cfg = triton_model_pb2.ModelConfig(name="test_model")
    ml_model.backend_config.Pack(triton_cfg)
    self.mock_model_controller.models = {"test_model": ml_model}
    runner.installed_models["test_model"] = ml_model
    self.mock_triton_stub.ModelConfig.return_value = (
        triton_pb2.ModelConfigResponse(config=triton_cfg)
    )

    request = triton_pb2.ModelInferRequest(model_name="test_model")
    request.raw_input_contents.append(b"dummy_bytes")

    response = runner.ModelInfer(request)

    self.assertEqual(response, expected_response)
    mock_run_shm_inference.assert_called_once_with(
        request=request,
        pool=runner.shm_pool,
        stub=self.mock_triton_stub,
        model_config=triton_cfg,
    )
    runner.stop()

  @mock.patch.object(triton_shm_utils, "run_inference", autospec=True)
  def test_model_infer_shm_fallback_on_error(self, mock_run_shm_inference):
    mock_shm.reset_mock()
    mock_shm.create_shared_memory_region.side_effect = (
        lambda name, path, size: f"handle_{name}"
    )

    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        use_shm=True,
        shm_pool_num=1,
        shm_byte_size=1024,
    )
    runner.start()

    ml_model = ml_model_pb2.MlModel()
    triton_cfg = triton_model_pb2.ModelConfig(name="test_model")
    ml_model.backend_config.Pack(triton_cfg)
    self.mock_model_controller.models = {"test_model": ml_model}

    mock_run_shm_inference.side_effect = ValueError("Exceeded SHM capacity")
    expected_response = triton_pb2.ModelInferResponse(model_name="test_model")
    self.mock_triton_stub.ModelInfer.return_value = expected_response

    request = triton_pb2.ModelInferRequest(model_name="test_model")
    request.raw_input_contents.append(b"dummy_bytes")
    inp = request.inputs.add()
    inp.name = "input_0"
    inp.parameters["shared_memory_region"].string_param = "in_region_0"

    response = runner.ModelInfer(request)

    self.assertEqual(response, expected_response)
    self.mock_triton_stub.ModelInfer.assert_called_once_with(request)
    # Ensure dangling SHM parameters were cleared before fallback
    self.assertNotIn("shared_memory_region", inp.parameters)
    runner.stop()

  @mock.patch.object(triton_shm_utils, "run_inference", autospec=True)
  def test_model_infer_no_fallback_when_request_mutated(
      self, mock_run_shm_inference
  ):
    mock_shm.reset_mock()
    mock_shm.create_shared_memory_region.side_effect = (
        lambda name, path, size: f"handle_{name}"
    )

    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
        use_shm=True,
        shm_pool_num=1,
        shm_byte_size=1024,
    )
    runner.start()

    ml_model = ml_model_pb2.MlModel()
    triton_cfg = triton_model_pb2.ModelConfig(name="test_model")
    ml_model.backend_config.Pack(triton_cfg)
    self.mock_model_controller.models = {"test_model": ml_model}

    def _mutate_and_raise(*args, **kwargs):
      req = kwargs.get("request")
      if req is not None:
        req.ClearField("raw_input_contents")
      raise MockRpcError(grpc.StatusCode.INTERNAL, "Triton error")

    mock_run_shm_inference.side_effect = _mutate_and_raise

    request = triton_pb2.ModelInferRequest(model_name="test_model")
    request.raw_input_contents.append(b"dummy_bytes")

    with self.assertRaises(MockRpcError):
      runner.ModelInfer(request)

    self.mock_triton_stub.ModelInfer.assert_not_called()
    runner.stop()

  def test_get_triton_model_config_prefers_exact_over_substring(self):
    runner = inference_runner.InferenceRunner(
        repo_path="/tmp/models",
        triton_stub=self.mock_triton_stub,
        model_controller=self.mock_model_controller,
    )
    model1 = ml_model_pb2.MlModel()
    model1.model_config.name = "detection"
    cfg1 = triton_model_pb2.ModelConfig(name="detection", max_batch_size=4)
    model1.backend_config.Pack(cfg1)

    model2 = ml_model_pb2.MlModel()
    model2.model_config.name = "detection_v2"
    cfg2 = triton_model_pb2.ModelConfig(name="detection_v2", max_batch_size=8)
    model2.backend_config.Pack(cfg2)

    self.mock_model_controller.models = {
        "com.intrinsic.detection_v2.v1": model2,
        "com.intrinsic.detection.v1": model1,
    }

    # Requesting "detection" should resolve to model1, not detection_v2
    resolved_cfg = runner._get_triton_model_config("detection")
    self.assertEqual(resolved_cfg.name, "detection")
    self.assertEqual(resolved_cfg.max_batch_size, 4)


if __name__ == "__main__":
  absltest.main()
