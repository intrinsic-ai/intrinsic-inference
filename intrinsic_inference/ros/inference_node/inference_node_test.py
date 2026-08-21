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

"""Unit tests for the InferenceNode ROS 2 node."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from inference_interfaces.srv import InferenceRPC
import rclpy
from tritonclient.grpc import service_pb2 as triton_pb2

from intrinsic_inference.core import inference_runner as inference_runner_lib
from intrinsic_inference.ros.inference_node import inference_node


class InferenceNodeTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    os.environ.setdefault("ROS_LOG_DIR", tempfile.gettempdir())
    os.environ.setdefault("HOME", tempfile.gettempdir())
    if not rclpy.ok():
      rclpy.init()

  @classmethod
  def tearDownClass(cls):
    if rclpy.ok():
      rclpy.shutdown()
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    self.mock_runner = mock.MagicMock(spec=inference_runner_lib.InferenceRunner)
    self.mock_runner.server_state = inference_runner_lib.ServerExtendedState(
        state=inference_runner_lib.ServerState.UNKNOWN, message="Created"
    )
    type(self.mock_runner).is_started = mock.PropertyMock(return_value=False)

    def start_side_effect():
      type(self.mock_runner).is_started = mock.PropertyMock(return_value=True)

    self.mock_runner.start.side_effect = start_side_effect

    def stop_side_effect():
      type(self.mock_runner).is_started = mock.PropertyMock(return_value=False)

    self.mock_runner.stop.side_effect = stop_side_effect

    self.node = inference_node.InferenceNode(inference_runner=self.mock_runner)

  def tearDown(self):
    self.node.destroy_node()
    super().tearDown()

  # -------------------- A. Lifecycle Tests --------------------

  def test_initial_state_is_uninitialized(self):
    self.assertEqual(self.node.status, inference_node.NodeState.UNINITIALIZED)

  def test_start_success_transitions_to_ready(self):
    self.mock_runner.server_state = inference_runner_lib.ServerExtendedState(
        state=inference_runner_lib.ServerState.READY, message="Server is ready"
    )
    self.node.start()
    self.assertEqual(self.node.status, inference_node.NodeState.READY)
    self.mock_runner.start.assert_called_once()

  def test_start_failure_sets_error_state(self):
    self.mock_runner.server_state = inference_runner_lib.ServerExtendedState(
        state=inference_runner_lib.ServerState.ERROR,
        message="Failed to connect",
    )
    self.node.start()
    self.assertEqual(self.node.status, inference_node.NodeState.ERROR)
    self.mock_runner.start.assert_called_once()

  def test_start_exception_sets_error_state(self):
    self.mock_runner.start.side_effect = RuntimeError(
        "Unexpected startup failure"
    )
    with self.assertRaises(RuntimeError):
      self.node.start()
    self.assertEqual(self.node.status, inference_node.NodeState.ERROR)

  def test_stop_transitions_to_uninitialized(self):
    self.mock_runner.server_state = inference_runner_lib.ServerExtendedState(
        state=inference_runner_lib.ServerState.READY, message="Server is ready"
    )
    self.node.start()
    self.assertEqual(self.node.status, inference_node.NodeState.READY)

    self.node.stop()
    self.assertEqual(self.node.status, inference_node.NodeState.UNINITIALIZED)
    self.mock_runner.stop.assert_called_once()

  # -------------------- B. Error Handling Tests --------------------

  def test_service_error_handling_on_runner_exception(self):
    self.mock_runner.ModelInfer.side_effect = RuntimeError("Triton unavailable")

    proto_req = triton_pb2.ModelInferRequest(model_name="nonexistent_model")
    req = InferenceRPC.Request()
    req.raw_request = list(proto_req.SerializeToString())
    res = InferenceRPC.Response()

    out_res = self.node.model_infer(req, res)

    self.assertFalse(out_res.success)
    self.assertIn("Triton unavailable", out_res.error_message)
    self.assertEqual(len(out_res.raw_response), 0)

  # -------------------- C. OIP Service Handler Tests --------------------

  def test_server_live_success(self):
    mock_response = triton_pb2.ServerLiveResponse(live=True)
    self.mock_runner.ServerLive.return_value = mock_response

    req = InferenceRPC.Request()
    req.raw_request = list(triton_pb2.ServerLiveRequest().SerializeToString())
    res = InferenceRPC.Response()

    out_res = self.node.server_live(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    parsed_proto = triton_pb2.ServerLiveResponse()
    parsed_proto.ParseFromString(bytes(out_res.raw_response))
    self.assertTrue(parsed_proto.live)
    self.mock_runner.ServerLive.assert_called_once()

  def test_server_ready_success(self):
    mock_response = triton_pb2.ServerReadyResponse(ready=True)
    self.mock_runner.ServerReady.return_value = mock_response

    req = InferenceRPC.Request()
    req.raw_request = list(triton_pb2.ServerReadyRequest().SerializeToString())
    res = InferenceRPC.Response()

    out_res = self.node.server_ready(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    parsed_proto = triton_pb2.ServerReadyResponse()
    parsed_proto.ParseFromString(bytes(out_res.raw_response))
    self.assertTrue(parsed_proto.ready)
    self.mock_runner.ServerReady.assert_called_once()

  def test_model_ready_success(self):
    mock_response = triton_pb2.ModelReadyResponse(ready=True)
    self.mock_runner.ModelReady.return_value = mock_response

    proto_req = triton_pb2.ModelReadyRequest(name="densenet_onnx", version="1")
    req = InferenceRPC.Request()
    req.raw_request = list(proto_req.SerializeToString())
    res = InferenceRPC.Response()

    out_res = self.node.model_ready(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    parsed_proto = triton_pb2.ModelReadyResponse()
    parsed_proto.ParseFromString(bytes(out_res.raw_response))
    self.assertTrue(parsed_proto.ready)
    self.mock_runner.ModelReady.assert_called_once_with(proto_req)

  def test_server_metadata_success(self):
    mock_response = triton_pb2.ServerMetadataResponse(
        name="triton", version="2.34.0", extensions=["binary_tensor_data"]
    )
    self.mock_runner.ServerMetadata.return_value = mock_response

    req = InferenceRPC.Request()
    req.raw_request = list(
        triton_pb2.ServerMetadataRequest().SerializeToString()
    )
    res = InferenceRPC.Response()

    out_res = self.node.server_metadata(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    parsed_proto = triton_pb2.ServerMetadataResponse()
    parsed_proto.ParseFromString(bytes(out_res.raw_response))
    self.assertEqual(parsed_proto.name, "triton")
    self.assertEqual(parsed_proto.version, "2.34.0")
    self.mock_runner.ServerMetadata.assert_called_once()

  def test_model_metadata_success(self):
    mock_response = triton_pb2.ModelMetadataResponse(
        name="densenet_onnx", versions=["1"], platform="onnxruntime_onnx"
    )
    self.mock_runner.ModelMetadata.return_value = mock_response

    proto_req = triton_pb2.ModelMetadataRequest(
        name="densenet_onnx", version="1"
    )
    req = InferenceRPC.Request()
    req.raw_request = list(proto_req.SerializeToString())
    res = InferenceRPC.Response()

    out_res = self.node.model_metadata(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    parsed_proto = triton_pb2.ModelMetadataResponse()
    parsed_proto.ParseFromString(bytes(out_res.raw_response))
    self.assertEqual(parsed_proto.name, "densenet_onnx")
    self.assertEqual(list(parsed_proto.versions), ["1"])
    self.mock_runner.ModelMetadata.assert_called_once_with(proto_req)

  def test_model_infer_success(self):
    mock_response = triton_pb2.ModelInferResponse(
        model_name="densenet_onnx", model_version="1"
    )
    self.mock_runner.ModelInfer.return_value = mock_response

    proto_req = triton_pb2.ModelInferRequest(
        model_name="densenet_onnx", model_version="1"
    )
    req = InferenceRPC.Request()
    req.raw_request = list(proto_req.SerializeToString())
    res = InferenceRPC.Response()

    out_res = self.node.model_infer(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    parsed_proto = triton_pb2.ModelInferResponse()
    parsed_proto.ParseFromString(bytes(out_res.raw_response))
    self.assertEqual(parsed_proto.model_name, "densenet_onnx")
    self.mock_runner.ModelInfer.assert_called_once_with(proto_req)


if __name__ == "__main__":
  absltest.main()
