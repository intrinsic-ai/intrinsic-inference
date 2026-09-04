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
import numpy as np
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterValue
import rclpy
from tensor_msgs.msg import ExperimentalTensor
from tritonclient.grpc import service_pb2 as triton_pb2

from intrinsic_inference.core import inference_runner as inference_runner_lib
from intrinsic_inference.ros.inference_interfaces.msg import NamedTensor
from intrinsic_inference.ros.inference_interfaces.srv import ModelInfer
from intrinsic_inference.ros.inference_interfaces.srv import ModelMetadata
from intrinsic_inference.ros.inference_interfaces.srv import ModelReady
from intrinsic_inference.ros.inference_interfaces.srv import ServerLive
from intrinsic_inference.ros.inference_interfaces.srv import ServerReady
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
    self.mock_runner.ServerMetadata.return_value = (
        triton_pb2.ServerMetadataResponse(
            name="triton", version="2.34.0", extensions=["binary_tensor_data"]
        )
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

    req = ModelInfer.Request(model_name="nonexistent_model")
    res = ModelInfer.Response()

    out_res = self.node.model_infer(req, res)

    self.assertFalse(out_res.success)
    self.assertIn("Triton unavailable", out_res.error_message)
    self.assertEqual(len(out_res.outputs), 0)

  def test_model_infer_invalid_param_datatype(self):
    req = ModelInfer.Request(model_name="densenet_onnx")
    req.parameters.append(
        Parameter(
            name="bad_param",
            value=ParameterValue(
                type=ParameterType.PARAMETER_BYTE_ARRAY,
                byte_array_value=[b"\x00"],
            ),
        )
    )
    res = ModelInfer.Response()
    out_res = self.node.model_infer(req, res)
    self.assertFalse(out_res.success)
    self.assertIn(
        "Unsupported Parameter datatype",
        out_res.error_message,
    )

  def testdlpack_to_triton_dtype_mappings_and_errors(self):
    for triton_dtype, (
        code,
        bits,
    ) in inference_node._TRITON_TO_DLPACK_DTYPE.items():
      self.assertEqual(
          inference_node.dlpack_to_triton_dtype(code, bits), triton_dtype
      )
      self.assertEqual(
          inference_node.triton_to_dlpack_dtype(triton_dtype), (code, bits)
      )

    with self.assertRaises(ValueError):
      inference_node.dlpack_to_triton_dtype(99, 99)

    with self.assertRaises(ValueError):
      inference_node.triton_to_dlpack_dtype("NONEXISTENT_TYPE")

  def test_model_infer_invalid_input_dtype(self):
    req = ModelInfer.Request(model_name="densenet_onnx")
    inp_tensor = ExperimentalTensor(
        dtype_code=99,
        dtype_bits=99,
        dtype_lanes=1,
        shape=[1],
        data=[0],
    )
    req.inputs.append(NamedTensor(name="bad_tensor", tensor=inp_tensor))
    res = ModelInfer.Response()
    out_res = self.node.model_infer(req, res)
    self.assertFalse(out_res.success)
    self.assertIn(
        "Unsupported DLPack datatype: dtype_code=99, dtype_bits=99",
        out_res.error_message,
    )

  def test_extract_contiguous_tensor_bytes_with_offset_and_strides(self):
    # Contiguous tensor with non-zero byte_offset
    full_array = np.array([0, 10, 20, 30, 40], dtype=np.float32)
    tensor_with_offset = ExperimentalTensor(
        dtype_code=2,
        dtype_bits=32,
        dtype_lanes=1,
        shape=[3],
        byte_offset=8,  # Start at element index 2 (value 20)
        data=list(full_array.tobytes()),
    )
    raw = inference_node.extract_contiguous_tensor_bytes(
        tensor_with_offset, np.float32
    )
    extracted = np.frombuffer(raw, dtype=np.float32)
    np.testing.assert_array_equal(
        extracted, np.array([20, 30, 40], dtype=np.float32)
    )

    # Non-contiguous strided tensor (Fortran column-major order)
    f_array = np.asfortranarray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    tensor_fortran = ExperimentalTensor(
        dtype_code=2,
        dtype_bits=32,
        dtype_lanes=1,
        shape=[2, 2],
        strides=[1, 2],  # Element strides for column-major [2, 2]
        data=list(f_array.tobytes(order="F")),
    )
    raw_f = inference_node.extract_contiguous_tensor_bytes(
        tensor_fortran, np.float32
    )
    extracted_f = np.frombuffer(raw_f, dtype=np.float32).reshape(2, 2)
    # Result must be in C-contiguous row-major order: [[1.0, 2.0], [3.0, 4.0]]
    np.testing.assert_array_equal(
        extracted_f, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )
    self.assertTrue(extracted_f.flags.c_contiguous)

  # -------------------- C. OIP Service Handler Tests --------------------

  def test_server_live_success(self):
    mock_response = triton_pb2.ServerLiveResponse(live=True)
    self.mock_runner.ServerLive.return_value = mock_response

    req = ServerLive.Request()
    res = ServerLive.Response()
    out_res = self.node.server_live(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    self.assertTrue(out_res.live)
    self.mock_runner.ServerLive.assert_called_once()

  def test_server_live_failure(self):
    self.mock_runner.ServerLive.side_effect = RuntimeError("Runner dead")

    req = ServerLive.Request()
    res = ServerLive.Response()
    out_res = self.node.server_live(req, res)

    self.assertFalse(out_res.success)
    self.assertIn("ServerLive error: Runner dead", out_res.error_message)

  def test_server_ready_success(self):
    mock_response = triton_pb2.ServerReadyResponse(ready=True)
    self.mock_runner.ServerReady.return_value = mock_response

    req = ServerReady.Request()
    res = ServerReady.Response()
    out_res = self.node.server_ready(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    self.assertTrue(out_res.ready)
    self.mock_runner.ServerReady.assert_called_once()

  def test_server_ready_failure(self):
    self.mock_runner.ServerReady.side_effect = RuntimeError("Runner not ready")

    req = ServerReady.Request()
    res = ServerReady.Response()
    out_res = self.node.server_ready(req, res)

    self.assertFalse(out_res.success)
    self.assertIn("ServerReady error: Runner not ready", out_res.error_message)

  def test_model_ready_success(self):
    mock_response = triton_pb2.ModelReadyResponse(ready=True)
    self.mock_runner.ModelReady.return_value = mock_response

    req = ModelReady.Request(model_name="densenet_onnx", model_version="1")
    res = ModelReady.Response()
    out_res = self.node.model_ready(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    self.assertTrue(out_res.ready)
    proto_req = triton_pb2.ModelReadyRequest(name="densenet_onnx", version="1")
    self.mock_runner.ModelReady.assert_called_once_with(proto_req)

  def test_model_ready_failure(self):
    self.mock_runner.ModelReady.side_effect = RuntimeError("Model not found")

    req = ModelReady.Request(model_name="densenet_onnx", model_version="1")
    res = ModelReady.Response()
    out_res = self.node.model_ready(req, res)

    self.assertFalse(out_res.success)
    self.assertIn("ModelReady error: Model not found", out_res.error_message)

  def test_server_metadata_read_only_parameters(self):
    mock_response = triton_pb2.ServerMetadataResponse(
        name="triton", version="2.34.0", extensions=["binary_tensor_data"]
    )
    self.mock_runner.ServerMetadata.return_value = mock_response
    self.mock_runner.server_state = inference_runner_lib.ServerExtendedState(
        state=inference_runner_lib.ServerState.READY, message="Server is ready"
    )

    self.node.start()

    self.assertTrue(self.node.has_parameter("server_name"))
    self.assertEqual(
        self.node.get_parameter("server_name")
        .get_parameter_value()
        .string_value,
        "triton",
    )
    self.assertTrue(self.node.has_parameter("server_version"))
    self.assertEqual(
        self.node.get_parameter("server_version")
        .get_parameter_value()
        .string_value,
        "2.34.0",
    )
    self.assertTrue(self.node.has_parameter("server_extensions"))
    self.assertEqual(
        list(
            self.node.get_parameter("server_extensions")
            .get_parameter_value()
            .string_array_value
        ),
        ["binary_tensor_data"],
    )

    # Verify read_only enforcement
    results = self.node.set_parameters([
        rclpy.parameter.Parameter(
            "server_name",
            rclpy.parameter.Parameter.Type.STRING,
            "modified_name",
        )
    ])
    self.assertFalse(results[0].successful)

  def test_model_metadata_success(self):
    mock_response = triton_pb2.ModelMetadataResponse(
        name="densenet_onnx", versions=["1"], platform="onnxruntime_onnx"
    )
    inp = mock_response.inputs.add(name="input0", datatype="FP32")
    inp.shape.extend([1, 224, 224, 3])
    out = mock_response.outputs.add(name="output0", datatype="FP32")
    out.shape.extend([1, 1000])

    self.mock_runner.ModelMetadata.return_value = mock_response

    req = ModelMetadata.Request(model_name="densenet_onnx", model_version="1")
    res = ModelMetadata.Response()
    out_res = self.node.model_metadata(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    self.assertEqual(out_res.name, "densenet_onnx")
    self.assertEqual(list(out_res.versions), ["1"])
    self.assertEqual(out_res.platform, "onnxruntime_onnx")
    self.assertEqual(len(out_res.inputs), 1)
    self.assertEqual(out_res.inputs[0].name, "input0")
    self.assertEqual(out_res.inputs[0].datatype, "FP32")
    self.assertEqual(out_res.inputs[0].dtype_code, 2)
    self.assertEqual(out_res.inputs[0].dtype_bits, 32)
    self.assertEqual(list(out_res.inputs[0].shape), [1, 224, 224, 3])
    self.assertEqual(len(out_res.outputs), 1)
    self.assertEqual(out_res.outputs[0].name, "output0")
    self.assertEqual(out_res.outputs[0].datatype, "FP32")
    self.assertEqual(out_res.outputs[0].dtype_code, 2)
    self.assertEqual(out_res.outputs[0].dtype_bits, 32)
    self.assertEqual(list(out_res.outputs[0].shape), [1, 1000])

    proto_req = triton_pb2.ModelMetadataRequest(
        name="densenet_onnx", version="1"
    )
    self.mock_runner.ModelMetadata.assert_called_once_with(proto_req)

  def test_model_metadata_failure(self):
    self.mock_runner.ModelMetadata.side_effect = RuntimeError(
        "Model metadata unavailable"
    )

    req = ModelMetadata.Request(model_name="densenet_onnx", model_version="1")
    res = ModelMetadata.Response()
    out_res = self.node.model_metadata(req, res)

    self.assertFalse(out_res.success)
    self.assertIn(
        "ModelMetadata error: Model metadata unavailable",
        out_res.error_message,
    )

  def test_model_infer_raw_outputs_success(self):
    mock_response = triton_pb2.ModelInferResponse(
        model_name="densenet_onnx",
        model_version="1",
        id="req_123",
        raw_output_contents=[
            np.array([1.5, 2.5], dtype=np.float32).tobytes(),
            np.array([10, 20], dtype=np.int64).tobytes(),
        ],
    )
    out0 = mock_response.outputs.add(name="out0", datatype="FP32")
    out0.shape.extend([1, 2])
    out1 = mock_response.outputs.add(name="out1", datatype="INT64")
    out1.shape.extend([2])

    mock_response.parameters["res_str"].string_param = "ok"
    mock_response.parameters["res_int"].int64_param = 100
    mock_response.parameters["res_double"].double_param = 2.718
    mock_response.parameters["res_bool"].bool_param = True

    self.mock_runner.ModelInfer.return_value = mock_response

    req = ModelInfer.Request(
        model_name="densenet_onnx",
        model_version="1",
        id="req_123",
        requested_outputs=["out0", "out1"],
    )
    inp_tensor = ExperimentalTensor(
        dtype_code=2,
        dtype_bits=32,
        dtype_lanes=1,
        shape=[1, 4],
        data=list(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()),
    )
    req.inputs.append(NamedTensor(name="in0", tensor=inp_tensor))
    req.parameters.append(
        Parameter(
            name="p_str",
            value=ParameterValue(
                type=ParameterType.PARAMETER_STRING,
                string_value="val",
            ),
        )
    )
    req.parameters.append(
        Parameter(
            name="p_int",
            value=ParameterValue(
                type=ParameterType.PARAMETER_INTEGER,
                integer_value=42,
            ),
        )
    )
    req.parameters.append(
        Parameter(
            name="p_double",
            value=ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE,
                double_value=3.14,
            ),
        )
    )
    req.parameters.append(
        Parameter(
            name="p_bool",
            value=ParameterValue(
                type=ParameterType.PARAMETER_BOOL,
                bool_value=True,
            ),
        )
    )

    res = ModelInfer.Response()
    out_res = self.node.model_infer(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    self.assertEqual(out_res.model_name, "densenet_onnx")
    self.assertEqual(out_res.model_version, "1")
    self.assertEqual(out_res.id, "req_123")

    self.assertEqual(len(out_res.outputs), 2)
    self.assertEqual(out_res.outputs[0].name, "out0")
    self.assertEqual(out_res.outputs[0].tensor.dtype_code, 2)
    self.assertEqual(out_res.outputs[0].tensor.dtype_bits, 32)
    self.assertEqual(list(out_res.outputs[0].tensor.shape), [1, 2])
    self.assertEqual(
        bytes(out_res.outputs[0].tensor.data),
        np.array([1.5, 2.5], dtype=np.float32).tobytes(),
    )

    self.assertEqual(out_res.outputs[1].name, "out1")
    self.assertEqual(out_res.outputs[1].tensor.dtype_code, 0)
    self.assertEqual(out_res.outputs[1].tensor.dtype_bits, 64)
    self.assertEqual(list(out_res.outputs[1].tensor.shape), [2])
    self.assertEqual(
        bytes(out_res.outputs[1].tensor.data),
        np.array([10, 20], dtype=np.int64).tobytes(),
    )

    param_dict = {p.name: (p.value.type, p.value) for p in out_res.parameters}
    self.assertEqual(param_dict["res_str"][0], ParameterType.PARAMETER_STRING)
    self.assertEqual(param_dict["res_str"][1].string_value, "ok")
    self.assertEqual(param_dict["res_int"][0], ParameterType.PARAMETER_INTEGER)
    self.assertEqual(param_dict["res_int"][1].integer_value, 100)
    self.assertEqual(
        param_dict["res_double"][0], ParameterType.PARAMETER_DOUBLE
    )
    self.assertAlmostEqual(
        param_dict["res_double"][1].double_value, 2.718, places=3
    )
    self.assertEqual(param_dict["res_bool"][0], ParameterType.PARAMETER_BOOL)
    self.assertEqual(param_dict["res_bool"][1].bool_value, True)

    self.mock_runner.ModelInfer.assert_called_once()
    called_proto = self.mock_runner.ModelInfer.call_args[0][0]
    self.assertEqual(called_proto.model_name, "densenet_onnx")
    self.assertEqual(called_proto.model_version, "1")
    self.assertEqual(called_proto.id, "req_123")
    self.assertEqual(len(called_proto.inputs), 1)
    self.assertEqual(called_proto.inputs[0].name, "in0")
    self.assertEqual(called_proto.inputs[0].datatype, "FP32")
    self.assertEqual(
        bytes(called_proto.raw_input_contents[0]),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes(),
    )
    self.assertEqual(called_proto.parameters["p_str"].string_param, "val")
    self.assertEqual(called_proto.parameters["p_int"].int64_param, 42)
    self.assertAlmostEqual(
        called_proto.parameters["p_double"].double_param, 3.14, places=2
    )
    self.assertEqual(called_proto.parameters["p_bool"].bool_param, True)

  def test_model_infer_contents_field_success(self):
    mock_response = triton_pb2.ModelInferResponse(
        model_name="densenet_onnx", model_version="1"
    )
    out0 = mock_response.outputs.add(name="out0", datatype="INT64")
    out0.shape.extend([2])
    out0.contents.int64_contents.extend([55, 66])

    self.mock_runner.ModelInfer.return_value = mock_response

    req = ModelInfer.Request(model_name="densenet_onnx", model_version="1")
    res = ModelInfer.Response()
    out_res = self.node.model_infer(req, res)

    self.assertTrue(out_res.success)
    self.assertEqual(out_res.error_message, "")
    self.assertEqual(len(out_res.outputs), 1)
    self.assertEqual(out_res.outputs[0].name, "out0")
    self.assertEqual(out_res.outputs[0].tensor.dtype_code, 0)
    self.assertEqual(out_res.outputs[0].tensor.dtype_bits, 64)
    self.assertEqual(
        bytes(out_res.outputs[0].tensor.data),
        np.array([55, 66], dtype=np.int64).tobytes(),
    )

  def test_model_infer_failure(self):
    self.mock_runner.ModelInfer.side_effect = RuntimeError("Inference failed")

    req = ModelInfer.Request(model_name="densenet_onnx")
    res = ModelInfer.Response()
    out_res = self.node.model_infer(req, res)

    self.assertFalse(out_res.success)
    self.assertIn("ModelInfer error: Inference failed", out_res.error_message)
    self.assertEqual(len(out_res.outputs), 0)


if __name__ == "__main__":
  absltest.main()
