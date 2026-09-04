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

This node embeds InferenceRunner directly and exposes OpenInferenceProtocol (OIP)
methods (ServerLive, ServerReady, ModelReady, ModelMetadata, ModelInfer)
as ROS 2 services and custom interface definitions.
"""

from __future__ import annotations

import dataclasses
import enum
import threading

from absl import logging
import numpy as np
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterValue
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor
from tritonclient.grpc import service_pb2 as triton_pb2

from intrinsic_inference.core import inference_runner as inference_runner_lib
from intrinsic_inference.core.utils import oip_mappings
from intrinsic_inference.ros.inference_interfaces.msg import NamedTensor
from intrinsic_inference.ros.inference_interfaces.msg import TensorMetadata
from intrinsic_inference.ros.inference_interfaces.srv import ModelInfer
from intrinsic_inference.ros.inference_interfaces.srv import ModelMetadata
from intrinsic_inference.ros.inference_interfaces.srv import ModelReady
from intrinsic_inference.ros.inference_interfaces.srv import ServerLive
from intrinsic_inference.ros.inference_interfaces.srv import ServerReady

# List of triton-supported tensor datatypes for both input and output:
# - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes
# List of DLPack-supported tensor datatypes:
# - https://github.com/ros2/rosidl_buffer_backends/blob/main/tensor_msgs/msg/ExperimentalTensor.msg
_TRITON_TO_DLPACK_DTYPE = {
    "BOOL": (6, 8),
    "INT8": (0, 8),
    "INT16": (0, 16),
    "INT32": (0, 32),
    "INT64": (0, 64),
    "UINT8": (1, 8),
    "UINT16": (1, 16),
    "UINT32": (1, 32),
    "UINT64": (1, 64),
    "FP16": (2, 16),
    "FP32": (2, 32),
    "FP64": (2, 64),
    "BF16": (4, 16),
}

_DLPACK_TO_TRITON_DTYPE = {
    dlpack: triton for triton, dlpack in _TRITON_TO_DLPACK_DTYPE.items()
}


def triton_to_dlpack_dtype(triton_dtype: str) -> tuple[int, int]:
  """Converts Triton datatype string to DLPack (dtype_code, dtype_bits)."""
  if triton_dtype not in _TRITON_TO_DLPACK_DTYPE:
    raise ValueError(f"Unsupported Triton datatype: {triton_dtype}")

  return _TRITON_TO_DLPACK_DTYPE[triton_dtype]


def dlpack_to_triton_dtype(dtype_code: int, dtype_bits: int) -> str:
  """Converts DLPack (dtype_code, dtype_bits) to a Triton datatype string."""
  key = (dtype_code, dtype_bits)
  if key not in _DLPACK_TO_TRITON_DTYPE:
    raise ValueError(
        f"Unsupported DLPack datatype: dtype_code={dtype_code},"
        f" dtype_bits={dtype_bits}"
    )

  return _DLPACK_TO_TRITON_DTYPE[key]


def extract_contiguous_tensor_bytes(
    tensor: ExperimentalTensor, np_dtype: np.dtype
) -> bytes:
  """Extracts contiguous row-major bytes from an ExperimentalTensor.

  Handles non-zero byte_offset and strided / non-contiguous layouts.
  """
  data_bytes = bytes(tensor.data)
  offset = getattr(tensor, "byte_offset", 0)
  strides = getattr(tensor, "strides", None)

  # Already contiguous
  if not strides:
    return data_bytes[offset:] if offset else data_bytes

  # Handle non-contiguous / strided path
  if offset:
    data_bytes = data_bytes[offset:]

  np_dtype = np.dtype(np_dtype)
  # DLPack strides are given in element counts. Convert to byte strides for NumPy.
  byte_strides = tuple(s * np_dtype.itemsize for s in strides)
  arr = np.lib.stride_tricks.as_strided(
      np.frombuffer(data_bytes, dtype=np_dtype),
      shape=tuple(tensor.shape),
      strides=byte_strides,
  )

  # Ensure contiguous row-major layout as required by Open Inference Protocol (OIP).
  return np.ascontiguousarray(arr).tobytes()


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

    # Expose OIP ROS 2 service interfaces.
    self._server_live_srv = self.create_service(
        ServerLive,
        "~/server_live",
        self.server_live,
    )
    self._server_ready_srv = self.create_service(
        ServerReady,
        "~/server_ready",
        self.server_ready,
    )
    self._model_ready_srv = self.create_service(
        ModelReady,
        "~/model_ready",
        self.model_ready,
    )
    self._model_metadata_srv = self.create_service(
        ModelMetadata,
        "~/model_metadata",
        self.model_metadata,
    )
    self._model_infer_srv = self.create_service(
        ModelInfer,
        "~/model_infer",
        self.model_infer,
    )

    self.get_logger().info(
        "InferenceNode created and exposing OIP ROS 2 services: "
        "/ServerLive, /ServerReady, /ModelReady, /ModelMetadata, /ModelInfer"
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

  def _declare_server_metadata_parameters(self) -> None:
    """Queries ServerMetadata from inference runner and declares read-only ROS 2 parameters."""
    try:
      proto_req = triton_pb2.ServerMetadataRequest()
      metadata = self.inference_runner.ServerMetadata(proto_req)
      if not self.has_parameter("server_name"):
        self.declare_parameter(
            "server_name",
            metadata.name,
            ParameterDescriptor(
                name="server_name",
                type=ParameterType.PARAMETER_STRING,
                description="Inference backend server name",
                read_only=True,
            ),
        )
      if not self.has_parameter("server_version"):
        self.declare_parameter(
            "server_version",
            metadata.version,
            ParameterDescriptor(
                name="server_version",
                type=ParameterType.PARAMETER_STRING,
                description="Inference backend server version",
                read_only=True,
            ),
        )
      if not self.has_parameter("server_extensions"):
        self.declare_parameter(
            "server_extensions",
            list(metadata.extensions),
            ParameterDescriptor(
                name="server_extensions",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Inference backend supported extensions",
                read_only=True,
            ),
        )
    except Exception as e:
      self.get_logger().error(
          f"Failed to declare server metadata parameters: {e}"
      )

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
          self._declare_server_metadata_parameters()
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
      request: ServerLive.Request,
      response: ServerLive.Response,
  ) -> ServerLive.Response:
    """ROS service handler for ServerLive query."""
    try:
      proto_req = triton_pb2.ServerLiveRequest()
      proto_res = self.inference_runner.ServerLive(proto_req)
      response.live = proto_res.live
      response.success = True
      response.error_message = ""
    except Exception as e:
      response.success = False
      response.error_message = f"ServerLive error: {e}"
      self.get_logger().error(response.error_message)
    return response

  def server_ready(
      self,
      request: ServerReady.Request,
      response: ServerReady.Response,
  ) -> ServerReady.Response:
    """ROS service handler for ServerReady query."""
    try:
      proto_req = triton_pb2.ServerReadyRequest()
      proto_res = self.inference_runner.ServerReady(proto_req)
      response.ready = proto_res.ready
      response.success = True
      response.error_message = ""
    except Exception as e:
      response.success = False
      response.error_message = f"ServerReady error: {e}"
      self.get_logger().error(response.error_message)
    return response

  def model_ready(
      self,
      request: ModelReady.Request,
      response: ModelReady.Response,
  ) -> ModelReady.Response:
    """ROS service handler for ModelReady query."""
    try:
      proto_req = triton_pb2.ModelReadyRequest(
          name=request.model_name,
          version=request.model_version,
      )
      proto_res = self.inference_runner.ModelReady(proto_req)
      response.ready = proto_res.ready
      response.success = True
      response.error_message = ""
    except Exception as e:
      response.success = False
      response.error_message = f"ModelReady error: {e}"
      self.get_logger().error(response.error_message)
    return response

  def model_metadata(
      self,
      request: ModelMetadata.Request,
      response: ModelMetadata.Response,
  ) -> ModelMetadata.Response:
    """ROS service handler for ModelMetadata query."""
    try:
      proto_req = triton_pb2.ModelMetadataRequest(
          name=request.model_name,
          version=request.model_version,
      )
      proto_res = self.inference_runner.ModelMetadata(proto_req)
      response.name = proto_res.name
      response.versions = list(proto_res.versions)
      response.platform = proto_res.platform

      # We return both OIP string datatype and DLPack code/bits for readability
      # and compatibility/ease-of-use with DLPack (tensor_msgs/msg/ExperimentalTensor).
      response.inputs = []
      for inp in proto_res.inputs:
        code, bits = triton_to_dlpack_dtype(inp.datatype)
        response.inputs.append(
            TensorMetadata(
                name=inp.name,
                datatype=inp.datatype,
                dtype_code=code,
                dtype_bits=bits,
                shape=list(inp.shape),
            )
        )
      response.outputs = []
      for out in proto_res.outputs:
        code, bits = triton_to_dlpack_dtype(out.datatype)
        response.outputs.append(
            TensorMetadata(
                name=out.name,
                datatype=out.datatype,
                dtype_code=code,
                dtype_bits=bits,
                shape=list(out.shape),
            )
        )
      response.success = True
      response.error_message = ""
    except Exception as e:
      response.success = False
      response.error_message = f"ModelMetadata error: {e}"
      self.get_logger().error(response.error_message)
    return response

  def model_infer(
      self,
      request: ModelInfer.Request,
      response: ModelInfer.Response,
  ) -> ModelInfer.Response:
    """ROS service handler for ModelInfer execution."""
    try:
      proto_req = triton_pb2.ModelInferRequest(
          model_name=request.model_name,
          model_version=request.model_version,
          id=request.id,
      )
      for inp in request.inputs:
        triton_dtype = dlpack_to_triton_dtype(
            inp.tensor.dtype_code, inp.tensor.dtype_bits
        )
        proto_req.inputs.add(
            name=inp.name,
            datatype=triton_dtype,
            shape=list(inp.tensor.shape),
        )
        np_dtype = oip_mappings.oip_to_numpy_type(triton_dtype)
        raw_bytes = extract_contiguous_tensor_bytes(inp.tensor, np_dtype)
        proto_req.raw_input_contents.append(raw_bytes)

      for out_name in request.requested_outputs:
        proto_req.outputs.add(name=out_name)

      for param in request.parameters:
        val = param.value
        if val.type == ParameterType.PARAMETER_BOOL:
          proto_req.parameters[param.name].bool_param = val.bool_value
        elif val.type == ParameterType.PARAMETER_INTEGER:
          proto_req.parameters[param.name].int64_param = val.integer_value
        elif val.type == ParameterType.PARAMETER_DOUBLE:
          proto_req.parameters[param.name].double_param = val.double_value
        elif val.type == ParameterType.PARAMETER_STRING:
          proto_req.parameters[param.name].string_param = val.string_value
        else:
          raise ValueError(
              f"Unsupported Parameter datatype '{val.type}' for parameter"
              f" '{param.name}'. Must be one of: PARAMETER_BOOL,"
              " PARAMETER_INTEGER, PARAMETER_DOUBLE, PARAMETER_STRING."
          )

      proto_res = self.inference_runner.ModelInfer(proto_req)
      response.model_name = proto_res.model_name
      response.model_version = proto_res.model_version
      response.id = proto_res.id

      use_raw_outputs = bool(proto_res.raw_output_contents)
      for idx, out_proto in enumerate(proto_res.outputs):
        code, bits = triton_to_dlpack_dtype(out_proto.datatype)
        if use_raw_outputs:
          raw_bytes = proto_res.raw_output_contents[idx]
        else:
          field_name = oip_mappings.oip_type_to_field(out_proto.datatype)
          np_dtype = oip_mappings.oip_to_numpy_type(out_proto.datatype)
          raw_bytes = np.array(
              getattr(out_proto.contents, field_name), dtype=np_dtype
          ).tobytes()

        # SIMD vector lane count per element (1 for standard scalar deep-learning tensors)
        lanes = 1
        exp_tensor = ExperimentalTensor(
            dtype_code=code,
            dtype_bits=bits,
            dtype_lanes=lanes,
            shape=list(out_proto.shape),
            data=raw_bytes,
        )
        response.outputs.append(
            NamedTensor(
                name=out_proto.name,
                tensor=exp_tensor,
            )
        )

      for key, param_val in proto_res.parameters.items():
        if param_val.HasField("bool_param"):
          param_value = ParameterValue(
              type=ParameterType.PARAMETER_BOOL,
              bool_value=param_val.bool_param,
          )
        elif param_val.HasField("int64_param"):
          param_value = ParameterValue(
              type=ParameterType.PARAMETER_INTEGER,
              integer_value=param_val.int64_param,
          )
        elif param_val.HasField("double_param"):
          param_value = ParameterValue(
              type=ParameterType.PARAMETER_DOUBLE,
              double_value=param_val.double_param,
          )
        elif param_val.HasField("string_param"):
          param_value = ParameterValue(
              type=ParameterType.PARAMETER_STRING,
              string_value=param_val.string_param,
          )
        elif param_val.HasField("uint64_param"):
          param_value = ParameterValue(
              type=ParameterType.PARAMETER_INTEGER,
              integer_value=param_val.uint64_param,
          )
        else:
          param_value = ParameterValue(type=ParameterType.PARAMETER_NOT_SET)

        response.parameters.append(Parameter(name=key, value=param_value))

      response.success = True
      response.error_message = ""
    except Exception as e:
      response.success = False
      response.error_message = f"ModelInfer error: {e}"
      self.get_logger().error(response.error_message)
    return response
