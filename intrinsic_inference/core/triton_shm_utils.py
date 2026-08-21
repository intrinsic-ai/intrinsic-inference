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

"""Utilities for managing Triton system shared memory pools via raw gRPC."""

from __future__ import annotations

import dataclasses
import itertools
import queue
from typing import Any

from absl import logging
import numpy as np
from tritonclient.grpc import model_config_pb2
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
import tritonclient.utils as triton_utils
import tritonclient.utils.shared_memory as shm

from intrinsic_inference.core import telemetry_base

_SHM_PARAM_KEYS = (
    "shared_memory_region",
    "shared_memory_byte_size",
    "shared_memory_offset",
)

_TRITON_TYPE_ITEMSIZE = {
    model_config_pb2.TYPE_BOOL: 1,
    model_config_pb2.TYPE_UINT8: 1,
    model_config_pb2.TYPE_INT8: 1,
    model_config_pb2.TYPE_INT16: 2,
    model_config_pb2.TYPE_UINT16: 2,
    model_config_pb2.TYPE_INT32: 4,
    model_config_pb2.TYPE_UINT32: 4,
    model_config_pb2.TYPE_INT64: 8,
    model_config_pb2.TYPE_UINT64: 8,
    model_config_pb2.TYPE_FP16: 2,
    model_config_pb2.TYPE_FP32: 4,
    model_config_pb2.TYPE_FP64: 8,
    model_config_pb2.TYPE_BF16: 2,
}


def _register_shm_region(
    stub: service_pb2_grpc.GRPCInferenceServiceStub,
    name: str,
    path: str,
    byte_size: int,
) -> Any:
  """Creates an OS shared memory region and registers it with Triton."""
  handle = shm.create_shared_memory_region(name, path, byte_size)
  try:
    stub.SystemSharedMemoryRegister(
        service_pb2.SystemSharedMemoryRegisterRequest(
            name=name, key=path, offset=0, byte_size=byte_size
        )
    )
  except Exception:
    shm.destroy_shared_memory_region(handle)
    raise
  return handle


def _unregister_shm_region(
    stub: service_pb2_grpc.GRPCInferenceServiceStub,
    name: str,
    handle: Any,
) -> None:
  """Unregisters a shared memory region from Triton and destroys its OS handle."""
  try:
    stub.SystemSharedMemoryUnregister(
        service_pb2.SystemSharedMemoryUnregisterRequest(name=name)
    )
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("Failed to unregister %s: %s", name, e)

  try:
    shm.destroy_shared_memory_region(handle)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning("Failed to destroy OS SHM %s: %s", name, e)


@dataclasses.dataclass
class SharedMemoryRegionData:
  """Container for shared memory region handles and identifiers.

  Attributes:
    id: Integer index of the region in the pool.
    in_name: Name of the registered input shared memory region.
    out_name: Name of the registered output shared memory region.
    in_handle: OS shared memory handle for the input region.
    out_handle: OS shared memory handle for the output region.
  """

  id: int
  in_name: str
  out_name: str
  in_handle: Any
  out_handle: Any


class RawGrpcSharedMemoryPool:
  """Pool of pre-allocated system shared memory regions registered with Triton.

  Attributes:
    stub: gRPC stub for Triton's GRPCInferenceService.
    pool_size: Number of shared memory region pairs in the pool.
    byte_size: Size in bytes of each shared memory region.
  """

  def __init__(
      self,
      stub: service_pb2_grpc.GRPCInferenceServiceStub,
      pool_size: int,
      byte_size: int,
  ) -> None:
    """Initializes and registers system shared memory regions with Triton.

    Args:
      stub: gRPC stub for communicating with Triton inference server.
      pool_size: Number of shared memory region pairs to allocate.
      byte_size: Size in bytes of each input and output region.
    """
    self.stub = stub
    self.pool_size = pool_size
    self.byte_size = byte_size

    self.available_regions: queue.Queue[SharedMemoryRegionData] = queue.Queue()
    self.all_regions: list[SharedMemoryRegionData] = []

    logging.info("Initializing raw gRPC pool with %d regions...", pool_size)
    try:
      for i in range(pool_size):
        in_name = f"in_region_{i}"
        out_name = f"out_region_{i}"

        in_handle = _register_shm_region(
            stub, in_name, f"/in_shm_{i}", byte_size
        )
        try:
          out_handle = _register_shm_region(
              stub, out_name, f"/out_shm_{i}", byte_size
          )
        except Exception:
          _unregister_shm_region(stub, in_name, in_handle)
          raise

        region_data = SharedMemoryRegionData(
            id=i,
            in_name=in_name,
            out_name=out_name,
            in_handle=in_handle,
            out_handle=out_handle,
        )
        self.all_regions.append(region_data)
        self.available_regions.put(region_data)
    except Exception:
      self.cleanup()
      raise

  def acquire(
      self, block: bool = True, timeout: float | None = None
  ) -> SharedMemoryRegionData:
    """Acquires an available shared memory region from the pool.

    Args:
      block: Whether to block until a region becomes available.
      timeout: Optional timeout in seconds when blocking.

    Returns:
      A SharedMemoryRegionData object representing the acquired region.
    """
    return self.available_regions.get(block=block, timeout=timeout)

  def release(self, region_data: SharedMemoryRegionData) -> None:
    """Releases a shared memory region back into the pool.

    Args:
      region_data: The SharedMemoryRegionData object to return.
    """
    self.available_regions.put(region_data)

  def cleanup(self) -> None:
    """Unregisters all shared memory regions from Triton and frees OS resources."""
    logging.info("Cleaning up raw gRPC shared memory pool...")
    for r in self.all_regions:
      _unregister_shm_region(self.stub, r.in_name, r.in_handle)
      _unregister_shm_region(self.stub, r.out_name, r.out_handle)

  def __enter__(self) -> RawGrpcSharedMemoryPool:
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.cleanup()


def _set_shm_params(
    tensor: Any, region_name: str, byte_size: int, offset: int
) -> None:
  """Populates shared memory parameters on a request tensor."""
  tensor.parameters["shared_memory_region"].string_param = region_name
  tensor.parameters["shared_memory_byte_size"].int64_param = byte_size
  tensor.parameters["shared_memory_offset"].int64_param = offset


def _clear_shm_params(parameters: Any) -> None:
  """Removes shared memory parameters from a parameter map."""
  for param_name in _SHM_PARAM_KEYS:
    parameters.pop(param_name, None)


def _get_tensor_num_elements(
    tensor_cfg: Any, batch_size: int, max_batch_size: int = 0
) -> int:
  """Returns num of elements in tensor if all dim defined, else returns -1."""
  dims = getattr(tensor_cfg, "dims", None)
  if not dims:
    return 0
  num_elements = 1
  if max_batch_size > 0:
    num_elements *= batch_size

  for dim_index, dim_value in enumerate(dims):
    # Batch dim case.
    if max_batch_size == 0 and dim_index == 0 and dim_value < 1:
      num_elements *= batch_size
      continue
    # Other dim.
    if dim_value < 1:
      return -1
    num_elements *= dim_value
  return num_elements


def _get_tensor_element_size(tensor_cfg: Any) -> int:
  """Returns element size in tensor."""
  dtype = getattr(tensor_cfg, "data_type", None)
  if dtype is None:
    logging.debug("No dtype available in tensor %s", tensor_cfg)
    return 0

  itemsize = _TRITON_TYPE_ITEMSIZE.get(dtype, 0)
  if not itemsize:
    logging.debug("Unrecognized dtype in tensor %s", tensor_cfg)
  return itemsize


def _compute_output_byte_size(
    tensor_cfg: Any, batch_size: int, max_batch_size: int = 0
) -> int | None:
  """Returns the total byte size of an output tensor or None if dynamic shape or unknown dtype."""
  num_elements = _get_tensor_num_elements(
      tensor_cfg, batch_size, max_batch_size
  )
  if num_elements < 1:
    logging.debug(
        "Dynamic shape output in output tensor %s, skipping shared memory"
        " usage.",
        tensor_cfg,
    )
    return None
  element_size = _get_tensor_element_size(tensor_cfg)
  if element_size < 1:
    logging.debug(
        "Invalid element size in output tensor %s, skipping shared memory"
        " usage.",
        tensor_cfg,
    )
    return None
  return num_elements * element_size


@telemetry_base.trace_span("triton_shm_utils.setup_input_shm")
def _setup_input_shm_params(
    request: service_pb2.ModelInferRequest,
    region: SharedMemoryRegionData,
    max_capacity: int,
) -> None:
  """Writes raw inputs to shared memory and sets SHM parameters on request inputs."""
  if len(request.raw_input_contents) != len(request.inputs):
    raise ValueError(
        f"Mismatch between number of inputs ({len(request.inputs)}) and"
        f" raw_input_contents ({len(request.raw_input_contents)})."
    )
  zero_copy_views = []
  current_offset = 0

  for i, input_tensor in enumerate(request.inputs):
    raw_bytes = request.raw_input_contents[i]
    actual_byte_size = len(raw_bytes)
    if current_offset + actual_byte_size > max_capacity:
      raise ValueError(
          f"Input tensor '{input_tensor.name}' byte offset"
          f" ({current_offset + actual_byte_size}) exceeds input shared"
          f" memory capacity ({max_capacity})."
      )

    zero_copy_views.append(np.frombuffer(raw_bytes, dtype=np.byte))
    _set_shm_params(
        input_tensor, region.in_name, actual_byte_size, current_offset
    )
    current_offset += actual_byte_size

  shm.set_shared_memory_region(region.in_handle, zero_copy_views)
  request.ClearField("raw_input_contents")


def _setup_output_shm_params(
    request: service_pb2.ModelInferRequest,
    region: SharedMemoryRegionData,
    model_config: model_config_pb2.ModelConfig,
    max_capacity: int,
    use_shm_for_outputs: bool = True,
) -> None:
  """Populates outputs if empty and sets SHM parameters on request outputs if enabled."""
  if not use_shm_for_outputs:
    logging.debug("Shared memory for outputs not enabled.")
    return

  if not request.outputs and not model_config.output:
    logging.warning("No outputs set in request nor in model config.")
    return

  if not request.outputs:
    for tensor_cfg in model_config.output:
      request.outputs.add(name=tensor_cfg.name)

  batch_size = (
      request.inputs[0].shape[0]
      if (request.inputs and request.inputs[0].shape)
      else 1
  )

  max_batch_size = getattr(model_config, "max_batch_size", 0)

  output_byte_size_map: dict[str, int] = {}
  for tensor_cfg in model_config.output:
    byte_size = _compute_output_byte_size(
        tensor_cfg, batch_size, max_batch_size
    )
    if byte_size is None:
      return
    output_byte_size_map[tensor_cfg.name] = byte_size

  offset = 0
  for out in request.outputs:
    byte_size = output_byte_size_map.get(out.name)
    if byte_size is None:
      continue
    if offset + byte_size > max_capacity:
      raise ValueError(
          f"Output tensor '{out.name}' byte offset ({offset + byte_size})"
          f" exceeds output shared memory capacity ({max_capacity})."
      )
    _set_shm_params(out, region.out_name, byte_size, offset)
    offset += byte_size


@telemetry_base.trace_span("triton_shm_utils.read_output_shm")
def _read_output_from_shm(
    request: service_pb2.ModelInferRequest,
    response: service_pb2.ModelInferResponse,
    region: SharedMemoryRegionData,
) -> None:
  """Extracts output tensor contents from system shared memory into response."""
  if not response.outputs or not request.outputs:
    return

  req_outputs_by_name = {out.name: out for out in request.outputs}
  if not any(
      "shared_memory_region" in out.parameters for out in request.outputs
  ):
    return

  new_raw_contents = []
  inline_idx = 0
  for out_resp in response.outputs:
    req_out = req_outputs_by_name.get(out_resp.name)
    if req_out and "shared_memory_region" in req_out.parameters:
      offset = req_out.parameters["shared_memory_offset"].int64_param
      dtype = triton_utils.triton_to_np_dtype(out_resp.datatype)
      arr = shm.get_contents_as_numpy(
          region.out_handle, dtype, list(out_resp.shape), offset
      )
      new_raw_contents.append(arr.tobytes())
      _clear_shm_params(out_resp.parameters)
    elif inline_idx < len(response.raw_output_contents):
      new_raw_contents.append(response.raw_output_contents[inline_idx])
      inline_idx += 1
    else:
      new_raw_contents.append(b"")

  del response.raw_output_contents[:]
  response.raw_output_contents.extend(new_raw_contents)


def clear_shm_parameters(request: service_pb2.ModelInferRequest) -> None:
  """Removes shared memory parameters from request inputs and outputs."""
  for tensor in itertools.chain(request.inputs, request.outputs):
    for param_name in [
        "shared_memory_region",
        "shared_memory_byte_size",
        "shared_memory_offset",
    ]:
      tensor.parameters.pop(param_name, None)


def run_inference(
    request: service_pb2.ModelInferRequest,
    pool: RawGrpcSharedMemoryPool,
    stub: service_pb2_grpc.GRPCInferenceServiceStub,
    model_config: model_config_pb2.ModelConfig,
    use_shm_for_outputs: bool = True,
) -> service_pb2.ModelInferResponse:
  """Executes a Triton gRPC ModelInferRequest using system shared memory.

  Acquires a shared memory region pair from the pool, extracts raw_input_contents
  from the request, writes them into the input shared memory region, attaches
  shared memory parameters to request inputs (and optionally outputs), clears
  raw_input_contents from the request, calls ModelInfer, and releases the region
  back to the pool.

  Args:
    request: The ModelInferRequest proto instance defining the model and tensor metadata.
    pool: Pre-allocated RawGrpcSharedMemoryPool instance.
    stub: gRPC stub for Triton's GRPCInferenceService.
    model_config: ModelConfig proto defining expected input/output model specs.
    use_shm_for_outputs: Whether to write outputs to system shared memory (True) or
      return them inline via standard gRPC response (False). Defaults to True.

  Returns:
    The ModelInferResponse proto returned by Triton.
  """
  try:
    region = pool.acquire(block=False)
  except queue.Empty as e:
    # pool.acquire uses a standard queue so we can check for this exception.
    raise RuntimeError("Failed to acquire Triton SHM region: pool empty") from e
  try:
    _setup_input_shm_params(
        request=request,
        region=region,
        max_capacity=pool.byte_size,
    )
    _setup_output_shm_params(
        request=request,
        region=region,
        model_config=model_config,
        max_capacity=pool.byte_size,
        use_shm_for_outputs=use_shm_for_outputs,
    )

    logging.debug(
        "Running gRPC ModelInfer for model '%s' using SHM region %s...",
        request.model_name,
        region.in_name,
    )
    response = stub.ModelInfer(request)
    _read_output_from_shm(request=request, response=response, region=region)
    return response
  finally:
    pool.release(region)
