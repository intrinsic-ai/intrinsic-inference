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

"""Utilities for creating and unpacking open inference protocol messages."""

import numpy as np
from specification.protocol import open_inference_grpc_pb2

from intrinsic_inference.core.utils import oip_mappings


def extract_np_tensor_from_oip_response(
    tensor_name: str,
    infer_response: open_inference_grpc_pb2.ModelInferResponse,
) -> np.ndarray:
  """Extracts a numpy array from an Open Inference Protocol ModelInferResponse."""
  for idx, tensor in enumerate(infer_response.outputs):
    if tensor.name == tensor_name:
      if tensor.HasField("contents"):
        np_dtype = oip_mappings.oip_to_numpy_type(tensor.datatype)
        field_name = oip_mappings.oip_type_to_field(tensor.datatype)
        return np.array(
            getattr(tensor.contents, field_name), dtype=np_dtype
        ).reshape(tensor.shape)
      return np.frombuffer(
          bytes(infer_response.raw_output_contents[idx]),
          dtype=oip_mappings.oip_to_numpy_type(tensor.datatype),
      ).reshape(tensor.shape)

  raise RuntimeError(f"Tensor {tensor_name} not found in inference response.")
