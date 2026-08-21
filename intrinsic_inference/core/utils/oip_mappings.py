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

"""Utilities for converting between NumPy and Open Inference Protocol datatypes."""

from typing import Dict

import numpy as np
import numpy.typing

# Ordered base mappings, last entry becomes default for converting OIP -> Numpy.
_BASE_MAPPINGS = [
    (np.bool_, "BOOL", "bool_contents"),
    (np.int8, "INT8", "int_contents"),
    (np.int16, "INT16", "int_contents"),
    (np.int32, "INT32", "int_contents"),
    (np.int64, "INT64", "int64_contents"),
    (np.uint8, "UINT8", "uint_contents"),
    (np.uint16, "UINT16", "uint_contents"),
    (np.uint32, "UINT32", "uint_contents"),
    (np.uint64, "UINT64", "uint64_contents"),
    (np.float16, "FP16", "fp32_contents"),
    (np.float32, "FP32", "fp32_contents"),
    (np.float64, "FP64", "fp64_contents"),
    # Fixed-width bytes as input from numpy can be safely stored as BYTES as
    # defined by OIP.
    (np.bytes_, "BYTES", "bytes_contents"),
    # Variable-length bytes (dtype='O') to match OIP BYTES definition which can
    # have variable length.
    (np.object_, "BYTES", "bytes_contents"),
]

# Map: Numpy datatype object -> OIP datatype string.
_NUMPY_TO_OIP: Dict[np.dtype, str] = {
    np.dtype(np_type): oip_str for np_type, oip_str, _ in _BASE_MAPPINGS
}

# Map: OIP datatype string -> Numpy dtype object
_OIP_TO_NUMPY: Dict[str, np.dtype] = {
    oip_str: np.dtype(np_type) for np_type, oip_str, _ in _BASE_MAPPINGS
}

# Map: OIP datatype string -> Field Name
_OIP_TO_FIELD: Dict[str, str] = {
    oip_str: field for _, oip_str, field in _BASE_MAPPINGS
}


def numpy_to_oip_type(dtype: numpy.typing.DTypeLike) -> str:
  """
  Get OIP string from a numpy dtype.

  Handles both fixed-width bytes (S#) and object arrays (O) for BYTES.

  Args:
    dtype: A numpy dtype object, class, or string (e.g., 'int8', np.int8).
  Returns:
    The OIP datatype string corresponding to the given numpy datatype object.
  Raises:
    TypeError: If the dtype is not supported.
  """
  try:
    # Normalize input to a standard dtype instance.
    norm_dtype = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f"Invalid numpy dtype input: {dtype}") from e

  # Direct lookup.
  result = _NUMPY_TO_OIP.get(norm_dtype)

  # Fallback: Lookup by base type class.
  if result is None:
    result = _NUMPY_TO_OIP.get(np.dtype(norm_dtype.type))

  if result is None:
    raise TypeError(f"Unsupported OIP mapping for numpy dtype: {norm_dtype}")

  return result


def oip_to_numpy_type(oip_type_str: str) -> np.dtype:
  """
  Get numpy dtype from OIP string.

  Note: 'BYTES' returns `np.object_` to support variable length data.

  Args:
    oip_type_str: OIP datatype string (e.g. UINT32, FP64).
  Returns:
    The numpy datatype corresponding to the given OIP datatype string.
  Raises:
    ValueError: If the OIP string is unknown.
  """
  result = _OIP_TO_NUMPY.get(oip_type_str)
  if result is None:
    raise ValueError(f"Unknown OIP type string: {oip_type_str}")
  return result


def oip_type_to_field(oip_type_str: str) -> str:
  """
  Get protobuf contents field name from OIP string.

  Args:
    oip_type_str: OIP datatype string (e.g. UINT32, FP64).
  Returns:
    The OIP proto field name corresponding to the given OIP datatype string.
  Raises:
    ValueError: If the OIP string is unknown.
  """
  result = _OIP_TO_FIELD.get(oip_type_str)
  if result is None:
    raise ValueError(f"Unknown OIP type string: {oip_type_str}")
  return result
