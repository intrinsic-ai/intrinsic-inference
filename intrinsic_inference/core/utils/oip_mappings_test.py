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

"""Unit tests for NumPy and Open Inference Protocol datatype mappings."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from intrinsic_inference.core.utils import oip_mappings


class OIPMappingsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("bool", np.bool_, "BOOL"),
      ("int8", np.int8, "INT8"),
      ("int16", np.int16, "INT16"),
      ("int32", np.int32, "INT32"),
      ("int64", np.int64, "INT64"),
      ("uint8", np.uint8, "UINT8"),
      ("uint16", np.uint16, "UINT16"),
      ("uint32", np.uint32, "UINT32"),
      ("uint64", np.uint64, "UINT64"),
      ("float16", np.float16, "FP16"),
      ("float32", np.float32, "FP32"),
      ("float64", np.float64, "FP64"),
      ("bytes", np.bytes_, "BYTES"),
      ("object", np.object_, "BYTES"),
      ("str_int8", "int8", "INT8"),
      ("dtype_int8", np.dtype("int8"), "INT8"),
  )
  def test_numpy_to_oip_type(self, np_type, expected_oip_type):
    self.assertEqual(oip_mappings.numpy_to_oip_type(np_type), expected_oip_type)

  def test_numpy_to_oip_type_invalid(self):
    with self.assertRaises(TypeError):
      oip_mappings.numpy_to_oip_type(np.complex128)
    with self.assertRaises(TypeError):
      oip_mappings.numpy_to_oip_type("invalid_type")

  @parameterized.named_parameters(
      ("BOOL", "BOOL", np.bool_),
      ("INT8", "INT8", np.int8),
      ("INT16", "INT16", np.int16),
      ("INT32", "INT32", np.int32),
      ("INT64", "INT64", np.int64),
      ("UINT8", "UINT8", np.uint8),
      ("UINT16", "UINT16", np.uint16),
      ("UINT32", "UINT32", np.uint32),
      ("UINT64", "UINT64", np.uint64),
      ("FP16", "FP16", np.float16),
      ("FP32", "FP32", np.float32),
      ("FP64", "FP64", np.float64),
      ("BYTES", "BYTES", np.object_),
  )
  def test_oip_to_numpy_type(self, oip_type, expected_np_type):
    self.assertEqual(oip_mappings.oip_to_numpy_type(oip_type), expected_np_type)

  def test_oip_to_numpy_type_invalid(self):
    with self.assertRaises(ValueError):
      oip_mappings.oip_to_numpy_type("INVALID")

  @parameterized.named_parameters(
      ("BOOL", "BOOL", "bool_contents"),
      ("INT8", "INT8", "int_contents"),
      ("UINT8", "UINT8", "uint_contents"),
      ("FP32", "FP32", "fp32_contents"),
      ("BYTES", "BYTES", "bytes_contents"),
  )
  def test_oip_type_to_field(self, oip_type, expected_field):
    self.assertEqual(oip_mappings.oip_type_to_field(oip_type), expected_field)

  def test_oip_type_to_field_invalid(self):
    with self.assertRaises(ValueError):
      oip_mappings.oip_type_to_field("INVALID")


if __name__ == "__main__":
  absltest.main()
