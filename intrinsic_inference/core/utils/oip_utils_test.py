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

"""Unit tests for Open Inference Protocol message unpacking utilities."""

from absl.testing import absltest
import numpy as np
from specification.protocol import open_inference_grpc_pb2

from intrinsic_inference.core.utils import oip_utils


class OIPUtilsTest(absltest.TestCase):

  def test_extract_np_tensor_from_oip_response_contents(self):
    response = open_inference_grpc_pb2.ModelInferResponse()
    tensor = response.outputs.add()
    tensor.name = "output0"
    tensor.datatype = "FP32"
    tensor.shape.extend([2, 2])
    tensor.contents.fp32_contents.extend([1.0, 2.0, 3.0, 4.0])

    result = oip_utils.extract_np_tensor_from_oip_response("output0", response)
    np.testing.assert_array_equal(
        result, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )

  def test_extract_np_tensor_from_oip_response_raw_contents(self):
    response = open_inference_grpc_pb2.ModelInferResponse()
    tensor = response.outputs.add()
    tensor.name = "output0"
    tensor.datatype = "UINT8"
    tensor.shape.extend([2, 2])
    # Raw contents for 1, 2, 3, 4 in uint8.
    response.raw_output_contents.append(b"\x01\x02\x03\x04")

    result = oip_utils.extract_np_tensor_from_oip_response("output0", response)
    np.testing.assert_array_equal(
        result, np.array([[1, 2], [3, 4]], dtype=np.uint8)
    )

  def test_extract_np_tensor_from_oip_response_not_found(self):
    response = open_inference_grpc_pb2.ModelInferResponse()
    tensor = response.outputs.add()
    tensor.name = "other_tensor"

    with self.assertRaisesRegex(
        RuntimeError, "Tensor missing_tensor not found"
    ):
      oip_utils.extract_np_tensor_from_oip_response("missing_tensor", response)


if __name__ == "__main__":
  absltest.main()
