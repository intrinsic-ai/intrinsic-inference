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

"""Tests for triton_shm_utils."""

import sys
from unittest import mock

# Hermetically mock tritonclient shared memory module to avoid loading C shared libraries in unit test env.
mock_shm = mock.MagicMock()
sys.modules["tritonclient.utils.shared_memory"] = mock_shm

from absl.testing import absltest
import numpy as np
from tritonclient.grpc import model_config_pb2
from tritonclient.grpc import service_pb2

from intrinsic_inference.core import telemetry_base
from intrinsic_inference.core import triton_shm_utils


class RawGrpcSharedMemoryPoolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_shm.reset_mock()
    mock_shm.create_shared_memory_region.side_effect = (
        lambda name, path, size: f"handle_{name}"
    )

  def test_pool_init_registers_shm_regions(self):
    mock_stub = mock.MagicMock()

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=2, byte_size=1024
    )

    self.assertEqual(pool.pool_size, 2)
    self.assertEqual(pool.byte_size, 1024)
    self.assertEqual(len(pool.all_regions), 2)
    self.assertEqual(pool.available_regions.qsize(), 2)

    # 2 regions * 2 (in & out) = 4 registrations
    self.assertEqual(mock_stub.SystemSharedMemoryRegister.call_count, 4)

  def test_acquire_and_release(self):
    mock_stub = mock.MagicMock()

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=2, byte_size=1024
    )

    region1 = pool.acquire()
    self.assertEqual(region1.id, 0)
    self.assertEqual(region1.in_name, "in_region_0")
    self.assertEqual(region1.out_name, "out_region_0")
    self.assertEqual(pool.available_regions.qsize(), 1)

    region2 = pool.acquire()
    self.assertEqual(region2.id, 1)
    self.assertEqual(pool.available_regions.qsize(), 0)

    pool.release(region1)
    self.assertEqual(pool.available_regions.qsize(), 1)

  def test_cleanup_unregisters_and_destroys(self):
    mock_stub = mock.MagicMock()

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=2, byte_size=1024
    )

    pool.cleanup()

    # 2 regions * 2 (in & out) = 4 unregistrations and 4 destroys
    self.assertEqual(mock_stub.SystemSharedMemoryUnregister.call_count, 4)
    self.assertEqual(mock_shm.destroy_shared_memory_region.call_count, 4)

  def test_context_manager_cleanup(self):
    mock_stub = mock.MagicMock()

    with triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=1, byte_size=512
    ) as pool:
      region = pool.acquire()
      self.assertEqual(region.id, 0)

    # Automatically cleaned up on context exit
    self.assertEqual(mock_stub.SystemSharedMemoryUnregister.call_count, 2)
    self.assertEqual(mock_shm.destroy_shared_memory_region.call_count, 2)

  def test_run_inference_attaches_shm_params_and_calls_stub(self):
    mock_stub = mock.MagicMock()
    mock_response = service_pb2.ModelInferResponse(model_name="my_model")
    mock_stub.ModelInfer.return_value = mock_response

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=1, byte_size=1024
    )

    request = service_pb2.ModelInferRequest(model_name="my_model")
    request.raw_input_contents.append(b"fake_bytes")
    inp = request.inputs.add()
    inp.name = "input_0"
    out = request.outputs.add()
    out.name = "output_0"

    model_config = model_config_pb2.ModelConfig(name="my_model")
    out_cfg = model_config.output.add()
    out_cfg.name = "output_0"
    out_cfg.data_type = model_config_pb2.TYPE_UINT8
    out_cfg.dims.extend([1024])

    response = triton_shm_utils.run_inference(
        request=request,
        pool=pool,
        stub=mock_stub,
        model_config=model_config,
    )

    self.assertEqual(response, mock_response)
    mock_shm.set_shared_memory_region.assert_called_once()
    handle, views = mock_shm.set_shared_memory_region.call_args.args
    self.assertEqual(handle, "handle_in_region_0")
    self.assertEqual(len(views), 1)
    np.testing.assert_array_equal(
        views[0], np.frombuffer(b"fake_bytes", dtype=np.byte)
    )
    # Inline raw_input_contents are cleared from request after copy to SHM
    self.assertEqual(len(request.raw_input_contents), 0)
    self.assertEqual(
        inp.parameters["shared_memory_region"].string_param, "in_region_0"
    )
    self.assertEqual(
        out.parameters["shared_memory_region"].string_param, "out_region_0"
    )
    mock_stub.ModelInfer.assert_called_once_with(request)
    # Region is safely released back to pool
    self.assertEqual(pool.available_regions.qsize(), 1)

  def test_run_inference_handles_empty_request_outputs(self):
    mock_stub = mock.MagicMock()
    mock_response = service_pb2.ModelInferResponse(model_name="my_model")
    mock_stub.ModelInfer.return_value = mock_response

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=1, byte_size=1024
    )

    request = service_pb2.ModelInferRequest(model_name="my_model")
    # request.outputs is intentionally left empty
    model_config = model_config_pb2.ModelConfig(name="my_model")
    out_cfg = model_config.output.add()
    out_cfg.name = "output_0"
    out_cfg.data_type = model_config_pb2.TYPE_UINT8
    out_cfg.dims.extend([1024])

    response = triton_shm_utils.run_inference(
        request=request,
        pool=pool,
        stub=mock_stub,
        model_config=model_config,
    )

    self.assertEqual(response, mock_response)
    self.assertEqual(len(request.outputs), 1)
    self.assertEqual(
        request.outputs[0].parameters["shared_memory_region"].string_param,
        "out_region_0",
    )
    self.assertEqual(
        request.outputs[0].parameters["shared_memory_byte_size"].int64_param,
        1024,
    )

  def test_run_inference_populates_outputs_from_model_config(self):
    mock_stub = mock.MagicMock()
    mock_response = service_pb2.ModelInferResponse(model_name="my_model")
    mock_stub.ModelInfer.return_value = mock_response

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=1, byte_size=2048
    )

    request = service_pb2.ModelInferRequest(model_name="my_model")

    model_config = model_config_pb2.ModelConfig(name="my_model")
    out1 = model_config.output.add()
    out1.name = "output_boxes"
    out1.data_type = model_config_pb2.TYPE_UINT8
    out1.dims.extend([1024])
    out2 = model_config.output.add()
    out2.name = "output_scores"
    out2.data_type = model_config_pb2.TYPE_UINT8
    out2.dims.extend([1024])

    response = triton_shm_utils.run_inference(
        request=request,
        pool=pool,
        stub=mock_stub,
        model_config=model_config,
    )

    self.assertEqual(response, mock_response)
    self.assertEqual(len(request.outputs), 2)
    self.assertEqual(request.outputs[0].name, "output_boxes")
    self.assertEqual(request.outputs[1].name, "output_scores")
    self.assertEqual(
        request.outputs[0].parameters["shared_memory_region"].string_param,
        "out_region_0",
    )
    self.assertEqual(
        request.outputs[1].parameters["shared_memory_region"].string_param,
        "out_region_0",
    )

  def test_run_inference_disables_shm_for_outputs(self):
    mock_stub = mock.MagicMock()
    mock_response = service_pb2.ModelInferResponse(model_name="my_model")
    mock_stub.ModelInfer.return_value = mock_response

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=1, byte_size=1024
    )

    request = service_pb2.ModelInferRequest(model_name="my_model")
    out = request.outputs.add()
    out.name = "output_0"

    model_config = model_config_pb2.ModelConfig(name="my_model")

    response = triton_shm_utils.run_inference(
        request=request,
        pool=pool,
        stub=mock_stub,
        model_config=model_config,
        use_shm_for_outputs=False,
    )

    self.assertEqual(response, mock_response)
    self.assertNotIn("shared_memory_region", out.parameters)

  def test_run_inference_records_telemetry_spans(self):
    mock_provider = mock.MagicMock(spec=telemetry_base.TelemetryProvider)
    mock_span = mock.MagicMock(spec=telemetry_base.Span)
    mock_provider.start_span.return_value = mock_span
    mock_span.__enter__.return_value = mock_span

    old_provider = telemetry_base.get_telemetry_provider()
    telemetry_base.set_telemetry_provider(mock_provider)
    try:
      mock_stub = mock.MagicMock()
      mock_response = service_pb2.ModelInferResponse(model_name="my_model")
      mock_stub.ModelInfer.return_value = mock_response

      pool = triton_shm_utils.RawGrpcSharedMemoryPool(
          stub=mock_stub, pool_size=1, byte_size=1024
      )

      request = service_pb2.ModelInferRequest(model_name="my_model")
      request.raw_input_contents.append(b"fake_bytes")
      inp = request.inputs.add()
      inp.name = "input_0"
      out = request.outputs.add()
      out.name = "output_0"

      model_config = model_config_pb2.ModelConfig(name="my_model")
      out_cfg = model_config.output.add()
      out_cfg.name = "output_0"
      out_cfg.data_type = model_config_pb2.TYPE_UINT8
      out_cfg.dims.extend([1024])

      triton_shm_utils.run_inference(
          request=request,
          pool=pool,
          stub=mock_stub,
          model_config=model_config,
      )

      started_spans = [
          call.args[0] for call in mock_provider.start_span.call_args_list
      ]
      self.assertEqual(
          started_spans,
          [
              "triton_shm_utils.setup_input_shm",
              "triton_shm_utils.read_output_shm",
          ],
      )
      mock_span.set_attribute.assert_not_called()
    finally:
      telemetry_base.set_telemetry_provider(old_provider)

  def test_run_inference_extracts_shm_outputs(self):
    mock_stub = mock.MagicMock()
    mock_response = service_pb2.ModelInferResponse(model_name="my_model")
    out_resp = mock_response.outputs.add()
    out_resp.name = "output_0"
    out_resp.datatype = "UINT8"
    out_resp.shape.extend([4])

    mock_stub.ModelInfer.return_value = mock_response

    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=mock_stub, pool_size=1, byte_size=1024
    )
    request = service_pb2.ModelInferRequest(model_name="my_model")
    model_config = model_config_pb2.ModelConfig(name="my_model")
    out_cfg = model_config.output.add()
    out_cfg.name = "output_0"
    out_cfg.data_type = model_config_pb2.TYPE_UINT8
    out_cfg.dims.extend([4])

    fake_np_data = np.array([10, 20, 30, 40], dtype=np.uint8)
    mock_shm.get_contents_as_numpy.return_value = fake_np_data

    response = triton_shm_utils.run_inference(
        request=request,
        pool=pool,
        stub=mock_stub,
        model_config=model_config,
        use_shm_for_outputs=True,
    )

    # Verify get_contents_as_numpy was called with correct arguments
    mock_shm.get_contents_as_numpy.assert_called_once_with(
        "handle_out_region_0", np.uint8, [4], 0
    )
    # Verify raw_output_contents has the extracted bytes
    self.assertEqual(len(response.raw_output_contents), 1)
    self.assertEqual(response.raw_output_contents[0], b"\n\x14\x1e(")
    # Verify shared memory parameters were stripped from response output
    self.assertNotIn("shared_memory_region", response.outputs[0].parameters)
    self.assertNotIn("shared_memory_byte_size", response.outputs[0].parameters)
    self.assertNotIn("shared_memory_offset", response.outputs[0].parameters)

  def test_read_output_from_shm_mixed_inline_and_shm(self):
    request = service_pb2.ModelInferRequest(model_name="my_model")
    # out_0: SHM
    req_out0 = request.outputs.add(name="out_0")
    req_out0.parameters["shared_memory_region"].string_param = "out_region_0"
    req_out0.parameters["shared_memory_offset"].int64_param = 0
    # out_1: inline
    request.outputs.add(name="out_1")
    # out_2: inline
    request.outputs.add(name="out_2")

    response = service_pb2.ModelInferResponse(model_name="my_model")
    resp_out0 = response.outputs.add(name="out_0", datatype="UINT8")
    resp_out0.shape.extend([2])
    resp_out0.parameters["shared_memory_region"].string_param = "out_region_0"
    response.outputs.add(name="out_1", datatype="UINT8")
    response.outputs.add(name="out_2", datatype="UINT8")
    response.raw_output_contents.extend([b"inline_1", b"inline_2"])

    fake_np_data = np.array([1, 2], dtype=np.uint8)
    mock_shm.get_contents_as_numpy.return_value = fake_np_data

    region = triton_shm_utils.SharedMemoryRegionData(
        id=0,
        in_name="in_region_0",
        out_name="out_region_0",
        in_handle="handle_in_region_0",
        out_handle="handle_out_region_0",
    )

    triton_shm_utils._read_output_from_shm(request, response, region)

    self.assertEqual(
        list(response.raw_output_contents),
        [fake_np_data.tobytes(), b"inline_1", b"inline_2"],
    )
    self.assertNotIn("shared_memory_region", resp_out0.parameters)

  def test_clear_shm_parameters(self):
    request = service_pb2.ModelInferRequest(model_name="my_model")
    inp = request.inputs.add(name="input_0")
    inp.parameters["shared_memory_region"].string_param = "in_region_0"
    inp.parameters["shared_memory_byte_size"].int64_param = 1024
    inp.parameters["shared_memory_offset"].int64_param = 0
    inp.parameters["custom_param"].string_param = "keep_me"

    out = request.outputs.add(name="output_0")
    out.parameters["shared_memory_region"].string_param = "out_region_0"
    out.parameters["shared_memory_byte_size"].int64_param = 1024
    out.parameters["shared_memory_offset"].int64_param = 0
    out.parameters["another_custom_param"].string_param = "keep_me_too"

    triton_shm_utils.clear_shm_parameters(request)

    self.assertNotIn("shared_memory_region", inp.parameters)
    self.assertNotIn("shared_memory_byte_size", inp.parameters)
    self.assertNotIn("shared_memory_offset", inp.parameters)
    self.assertIn("custom_param", inp.parameters)
    self.assertEqual(inp.parameters["custom_param"].string_param, "keep_me")

    self.assertNotIn("shared_memory_region", out.parameters)
    self.assertNotIn("shared_memory_byte_size", out.parameters)
    self.assertNotIn("shared_memory_offset", out.parameters)
    self.assertIn("another_custom_param", out.parameters)
    self.assertEqual(
        out.parameters["another_custom_param"].string_param, "keep_me_too"
    )


if __name__ == "__main__":
  absltest.main()
