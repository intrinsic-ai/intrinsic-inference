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

"""Tests for model_controller_triton."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
import grpc
from tritonclient.grpc import model_config_pb2

from intrinsic_inference.core import model_assets_manager_base
from intrinsic_inference.core import model_controller_triton
from intrinsic_inference.core.v1 import ml_model_pb2


class ModelControllerTritonTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.repo_path = self.enter_context(tempfile.TemporaryDirectory())

    # Mock the ModelAssetsManager dependency.
    self.mock_model_assets_manager = mock.MagicMock(
        spec=model_assets_manager_base.ModelAssetsManagerBase
    )
    self.mock_triton_stub = mock.MagicMock()

    self.controller = model_controller_triton.ModelControllerTriton(
        repo_path=self.repo_path,
        model_assets_manager=self.mock_model_assets_manager,
        triton_stub=self.mock_triton_stub,
    )

  def test_init_succeeds(self) -> None:
    self.assertIsNotNone(self.controller)
    self.assertEqual(
        self.controller._model_assets_manager, self.mock_model_assets_manager
    )  # pylint: disable=protected-access
    self.assertEqual(
        self.controller._triton_stub, self.mock_triton_stub
    )  # pylint: disable=protected-access
    self.assertEqual(self.controller.models, {})

  def test_reconcile_load_model_succeeds(self) -> None:
    model = ml_model_pb2.MlModel()
    model.model_config.name = "com.example.test_model"
    triton_config = model_config_pb2.ModelConfig(
        name="com.example.test_model",
        max_batch_size=8,
    )
    model.backend_config.Pack(triton_config)

    # Mock provider to return this model when listed.
    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.test_model.v1": model
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify Triton config was written locally.
    expected_config_path = os.path.join(
        self.repo_path, "com.example.test_model", "config.pbtxt"
    )
    self.assertTrue(os.path.exists(expected_config_path))
    with open(expected_config_path, "r", encoding="utf-8") as f:
      content = f.read()
      self.assertIn('name: "com.example.test_model"', content)
      self.assertIn("max_batch_size: 8", content)

    # Verify model assets provider was called to create the asset.
    self.mock_model_assets_manager.create_model_asset.assert_called_once_with(
        model
    )

    # Verify Triton was called to load the model.
    self.mock_triton_stub.RepositoryModelLoad.assert_called_once()
    request = self.mock_triton_stub.RepositoryModelLoad.call_args[0][0]
    self.assertEqual(request.model_name, "com.example.test_model")

    # Verify controller.models updated.
    self.assertIn("com.example.test_model.v1", self.controller.models)
    self.assertEqual(self.controller.models["com.example.test_model.v1"], model)

  def test_reconcile_unload_model_succeeds(self) -> None:
    # Pre-populate controller.models.
    model = ml_model_pb2.MlModel()
    model.model_config.name = "com.example.test_model"
    self.controller._models = {"com.example.test_model.v1": model}  # pylint: disable=protected-access

    # Reconcile to empty state (no models installed).
    self.mock_model_assets_manager.list_model_assets.return_value = {}

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify Triton was called to unload the model.
    self.mock_triton_stub.RepositoryModelUnload.assert_called_once()
    request = self.mock_triton_stub.RepositoryModelUnload.call_args[0][0]
    self.assertEqual(request.model_name, "com.example.test_model")

    # Verify model assets provider was called to delete the asset.
    self.mock_model_assets_manager.delete_model_asset.assert_called_once_with(
        model
    )
    self.assertNotIn("com.example.test_model.v1", self.controller.models)

  def test_reconcile_reload_model_succeeds(self) -> None:
    # Setup old model in controller.
    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "com.example.test_model"
    old_model.model_data["model.onnx"].reference = "gs://12345"
    triton_config = model_config_pb2.ModelConfig(name="com.example.test_model")
    old_model.backend_config.Pack(triton_config)
    self.controller._models = {"com.example.test_model.v1": old_model}  # pylint: disable=protected-access

    # New model (different reference -> full reload).
    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "com.example.test_model"
    new_model.model_data["model.onnx"].reference = "gs://67890"
    triton_config = model_config_pb2.ModelConfig(name="com.example.test_model")
    new_model.backend_config.Pack(triton_config)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.test_model.v1": new_model
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify Triton was NOT called to unload (reload handles it internally).
    self.mock_triton_stub.RepositoryModelUnload.assert_not_called()

    # Verify model assets provider was NOT called to delete (differential update is used instead).
    self.mock_model_assets_manager.delete_model_asset.assert_not_called()

    # Verify Triton config was written.
    expected_config_path = os.path.join(
        self.repo_path, "com.example.test_model", "config.pbtxt"
    )
    self.assertTrue(os.path.exists(expected_config_path))

    # Verify model assets provider was called to update the asset.
    self.mock_model_assets_manager.update_model_asset.assert_called_once_with(
        old_model, new_model
    )

    # Verify Triton was called to load the model.
    self.mock_triton_stub.RepositoryModelLoad.assert_called_once()
    load_request = self.mock_triton_stub.RepositoryModelLoad.call_args[0][0]
    self.assertEqual(load_request.model_name, "com.example.test_model")

    self.assertEqual(
        self.controller.models["com.example.test_model.v1"], new_model
    )

  def test_reconcile_reload_model_config_only_succeeds(self) -> None:
    # Setup old model in controller.
    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "com.example.test_model"
    old_model.model_data["model.onnx"].reference = "gs://12345"
    triton_config = model_config_pb2.ModelConfig(
        name="com.example.test_model",
        max_batch_size=8,
    )
    old_model.backend_config.Pack(triton_config)
    self.controller._models = {"com.example.test_model.v1": old_model}  # pylint: disable=protected-access

    # New model (only config changed).
    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "com.example.test_model"
    new_model.model_data["model.onnx"].reference = (
        "gs://12345"  # Same reference.
    )
    triton_config = model_config_pb2.ModelConfig(
        name="com.example.test_model",
        max_batch_size=16,
    )
    new_model.backend_config.Pack(triton_config)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.test_model.v1": new_model
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify Triton unload was not called.
    self.mock_triton_stub.RepositoryModelUnload.assert_not_called()

    # Verify model assets provider delete was not called.
    self.mock_model_assets_manager.delete_model_asset.assert_not_called()

    # Verify model assets provider update was not called (config-only reload).
    self.mock_model_assets_manager.update_model_asset.assert_not_called()

    # Verify config was written.
    expected_config_path = os.path.join(
        self.repo_path, "com.example.test_model", "config.pbtxt"
    )
    self.assertTrue(os.path.exists(expected_config_path))

    # Verify Triton was called to load the model (triggers reload).
    self.mock_triton_stub.RepositoryModelLoad.assert_called_once()
    load_request = self.mock_triton_stub.RepositoryModelLoad.call_args[0][0]
    self.assertEqual(load_request.model_name, "com.example.test_model")

  def test_reconcile_load_model_empty_config_file_exists_succeeds(self) -> None:
    model = ml_model_pb2.MlModel()
    model.model_config.name = "com.example.test_model"

    # Pre-create config.pbtxt on disk to simulate local repo case.
    model_dir = os.path.join(self.repo_path, "com.example.test_model")
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write("dummy config content")

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.test_model.v1": model
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify Triton was called to load.
    self.mock_triton_stub.RepositoryModelLoad.assert_called_once()

    # Verify config.pbtxt was NOT overwritten.
    with open(config_path, "r", encoding="utf-8") as f:
      self.assertEqual(f.read(), "dummy config content")

  def test_reconcile_load_model_empty_config_file_missing_fails(self) -> None:
    model = ml_model_pb2.MlModel()
    model.model_config.name = "com.example.test_model"

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.test_model.v1": model
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    self.mock_triton_stub.RepositoryModelLoad.assert_not_called()
    self.assertNotIn("com.example.test_model.v1", self.controller.models)


if __name__ == "__main__":
  absltest.main()
