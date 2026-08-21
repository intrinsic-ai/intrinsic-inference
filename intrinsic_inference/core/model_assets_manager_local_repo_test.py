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

"""Tests for model_assets_manager_local_repo."""

import os
import tempfile

from absl.testing import absltest

from intrinsic_inference.core import model_assets_manager_local_repo
from intrinsic_inference.core.v1 import ml_model_pb2


class ModelAssetsManagerLocalRepoTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.repo_path = self.enter_context(tempfile.TemporaryDirectory())
    self.manager = model_assets_manager_local_repo.ModelAssetsManagerLocalRepo(
        self.repo_path
    )

  def _create_dir(self, *parts):
    path = os.path.join(self.repo_path, *parts)
    os.makedirs(path, exist_ok=True)
    return path

  def _create_file(self, *parts):
    path = os.path.join(self.repo_path, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      f.write("")
    return path

  def test_list_model_assets_empty(self) -> None:
    assets = self.manager.list_model_assets()
    self.assertEqual(assets, {})

  def test_list_model_assets_with_models(self) -> None:
    # Setup: repo/model_a/1/
    self._create_dir("model_a", "1")
    # Setup: repo/model_b/2/
    self._create_dir("model_b", "2")

    assets = self.manager.list_model_assets()

    self.assertIn("model_a.1", assets)
    self.assertIn("model_b.2", assets)
    self.assertEqual(assets["model_a.1"].model_config.name, "model_a")
    self.assertEqual(assets["model_a.1"].model_config.version, "1")
    self.assertEqual(assets["model_b.2"].model_config.name, "model_b")
    self.assertEqual(assets["model_b.2"].model_config.version, "2")

  def test_list_model_assets_ignores_non_dirs(self) -> None:
    # Setup: repo/model_a/1/ (valid)
    self._create_dir("model_a", "1")
    # Setup: repo/some_file.txt (invalid model)
    self._create_file("some_file.txt")
    # Setup: repo/model_b/some_file.txt (invalid version)
    self._create_file("model_b", "some_file.txt")

    assets = self.manager.list_model_assets()

    self.assertIn("model_a.1", assets)
    self.assertNotIn("some_file.txt", assets)
    self.assertNotIn("model_b.some_file.txt", assets)
    self.assertFalse(any(k.startswith("model_b") for k in assets.keys()))

  def test_create_model_asset_success(self) -> None:
    self._create_dir("model_a", "1")
    # We must call list_model_assets first to populate the cache in current implementation
    self.manager.list_model_assets()

    model = ml_model_pb2.MlModel()
    model.model_config.name = "model_a"
    model.model_config.version = "1"

    # Should not raise
    self.manager.create_model_asset(model)

  def test_create_model_asset_fails_if_not_exists(self) -> None:
    self._create_dir("model_a", "1")
    self.manager.list_model_assets()

    model = ml_model_pb2.MlModel()
    model.model_config.name = "non_existent"
    model.model_config.version = "1"

    with self.assertRaises(ValueError) as context:
      self.manager.create_model_asset(model)
    self.assertIn("Model non_existent.1 does not exist", str(context.exception))

  def test_update_model_asset_success(self) -> None:
    self._create_dir("model_a", "1")
    self.manager.list_model_assets()

    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "model_a"
    old_model.model_config.version = "1"
    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "model_a"
    new_model.model_config.version = "1"

    # Should not raise
    self.manager.update_model_asset(old_model, new_model)

  def test_update_model_asset_fails_if_old_not_exists(self) -> None:
    self._create_dir("model_a", "1")
    self.manager.list_model_assets()

    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "non_existent"
    old_model.model_config.version = "1"
    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "model_a"
    new_model.model_config.version = "1"

    with self.assertRaises(ValueError) as context:
      self.manager.update_model_asset(old_model, new_model)
    self.assertIn(
        "Old model non_existent.1 does not exist", str(context.exception)
    )

  def test_update_model_asset_fails_if_new_not_exists(self) -> None:
    self._create_dir("model_a", "1")
    self.manager.list_model_assets()

    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "model_a"
    old_model.model_config.version = "1"
    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "non_existent"
    new_model.model_config.version = "1"

    with self.assertRaises(ValueError) as context:
      self.manager.update_model_asset(old_model, new_model)
    self.assertIn(
        "New model non_existent.1 does not exist", str(context.exception)
    )

  def test_delete_model_asset_success(self) -> None:
    self._create_dir("model_a", "1")
    self.manager.list_model_assets()

    model = ml_model_pb2.MlModel()
    model.model_config.name = "model_a"
    model.model_config.version = "1"

    # Should not raise
    self.manager.delete_model_asset(model)

  def test_delete_model_asset_fails_if_not_exists(self) -> None:
    self._create_dir("model_a", "1")
    self.manager.list_model_assets()

    model = ml_model_pb2.MlModel()
    model.model_config.name = "non_existent"
    model.model_config.version = "1"

    with self.assertRaises(ValueError) as context:
      self.manager.delete_model_asset(model)
    self.assertIn("Model non_existent.1 does not exist", str(context.exception))


if __name__ == "__main__":
  absltest.main()
