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

"""Tests for model_assets_managers_factory."""

from absl.testing import absltest

from intrinsic_inference.core import model_assets_manager_base
from intrinsic_inference.core import model_assets_manager_local_repo
from intrinsic_inference.core import model_assets_managers_factory


class FakeManager(model_assets_manager_base.ModelAssetsManagerBase):

  def __init__(self, repo_path: str, custom_arg: str = "default"):
    super().__init__(repo_path)
    self.custom_arg = custom_arg

  def list_model_assets(self):
    return {}

  def create_model_asset(self, model_asset):
    pass

  def update_model_asset(self, old_model, new_model):
    pass

  def delete_model_asset(self, model_asset):
    pass


class ModelAssetsManagersFactoryTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    # Clear registry or keep track of what we add to avoid pollution.
    self.original_registry = dict(model_assets_managers_factory._registry)

  def tearDown(self) -> None:
    model_assets_managers_factory._registry = self.original_registry
    super().tearDown()

  def test_register_and_create_custom_manager(self) -> None:
    model_assets_managers_factory.register("fake", FakeManager)

    manager = model_assets_managers_factory.create(
        "fake", repo_path="/tmp/fake", custom_arg="test"
    )

    self.assertIsInstance(manager, FakeManager)
    self.assertEqual(manager.repo_path, "/tmp/fake")
    self.assertEqual(manager.custom_arg, "test")

  def test_create_fails_if_not_registered(self) -> None:
    with self.assertRaises(ValueError) as context:
      model_assets_managers_factory.create("non_existent", repo_path="/tmp")
    self.assertIn("is not registered", str(context.exception))

  def test_defaults_are_registered(self) -> None:
    self.assertIn("local", model_assets_managers_factory._registry)

  def test_create_local_manager_success(self) -> None:
    manager = model_assets_managers_factory.create(
        "local",
        repo_path="/tmp/local",
    )

    self.assertIsInstance(
        manager, model_assets_manager_local_repo.ModelAssetsManagerLocalRepo
    )
    self.assertEqual(manager.repo_path, "/tmp/local")


if __name__ == "__main__":
  absltest.main()
