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

"""Tests for model_assets_manager_base."""

from absl.testing import absltest

from intrinsic_inference.core import model_assets_manager_base
from intrinsic_inference.core.v1 import ml_model_pb2


class ConcreteModelAssetsManager(
    model_assets_manager_base.ModelAssetsManagerBase
):
  """Concrete implementation of ModelAssetsManagerBase for testing."""

  def list_model_assets(self) -> dict[str, ml_model_pb2.MlModel]:
    return {}

  def create_model_asset(self, ml_model: ml_model_pb2.MlModel) -> None:
    pass

  def update_model_asset(
      self, old_model: ml_model_pb2.MlModel, new_model: ml_model_pb2.MlModel
  ) -> None:
    pass

  def delete_model_asset(self, ml_model: ml_model_pb2.MlModel) -> None:
    pass


class ModelAssetsManagerBaseTest(absltest.TestCase):

  def test_cannot_instantiate_base_class(self) -> None:
    with self.assertRaises(TypeError):
      # ModelAssetsManagerBase is abstract and should not be instantiable.
      model_assets_manager_base.ModelAssetsManagerBase(repo_path="/tmp")

  def test_concrete_subclass_instantiation_succeeds(self) -> None:
    repo_path = "/tmp/test_repo"
    manager = ConcreteModelAssetsManager(repo_path=repo_path)
    self.assertIsNotNone(manager)
    self.assertEqual(manager.repo_path, repo_path)


if __name__ == "__main__":
  absltest.main()
