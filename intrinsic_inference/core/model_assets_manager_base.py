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

"""Base class definition for model assets management."""

import abc

from intrinsic_inference.core.v1 import ml_model_pb2


class ModelAssetsManagerBase(abc.ABC):
  """Base class for managing model assets.

  Handles the lifecycle of model files on disk, including listing, creating,
  updating, and deleting model assets. Subclasses holding long-lived background
  resources (e.g., thread pools, network channels, or file descriptors) should
  override `close()` to ensure proper resource cleanup upon shutdown.

  Supports the Python context manager protocol (`with manager:`), which
  automatically triggers `close()` upon context block exit.

  Attributes:
    repo_path: The root directory where model assets are stored.
  """

  def __init__(self, repo_path: str):
    """Initializes the ModelAssetsManagerBase.

    Args:
      repo_path: The root directory where model assets are stored.
    """
    self.repo_path = repo_path

  @abc.abstractmethod
  def list_model_assets(
      self,
  ) -> dict[str, ml_model_pb2.MlModel]:
    """Lists available model assets.

    Returns:
      A dictionary mapping model identifier strings (e.g.,
      'package.name.version') to their corresponding MlModel proto instances.
    """

  @abc.abstractmethod
  def create_model_asset(self, ml_model: ml_model_pb2.MlModel) -> None:
    """Creates model-related files required for inference.

    Args:
      ml_model: The MlModel proto containing configuration and data
        references to download and set up locally.
    """

  @abc.abstractmethod
  def update_model_asset(
      self, old_model: ml_model_pb2.MlModel, new_model: ml_model_pb2.MlModel
  ) -> None:
    """Updates model-related files for an existing model.

    This method should perform differential updates, deleting files that are
    no longer referenced and downloading new or changed files.

    Args:
      old_model: The previously loaded MlModel proto.
      new_model: The new MlModel proto to update to.
    """

  @abc.abstractmethod
  def delete_model_asset(self, ml_model: ml_model_pb2.MlModel) -> None:
    """Deletes model-related files of an existing model.

    Args:
      ml_model: The MlModel proto of the model to delete.
    """

  def close(self) -> None:
    """Frees resources associated with the manager (e.g., thread pools, gRPC channels).

    Subclasses maintaining background threads, persistent thread pools, or network
    clients should override this method to perform cleanup operations. The
    default implementation is a no-op.
    """
    pass

  def __enter__(self):
    """Enters the context manager block, returning the manager instance."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exits the context manager block, automatically invoking close()."""
    self.close()
