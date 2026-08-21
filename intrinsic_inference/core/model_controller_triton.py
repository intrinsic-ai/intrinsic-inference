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

"""Handles model loading and unloading for the Triton inference backend."""

import enum
import os

from absl import logging
from google.protobuf import text_format
import grpc
from tritonclient.grpc import model_config_pb2 as triton_model_pb2
from tritonclient.grpc import service_pb2 as triton_pb2
from tritonclient.grpc import service_pb2_grpc as triton_pb2_grpc

from intrinsic_inference.core import model_assets_manager_base
from intrinsic_inference.core import model_controller_base
from intrinsic_inference.core.v1 import ml_model_pb2


@enum.unique
class TritonAction(enum.Enum):
  LOAD = "load"
  RELOAD = "reload"
  UNLOAD = "unload"


class ModelControllerTriton(model_controller_base.ModelControllerBase):
  """Controls models for the Triton inference backend.

  Attributes:
    models: A mapping of model names to their loaded MlModel proto instances.
  """

  def __init__(
      self,
      repo_path: str,
      model_assets_manager: model_assets_manager_base.ModelAssetsManagerBase,
      triton_stub: triton_pb2_grpc.GRPCInferenceServiceStub,
  ) -> None:
    super().__init__(model_assets_manager=model_assets_manager)
    self._repo_path = repo_path
    self._triton_stub = triton_stub

  def _validate_backend_config(self, model_asset: ml_model_pb2.MlModel) -> None:
    """Validates that the Triton config is present."""
    super()._validate_backend_config(model_asset)
    triton_config = triton_model_pb2.ModelConfig()
    if not model_asset.backend_config.Is(triton_config.DESCRIPTOR):
      error_msg = (
          "Unsupported backend config field! Triton backend requires a Triton"
          " ModelConfig proto in backend_config."
      )
      logging.error(error_msg)
      raise RuntimeError(error_msg)

  def _are_backend_config_equal(
      self, old_model: ml_model_pb2.MlModel, new_model: ml_model_pb2.MlModel
  ) -> bool:
    """Triton supports config-only reload if only triton_config changed."""
    old_has_config = old_model.HasField(
        "backend_config"
    ) and old_model.backend_config.Is(triton_model_pb2.ModelConfig.DESCRIPTOR)
    new_has_config = new_model.HasField(
        "backend_config"
    ) and new_model.backend_config.Is(triton_model_pb2.ModelConfig.DESCRIPTOR)

    if not old_has_config and not new_has_config:
      # Neither model has a Triton ModelConfig in backend_config. Falls back to
      # raw field comparison.
      return old_model.backend_config == new_model.backend_config

    if old_has_config != new_has_config:
      return False

    old_backend_config = triton_model_pb2.ModelConfig()
    new_backend_config = triton_model_pb2.ModelConfig()
    if not old_model.backend_config.Unpack(
        old_backend_config
    ) or not new_model.backend_config.Unpack(new_backend_config):
      logging.error("Failed to unpack Triton backend_config!")
      # Raise an error as we should not load a model with an invalid backend
      # config. This propagates back to reconcile_models.
      raise RuntimeError(
          "Error unpacking Triton backend_config for model"
          f" {old_model.model_config.name}! Check that the config is valid."
      )
    return old_backend_config == new_backend_config

  def _run_triton_action(
      self, model_proto: ml_model_pb2.MlModel, action: TritonAction
  ) -> None:
    """Helper to load or reload a model in Triton."""
    model_name = model_proto.model_config.name
    if action in [TritonAction.LOAD, TritonAction.RELOAD]:
      self._write_triton_config(model_proto)
      request = triton_pb2.RepositoryModelLoadRequest(model_name=model_name)
      action_fn = self._triton_stub.RepositoryModelLoad
    elif action == TritonAction.UNLOAD:
      request = triton_pb2.RepositoryModelUnloadRequest(model_name=model_name)
      action_fn = self._triton_stub.RepositoryModelUnload
    else:
      raise ValueError("Unknown triton action requested.")

    try:
      action_fn(request)
    except grpc.RpcError as e:
      logging.error(
          "Error %sing model '%s': %s", action.value, model_name, e.details()
      )
      raise
    logging.info("Successfully %sed model '%s'", action.value, model_name)

  def _load_model_impl(self, model_proto: ml_model_pb2.MlModel) -> None:
    """Loads a model into the Triton inference backend."""
    self._run_triton_action(model_proto, action=TritonAction.LOAD)

  def _unload_model_impl(self, model_proto: ml_model_pb2.MlModel) -> None:
    """Unloads a model from the Triton inference backend."""
    self._run_triton_action(model_proto, action=TritonAction.UNLOAD)

  def _reload_model_impl(self, model_proto: ml_model_pb2.MlModel) -> None:
    """Reloads a model from the Triton inference backend.

    The difference to unload->load is that unchanged files are not deleted.
    If only the Triton config changed, it performs a config-only reload.

    Args:
      model_proto: The model to reload.
    """
    self._run_triton_action(model_proto, action=TritonAction.RELOAD)

  def _write_triton_config(self, model_proto: ml_model_pb2.MlModel) -> None:
    model_name = model_proto.model_config.name
    config_path = os.path.join(self._repo_path, model_name, "config.pbtxt")

    triton_config = triton_model_pb2.ModelConfig()
    # If backend_config is populated and is of correct type, try to unpack and
    # write it.
    if model_proto.backend_config.Is(triton_config.DESCRIPTOR):
      logging.info("Writing Triton config for model '%s'...", model_name)
      if model_proto.backend_config.Unpack(triton_config):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
          f.write(text_format.MessageToString(triton_config))
        return

    # If we couldn't unpack from proto, check if it already exists on disk.
    if os.path.exists(config_path):
      logging.info(
          "Triton config for model '%s' already exists on disk at %s, skipping"
          " write.",
          model_name,
          config_path,
      )
      return

    # Otherwise, we cannot proceed.
    error_msg = (
        f"Failed to obtain Triton config for model '{model_name}'."
        " backend_config was empty/invalid and no config.pbtxt was found on"
        " disk."
    )
    logging.error(error_msg)
    raise RuntimeError(error_msg)
