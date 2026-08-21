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

"""Factory for creating model assets managers."""

from typing import Any
from typing import Callable
from typing import Dict
from typing import overload

from absl import logging

from intrinsic_inference.core import model_assets_manager_base

CreatorFn = Callable[..., model_assets_manager_base.ModelAssetsManagerBase]

_registry: Dict[str, CreatorFn] = {}


@overload
def register(name: str) -> Callable[[CreatorFn], CreatorFn]:
  ...


@overload
def register(name: str, creator: CreatorFn) -> None:
  ...


def register(name: str, creator: CreatorFn | None = None) -> Any:
  """Registers a new model assets manager creator.

  Can be used as a decorator or as a regular function.
  Usage as decorator:
    @register("my_manager")
    class MyManager(ModelAssetsManagerBase):
      ...

  Usage as function:
    register("my_manager", MyManager)

  Args:
    name: The name/type of the manager.
    creator: (Optional) The creator callable. If not provided, returns a
      decorator.

  Returns:
    A decorator function if creator is not provided, otherwise None.
  """

  def decorator(cls: CreatorFn) -> CreatorFn:
    if name in _registry:
      logging.warning("Manager '%s' is already registered, overwriting.", name)
    _registry[name] = cls
    return cls

  if creator is None:
    return decorator
  else:
    decorator(creator)


def create(
    name: str, **kwargs: Any
) -> model_assets_manager_base.ModelAssetsManagerBase:
  """Creates a model assets manager instance.

  Args:
    name: The name/type of the manager to create.
    **kwargs: Keyword arguments to pass to the registered creator.

  Returns:
    An instance of ModelAssetsManagerBase.

  Raises:
    ValueError: If the manager name is not registered.
  """
  if name not in _registry:
    raise ValueError(
        f"Manager '{name}' is not registered. Available:"
        f" {list(_registry.keys())}"
    )
  return _registry[name](**kwargs)
