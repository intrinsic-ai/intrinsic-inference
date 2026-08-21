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

"""Telemetry base for core inference.

This module provides framework-agnostic telemetry abstractions (Spans and
Providers) and decorators to record business logic spans. Subclass these
abstractions to create a concrete implementation for a specific telemetry
framework.
"""

import abc
import functools
from typing import Any
from typing import Callable
from typing import cast
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Span(abc.ABC):
  """Abstract base class for a telemetry span.

  This class is implemented as a context manager and needs to be subclassed to
  create framework-specific spans for a specific telemetry framework.
  """

  @abc.abstractmethod
  def __enter__(self) -> "Span":
    pass

  @abc.abstractmethod
  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    pass

  @abc.abstractmethod
  def set_attribute(self, key: str, value: Any) -> None:
    pass


class TelemetryProvider(abc.ABC):
  """Abstract base class for a telemetry provider.

  The start_span method needs to be implemented to record business logic in the
  telemetry framework and should return an object that is a concrete
  implementation of the Span class for a specific telemetry framework.
  """

  @abc.abstractmethod
  def start_span(self, name: str) -> Span:
    pass


class NoOpSpan(Span):
  """A No-Op implementation of a Span."""

  def __enter__(self) -> "NoOpSpan":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    pass

  def set_attribute(self, key: str, value: Any) -> None:
    pass


class NoOpTelemetryProvider(TelemetryProvider):
  """A No-Op implementation of a TelemetryProvider."""

  def start_span(self, name: str) -> Span:
    return NoOpSpan()


# Global provider, defaults to No-Op.
_PROVIDER: TelemetryProvider = NoOpTelemetryProvider()


def set_telemetry_provider(provider: TelemetryProvider) -> None:
  """Sets the global telemetry provider.

  Needs to be called to change the default no-op provider to a custom
  framework-specific implementation. This should ideally be done early in the
  initialization flow.
  Args:
    provider: The new telemetry provider.
  """
  global _PROVIDER
  _PROVIDER = provider


def get_telemetry_provider() -> TelemetryProvider:
  """Returns the currently set global telemetry provider."""
  return _PROVIDER


def trace_span(name: str) -> Callable[[F], F]:
  """Decorator to trace a function call as a span.

  Uses the global telemetry provider or, if not explicitly set, the NoOp
  implementation.

  Args:
    name: The name of the span.

  Returns:
    A decorator that wraps the function in a span.
  """

  def decorator(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      provider = get_telemetry_provider()
      with provider.start_span(name):
        return func(*args, **kwargs)

    return cast(F, wrapper)

  return decorator
