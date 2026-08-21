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

"""OpenTelemetry implementation of the telemetry base."""

from __future__ import annotations

from typing import Any

from absl import logging
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from intrinsic_inference.core import telemetry_base


class OTelSpan(telemetry_base.Span):
  """OpenTelemetry implementation of a Span."""

  def __init__(self, otel_span: Any):
    # _otel_span is typically the context manager wrapper returned by
    # tracer.start_as_current_span(), used to control the context lifecycle.
    self._otel_span = otel_span
    # _active_span holds the actual trace.Span returned by
    # _otel_span.__enter__(), where attributes can be set inside a 'with span:'
    # block.
    self._active_span: Any | None = None

  def __enter__(self) -> OTelSpan:
    self._active_span = self._otel_span.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    try:
      self._otel_span.__exit__(exc_type, exc_val, exc_tb)
    finally:
      self._active_span = None

  def set_attribute(self, key: str, value: Any) -> None:
    if self._active_span is not None and hasattr(
        self._active_span, "set_attribute"
    ):
      self._active_span.set_attribute(key, value)
    elif hasattr(self._otel_span, "set_attribute"):
      self._otel_span.set_attribute(key, value)


class OTelTelemetryProvider(telemetry_base.TelemetryProvider):
  """OpenTelemetry implementation of a TelemetryProvider."""

  def __init__(self, tracer_name: str = "ai.intrinsic.ml"):
    self._tracer = trace.get_tracer(tracer_name)

  def start_span(self, name: str) -> telemetry_base.Span:
    otel_span = self._tracer.start_as_current_span(name)
    return OTelSpan(otel_span)


def initialize_telemetry(tracer_name: str = "inference_core") -> None:
  """Initializes the telemetry provider with OpenTelemetry."""
  telemetry_base.set_telemetry_provider(OTelTelemetryProvider(tracer_name))


def setup_tracing(
    service_name: str,
    endpoint: str,
    service_namespace: str | None = None,
) -> None:
  """Initializes OpenTelemetry tracing and auto-instruments gRPC.

  This setup is idempotent (does nothing if a TracerProvider is already
  registered) and catches initialization errors to prevent crashing.

  Args:
    service_name: Name of the service to register (used to group spans).
    endpoint: Custom endpoint for the OTLP collector.
    service_namespace: Optional namespace grouping to classify microservices.
  """
  if isinstance(trace.get_tracer_provider(), TracerProvider):
    logging.debug("OpenTelemetry tracing already initialized.")
    return

  try:
    resource_attributes = {"service.name": service_name}
    if service_namespace:
      resource_attributes["service.namespace"] = service_namespace
    resource = Resource.create(resource_attributes)
    provider = TracerProvider(resource=resource)

    logging.info(
        "Initializing OpenTelemetry tracing for %s targeting OTLP/gRPC: %s",
        service_name,
        endpoint,
    )
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    GrpcInstrumentorClient().instrument()
    GrpcInstrumentorServer().instrument()

    logging.info("OpenTelemetry tracing successfully enabled.")
  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Failed to initialize OpenTelemetry tracing: %s. Tracing disabled.", e
    )
