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

"""Setup utilities for local file-based OpenTelemetry tracing.

This module provides a convenient way to set up a local OpenTelemetry tracing
pipeline that exports traces directly to a JSON file in Trace Event Format.
This is useful for standalone or development environments where running a
collector (like Jaeger) is not desired.

Example usage:
  from intrinsic_inference.core import simple_tracer

  simple_tracer.setup_local_pipeline(
      file_path="/tmp/traces.json",
      service_name="my-service",
  )
  # Traces will be written to /tmp/traces.json automatically on process exit.
  # This file can be loaded into https://ui.perfetto.dev/ or chrome://tracing.
"""

from collections import deque
import json
import os
import threading
from typing import Sequence

from absl import logging
from opentelemetry import trace
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

from intrinsic_inference.core import telemetry_otel

_DEFAULT_SERVICE_NAME = "inference_service"
_TRACER_NAME = "inference_core"


class TraceEventExporter(SpanExporter):
  """Exports spans in Trace Event Format (JSON) to a local file.

  Stores the last `max_events` trace events and exports them to a Json file on
  shutdown.
  Compatible with chrome://tracing and https://ui.perfetto.dev/
  """

  def __init__(self, file_path: str, max_events: int = 10000):
    self._file_path = file_path
    self._lock = threading.Lock()
    self._events = deque(maxlen=max_events)

  def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
    with self._lock:
      for span in spans:
        if span.start_time is None or span.end_time is None:
          continue
        # Convert nano to microseconds for Trace Event Format compatibility.
        start_time_us = span.start_time / 1000
        duration_us = (span.end_time - span.start_time) / 1000

        event = {
            "name": span.name,
            "cat": "otel",
            "ph": "X",
            "ts": start_time_us,
            "dur": duration_us,
            "pid": span.resource.attributes.get("process.id", os.getpid()),
            "tid": threading.get_ident(),
            "args": dict(span.attributes) if span.attributes else {},
        }
        self._events.append(event)
    return SpanExportResult.SUCCESS

  def shutdown(self) -> None:
    with self._lock:
      # Ensure parent directory exists
      if self._file_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(self._file_path)), exist_ok=True
        )
        with open(self._file_path, "w", encoding="utf-8") as f:
          json.dump(list(self._events), f, indent=2)
        logging.info("Traces successfully exported to %s", self._file_path)


def setup_local_pipeline(
    file_path: str,
    service_name: str = _DEFAULT_SERVICE_NAME,
    max_events: int = 10000,
) -> None:
  """Sets up a minimal OpenTelemetry pipeline exporting traces to a local file.

  Args:
    file_path: Path to the file where traces will be written on shutdown.
    service_name: Name of the service as it will show up in the trace.
    max_events: Maximum number of events to keep in memory. Older events will
      be dropped.
  """
  # Configure the resource.
  resource = Resource.create({"service.name": service_name})
  provider = TracerProvider(resource=resource)

  exporter = TraceEventExporter(file_path, max_events=max_events)
  provider.add_span_processor(SimpleSpanProcessor(exporter))

  # Initialize Tracer.
  trace.set_tracer_provider(provider)

  # Automatically instrument gRPC client and server.
  GrpcInstrumentorClient().instrument()
  GrpcInstrumentorServer().instrument()

  telemetry_otel.initialize_telemetry(_TRACER_NAME)
