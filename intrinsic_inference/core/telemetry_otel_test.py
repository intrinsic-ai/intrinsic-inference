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

"""Tests for telemetry_otel."""

from unittest import mock

from absl.testing import absltest
from opentelemetry import trace
from opentelemetry.sdk import trace as sdk_trace

from intrinsic_inference.core import telemetry_otel


class TelemetryOTelTest(absltest.TestCase):

  def test_otel_span_set_attribute_inside_context(self):
    # Set up real OTel SDK tracer provider so start_as_current_span returns real
    # AgnosticContextManager
    provider = sdk_trace.TracerProvider()
    trace.set_tracer_provider(provider)

    otel_provider = telemetry_otel.OTelTelemetryProvider("test_tracer")
    span = otel_provider.start_span("test_span")

    # Inside the with block, set_attribute should not raise an AttributeError on
    # _AgnosticContextManager
    with span:
      span.set_attribute("key_inside", "value_inside")

  def test_otel_span_set_attribute_mocked(self):
    mock_otel_span = mock.MagicMock()
    mock_active_span = mock.MagicMock()
    mock_otel_span.__enter__.return_value = mock_active_span

    span = telemetry_otel.OTelSpan(mock_otel_span)
    with span:
      span.set_attribute("foo", "bar")

    mock_active_span.set_attribute.assert_called_once_with("foo", "bar")

  def test_setup_tracing(self):
    telemetry_otel.setup_tracing(
        service_name="test_service", endpoint="endpoint.example.com"
    )
    self.assertIsInstance(trace.get_tracer_provider(), sdk_trace.TracerProvider)


if __name__ == "__main__":
  absltest.main()
