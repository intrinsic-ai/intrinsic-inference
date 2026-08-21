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

load("@intrinsic_inference_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library(
    name = "inference_runner",
    srcs = ["inference_runner.py"],
    deps = [
        ":model_controller_base",
        ":telemetry_base",
        ":triton_shm_utils",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        requirement("tritonclient"),
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "inference_runner_test",
    srcs = ["inference_runner_test.py"],
    deps = [
        ":inference_runner",
        ":model_controller_base",
        ":triton_shm_utils",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        requirement("tritonclient"),
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "model_assets_manager_base",
    srcs = ["model_assets_manager_base.py"],
    deps = [
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
    ],
)

py_test(
    name = "model_assets_manager_base_test",
    srcs = ["model_assets_manager_base_test.py"],
    deps = [
        ":model_assets_manager_base",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "model_assets_manager_local_repo",
    srcs = ["model_assets_manager_local_repo.py"],
    deps = [
        ":model_assets_manager_base",
        ":model_assets_managers_factory",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "model_assets_manager_local_repo_test",
    srcs = ["model_assets_manager_local_repo_test.py"],
    deps = [
        ":model_assets_manager_local_repo",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "model_assets_managers_factory",
    srcs = ["model_assets_managers_factory.py"],
    deps = [
        ":model_assets_manager_base",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "model_assets_managers_factory_test",
    srcs = ["model_assets_managers_factory_test.py"],
    deps = [
        ":model_assets_manager_base",
        ":model_assets_manager_local_repo",
        ":model_assets_managers_factory",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "model_controller_base",
    srcs = ["model_controller_base.py"],
    deps = [
        ":model_assets_manager_base",
        ":telemetry_base",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "model_controller_base_test",
    srcs = ["model_controller_base_test.py"],
    deps = [
        ":model_assets_manager_base",
        ":model_controller_base",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "model_controller_triton",
    srcs = ["model_controller_triton.py"],
    deps = [
        ":model_assets_manager_base",
        ":model_controller_base",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        requirement("tritonclient"),
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "telemetry_base",
    srcs = ["telemetry_base.py"],
)

py_library(
    name = "telemetry_otel",
    srcs = ["telemetry_otel.py"],
    deps = [
        ":telemetry_base",
        requirement("opentelemetry-api"),
        requirement("opentelemetry-exporter-otlp-proto-grpc"),
        requirement("opentelemetry-instrumentation-grpc"),
        requirement("opentelemetry-sdk"),
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "telemetry_otel_test",
    srcs = ["telemetry_otel_test.py"],
    deps = [
        ":telemetry_otel",
        requirement("opentelemetry-api"),
        requirement("opentelemetry-sdk"),
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "simple_tracer",
    srcs = ["simple_tracer.py"],
    deps = [
        ":telemetry_otel",
        requirement("opentelemetry-api"),
        requirement("opentelemetry-instrumentation-grpc"),
        requirement("opentelemetry-sdk"),
    ],
)

py_test(
    name = "model_controller_triton_test",
    srcs = ["model_controller_triton_test.py"],
    deps = [
        ":model_assets_manager_base",
        ":model_controller_triton",
        "//intrinsic_inference/core/v1:ml_model_py_pb2",
        requirement("tritonclient"),
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "triton_shm_utils",
    srcs = ["triton_shm_utils.py"],
    deps = [
        ":telemetry_base",
        requirement("numpy"),
        requirement("setuptools"),
        requirement("tritonclient"),
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "triton_shm_utils_test",
    srcs = ["triton_shm_utils_test.py"],
    deps = [
        ":telemetry_base",
        ":triton_shm_utils",
        requirement("numpy"),
        requirement("setuptools"),
        requirement("tritonclient"),
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)
