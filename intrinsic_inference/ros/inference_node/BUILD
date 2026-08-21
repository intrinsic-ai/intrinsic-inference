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

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library(
    name = "inference_node",
    srcs = ["inference_node.py"],
    deps = [
        "//intrinsic_inference/core:inference_runner",
        "//intrinsic_inference/ros:py_inference_interfaces",
        "@intrinsic_inference_pip_deps//tritonclient",
        "@com_google_absl_py//absl/logging",
        "@ros2_rclpy//:rclpy",
    ],
)

py_binary(
    name = "inference_node_main",
    srcs = ["inference_node_main.py"],
    deps = [
        ":inference_node",
        "//intrinsic_inference/core:inference_runner",
        "//intrinsic_inference/core:model_assets_manager_local_repo",
        "//intrinsic_inference/core:model_controller_triton",
        "@intrinsic_inference_pip_deps//tritonclient",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/logging",
        "@ros2_rclpy//:rclpy",
    ],
)

py_test(
    name = "inference_node_test",
    srcs = ["inference_node_test.py"],
    deps = [
        ":inference_node",
        "//intrinsic_inference/core:inference_runner",
        "//intrinsic_inference/ros:py_inference_interfaces",
        "@intrinsic_inference_pip_deps//tritonclient",
        "@com_google_absl_py//absl/testing:absltest",
        "@ros2_rclpy//:rclpy",
    ],
)
