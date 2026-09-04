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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "oip_mappings",
    srcs = ["oip_mappings.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@intrinsic_inference_pip_deps//numpy",
    ],
)

py_library(
    name = "oip_utils",
    srcs = ["oip_utils.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":oip_mappings",
        "@intrinsic_inference_pip_deps//numpy",
        "@open_inference_protocol//:open_inference_grpc_py_pb2",
    ],
)

py_test(
    name = "oip_mappings_test",
    srcs = ["oip_mappings_test.py"],
    deps = [
        ":oip_mappings",
        "@intrinsic_inference_pip_deps//numpy",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "oip_utils_test",
    srcs = ["oip_utils_test.py"],
    deps = [
        ":oip_utils",
        "@intrinsic_inference_pip_deps//numpy",
        "@com_google_absl_py//absl/testing:absltest",
        "@open_inference_protocol//:open_inference_grpc_py_pb2",
    ],
)
