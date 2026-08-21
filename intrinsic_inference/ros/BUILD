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

load(
    "@com_github_mvukov_rules_ros2//ros2:interfaces.bzl",
    "cpp_ros2_interface_library",
    "py_ros2_interface_library",
    "ros2_interface_library",
)

package(default_visibility = ["//visibility:public"])

ros2_interface_library(
    name = "inference_interfaces",
    srcs = glob([
        "inference_interfaces/srv/*.srv",
    ]),
)

py_ros2_interface_library(
    name = "py_inference_interfaces",
    deps = [":inference_interfaces"],
)

cpp_ros2_interface_library(
    name = "cpp_inference_interfaces",
    deps = [":inference_interfaces"],
)
