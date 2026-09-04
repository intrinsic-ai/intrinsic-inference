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

# This CMakeLists.txt is used to generate ROS 2 interfaces when building in an
# external ROS 2 / colcon workspace (e.g. downstream ROS packages) where Bazel
# is not used. Within this repository, interfaces are built via Bazel (see
# intrinsic_inference/ros/BUILD).

cmake_minimum_required(VERSION 3.14)
project(inference_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(tensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NamedTensor.msg"
  "msg/TensorMetadata.msg"
  "srv/ServerLive.srv"
  "srv/ServerReady.srv"
  "srv/ModelInfer.srv"
  "srv/ModelReady.srv"
  "srv/ModelMetadata.srv"
  DEPENDENCIES rcl_interfaces tensor_msgs
)

ament_package()
