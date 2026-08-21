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

"""Entrypoint binary to start ROS 2 InferenceNode with InferenceRunner."""

from __future__ import annotations

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import grpc
import rclpy
from rclpy.executors import MultiThreadedExecutor
from tritonclient.grpc import service_pb2_grpc as triton_pb2_grpc

from intrinsic_inference.core import inference_runner as inference_runner_lib
from intrinsic_inference.core import model_assets_manager_local_repo
from intrinsic_inference.core import model_controller_triton
from intrinsic_inference.ros.inference_node import inference_node

_TRITON_GRPC_URL = flags.DEFINE_string(
    "triton_grpc_url",
    "127.0.0.1:8001",
    "Address of the Triton inference server gRPC endpoint.",
)
_REPO_PATH = flags.DEFINE_string(
    "repo_path",
    "/tmp/model_repository",
    "Local filesystem path to the model repository directory.",
)
_POLL_MODELS_INTERVAL = flags.DEFINE_float(
    "poll_models_interval",
    5.0,
    "Interval in seconds for polling the model repository.",
)
_USE_SHM = flags.DEFINE_bool(
    "use_shm",
    False,
    "Whether to use system shared memory for ModelInfer calls.",
)


def main(argv: Sequence[str]) -> None:
  logging.info("Initializing InferenceRunner for ROS InferenceNode...")

  repo_path = _REPO_PATH.value
  if not os.path.exists(repo_path):
    logging.info(
        "Model repository folder does not exist, creating %s...", repo_path
    )
    os.makedirs(repo_path, exist_ok=True)

  triton_channel = grpc.insecure_channel(_TRITON_GRPC_URL.value)
  triton_stub = triton_pb2_grpc.GRPCInferenceServiceStub(triton_channel)

  model_assets_manager = (
      model_assets_manager_local_repo.ModelAssetsManagerLocalRepo(
          repo_path=repo_path
      )
  )

  model_controller = model_controller_triton.ModelControllerTriton(
      repo_path=repo_path,
      model_assets_manager=model_assets_manager,
      triton_stub=triton_stub,
  )

  inference_runner = inference_runner_lib.InferenceRunner(
      repo_path=repo_path,
      triton_stub=triton_stub,
      model_controller=model_controller,
      poll_models_interval=_POLL_MODELS_INTERVAL.value,
      use_shm=_USE_SHM.value,
  )

  logging.info("Starting ROS 2 InferenceNode (and InferenceRunner)...")
  rclpy.init(args=list(argv))
  node = inference_node.InferenceNode(inference_runner=inference_runner)
  node.start()

  try:
    rclpy.spin(node, executor=MultiThreadedExecutor())
  except KeyboardInterrupt:
    pass
  finally:
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  app.run(main)
