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

"""Tests for model_controller_base."""

import threading
import time
from unittest import mock

from absl.testing import absltest
from google.protobuf import wrappers_pb2

from intrinsic_inference.core import model_assets_manager_base
from intrinsic_inference.core import model_controller_base
from intrinsic_inference.core.v1 import ml_model_pb2


# Fake implementation of ModelControllerBase to test generic logic.
class FakeModelController(model_controller_base.ModelControllerBase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.load_calls = []
    self.unload_calls = []
    self.reload_calls = []

  def _validate_backend_config(self, model_asset: ml_model_pb2.MlModel) -> None:
    super()._validate_backend_config(model_asset)
    # Generic validation is enough for fake.

  def _are_backend_config_equal(
      self, old_model: ml_model_pb2.MlModel, new_model: ml_model_pb2.MlModel
  ) -> bool:
    return old_model.backend_config == new_model.backend_config

  def _load_model_impl(self, model_proto: ml_model_pb2.MlModel) -> None:
    self.load_calls.append(model_proto)

  def _unload_model_impl(self, model_proto: ml_model_pb2.MlModel) -> None:
    self.unload_calls.append(model_proto)

  def _reload_model_impl(self, model_proto: ml_model_pb2.MlModel) -> None:
    self.reload_calls.append(model_proto)


class ModelControllerBaseTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.mock_model_assets_manager = mock.MagicMock(
        spec=model_assets_manager_base.ModelAssetsManagerBase
    )

    self.controller = FakeModelController(
        model_assets_manager=self.mock_model_assets_manager
    )

  def test_init_succeeds(self) -> None:
    self.assertIsNotNone(self.controller)
    self.assertEqual(
        self.controller._model_assets_manager, self.mock_model_assets_manager
    )  # pylint: disable=protected-access
    self.assertEqual(self.controller.models, {})

  def test_reconcile_models_add_only(self) -> None:
    model1 = ml_model_pb2.MlModel()
    model1.model_config.name = "com.example.model1"
    dummy_config = wrappers_pb2.StringValue(value="config_bar")
    model1.backend_config.Pack(dummy_config)
    model1.model_data["file1"].reference = "file://1"

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": model1
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    self.assertEqual(self.controller.load_calls, [model1])
    self.assertIn("com.example.model1.v1", self.controller.models)
    self.assertEqual(self.controller.models["com.example.model1.v1"], model1)

  def test_reconcile_models_remove_only(self) -> None:
    model1 = ml_model_pb2.MlModel()
    model1.model_config.name = "com.example.model1"
    dummy_config = wrappers_pb2.StringValue(value="config_bar")
    model1.backend_config.Pack(dummy_config)
    self.controller._models = {"com.example.model1.v1": model1}  # pylint: disable=protected-access

    self.mock_model_assets_manager.list_model_assets.return_value = {}

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    self.assertEqual(self.controller.unload_calls, [model1])
    self.assertNotIn("com.example.model1.v1", self.controller.models)

  def test_reconcile_models_full_reload(self) -> None:
    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "com.example.model1"
    config1 = wrappers_pb2.StringValue(value="config_bar")
    old_model.backend_config.Pack(config1)
    old_model.model_data["file1"].reference = "file://1"
    self.controller._models = {"com.example.model1.v1": old_model}  # pylint: disable=protected-access

    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "com.example.model1"
    new_model.backend_config.Pack(config1)  # Same config.
    new_model.model_data["file1"].reference = (
        "file://2"  # Changed reference -> full reload.
    )

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": new_model
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify update_model_asset was called because files changed.
    self.mock_model_assets_manager.update_model_asset.assert_called_once_with(
        old_model, new_model
    )
    self.assertEqual(self.controller.reload_calls, [new_model])
    self.assertEqual(self.controller.models["com.example.model1.v1"], new_model)

  def test_reconcile_models_config_only_reload(self) -> None:
    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "com.example.model1"
    config1 = wrappers_pb2.StringValue(value="config_bar")
    old_model.backend_config.Pack(config1)
    old_model.model_data["file1"].reference = "file://1"
    self.controller._models = {"com.example.model1.v1": old_model}  # pylint: disable=protected-access

    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "com.example.model1"
    config2 = wrappers_pb2.StringValue(value="config_baz")  # Different config.
    new_model.backend_config.Pack(config2)
    new_model.model_data["file1"].reference = "file://1"  # Same reference.

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": new_model
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify update_model_asset was NOT called because only config changed.
    self.mock_model_assets_manager.update_model_asset.assert_not_called()
    self.assertEqual(self.controller.reload_calls, [new_model])
    self.assertEqual(self.controller.models["com.example.model1.v1"], new_model)

  def test_reconcile_models_thread_safety(self) -> None:
    active_calls = 0
    max_active_calls = 0
    calls_lock = threading.Lock()

    call_count = 0

    def mock_list_model_assets():
      nonlocal call_count, active_calls, max_active_calls
      with calls_lock:
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)

      time.sleep(0.1)

      with calls_lock:
        active_calls -= 1
        model_name = f"com.example.model_{call_count}"
        call_count += 1

      model = ml_model_pb2.MlModel()
      model.model_config.name = model_name
      dummy_config = wrappers_pb2.StringValue(value="config_bar")
      model.backend_config.Pack(dummy_config)
      return {f"com.example.model_{call_count-1}.v1": model}

    self.mock_model_assets_manager.list_model_assets = mock_list_model_assets

    threads = []
    for _ in range(5):
      t = threading.Thread(target=self.controller.reconcile_models)
      threads.append(t)

    for t in threads:
      t.start()

    for t in threads:
      t.join()

    self.controller.wait_for_idle()
    self.assertEqual(max_active_calls, 1)
    self.assertEqual(active_calls, 0)

  def test_reconcile_models_partial_failure(self) -> None:
    model_old = ml_model_pb2.MlModel()
    model_old.model_config.name = "com.example.model_old"
    dummy_config = wrappers_pb2.StringValue(value="config_bar")
    model_old.backend_config.Pack(dummy_config)
    self.controller._models = {"com.example.model_old.v1": model_old}  # pylint: disable=protected-access

    model_ok = ml_model_pb2.MlModel()
    model_ok.model_config.name = "com.example.model_ok"
    model_ok.backend_config.Pack(dummy_config)

    model_fail = ml_model_pb2.MlModel()
    model_fail.model_config.name = "com.example.model_fail"
    model_fail.backend_config.Pack(dummy_config)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model_ok.v1": model_ok,
        "com.example.model_fail.v1": model_fail,
    }

    def mock_load_model_impl(model_proto: ml_model_pb2.MlModel) -> None:
      if model_proto.model_config.name == "com.example.model_fail":
        raise RuntimeError("Simulated load failure")
      self.controller.load_calls.append(model_proto)

    self.controller._load_model = mock_load_model_impl

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    # Verify unload succeeded and was reflected.
    self.assertEqual(self.controller.unload_calls, [model_old])
    self.assertNotIn("com.example.model_old.v1", self.controller.models)

    # Verify state based on what actually succeeded in load_model.
    called_names = [m.model_config.name for m in self.controller.load_calls]
    if "com.example.model_ok" in called_names:
      self.assertIn("com.example.model_ok.v1", self.controller.models)
      self.assertEqual(
          self.controller.models["com.example.model_ok.v1"], model_ok
      )
    else:
      self.assertNotIn("com.example.model_ok.v1", self.controller.models)

    self.assertNotIn("com.example.model_fail.v1", self.controller.models)
    self.assertIn("com.example.model_fail.v1", self.controller.model_states)
    self.assertEqual(
        self.controller.model_states["com.example.model_fail.v1"].state,
        model_controller_base.ModelState.FAILED,
    )
    self.assertEqual(
        self.controller.model_states["com.example.model_fail.v1"].message,
        "Failed to load model com.example.model_fail.v1: Simulated load"
        " failure",
    )

  def test_validate_backend_config_fails_when_missing(self) -> None:
    model = ml_model_pb2.MlModel()
    model.model_config.name = "test_model"
    # No backend_config set.

    with self.assertRaises(RuntimeError) as context:
      self.controller._validate_backend_config(model)
    self.assertIn("No backend config specified", str(context.exception))

  def test_model_states_enum_and_deduplication(self) -> None:
    model1 = ml_model_pb2.MlModel()
    model1.model_config.name = "com.example.model1"
    dummy_config = wrappers_pb2.StringValue(value="config_bar")
    model1.backend_config.Pack(dummy_config)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": model1
    }

    self.controller.reconcile_models()
    # While loading, status should be LOADING or READY.
    statuses = self.controller.model_states
    self.assertIn("com.example.model1.v1", statuses)
    self.assertIn(
        statuses["com.example.model1.v1"].state,
        (
            model_controller_base.ModelState.LOADING,
            model_controller_base.ModelState.READY,
        ),
    )

    self.controller.wait_for_idle()
    statuses = self.controller.model_states
    self.assertEqual(
        statuses["com.example.model1.v1"].state,
        model_controller_base.ModelState.READY,
    )
    self.assertEqual(statuses["com.example.model1.v1"].proto, model1)

    # Calling reconcile again should deduplicate and not invoke load again.
    self.controller.reconcile_models()
    self.controller.wait_for_idle()
    self.assertEqual(len(self.controller.load_calls), 1)

  def test_reconcile_recovers_failed_model_when_config_changes(self) -> None:
    model_fail = ml_model_pb2.MlModel()
    model_fail.model_config.name = "com.example.recoverable"
    model_fail.model_config.version = "v1"
    dummy_config = wrappers_pb2.StringValue(value="config_bad")
    model_fail.backend_config.Pack(dummy_config)
    model_fail.model_data["file1"].reference = "file://bad_reference"

    # 1. Initial failure
    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.recoverable.v1": model_fail
    }

    def mock_load_model_impl(model_proto: ml_model_pb2.MlModel) -> None:
      if model_proto.model_data["file1"].reference == "file://bad_reference":
        raise RuntimeError("CAS unavailable")
      self.controller.load_calls.append(model_proto)

    self.controller._load_model_impl = mock_load_model_impl

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    self.assertEqual(
        self.controller.model_states["com.example.recoverable.v1"].state,
        model_controller_base.ModelState.FAILED,
    )
    self.assertNotIn("com.example.recoverable.v1", self.controller.models)

    # 2. Same config -> should NOT retry
    self.controller.reconcile_models()
    self.controller.wait_for_idle()
    self.assertEqual(len(self.controller.load_calls), 0)

    # 3. New config pushed -> MUST retry and succeed
    model_fixed = ml_model_pb2.MlModel()
    model_fixed.model_config.name = "com.example.recoverable"
    model_fixed.model_config.version = "v1"
    model_fixed.backend_config.Pack(dummy_config)
    model_fixed.model_data["file1"].reference = "file://good_reference"

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.recoverable.v1": model_fixed
    }
    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    self.assertEqual(self.controller.load_calls, [model_fixed])
    self.assertEqual(
        self.controller.model_states["com.example.recoverable.v1"].state,
        model_controller_base.ModelState.READY,
    )
    self.assertIn("com.example.recoverable.v1", self.controller.models)

  def test_reconcile_reload_failure_sets_failed_state(self) -> None:
    old_model = ml_model_pb2.MlModel()
    old_model.model_config.name = "com.example.model1"
    config1 = wrappers_pb2.StringValue(value="config_v1")
    old_model.backend_config.Pack(config1)
    self.controller._models = {"com.example.model1.v1": old_model}

    new_model = ml_model_pb2.MlModel()
    new_model.model_config.name = "com.example.model1"
    config2 = wrappers_pb2.StringValue(value="config_v2")
    new_model.backend_config.Pack(config2)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": new_model
    }

    self.controller._reload_model_impl = mock.MagicMock(
        side_effect=RuntimeError("Reload failed")
    )

    self.controller.reconcile_models()
    self.controller.wait_for_idle()

    self.assertEqual(
        self.controller.model_states["com.example.model1.v1"].state,
        model_controller_base.ModelState.FAILED,
    )
    self.assertIn(
        "Failed to reload model com.example.model1.v1: Reload failed",
        self.controller.model_states["com.example.model1.v1"].message,
    )

  def test_start_stop_start_executor_lifecycle(self) -> None:
    model1 = ml_model_pb2.MlModel()
    model1.model_config.name = "com.example.model1"
    dummy_config = wrappers_pb2.StringValue(value="config_bar")
    model1.backend_config.Pack(dummy_config)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": model1
    }

    # 1. First run
    self.controller.reconcile_models()
    self.controller.wait_for_idle()
    self.assertEqual(self.controller.load_calls, [model1])

    # 2. Stop controller (shuts down executor)
    self.controller.stop(wait=True)

    # 3. Restart controller and reconcile new model
    self.controller.start()

    model2 = ml_model_pb2.MlModel()
    model2.model_config.name = "com.example.model2"
    model2.backend_config.Pack(dummy_config)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": model1,
        "com.example.model2.v1": model2,
    }

    self.controller.reconcile_models()
    self.controller.wait_for_idle()
    self.assertEqual(self.controller.load_calls, [model1, model2])
    self.assertIn("com.example.model2.v1", self.controller.models)

  def test_stop_wait_true_does_not_deadlock(self) -> None:
    """This test verifies that a thread pool stop doesn't cause deadlocks.

    A deadlock could potentially happen if the stop method acquires the lock
    while a worker task is in progress and waits for that worker to finish.
    After the worker finishes it tries itself to acquire the lock to update
    the current state leading to a deadlock.
    """
    model1 = ml_model_pb2.MlModel()
    model1.model_config.name = "com.example.model1"
    dummy_config = wrappers_pb2.StringValue(value="config_bar")
    model1.backend_config.Pack(dummy_config)

    self.mock_model_assets_manager.list_model_assets.return_value = {
        "com.example.model1.v1": model1
    }

    in_load_event = threading.Event()
    release_load_event = threading.Event()

    def slow_load(model_proto: ml_model_pb2.MlModel) -> None:
      in_load_event.set()
      release_load_event.wait(timeout=2.0)
      self.controller.load_calls.append(model_proto)

    # Patch in the slow_load function to be used during reconciliation.
    self.controller._load_model_impl = slow_load

    self.controller.reconcile_models()
    self.assertTrue(in_load_event.wait(timeout=1.0))

    # Trigger release after slight delay from another thread while
    # stop(wait=True) runs.
    def trigger_release():
      time.sleep(0.05)
      release_load_event.set()

    t = threading.Thread(target=trigger_release)
    t.start()

    # stop(wait=True) must not deadlock when the worker thread acquires
    # self._lock at completion.
    self.controller.stop(wait=True)
    t.join()

    self.assertEqual(self.controller.load_calls, [model1])
    self.assertIn("com.example.model1.v1", self.controller.models)

  def test_get_model_resolution(self) -> None:
    model1 = ml_model_pb2.MlModel()
    model1.model_config.name = "detection"
    dummy_config = wrappers_pb2.StringValue(value="config_bar")
    model1.backend_config.Pack(dummy_config)

    model2 = ml_model_pb2.MlModel()
    model2.model_config.name = "classification"
    model2.backend_config.Pack(dummy_config)

    self.controller._models = {
        "com.intrinsic.detection.v1": model1,
        "classification": model2,
    }

    # Exact key match.
    self.assertEqual(self.controller.get_model("classification"), model2)

    # Exact model_config.name match.
    self.assertEqual(self.controller.get_model("detection"), model1)

    # Prefix/suffix match.
    self.assertEqual(
        self.controller.get_model("com.intrinsic.detection.v1"), model1
    )

    # Non-existent model.
    self.assertIsNone(self.controller.get_model("unknown_model"))


if __name__ == "__main__":
  absltest.main()
