import subprocess
import unittest

from fedops_silo_baseline.launcher_app import (
    _read_int,
    _read_str,
    _run_participation,
    _terminate_process,
)


class FakeProcess:
    def __init__(self, exit_code=None):
        self.pid = 123
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout):
        del timeout
        if not self.terminated and not self.killed:
            raise subprocess.TimeoutExpired("fake", timeout=1)
        self.exit_code = 0 if self.terminated else self.exit_code
        return self.exit_code


class LauncherLifecycleTest(unittest.TestCase):
    def test_run_config_readers_apply_defaults_and_convert_values(self):
        self.assertEqual(_read_str({}, "mode", "validate"), "validate")
        self.assertEqual(_read_str({"mode": "participate"}, "mode", "validate"), "participate")
        self.assertEqual(_read_int({}, "manager_port", 8004), 8004)
        self.assertEqual(_read_int({"manager_port": "9000"}, "manager_port", 8004), 9000)

    def test_placeholder_task_id_is_rejected_before_process_start(self):
        with self.assertRaisesRegex(ValueError, "real task_id"):
            _run_participation({"task_id": "task_id"})

    def test_placeholder_runtime_key_is_rejected_before_process_start(self):
        with self.assertRaisesRegex(ValueError, "real runtime_key"):
            _run_participation({"task_id": "507f1f77bcf86cd799439011"})

    def test_process_termination_is_idempotent(self):
        running = FakeProcess()
        _terminate_process(running, "test process")
        self.assertTrue(running.terminated)
        self.assertEqual(running.poll(), 0)

        already_stopped = FakeProcess(exit_code=0)
        _terminate_process(already_stopped, "test process")
        self.assertFalse(already_stopped.terminated)


if __name__ == "__main__":
    unittest.main()
