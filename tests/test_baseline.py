import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "federated-task-baseline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASELINE))

from fedops.client.parameter_contract import describe_parameters, verify_parameter_round_trip
from federated_task.client_main import main as client_main
from federated_task.config import load_config, validate_config
from federated_task.data_preparation import build_smoke_loaders, describe_input_features
from federated_task.model import build_model
from federated_task.server_main import main as server_main
from federated_task.training import (
    MODEL_MANIFEST_PATH,
    MODEL_PATH,
    export_initial_model,
    train_model,
)
from federated_task.task_check import check_readiness
from tools.build_release import build_manifest


class BaselineContractTest(unittest.TestCase):
    def test_config_and_data_contract(self):
        config = load_config()
        self.assertEqual(config["dataset"]["name"], "MNIST")
        self.assertEqual(describe_input_features()["features"][0]["shape"], [1, 28, 28])
        config["batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_config(config)

    def test_model_training_and_parameter_round_trip(self):
        model = build_model({"output_size": 10})
        self.assertEqual(list(model(torch.zeros(2, 1, 28, 28)).shape), [2, 10])
        train_loader, _ = build_smoke_loaders(sample_count=16, batch_size=4, seed=42)
        loss = train_model(
            model,
            train_loader,
            epochs=1,
            learning_rate=0.001,
            device=torch.device("cpu"),
            max_batches=1,
        )
        self.assertGreaterEqual(loss, 0)
        result = verify_parameter_round_trip(
            model,
            lambda: build_model({"output_size": 10}),
            "Pytorch",
        )
        self.assertTrue(result["ok"])
        self.assertGreater(result["payloadBytes"], 0)
        self.assertFalse(any("bn" in item["name"] for item in describe_parameters(model, "Pytorch")))

    def test_release_manifest_contains_only_user_files(self):
        manifest = build_manifest()
        self.assertEqual(manifest["baseline"]["release_version"], "0.4.0")
        paths = {entry["path"] for entry in manifest["files"]}
        self.assertIn("federated_task/task_check.py", paths)
        self.assertIn("federated_task/server_main.py", paths)
        self.assertIn("federated_task/client_main.py", paths)
        self.assertIn("model_release/manifest.json", paths)
        self.assertNotIn("federated_task/federated_learning/parameters.py", paths)
        self.assertFalse(any(path.startswith("tests/") or path.startswith("tools/") for path in paths))
        for entry in manifest["files"]:
            self.assertTrue((BASELINE / entry["path"]).is_file())

    def test_fedops_1_2_client_and_server_entrypoints_are_present(self):
        self.assertTrue(callable(client_main))
        self.assertTrue(callable(server_main))


class ReadinessTest(unittest.TestCase):
    def setUp(self):
        self.original_manifest = MODEL_MANIFEST_PATH.read_text(encoding="utf-8")

    def tearDown(self):
        MODEL_MANIFEST_PATH.write_text(self.original_manifest, encoding="utf-8")
        MODEL_PATH.unlink(missing_ok=True)

    def test_release_and_participation_readiness(self):
        manifest = export_initial_model(synthetic=True, max_batches=1)
        self.assertEqual(manifest["status"], "ready")
        release = check_readiness(mode="release", sample_count=16, max_batches=1)
        self.assertTrue(release["ok"])
        self.assertEqual(release["mode"], "release")
        participation = check_readiness(
            mode="participation",
            sample_count=16,
            max_batches=1,
            allow_synthetic_participation=True,
        )
        self.assertTrue(participation["ok"])
        self.assertFalse(participation["privacy"]["rawDataUploaded"])

    def test_participation_requires_local_data_by_default(self):
        export_initial_model(synthetic=True, max_batches=1)
        with self.assertRaisesRegex(ValueError, "local data binding"):
            check_readiness(mode="participation", max_batches=1)


if __name__ == "__main__":
    unittest.main()
