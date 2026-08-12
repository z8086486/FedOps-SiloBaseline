import inspect
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "federated-task-baseline"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASELINE))

from federated_task.client_main import main as client_main
from federated_task.config import load_config, validate_config
from federated_task.data_preparation import (
    build_contract_probe,
    build_smoke_loaders,
    describe_input_features,
    load_partition,
)
from federated_task.model import build_model, run_model, validate_model_output
from federated_task.server_main import main as server_main
from federated_task.tool import build_tool_smoke_payload, predict
from federated_task.training import evaluate_model, normalize_evaluation, train_model
from tools.build_release import build_manifest


class BaselineContractTest(unittest.TestCase):
    def test_generic_config_is_valid_and_has_no_mnist_contract(self):
        config = load_config()
        self.assertEqual(config["model_type"], "Pytorch")
        self.assertEqual(config["dataset"]["name"], "replace-with-dataset-name")
        self.assertNotIn("output_size", config["model"])
        config["batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_config(config)

    def test_user_function_signatures_are_fixed(self):
        self.assertEqual(
            list(inspect.signature(build_model).parameters),
            ["config"],
        )
        self.assertEqual(
            list(inspect.signature(load_partition).parameters),
            ["dataset", "validation_split", "batch_size", "data_root", "seed", "download"],
        )
        self.assertEqual(
            list(inspect.signature(train_model).parameters),
            ["model", "loader", "epochs", "learning_rate", "device", "max_batches"],
        )
        self.assertEqual(
            list(inspect.signature(evaluate_model).parameters),
            ["model", "loader", "device", "max_batches"],
        )
        self.assertEqual(
            list(inspect.signature(predict).parameters),
            ["payload", "model_path"],
        )

    def test_blank_user_contracts_fail_explicitly(self):
        calls = [
            lambda: build_model({}),
            lambda: run_model(None, None),
            lambda: validate_model_output(None, {}),
            describe_input_features,
            lambda: build_contract_probe(2),
            lambda: build_smoke_loaders(sample_count=8, batch_size=4, seed=42),
            build_tool_smoke_payload,
            lambda: predict({}),
            lambda: train_model(
                None,
                [],
                epochs=1,
                learning_rate=0.001,
                device=torch.device("cpu"),
            ),
        ]
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(NotImplementedError, "Implement"):
                call()

    def test_evaluation_output_contract_is_generic(self):
        result = normalize_evaluation((0.5, 0.75, {"custom_score": 0.25}))
        self.assertEqual(result, (0.5, 0.75, {"custom_score": 0.25}))
        with self.assertRaisesRegex(ValueError, "metrics"):
            normalize_evaluation((0.5, 0.75, None))

    def test_release_manifest_contains_only_workspace_files(self):
        manifest = build_manifest()
        self.assertEqual(manifest["baseline"]["release_version"], "0.6.0")
        paths = {entry["path"] for entry in manifest["files"]}
        self.assertIn("federated_task/task_check.py", paths)
        self.assertIn("federated_task/server_main.py", paths)
        self.assertIn("federated_task/client_main.py", paths)
        self.assertIn("model_release/manifest.json", paths)
        self.assertFalse(any(path.startswith("tests/") or path.startswith("tools/") for path in paths))
        for entry in manifest["files"]:
            self.assertTrue((BASELINE / entry["path"]).is_file())

    def test_tool_manifest_is_valid_json_template(self):
        manifest = json.loads(
            (BASELINE / "federated_task/manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["entrypoint"], "federated_task.tool:predict")
        self.assertIn("replace_with_input", manifest["input"]["properties"])

    def test_fedops_client_and_server_entrypoints_are_present(self):
        self.assertTrue(callable(client_main))
        self.assertTrue(callable(server_main))

    def test_authoring_boundaries_are_visible_in_workspace_files(self):
        readme = (BASELINE / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Start here", readme)
        self.assertIn("## What can be edited", readme)
        self.assertIn("## Fixed Python contracts", readme)
        for relative in (
            "federated_task/model.py",
            "federated_task/data_preparation.py",
            "federated_task/training.py",
            "federated_task/tool.py",
        ):
            source = (BASELINE / relative).read_text(encoding="utf-8")
            self.assertIn("FEDOPS CONTRACT", source, relative)
        training_source = (BASELINE / "federated_task/training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("FEDOPS RUNTIME - DO NOT EDIT", training_source)


if __name__ == "__main__":
    unittest.main()
