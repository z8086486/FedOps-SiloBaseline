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

from federated_task.config import load_config, resolve_runtime_config, validate_config
from federated_task.federated_learning.client_main import main as client_main
from federated_task.federated_learning.server_main import main as server_main
from federated_task.local_training.data_preparation import (
    build_contract_probe,
    build_smoke_loaders,
    describe_input_features,
    load_partition,
)
from federated_task.local_training.model import build_model, run_model, validate_model_output
from federated_task.local_training.training import evaluate_model, train_model
from federated_task.runtime.model_release import normalize_evaluation
from federated_task.tool_ai.tool import build_tool_smoke_payload, predict
from tools.build_release import build_manifest


class BaselineContractTest(unittest.TestCase):
    def test_generic_config_is_valid_and_has_no_mnist_contract(self):
        config = load_config()
        self.assertEqual(config["model_type"], "Pytorch")
        self.assertEqual(config["dataset"]["name"], "replace-with-dataset-name")
        self.assertNotIn("output_size", config["model"])
        config["local_training"]["batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_config(config)

    def test_campaign_overlay_is_validated_and_normalized(self):
        config = load_config()
        source = {
            key: config[key]
            for key in (
                "schema_version", "random_seed", "model_type", "model", "dataset",
                "local_training", "federation", "monitoring",
            )
        }
        runtime = resolve_runtime_config(source, {
            "schemaVersion": 1,
            "rounds": 7,
            "clientsPerRound": 3,
            "strategy": {"name": "FedAvg", "parameters": {"fraction_fit": 0.8}},
        })
        self.assertEqual(runtime["num_rounds"], 7)
        self.assertEqual(runtime["clients_per_round"], 3)
        self.assertEqual(runtime["server"]["strategy"]["min_fit_clients"], 3)
        self.assertEqual(runtime["server"]["strategy"]["fraction_fit"], 0.8)
        with self.assertRaisesRegex(ValueError, "not supported"):
            resolve_runtime_config(source, {"strategy": "FedAdam"})

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
        self.assertEqual(manifest["baseline"]["release_version"], "0.10.0")
        self.assertEqual(manifest["compatibility"]["agent_studio_task_schema"], 3)
        self.assertEqual(manifest["compatibility"]["fedops_participation"], "==1.1.30.15")
        paths = {entry["path"] for entry in manifest["files"]}
        by_path = {entry["path"]: entry for entry in manifest["files"]}
        self.assertIn("federated_task/task_readiness/check.py", paths)
        self.assertIn("federated_task/federated_learning/server_main.py", paths)
        self.assertIn("federated_task/federated_learning/client_main.py", paths)
        self.assertIn("federated_task/local_training/model.py", paths)
        self.assertIn("federated_task/tool_ai/manifest.json", paths)
        self.assertIn("model_release/manifest.json", paths)
        self.assertIn("requirements.txt", paths)
        self.assertFalse(any(path.startswith("tests/") or path.startswith("tools/") for path in paths))
        self.assertTrue(by_path["federated_task/local_training/model.py"]["editable"])
        self.assertTrue(by_path["federated_task/tool_ai/manifest.json"]["editable"])
        self.assertTrue(by_path["requirements.txt"]["editable"])
        self.assertFalse(by_path["pyproject.toml"]["editable"])
        self.assertFalse(by_path["uv.lock"]["editable"])
        self.assertFalse(by_path["federated_task/federated_learning/client_main.py"]["editable"])
        self.assertFalse(by_path["federated_task/runtime/model_release.py"]["editable"])
        for entry in manifest["files"]:
            self.assertTrue((BASELINE / entry["path"]).is_file())

    def test_tool_manifest_is_valid_json_template(self):
        manifest = json.loads(
            (BASELINE / "federated_task/tool_ai/manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["features"], ["replace_with_feature_name"])
        self.assertIn("description", manifest["output"])
        self.assertEqual(manifest["output"]["labels"], [])

    def test_fedops_client_and_server_entrypoints_are_present(self):
        self.assertTrue(callable(client_main))
        self.assertTrue(callable(server_main))

    def test_authoring_boundaries_are_visible_in_workspace_files(self):
        readme = (BASELINE / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Start here", readme)
        self.assertIn("## What can be edited", readme)
        self.assertIn("## Fixed Python contracts", readme)
        self.assertIn("requirements.txt", readme)
        for relative in (
            "federated_task/local_training/model.py",
            "federated_task/local_training/data_preparation.py",
            "federated_task/local_training/training.py",
            "federated_task/tool_ai/tool.py",
        ):
            source = (BASELINE / relative).read_text(encoding="utf-8")
            self.assertIn("FEDOPS CONTRACT", source, relative)
        runtime_source = (BASELINE / "federated_task/runtime/model_release.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("FEDOPS RUNTIME FILE", runtime_source)
        self.assertNotIn("FEDOPS RUNTIME - DO NOT EDIT", (
            BASELINE / "federated_task/local_training/training.py"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
