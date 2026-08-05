"""Configuration loading and contract validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "conf" / "config.toml"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required_scalars = {
        "random_seed": int,
        "learning_rate": (int, float),
        "model_type": str,
        "task_id": str,
        "runtime_key": str,
        "num_epochs": int,
        "batch_size": int,
        "num_rounds": int,
        "clients_per_round": int,
    }
    for key, expected_type in required_scalars.items():
        value = config.get(key)
        if not isinstance(value, expected_type):
            raise ValueError(f"config.toml: {key} must be {expected_type}, got {value!r}")

    for key in ("task_id", "runtime_key"):
        if not config[key].strip():
            raise ValueError(f"config.toml: {key} must not be empty")

    if config["learning_rate"] <= 0:
        raise ValueError("config.toml: learning_rate must be greater than zero")
    for key in ("num_epochs", "batch_size", "num_rounds", "clients_per_round"):
        if config[key] < 1:
            raise ValueError(f"config.toml: {key} must be at least 1")

    model = config.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("output_size"), int):
        raise ValueError("config.toml: model.output_size must be an integer")
    if model["output_size"] < 2:
        raise ValueError("config.toml: model.output_size must be at least 2")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("name") != "MNIST":
        raise ValueError("config.toml: the starter dataset.name must be MNIST")
    validation_split = dataset.get("validation_split")
    if not isinstance(validation_split, (int, float)) or not 0 < validation_split < 1:
        raise ValueError("config.toml: dataset.validation_split must be between 0 and 1")

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("config.toml: runtime section is required")
    for key in ("server_manager_url", "federated_server_host"):
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            raise ValueError(f"config.toml: runtime.{key} must be a non-empty string")
