"""Load the same Hydra YAML used by FedOps client and server entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "conf" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = path or DEFAULT_CONFIG_PATH
    config = OmegaConf.to_container(OmegaConf.load(selected), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain an object")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "random_seed": int,
        "learning_rate": (int, float),
        "model_type": str,
        "num_epochs": int,
        "batch_size": int,
        "num_rounds": int,
        "clients_per_round": int,
    }
    for key, expected in required.items():
        if not isinstance(config.get(key), expected):
            raise ValueError(f"config.yaml: {key} has an invalid value")
    if config["model_type"] != "Pytorch":
        raise ValueError("config.yaml: this starter model_type must be Pytorch")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("config.yaml: learning_rate must be greater than zero")
    for key in ("num_epochs", "batch_size", "num_rounds", "clients_per_round"):
        if int(config[key]) < 1:
            raise ValueError(f"config.yaml: {key} must be at least 1")

    model = config.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("output_size"), int):
        raise ValueError("config.yaml: model.output_size must be an integer")
    if model["output_size"] < 2:
        raise ValueError("config.yaml: model.output_size must be at least 2")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("name") != "MNIST":
        raise ValueError("config.yaml: starter dataset.name must be MNIST")
    split = dataset.get("validation_split")
    if not isinstance(split, (int, float)) or not 0 < float(split) < 1:
        raise ValueError("config.yaml: dataset.validation_split must be between 0 and 1")

    wandb = config.get("wandb")
    if not isinstance(wandb, dict) or not isinstance(wandb.get("use"), bool):
        raise ValueError("config.yaml: wandb.use must be boolean")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("manager_port"), int):
        raise ValueError("config.yaml: runtime.manager_port must be an integer")
