"""FEDOPS RUNTIME FILE - fixed shared config loader; normal Task authors do not edit."""

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
    if not isinstance(model, dict) or not model:
        raise ValueError("config.yaml: model must be a non-empty object")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("config.yaml: dataset must be an object")
    if not isinstance(dataset.get("name"), str) or not dataset["name"].strip():
        raise ValueError("config.yaml: dataset.name must be a non-empty string")
    if not isinstance(dataset.get("root"), str) or not dataset["root"].strip():
        raise ValueError("config.yaml: dataset.root must be a non-empty string")
    split = dataset.get("validation_split")
    if not isinstance(split, (int, float)) or not 0 < float(split) < 1:
        raise ValueError("config.yaml: dataset.validation_split must be between 0 and 1")
    if not isinstance(dataset.get("download"), bool):
        raise ValueError("config.yaml: dataset.download must be boolean")

    wandb = config.get("wandb")
    if not isinstance(wandb, dict) or not isinstance(wandb.get("use"), bool):
        raise ValueError("config.yaml: wandb.use must be boolean")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("config.yaml: runtime must be an object")
    for port_name in ("manager_port", "client_port"):
        port = runtime.get(port_name)
        if not isinstance(port, int) or not 0 < port < 65536:
            raise ValueError(f"config.yaml: runtime.{port_name} must be a valid port")
