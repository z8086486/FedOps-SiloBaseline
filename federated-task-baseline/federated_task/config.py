"""FEDOPS RUNTIME FILE - Release config and Campaign overlay resolver."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "conf" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = path or DEFAULT_CONFIG_PATH
    source = OmegaConf.to_container(OmegaConf.load(selected), resolve=True)
    if not isinstance(source, dict):
        raise ValueError("config.yaml must contain an object")
    validate_config(source)
    return resolve_runtime_config(source)


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version": int,
        "random_seed": int,
        "model_type": str,
    }
    for key, expected in required.items():
        if not isinstance(config.get(key), expected):
            raise ValueError(f"config.yaml: {key} has an invalid value")
    if config["model_type"] != "Pytorch":
        raise ValueError("config.yaml: this starter model_type must be Pytorch")
    if config["schema_version"] != 2:
        raise ValueError("config.yaml: schema_version must be 2")

    model = config.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("config.yaml: model must be a non-empty object")
    if not isinstance(model.get("display_name"), str) or not model["display_name"].strip():
        raise ValueError("config.yaml: model.display_name must be a non-empty string")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("config.yaml: dataset must be an object")
    if not isinstance(dataset.get("name"), str) or not dataset["name"].strip():
        raise ValueError("config.yaml: dataset.name must be a non-empty string")
    split = dataset.get("validation_split")
    if not isinstance(split, (int, float)) or not 0 < float(split) < 1:
        raise ValueError("config.yaml: dataset.validation_split must be between 0 and 1")
    if not isinstance(dataset.get("download"), bool):
        raise ValueError("config.yaml: dataset.download must be boolean")

    local_training = config.get("local_training")
    if not isinstance(local_training, dict):
        raise ValueError("config.yaml: local_training must be an object")
    learning_rate = local_training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0:
        raise ValueError("config.yaml: local_training.learning_rate must be greater than zero")
    for key in ("epochs", "batch_size"):
        value = local_training.get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"config.yaml: local_training.{key} must be at least 1")

    federation = config.get("federation")
    if not isinstance(federation, dict):
        raise ValueError("config.yaml: federation must be an object")
    strategies = federation.get("supported_strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("config.yaml: federation.supported_strategies must not be empty")
    names: set[str] = set()
    for strategy in strategies:
        if not isinstance(strategy, dict):
            raise ValueError("config.yaml: each supported strategy must be an object")
        name = str(strategy.get("name") or "").strip()
        target = str(strategy.get("target") or "").strip()
        if not name or not target:
            raise ValueError("config.yaml: supported strategy name and target are required")
        if name in names:
            raise ValueError("config.yaml: supported strategy names must be unique")
        names.add(name)
    recommended = federation.get("recommended_campaign")
    if not isinstance(recommended, dict):
        raise ValueError("config.yaml: federation.recommended_campaign must be an object")
    for key in ("rounds", "clients_per_round"):
        if not isinstance(recommended.get(key), int) or recommended[key] < 1:
            raise ValueError(f"config.yaml: federation.recommended_campaign.{key} must be at least 1")
    if recommended.get("strategy") not in names:
        raise ValueError("config.yaml: recommended strategy is not supported")

    monitoring = config.get("monitoring")
    if not isinstance(monitoring, dict) or not isinstance(monitoring.get("wandb"), bool):
        raise ValueError("config.yaml: monitoring.wandb must be boolean")


def _campaign_from_environment() -> dict[str, Any] | None:
    raw = os.environ.get("FEDOPS_CAMPAIGN_CONFIG", "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("FEDOPS_CAMPAIGN_CONFIG must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("FEDOPS_CAMPAIGN_CONFIG must contain an object")
    return value


def resolve_runtime_config(
    source: dict[str, Any],
    campaign: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert the Release contract to the FedOps-compatible runtime shape."""
    selected = campaign if campaign is not None else _campaign_from_environment()
    recommended = source["federation"]["recommended_campaign"]
    selected = selected or {}
    schema_version = selected.get("schemaVersion", 1)
    if schema_version != 1:
        raise ValueError("Campaign overlay schemaVersion must be 1")
    rounds = int(selected.get("rounds", recommended["rounds"]))
    clients = int(selected.get("clientsPerRound", recommended["clients_per_round"]))
    if rounds < 1 or clients < 1:
        raise ValueError("Campaign rounds and clientsPerRound must be at least 1")

    strategy_value = selected.get("strategy", recommended["strategy"])
    if isinstance(strategy_value, str):
        strategy_name = strategy_value
        parameters: dict[str, Any] = {}
    elif isinstance(strategy_value, dict):
        strategy_name = str(strategy_value.get("name") or "")
        parameters = strategy_value.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("Campaign strategy.parameters must be an object")
    else:
        raise ValueError("Campaign strategy must be a name or object")
    supported = {
        item["name"]: item for item in source["federation"]["supported_strategies"]
    }
    if strategy_name not in supported:
        raise ValueError(f"Campaign strategy {strategy_name!r} is not supported by this Release")
    allowed_parameters = {"fraction_fit", "fraction_evaluate"}
    unknown = set(parameters) - allowed_parameters
    if unknown:
        raise ValueError(f"Campaign strategy has unsupported parameters: {', '.join(sorted(unknown))}")

    strategy = {
        "_target_": supported[strategy_name]["target"],
        "fraction_fit": float(parameters.get("fraction_fit", 1.0)),
        "fraction_evaluate": float(parameters.get("fraction_evaluate", 1.0)),
        "min_fit_clients": clients,
        "min_available_clients": clients,
        "min_evaluate_clients": clients,
    }
    training = source["local_training"]
    dataset = dict(source["dataset"])
    dataset["root"] = "./dataset"
    return {
        **source,
        "learning_rate": float(training["learning_rate"]),
        "num_epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "num_rounds": rounds,
        "clients_per_round": clients,
        "task_id": "task_id",
        "dataset": dataset,
        "wandb": {"use": bool(source["monitoring"]["wandb"]), "key": "", "account": "", "project": "fedops"},
        "runtime": {"manager_port": 8004, "client_port": 8003},
        "server": {"strategy": strategy},
        "campaign": {
            "schemaVersion": 1,
            "rounds": rounds,
            "clientsPerRound": clients,
            "strategy": {"name": strategy_name, "parameters": parameters},
        },
    }
