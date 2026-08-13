"""FEDOPS RUNTIME FILE - fixed Model Release and FedOps callback adapters."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from fedops.client.parameter_contract import parameter_signature

from ..config import load_config
from ..local_training.data_preparation import build_smoke_loaders, load_partition
from ..local_training.model import build_model
from ..local_training.training import evaluate_model, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_RELEASE_DIR = PROJECT_ROOT / "model_release"
MODEL_PATH = MODEL_RELEASE_DIR / "model.safetensors"
MODEL_MANIFEST_PATH = MODEL_RELEASE_DIR / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def normalize_evaluation(
    result: tuple[float, float, dict[str, float]],
) -> tuple[float, float, dict[str, float]]:
    """Validate the user evaluation result before runtime or release use."""
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("evaluate_model() must return (loss, primary_metric, metrics)")
    loss, primary_metric, metrics = result
    if not isinstance(metrics, dict):
        raise ValueError("evaluate_model() metrics must be a dictionary")
    normalized_metrics = {
        str(name): _finite_float(value, f"metric {name!r}")
        for name, value in metrics.items()
    }
    return (
        _finite_float(loss, "evaluation loss"),
        _finite_float(primary_metric, "primary metric"),
        normalized_metrics,
    )


def load_released_model(path: Path = MODEL_PATH) -> nn.Module:
    """Construct the Task model and load one exact safetensors release."""
    config = load_config()
    model = build_model(config["model"])
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model


def export_initial_model(
    *,
    data_root: str | None = None,
    synthetic: bool = False,
    download: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Train and export the owner Initial Model and immutable identity metadata."""
    config = load_config()
    seed = int(config["random_seed"])
    torch.manual_seed(seed)
    if synthetic:
        train_loader, validation_loader = build_smoke_loaders(
            sample_count=32,
            batch_size=min(int(config["batch_size"]), 8),
            seed=seed,
        )
    else:
        root = data_root or str(config["dataset"]["root"])
        train_loader, validation_loader, _ = load_partition(
            dataset=str(config["dataset"]["name"]),
            validation_split=float(config["dataset"]["validation_split"]),
            batch_size=int(config["batch_size"]),
            data_root=root,
            seed=seed,
            download=download,
        )
    model = build_model(config["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loss = _finite_float(
        train_model(
            model,
            train_loader,
            epochs=int(config["num_epochs"]),
            learning_rate=float(config["learning_rate"]),
            device=device,
            max_batches=max_batches,
        ),
        "training loss",
    )
    validation_loss, primary_metric, additional_metrics = normalize_evaluation(
        evaluate_model(
            model,
            validation_loader,
            device=device,
            max_batches=max_batches,
        )
    )

    MODEL_RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    if not state:
        raise ValueError("build_model() produced no serializable model parameters")
    save_file(state, str(MODEL_PATH))
    signature = parameter_signature(model, str(config["model_type"]))
    manifest = {
        "schemaVersion": 1,
        "status": "ready",
        "role": "registry-bootstrap-model",
        "origin": "centrally-trained",
        "framework": "pytorch",
        "format": "safetensors",
        "artifact": MODEL_PATH.name,
        "architecture": "federated_task.local_training.model:build_model",
        "displayName": str(config["model"]["display_name"]),
        "size": MODEL_PATH.stat().st_size,
        "sha256": _sha256(MODEL_PATH),
        "parameterSignature": signature,
        "inputContract": "federated_task/tool_ai/manifest.json",
        "trainingData": "synthetic-smoke" if synthetic else "local-only",
        "metrics": {
            "trainingLoss": train_loss,
            "validationLoss": validation_loss,
            "primaryMetric": primary_metric,
            "additionalMetrics": additional_metrics,
        },
    }
    MODEL_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def train_torch():
    """Return the callback shape required by the FedOps client."""
    def callback(model, train_loader, epochs, cfg, hp=None):
        learning_rate = (
            float(hp["learning_rate"])
            if hp and hp.get("learning_rate") is not None
            else float(cfg.learning_rate)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_model(
            model,
            train_loader,
            epochs=int(epochs),
            learning_rate=learning_rate,
            device=device,
        )
        return model

    return callback


def test_torch():
    """Return the callback shape required by the FedOps client and server."""
    def callback(model, test_loader, cfg):
        del cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return normalize_evaluation(evaluate_model(model, test_loader, device=device))

    return callback
