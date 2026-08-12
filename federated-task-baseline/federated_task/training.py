"""Training callbacks shared by local development and FedOps 1.2 FL runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from fedops.client.parameter_contract import parameter_signature

from .config import load_config
from .data_preparation import build_smoke_loaders, load_partition
from .model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_RELEASE_DIR = PROJECT_ROOT / "model_release"
MODEL_PATH = MODEL_RELEASE_DIR / "model.safetensors"
MODEL_MANIFEST_PATH = MODEL_RELEASE_DIR / "manifest.json"


def train_model(
    model: nn.Module,
    loader: Iterable,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0.0
    completed = 0
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            completed += 1
            if max_batches is not None and completed >= max_batches:
                model.to("cpu")
                return total_loss / completed
    model.to("cpu")
    return total_loss / max(1, completed)


def _weighted_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels, predictions = labels.cpu(), predictions.cpu()
    if not labels.numel():
        return 0.0
    total = 0.0
    for class_id in torch.unique(labels).tolist():
        expected, predicted = labels == class_id, predictions == class_id
        true_positive = int((expected & predicted).sum())
        false_positive = int((~expected & predicted).sum())
        false_negative = int((expected & ~predicted).sum())
        support = int(expected.sum())
        denominator = (2 * true_positive) + false_positive + false_negative
        total += ((2 * true_positive / denominator) if denominator else 0.0) * support
    return total / int(labels.numel())


def evaluate_model(
    model: nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, dict[str, float]]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    completed = 0
    labels_seen: list[torch.Tensor] = []
    predictions_seen: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += float(criterion(outputs, labels).item())
            predictions = outputs.argmax(dim=1)
            total_samples += int(labels.size(0))
            correct += int((predictions == labels).sum().item())
            completed += 1
            labels_seen.append(labels)
            predictions_seen.append(predictions)
            if max_batches is not None and completed >= max_batches:
                break
    model.to("cpu")
    labels_tensor = torch.cat(labels_seen) if labels_seen else torch.empty(0, dtype=torch.long)
    predictions_tensor = (
        torch.cat(predictions_seen) if predictions_seen else torch.empty(0, dtype=torch.long)
    )
    return (
        total_loss / max(1, completed),
        correct / max(1, total_samples),
        {"f1_score": _weighted_f1(labels_tensor, predictions_tensor)},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_released_model(path: Path = MODEL_PATH) -> nn.Module:
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
    train_loss = train_model(
        model,
        train_loader,
        epochs=int(config["num_epochs"]),
        learning_rate=float(config["learning_rate"]),
        device=device,
        max_batches=max_batches,
    )
    validation_loss, accuracy, metrics = evaluate_model(
        model,
        validation_loader,
        device=device,
        max_batches=max_batches,
    )
    values = [train_loss, validation_loss, accuracy, metrics["f1_score"]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("local training produced a non-finite metric")

    MODEL_RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
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
        "architecture": "federated_task.model:build_model",
        "size": MODEL_PATH.stat().st_size,
        "sha256": _sha256(MODEL_PATH),
        "parameterSignature": signature,
        "inputContract": "federated_task/manifest.json",
        "trainingData": "synthetic-smoke" if synthetic else "local-only",
        "metrics": {
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "accuracy": accuracy,
            "weightedF1": metrics["f1_score"],
        },
    }
    MODEL_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def train_torch():
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
    def callback(model, test_loader, cfg):
        del cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return evaluate_model(model, test_loader, device=device)

    return callback
