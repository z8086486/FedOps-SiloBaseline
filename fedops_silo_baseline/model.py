"""Owner-editable MNIST model, training, and evaluation contract."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as functional


class MNISTClassifier(nn.Module):
    """Small convolutional classifier used by the starter Task."""

    def __init__(self, output_size: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.pool(functional.relu(self.conv1(inputs)))
        values = self.pool(functional.relu(self.conv2(values)))
        values = torch.flatten(values, start_dim=1)
        values = functional.relu(self.fc1(values))
        return self.fc2(values)


def build_model(config: Optional[Dict[str, Any]] = None) -> MNISTClassifier:
    model_config = config or {}
    return MNISTClassifier(output_size=int(model_config.get("output_size", 10)))


def train_model(
    model: nn.Module,
    loader: Iterable,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """Train locally and return mean cross-entropy loss."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0.0
    completed_batches = 0

    for _ in range(epochs):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            completed_batches += 1
            if max_batches is not None and completed_batches >= max_batches:
                model.to("cpu")
                return total_loss / completed_batches

    model.to("cpu")
    return total_loss / max(1, completed_batches)


def _weighted_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels = labels.to("cpu")
    predictions = predictions.to("cpu")
    sample_count = int(labels.numel())
    if sample_count == 0:
        return 0.0

    weighted_total = 0.0
    for class_id in torch.unique(labels).tolist():
        expected = labels == class_id
        predicted = predictions == class_id
        true_positive = int((expected & predicted).sum().item())
        false_positive = int((~expected & predicted).sum().item())
        false_negative = int((expected & ~predicted).sum().item())
        support = int(expected.sum().item())
        denominator = (2 * true_positive) + false_positive + false_negative
        class_f1 = (2 * true_positive / denominator) if denominator else 0.0
        weighted_total += class_f1 * support
    return weighted_total / sample_count


def evaluate_model(
    model: nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate locally and return loss, accuracy, and weighted F1."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    completed_batches = 0
    labels_seen = []
    predictions_seen = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            total_loss += float(criterion(outputs, labels).item())
            predictions = outputs.argmax(dim=1)
            total_samples += int(labels.size(0))
            correct += int((predictions == labels).sum().item())
            completed_batches += 1
            labels_seen.append(labels)
            predictions_seen.append(predictions)
            if max_batches is not None and completed_batches >= max_batches:
                break

    model.to("cpu")
    labels_tensor = torch.cat(labels_seen) if labels_seen else torch.empty(0, dtype=torch.long)
    predictions_tensor = (
        torch.cat(predictions_seen) if predictions_seen else torch.empty(0, dtype=torch.long)
    )
    return (
        total_loss / max(1, completed_batches),
        correct / max(1, total_samples),
        {"f1_score": _weighted_f1(labels_tensor, predictions_tensor)},
    )


def train_torch():
    """Return the callback expected by ``fedops.client.app.FLClientTask``."""

    def callback(model, train_loader, epochs, cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_model(
            model,
            train_loader,
            epochs=int(epochs),
            learning_rate=float(cfg.learning_rate),
            device=device,
        )
        return model

    return callback


def test_torch():
    """Return the evaluation callback expected by FedOps."""

    def callback(model, test_loader, cfg):
        del cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return evaluate_model(model, test_loader, device=device)

    return callback
