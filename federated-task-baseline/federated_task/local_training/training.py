"""Task-owner local training and evaluation hooks."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


# Add Task-specific losses, metrics, and optimizer helpers in this file.

# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - update ``model`` in place and honor ``max_batches`` when provided.
def train_model(
    model: nn.Module,
    loader: Iterable,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Train ``model`` with local batches and return the mean training loss.

    Args:
        model: The local model to update in place.
        loader: Batches shaped as ``(inputs, targets)``.
        epochs: Number of local epochs.
        learning_rate: Effective local learning rate.
        device: Selected CPU, CUDA, or accelerator device.
        max_batches: Optional readiness-only limit; honor it when provided.

    Returns:
        One finite ``float`` representing mean training loss. Move the model
        back to CPU before returning so FedOps can serialize its parameters.

    Example implementation outline::

        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for _ in range(epochs):
            for inputs, targets in loader:
                # move values to device, compute loss, backward, and optimizer.step()
                # stop once max_batches is reached when it is not None
                ...
        model.to("cpu")
        return float(mean_loss)
    """
    del model, loader, epochs, learning_rate, device, max_batches
    raise NotImplementedError(
        "Implement federated_task.local_training.training.train_model() with the Task loss and optimizer"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - use the Task's loss and documented primary/additional metrics.
def evaluate_model(
    model: nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Evaluate a model and return the fixed FedOps evaluation tuple.

    Returns:
        Exactly ``(loss, primary_metric, additional_metrics)`` where the first
        two values are finite floats and ``additional_metrics`` is a mapping of
        metric names to finite float values. The primary metric may be accuracy,
        F1, MAE, RMSE, or another Task-appropriate measure documented in README.

    Example return::

        return float(mean_loss), float(accuracy), {"weighted_f1": float(f1)}
    """
    del model, loader, device, max_batches
    raise NotImplementedError(
        "Implement federated_task.local_training.training.evaluate_model() with Task metrics"
    )
