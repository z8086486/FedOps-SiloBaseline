"""Task model shared by local training, FedOps, and Tool AI inference."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as functional


class MNISTClassifier(nn.Module):
    def __init__(self, output_size: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Keep the FedOps 1.2 MNIST architecture checkpoint-compatible.
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.pool(functional.relu(self.conv1(inputs)))
        values = self.pool(functional.relu(self.conv2(values)))
        values = torch.flatten(values, start_dim=1)
        return self.fc2(functional.relu(self.fc1(values)))


def build_model(config: dict[str, Any] | None = None) -> MNISTClassifier:
    model_config = config or {}
    return MNISTClassifier(output_size=int(model_config.get("output_size", 10)))
