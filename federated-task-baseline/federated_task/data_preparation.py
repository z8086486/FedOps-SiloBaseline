"""Task data contract shared by local, client, server, and Tool AI paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


def describe_input_features() -> dict[str, Any]:
    return {
        "features": [{
            "name": "image",
            "shape": [1, 28, 28],
            "dtype": "float32",
            "range": [-1.0, 1.0],
            "normalization": {"mean": [0.5], "std": [0.5]},
        }],
        "label": {
            "name": "digit",
            "dtype": "int64",
            "classes": list(range(10)),
        },
        "raw_data_upload": False,
    }


def preprocess(sample: dict[str, Any]) -> torch.Tensor:
    image = transforms.ToTensor()(np.array(sample["image"], dtype=np.uint8, copy=True))
    return transforms.Normalize((0.5,), (0.5,))(image)


def load_partition(
    dataset: str,
    validation_split: float,
    batch_size: int,
    *,
    data_root: str,
    seed: int = 42,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if dataset.upper() != "MNIST":
        raise ValueError(f"This starter supports MNIST, received {dataset!r}")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    full_dataset = datasets.MNIST(
        root=str(Path(data_root)),
        train=True,
        download=download,
        transform=lambda image: preprocess({"image": image}),
    )
    validation_size = max(1, int(validation_split * len(full_dataset)))
    test_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - validation_size - test_size
    if train_size < 1:
        raise ValueError("validation_split leaves no samples for local training")
    generator = torch.Generator().manual_seed(seed)
    train_data, validation_data, test_data = random_split(
        full_dataset,
        [train_size, validation_size, test_size],
        generator=generator,
    )
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(validation_data, batch_size=batch_size),
        DataLoader(test_data, batch_size=batch_size),
    )


def build_smoke_loaders(
    *, sample_count: int = 32, batch_size: int = 8, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    if sample_count < 8:
        raise ValueError("sample_count must be at least 8")
    generator = torch.Generator().manual_seed(seed)
    images = (torch.rand(sample_count, 1, 28, 28, generator=generator) * 2.0) - 1.0
    labels = torch.arange(sample_count, dtype=torch.long) % 10
    dataset = TensorDataset(images, labels)
    validation_size = max(2, sample_count // 4)
    train_size = sample_count - validation_size
    train_data, validation_data = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(
            train_data,
            batch_size=min(batch_size, train_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        DataLoader(validation_data, batch_size=min(batch_size, validation_size)),
    )


def gl_model_torch_validation(
    batch_size: int,
    *,
    data_root: str,
    download: bool = False,
) -> DataLoader:
    """Load the server-side validation set used by the FedOps 1.2 server path."""
    dataset = datasets.MNIST(
        root=str(Path(data_root)),
        train=False,
        download=download,
        transform=lambda image: preprocess({"image": image}),
    )
    return DataLoader(dataset, batch_size=batch_size)
