"""Owner-editable local input and data preparation contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


def describe_input_features() -> Dict[str, Any]:
    """Describe the model input and label contract published with the Task."""
    return {
        "features": [{
            "name": "image",
            "shape": [1, 28, 28],
            "dtype": "float32",
            "range": [-1.0, 1.0],
            "normalization": {
                "mean": [0.5],
                "std": [0.5],
            },
        }],
        "label": {
            "name": "digit",
            "dtype": "int64",
            "classes": list(range(10)),
        },
        "raw_data_upload": False,
    }


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def preprocess(
    dataset_path: str,
    batch_size: int,
) -> DataLoader:
    """Load raw MNIST inputs from JSON and return an inference DataLoader."""
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError("dataset_path must be a non-empty string")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    input_path = Path(dataset_path)
    if not input_path.is_file():
        raise ValueError(f"dataset_path must point to a file: {dataset_path}")
    try:
        with input_path.open("r", encoding="utf-8") as file:
            inputs = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"dataset_path must contain valid UTF-8 JSON: {dataset_path}"
        ) from error
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("dataset JSON must be a non-empty array")

    samples = []
    for row_index, row in enumerate(inputs):
        if not isinstance(row, dict):
            raise ValueError(f"input {row_index} must be an object")
        if "image" not in row:
            raise ValueError(f"input {row_index} is missing feature: image")

        try:
            image = torch.as_tensor(row["image"], dtype=torch.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"input {row_index} image must be a numeric array"
            ) from error

        if list(image.shape) == [28, 28]:
            image = image.unsqueeze(0)
        if list(image.shape) != [1, 28, 28]:
            raise ValueError(
                f"input {row_index} image must have shape [28, 28] "
                f"or [1, 28, 28], received {list(image.shape)}"
            )
        if not bool(torch.isfinite(image).all().item()):
            raise ValueError(f"input {row_index} image contains a non-finite value")
        if bool(((image < 0) | (image > 255)).any().item()):
            raise ValueError(
                f"input {row_index} pixel values must be between 0 and 255"
            )

        image = image.div(255.0)
        image = image.sub(0.5).div(0.5)
        samples.append({"inputs": image})

    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def load_partition(
    dataset: str,
    validation_split: float,
    batch_size: int,
    *,
    data_root: str = "./dataset/mnist",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load local MNIST and deterministically create train/validation/test splits."""
    if dataset.upper() != "MNIST":
        raise ValueError(f"This starter supports MNIST, received {dataset!r}")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    full_dataset = datasets.MNIST(
        root=str(Path(data_root)),
        train=True,
        download=True,
        transform=mnist_transform(),
    )
    test_fraction = 0.2
    validation_size = max(1, int(validation_split * len(full_dataset)))
    test_size = max(1, int(test_fraction * len(full_dataset)))
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


def build_validation_loaders(
    *,
    sample_count: int = 32,
    batch_size: int = 8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create network-free MNIST-shaped data for contract validation."""
    if sample_count < 8:
        raise ValueError("sample_count must be at least 8")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

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
