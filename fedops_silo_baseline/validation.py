"""Network-free validation for owner-edited Baseline projects."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .config import load_config
from .data_preparation import build_validation_loaders, describe_input_features
from .model import build_model, evaluate_model, train_model


def validate_baseline(
    *,
    config_path: Optional[Path] = None,
    sample_count: int = 32,
    max_batches: int = 2,
) -> Dict[str, Any]:
    """Validate config, input shape, model output, train, and evaluation."""
    if max_batches < 1:
        raise ValueError("max_batches must be at least 1")
    config = load_config(config_path)
    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    contract = describe_input_features()
    feature = contract["features"][0]
    input_shape = feature["shape"]
    output_size = int(config["model"]["output_size"])
    expected_classes = len(contract["label"]["classes"])
    if output_size != expected_classes:
        raise ValueError(
            f"model.output_size={output_size} does not match {expected_classes} label classes"
        )

    train_loader, validation_loader = build_validation_loaders(
        sample_count=sample_count,
        batch_size=min(int(config["batch_size"]), 8),
        seed=seed,
    )
    model = build_model(config["model"])
    probe = torch.zeros(2, *input_shape)
    output = model(probe)
    if list(output.shape) != [2, output_size]:
        raise ValueError(
            f"model output shape {list(output.shape)} does not match [2, {output_size}]"
        )

    device = torch.device("cpu")
    train_loss = train_model(
        model,
        train_loader,
        epochs=1,
        learning_rate=float(config["learning_rate"]),
        device=device,
        max_batches=max_batches,
    )
    validation_loss, validation_accuracy, metrics = evaluate_model(
        model,
        validation_loader,
        device=device,
        max_batches=max_batches,
    )
    numeric_values = [
        train_loss,
        validation_loss,
        validation_accuracy,
        metrics["f1_score"],
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("validation produced a non-finite metric")

    return {
        "ok": True,
        "mode": "validate",
        "task_id": config["task_id"],
        "model": type(model).__name__,
        "input_shape": input_shape,
        "output_shape": list(output.shape),
        "samples": sample_count,
        "max_batches": max_batches,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "weighted_f1": metrics["f1_score"],
        "raw_data_uploaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a FedOps Silo Baseline")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=2)
    args = parser.parse_args()
    try:
        result = validate_baseline(
            config_path=args.config,
            sample_count=args.samples,
            max_batches=args.max_batches,
        )
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
