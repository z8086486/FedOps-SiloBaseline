"""FEDOPS RUNTIME FILE - structured local-training progress events."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any


EVENT_PREFIX = "FEDOPS_PROGRESS "
STAGE_RANGES = {
    "preparing": (0.0, 5.0),
    "loading-data": (5.0, 10.0),
    "training": (10.0, 75.0),
    "evaluating": (75.0, 92.0),
    "exporting": (92.0, 99.0),
    "completed": (100.0, 100.0),
}


def emit_progress(
    stage: str,
    fraction: float,
    message: str,
    *,
    metrics: Mapping[str, float] | None = None,
    epoch: int | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    total_batches: int | None = None,
) -> None:
    """Emit one machine-readable event without changing Task hook signatures."""
    if stage not in STAGE_RANGES:
        raise ValueError(f"Unsupported progress stage: {stage}")
    bounded_fraction = min(1.0, max(0.0, float(fraction)))
    start, end = STAGE_RANGES[stage]
    normalized_metrics: dict[str, float] = {}
    for name, value in (metrics or {}).items():
        number = float(value)
        if not math.isfinite(number):
            continue
        normalized_metrics[str(name)] = number
    event: dict[str, Any] = {
        "schemaVersion": 1,
        "stage": stage,
        "percent": start + ((end - start) * bounded_fraction),
        "message": str(message),
        "timestamp": datetime.now(UTC).isoformat(),
        "metrics": normalized_metrics,
    }
    optional = {
        "epoch": epoch,
        "epochs": epochs,
        "batch": batch,
        "totalBatches": total_batches,
    }
    event.update({name: int(value) for name, value in optional.items() if value is not None})
    print(EVENT_PREFIX + json.dumps(event, separators=(",", ":")), flush=True)


def emit_training_metrics(
    *,
    completed_batches: int,
    total_batches: int,
    metrics: Mapping[str, float],
    epoch: int | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    batches_per_epoch: int | None = None,
) -> None:
    """Optional helper for Task code to report live loss or other metrics."""
    emit_progress(
        "training",
        completed_batches / max(1, total_batches),
        "Training local model",
        metrics=metrics,
        epoch=epoch,
        epochs=epochs,
        batch=batch,
        total_batches=batches_per_epoch,
    )


def emit_evaluation_metrics(
    *,
    completed_batches: int,
    total_batches: int,
    metrics: Mapping[str, float],
    batch: int | None = None,
) -> None:
    """Optional helper for Task code to report live validation metrics."""
    emit_progress(
        "evaluating",
        completed_batches / max(1, total_batches),
        "Evaluating local model",
        metrics=metrics,
        batch=batch,
        total_batches=total_batches,
    )


class ProgressLoader(Iterable):
    """Transparent loader wrapper that guarantees progress even without metrics."""

    def __init__(
        self,
        loader: Iterable,
        *,
        stage: str,
        iterations: int = 1,
        max_batches: int | None = None,
    ) -> None:
        self.loader = loader
        self.stage = stage
        self.iterations = max(1, int(iterations))
        try:
            batches_per_iteration = len(loader)  # type: ignore[arg-type]
        except TypeError:
            batches_per_iteration = 0
        expected = batches_per_iteration * self.iterations
        self.total = min(expected, max_batches) if max_batches and expected else expected
        self.completed = 0
        self.iteration = 0

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[Any]:
        self.iteration += 1
        try:
            batch_count = len(self.loader)  # type: ignore[arg-type]
        except TypeError:
            batch_count = 0
        for batch_index, value in enumerate(self.loader, start=1):
            self.completed += 1
            emit_progress(
                self.stage,
                self.completed / max(1, self.total or self.completed),
                "Training local model" if self.stage == "training" else "Evaluating local model",
                epoch=self.iteration if self.stage == "training" else None,
                epochs=self.iterations if self.stage == "training" else None,
                batch=batch_index,
                total_batches=batch_count or None,
            )
            yield value

