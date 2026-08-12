"""User-editable Agent Builder Tool AI contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def predict(payload: dict[str, Any], model_path: str | Path | None = None) -> dict[str, Any]:
    """Run one inference with an Initial or Global Model release.

    Args:
        payload: JSON object matching ``federated_task/manifest.json`` input.
        model_path: Optional selected model artifact. When omitted, load the
        Task's local ``model_release/model.safetensors`` artifact.

    Returns:
        A JSON-serializable dictionary matching the Tool manifest output.

    Use ``data_preparation.preprocess()`` and
    ``training.load_released_model()`` so local training, federated learning,
    and Agent Builder inference use one model and preprocessing contract.
    """
    del payload, model_path
    raise NotImplementedError(
        "Implement federated_task.tool.predict() and match manifest.json"
    )


def build_tool_smoke_payload() -> dict[str, Any]:
    """Return one non-sensitive JSON payload accepted by :func:`predict`."""
    raise NotImplementedError(
        "Implement federated_task.tool.build_tool_smoke_payload()"
    )
