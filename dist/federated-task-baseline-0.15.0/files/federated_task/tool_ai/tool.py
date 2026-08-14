"""User-owned Agent Builder Tool AI hooks with fixed JSON contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - keep implementation consistent with manifest.json and local training.
def predict(payload: dict[str, Any], model_path: str | Path | None = None) -> dict[str, Any]:
    """Run one inference with an Initial or Global Model release.

    Args:
        payload: JSON object containing every feature named in ``manifest.json``.
        model_path: Optional selected model artifact. When omitted, load the
        Task's local ``model_release/model.safetensors`` artifact.

    Returns:
        A JSON-serializable dictionary matching the Tool manifest output.

    Use ``data_preparation.preprocess()`` and
    ``training.load_released_model()`` so local training, federated learning,
    and Agent Builder inference use one model and preprocessing contract.

    Example implementation outline::

        model = load_released_model(Path(model_path) if model_path else MODEL_PATH)
        inputs = preprocess(payload)
        output = run_model(model, add_batch_dimension(inputs))
        return {"prediction": convert_to_json_value(output)}
    """
    del payload, model_path
    raise NotImplementedError(
        "Implement federated_task.tool_ai.tool.predict() and match manifest.json"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - return safe example JSON matching manifest.json input.
def build_tool_smoke_payload() -> dict[str, Any]:
    """Return one non-sensitive JSON payload accepted by :func:`predict`.

    Example implementation::

        return {"features": [0.0] * 8}
    """
    raise NotImplementedError(
        "Implement federated_task.tool_ai.tool.build_tool_smoke_payload()"
    )
