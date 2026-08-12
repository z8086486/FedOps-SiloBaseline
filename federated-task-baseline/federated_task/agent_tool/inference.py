"""Load the selected Model Release and run one prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..local_training.data_preparation import preprocess
from ..local_training.train import MODEL_PATH, load_released_model


def predict(payload: dict[str, Any], model_path: str | Path | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict) or "image" not in payload:
        raise ValueError("input must contain an image")
    selected_path = Path(model_path) if model_path else MODEL_PATH
    model = load_released_model(selected_path)
    model.eval()
    input_tensor = preprocess({"image": payload["image"]}).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
    label = int(probabilities.argmax().item())
    return {"label": label, "confidence": float(probabilities[label].item())}
