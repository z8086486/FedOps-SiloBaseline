"""Framework-neutral parameter signature and round-trip checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


def parameter_descriptor(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "elements": int(tensor.numel()),
        }
        for name, tensor in model.state_dict().items()
    ]


def parameter_signature(model: nn.Module) -> dict[str, Any]:
    tensors = parameter_descriptor(model)
    serialized = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schemaVersion": 1,
        "algorithm": "sha256",
        "fingerprint": hashlib.sha256(serialized).hexdigest(),
        "tensorCount": len(tensors),
        "elementCount": sum(item["elements"] for item in tensors),
        "tensors": tensors,
    }


def serialize_parameters(model: nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def apply_parameters(model: nn.Module, values: Sequence[np.ndarray]) -> None:
    state = model.state_dict()
    if len(values) != len(state):
        raise ValueError("parameter tensor count does not match the model")
    restored = {}
    for (name, expected), value in zip(state.items(), values):
        tensor = torch.from_numpy(np.asarray(value))
        if tuple(tensor.shape) != tuple(expected.shape):
            raise ValueError(f"parameter shape mismatch: {name}")
        if tensor.dtype != expected.dtype:
            raise ValueError(f"parameter dtype mismatch: {name}")
        restored[name] = tensor
    model.load_state_dict(restored, strict=True)


def parameter_update(before: Sequence[np.ndarray], after: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(before) != len(after):
        raise ValueError("parameter update tensor count mismatch")
    updates = []
    for old, new in zip(before, after):
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError("parameter update structure mismatch")
        updates.append(new - old)
    return updates


def verify_round_trip(model: nn.Module, model_factory) -> dict[str, Any]:
    values = serialize_parameters(model)
    restored = model_factory()
    apply_parameters(restored, values)
    restored_values = serialize_parameters(restored)
    if not all(np.array_equal(left, right) for left, right in zip(values, restored_values)):
        raise ValueError("parameter round-trip changed model values")
    return {
        "ok": True,
        "signature": parameter_signature(model),
        "payloadBytes": sum(value.nbytes for value in values),
    }
