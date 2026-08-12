"""User-editable model contract for one FedOps Federated Task.

Keep the public function names, arguments, and return values. Replace each
``NotImplementedError`` with the Task model implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn


def build_model(config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build a new model instance with the Federated Task architecture.

    Args:
        config: The ``model`` object from ``conf/config.yaml``.

    Returns:
        A new ``torch.nn.Module``. Every owner, participant, and aggregation
        server must construct the same parameter names, shapes, and dtypes.

    Implementation guidance:
        Define the model class in this file (or import it from another source
        file) and return it here. Do not load participant data or contact the
        FedOps server in this function.
    """
    del config
    raise NotImplementedError(
        "Implement federated_task.model.build_model() with the Task model architecture"
    )


def run_model(model: nn.Module, inputs: Any) -> Any:
    """Run one forward pass for readiness and Tool-compatible validation.

    Args:
        model: A model returned by :func:`build_model`.
        inputs: One batched value returned by ``build_contract_probe()``.

    Returns:
        The raw model output. For one tensor input this is normally
        ``model(inputs)``. For multiple inputs it may be
        ``model(*inputs)`` or ``model(**inputs)``.
    """
    del model, inputs
    raise NotImplementedError(
        "Implement federated_task.model.run_model() for the Task input structure"
    )


def validate_model_output(output: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a probe output and return a JSON-serializable summary.

    Args:
        output: The result of :func:`run_model` for a batched contract probe.
        config: The complete Task configuration.

    Returns:
        A JSON-serializable dictionary describing the verified output, for
        example ``{"shape": [2, 10], "dtype": "float32"}``.

    Raise ``ValueError`` when the output cannot satisfy the Task's documented
    output contract.
    """
    del output, config
    raise NotImplementedError(
        "Implement federated_task.model.validate_model_output() for readiness"
    )
