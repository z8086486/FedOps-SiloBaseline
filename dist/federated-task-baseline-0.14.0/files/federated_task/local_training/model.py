"""User-owned model definition with fixed FedOps integration hooks.

Add model classes and helpers freely. Functions marked ``FEDOPS CONTRACT`` are
imported by local training, Readiness, clients, servers, and Tool inference; keep
their names, arguments, and return structures and replace only their bodies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn


# USER IMPLEMENTATION ---------------------------------------------------------
# Add the Task's torch.nn.Module class above or below this marker.
# Example only:
#
# class TaskModel(nn.Module):
#     def __init__(self, input_size: int, output_size: int):
#         super().__init__()
#         self.network = nn.Linear(input_size, output_size)
#
#     def forward(self, inputs):
#         return self.network(inputs)


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - replace only this implementation body and add helpers as needed.
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

    Example implementation::

        values = config or {}
        return TaskModel(
            input_size=int(values.get("input_size", 8)),
            output_size=int(values.get("output_size", 2)),
        )
    """
    del config
    raise NotImplementedError(
        "Implement federated_task.local_training.model.build_model() with the Task model architecture"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - adapt only how this Task passes its input structure to the model.
def run_model(model: nn.Module, inputs: Any) -> Any:
    """Run one forward pass for readiness and Tool-compatible validation.

    Args:
        model: A model returned by :func:`build_model`.
        inputs: One batched value returned by ``build_contract_probe()``.

    Returns:
        The raw model output. For one tensor input this is normally
        ``model(inputs)``. For multiple inputs it may be
        ``model(*inputs)`` or ``model(**inputs)``.

    Example implementation for one tensor input::

        return model(inputs)
    """
    del model, inputs
    raise NotImplementedError(
        "Implement federated_task.local_training.model.run_model() for the Task input structure"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - validate the Task-specific output without returning tensor values.
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

    Example implementation for ``[batch, classes]`` logits::

        expected = int(config["model"]["output_size"])
        if output.ndim != 2 or output.shape[1] != expected:
            raise ValueError("model output must be [batch, output_size]")
        return {"shape": list(output.shape), "dtype": str(output.dtype)}
    """
    del output, config
    raise NotImplementedError(
        "Implement federated_task.local_training.model.validate_model_output() for readiness"
    )
