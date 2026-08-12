"""User-editable local-data contract for one FedOps Federated Task.

Raw data must remain on the Agent Studio device. Keep the function names,
arguments, and return values below while replacing the implementation gaps.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import DataLoader


def describe_input_features() -> dict[str, Any]:
    """Describe model features, labels, shape, dtype, and preprocessing.

    Returns:
        A JSON-serializable dictionary with at least ``features``, ``label``,
        and ``raw_data_upload``. ``raw_data_upload`` must remain ``False``.

    Example shape only::

        {
            "features": [{"name": "feature", "shape": [8], "dtype": "float32"}],
            "label": {"name": "target", "dtype": "int64", "classes": [0, 1]},
            "raw_data_upload": False,
        }
    """
    raise NotImplementedError(
        "Implement federated_task.data_preparation.describe_input_features()"
    )


def preprocess(sample: Mapping[str, Any]) -> Any:
    """Convert one raw sample into the input structure expected by the model.

    Args:
        sample: One sample read from the owner's or participant's local data.

    Returns:
        A tensor, tuple/list of tensors, or mapping of tensors accepted by
        ``run_model()`` and the training implementation.
    """
    del sample
    raise NotImplementedError(
        "Implement federated_task.data_preparation.preprocess()"
    )


def load_partition(
    dataset: str,
    validation_split: float,
    batch_size: int,
    *,
    data_root: str,
    seed: int = 42,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load local data and return train, validation, and test loaders.

    Args:
        dataset: Logical dataset name from ``conf/config.yaml``.
        validation_split: Fraction reserved for local validation.
        batch_size: Batch size for all returned loaders.
        data_root: Agent Studio's local data binding. Never upload or report it.
        seed: Deterministic split/shuffle seed.
        download: Whether this Task explicitly permits downloading public data.

    Returns:
        Exactly ``(train_loader, validation_loader, test_loader)``. Every batch
        must have the form ``(inputs, targets)`` expected by ``training.py``.
    """
    del dataset, validation_split, batch_size, data_root, seed, download
    raise NotImplementedError(
        "Implement federated_task.data_preparation.load_partition() with local-only data"
    )


def build_smoke_loaders(
    *,
    sample_count: int = 32,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build non-sensitive synthetic loaders for Release Readiness.

    Returns:
        Exactly ``(train_loader, validation_loader)`` with the same batch
        structure, dtypes, and shapes as :func:`load_partition`.

    Do not read private data here. This hook proves the executable contract
    before a participant connects their own local dataset.
    """
    del sample_count, batch_size, seed
    raise NotImplementedError(
        "Implement federated_task.data_preparation.build_smoke_loaders()"
    )


def build_contract_probe(batch_size: int = 2) -> Any:
    """Build one batched model input without reading real user data.

    Returns:
        The exact input structure accepted by ``model.run_model()``. The first
        dimension of tensor values should equal ``batch_size``.
    """
    del batch_size
    raise NotImplementedError(
        "Implement federated_task.data_preparation.build_contract_probe()"
    )


def gl_model_torch_validation(
    batch_size: int,
    *,
    data_root: str,
    download: bool = False,
) -> DataLoader:
    """Load the aggregation server's permitted global-validation dataset.

    Returns:
        One ``DataLoader`` with the same ``(inputs, targets)`` batch contract.

    This must not depend on a participant's private dataset. Use only an
    owner-controlled or explicitly licensed central validation source.
    """
    del batch_size, data_root, download
    raise NotImplementedError(
        "Implement federated_task.data_preparation.gl_model_torch_validation()"
    )
