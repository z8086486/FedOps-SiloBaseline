"""User-owned local-data adapter with fixed FedOps integration hooks.

Raw data remains on the Agent Studio device. Keep every ``FEDOPS CONTRACT``
function name, argument, default value, keyword-only marker, and return structure.
Real and synthetic loaders must yield the same ``(inputs, targets)`` batch shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import DataLoader


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - document the exact feature and label contract as JSON-safe metadata.
def describe_input_features() -> dict[str, Any]:
    """Describe model features, labels, shape, dtype, and preprocessing.

    Returns:
        A JSON-serializable dictionary with at least ``features``, ``label``,
        and ``raw_data_upload``. ``raw_data_upload`` must remain ``False``.

    Example implementation shape only::

        {
            "features": [{"name": "feature", "shape": [8], "dtype": "float32"}],
            "label": {"name": "target", "dtype": "int64", "classes": [0, 1]},
            "raw_data_upload": False,
        }
    """
    raise NotImplementedError(
        "Implement federated_task.local_training.data_preparation.describe_input_features()"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/RETURN TYPE.
# EDIT HERE - convert one raw sample; never upload or report the source path.
def preprocess(sample: Mapping[str, Any]) -> Any:
    """Convert one raw sample into the input structure expected by the model.

    Args:
        sample: One sample read from the owner's or participant's local data.

    Returns:
        A tensor, tuple/list of tensors, or mapping of tensors accepted by
        ``run_model()`` and the training implementation.

    Example implementation for numeric features::

        return torch.tensor(sample["features"], dtype=torch.float32)
    """
    del sample
    raise NotImplementedError(
        "Implement federated_task.local_training.data_preparation.preprocess()"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/DEFAULTS/KEYWORD-ONLY MARKER.
# EDIT HERE - open only the user-selected local data binding and create three loaders.
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
        data_root: Agent Studio's Task-specific ``.local-data`` directory.
            Treat this argument as the only dataset root; do not replace it with
            a user-specific absolute path or a path inside the releaseable source.
        seed: Deterministic split/shuffle seed.
        download: Whether this Task explicitly permits downloading public data.

    Returns:
        Exactly ``(train_loader, validation_loader, test_loader)``. Every batch
        must have the form ``(inputs, targets)`` expected by ``training.py``.

    Example implementation outline::

        # The user placed files with Workspace > Task Test > Open Data Folder.
        # For example: Path(data_root) / "train.csv"
        rows = read_local_rows(data_root)
        dataset_object = TaskDataset(rows, transform=preprocess)
        train_data, validation_data, test_data = deterministic_split(
            dataset_object, validation_split, seed
        )
        return (
            DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(validation_data, batch_size=batch_size),
            DataLoader(test_data, batch_size=batch_size),
        )
    """
    del dataset, validation_split, batch_size, data_root, seed, download
    raise NotImplementedError(
        "Implement federated_task.local_training.data_preparation.load_partition() with local-only data"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/DEFAULTS/KEYWORD-ONLY MARKER.
# EDIT HERE - create non-sensitive fake data with the exact real batch contract.
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

    Example outline: construct tensors with the same input/target shape and
    dtype as ``load_partition()``, wrap them in ``TensorDataset``, and return
    separate training and validation ``DataLoader`` objects.
    """
    del sample_count, batch_size, seed
    raise NotImplementedError(
        "Implement federated_task.local_training.data_preparation.build_smoke_loaders()"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/DEFAULT VALUE.
# EDIT HERE - return model inputs only, not labels and not private data.
def build_contract_probe(batch_size: int = 2) -> Any:
    """Build one batched model input without reading real user data.

    Returns:
        The exact input structure accepted by ``model.run_model()``. The first
        dimension of tensor values should equal ``batch_size``.

    Example implementation for eight numeric features::

        return torch.zeros(batch_size, 8, dtype=torch.float32)
    """
    del batch_size
    raise NotImplementedError(
        "Implement federated_task.local_training.data_preparation.build_contract_probe()"
    )


# FEDOPS CONTRACT - DO NOT RENAME OR CHANGE ARGUMENTS/DEFAULTS/KEYWORD-ONLY MARKER.
# EDIT HERE - use only owner-controlled/licensed server validation data.
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

    Example outline: open the permitted validation dataset below ``data_root``,
    apply the same preprocessing, and return one non-shuffled ``DataLoader``.
    """
    del batch_size, data_root, download
    raise NotImplementedError(
        "Implement federated_task.local_training.data_preparation.gl_model_torch_validation()"
    )
