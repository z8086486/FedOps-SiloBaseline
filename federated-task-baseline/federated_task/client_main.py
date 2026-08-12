"""FEDOPS RUNTIME FILE - fixed client entrypoint; normal Task authors do not edit."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from .config import load_config
from . import data_preparation
from .model import build_model
from .training import MODEL_PATH, load_released_model, test_torch, train_torch


def _participation_imports():
    try:
        from fedops.client import client_utils
        from fedops.client.app import FLClientTask
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("Install participation dependencies with: uv sync --extra participate") from error
    return client_utils, FLClientTask, OmegaConf


def main() -> None:
    client_utils, FLClientTask, OmegaConf = _participation_imports()
    raw_config = load_config()
    config = OmegaConf.create(raw_config)
    runtime_key = os.environ.get("FEDOPS_RUNTIME_KEY", "").strip()
    if not runtime_key:
        raise ValueError("FEDOPS_RUNTIME_KEY is required")
    config.task_id = runtime_key
    data_root = os.environ.get("FEDOPS_LOCAL_DATA_DIR", "").strip()
    if not data_root:
        raise ValueError("FEDOPS_LOCAL_DATA_DIR is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(int(config.random_seed))
    np.random.seed(int(config.random_seed))
    torch.manual_seed(int(config.random_seed))
    train_loader, validation_loader, test_loader = data_preparation.load_partition(
        dataset=str(config.dataset.name),
        validation_split=float(config.dataset.validation_split),
        batch_size=int(config.batch_size),
        data_root=data_root,
        seed=int(config.random_seed),
        download=False,
    )
    local_model = load_released_model() if MODEL_PATH.is_file() else build_model(dict(config.model))
    local_versions = client_utils.local_model_directory(runtime_key)
    if local_versions:
        local_model = client_utils.download_local_model(
            model_type=str(config.model_type),
            task_id=runtime_key,
            listdir=local_versions,
            model=local_model,
        )
    registration = {
        "train_loader": train_loader,
        "val_loader": validation_loader,
        "test_loader": test_loader,
        "model": local_model,
        "model_name": type(local_model).__name__,
        "train_torch": train_torch(),
        "test_torch": test_torch(),
    }
    FLClientTask(config, registration).start()


if __name__ == "__main__":
    main()
