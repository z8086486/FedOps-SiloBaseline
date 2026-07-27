"""FedOps local training client for participation mode."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from .config import load_config
from . import data_preparation, model as model_contract


def _participation_imports():
    try:
        from fedops.client import client_utils
        from fedops.client.app import FLClientTask
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError(
            "Participation dependencies are missing. "
            "Install with: python -m pip install -e '.[participate]'"
        ) from error
    return client_utils, FLClientTask, OmegaConf


def main() -> None:
    client_utils, FLClientTask, OmegaConf = _participation_imports()
    raw_config = load_config()
    config = OmegaConf.create(raw_config)
    runtime_key = os.environ.get("FEDOPS_TASK_ID", str(config.runtime_key)).strip()
    if not runtime_key or runtime_key == "task_id":
        raise ValueError("FEDOPS_TASK_ID must contain a real runtime key")
    config.task_id = runtime_key

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    random.seed(int(config.random_seed))
    np.random.seed(int(config.random_seed))
    torch.manual_seed(int(config.random_seed))

    train_loader, validation_loader, test_loader = data_preparation.load_partition(
        dataset=str(config.dataset.name),
        validation_split=float(config.dataset.validation_split),
        batch_size=int(config.batch_size),
        data_root=str(config.dataset.root),
        seed=int(config.random_seed),
    )
    local_model = model_contract.build_model(
        {"output_size": int(config.model.output_size)}
    )
    model_type = str(config.model_type)
    model_name = type(local_model).__name__

    local_versions = client_utils.local_model_directory(runtime_key)
    if local_versions:
        logging.info("Loading the latest local model for runtime %s", runtime_key)
        local_model = client_utils.download_local_model(
            model_type=model_type,
            task_id=runtime_key,
            listdir=local_versions,
            model=local_model,
        )

    registration = {
        "train_loader": train_loader,
        "val_loader": validation_loader,
        "test_loader": test_loader,
        "model": local_model,
        "model_name": model_name,
        "train_torch": model_contract.train_torch(),
        "test_torch": model_contract.test_torch(),
    }
    logging.info("Starting FedOps client for runtime %s", runtime_key)
    FLClientTask(config, registration).start()


if __name__ == "__main__":
    main()
