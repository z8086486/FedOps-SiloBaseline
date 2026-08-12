"""FedOps 1.2-compatible aggregation-server entrypoint for this Task."""

from __future__ import annotations

import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from fedops.server.app import FLServer

from .data_preparation import gl_model_torch_validation
from .training import MODEL_PATH, load_released_model, test_torch


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    model = load_released_model() if MODEL_PATH.is_file() else instantiate(config.model)
    data_root = os.environ.get("FEDOPS_SERVER_DATA_DIR", str(config.dataset.root))
    validation_loader = gl_model_torch_validation(
        batch_size=int(config.batch_size),
        data_root=data_root,
        download=bool(config.dataset.download),
    )
    FLServer(
        cfg=config,
        model=model,
        model_name=type(model).__name__,
        model_type=str(config.model_type),
        gl_val_loader=validation_loader,
        test_torch=test_torch(),
    ).start()


if __name__ == "__main__":
    main()
