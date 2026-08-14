"""FEDOPS RUNTIME FILE - fixed aggregation-server entrypoint; authors do not edit."""

from __future__ import annotations

import os

from omegaconf import OmegaConf

from fedops.server.app import FLServer

from ..config import load_config
from ..local_training.data_preparation import gl_model_torch_validation
from ..local_training.model import build_model
from ..runtime.model_release import MODEL_PATH, load_released_model, test_torch


def main() -> None:
    config = OmegaConf.create(load_config())
    model = load_released_model() if MODEL_PATH.is_file() else build_model(dict(config.model))
    data_root = os.environ.get("FEDOPS_SERVER_DATA_DIR", str(config.dataset.root))
    validation_loader = gl_model_torch_validation(
        batch_size=int(config.batch_size),
        data_root=data_root,
        download=bool(config.dataset.download),
    )
    evaluation_max_batches = config.server_evaluation.max_batches
    FLServer(
        cfg=config,
        model=model,
        model_name=type(model).__name__,
        model_type=str(config.model_type),
        gl_val_loader=validation_loader,
        test_torch=test_torch(
            int(evaluation_max_batches)
            if evaluation_max_batches is not None
            else None
        ),
    ).start()


if __name__ == "__main__":
    main()
