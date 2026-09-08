"""FEDOPS RUNTIME FILE - fixed aggregation-server entrypoint; authors do not edit."""

from __future__ import annotations

import os

from omegaconf import OmegaConf

from fedops.server.app import FLServer
from fedops.server.evaluation import prepare_validation_loader

from ..config import load_config
from ..local_training.data_preparation import gl_model_torch_validation
from ..local_training.model import build_model
from ..runtime.model_release import MODEL_PATH, load_released_model, test_torch


def main() -> None:
    config = OmegaConf.create(load_config())
    model = load_released_model() if MODEL_PATH.is_file() else build_model(dict(config.model))
    validation_loader = prepare_validation_loader(config, gl_model_torch_validation)
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
