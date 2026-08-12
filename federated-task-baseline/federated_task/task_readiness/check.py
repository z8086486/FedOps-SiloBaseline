"""Check one Task contract in Owner-release or participant-local context."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..agent_tool.inference import predict
from ..config import load_config
from ..federated_learning.parameters import (
    parameter_signature,
    parameter_update,
    serialize_parameters,
    verify_round_trip,
)
from ..local_training.data_preparation import (
    build_smoke_loaders,
    describe_input_features,
    load_partition,
)
from ..local_training.model import build_model
from ..local_training.train import (
    MODEL_MANIFEST_PATH,
    MODEL_PATH,
    evaluate_model,
    load_released_model,
    train_model,
)


CHECKER_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PARTS = {".git", ".venv", "__pycache__", ".fedops-studio", "dataset", "datasets"}
REQUIRED_README_HEADINGS = {
    "## Intended use",
    "## Local data setup",
    "## Federated participation",
    "## Limitations",
    "## Privacy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root: Path = PROJECT_ROOT) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            continue
        if relative.as_posix() == "model_release/model.safetensors":
            continue
        entries.append([relative.as_posix(), path.stat().st_size, _sha256(path)])
    return hashlib.sha256(
        json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _check_project_files() -> dict[str, Any]:
    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "federated_task/config.toml",
        "federated_task/local_training/model.py",
        "federated_task/local_training/data_preparation.py",
        "federated_task/local_training/train.py",
        "federated_task/federated_learning/client_main.py",
        "federated_task/federated_learning/client_manager_main.py",
        "federated_task/federated_learning/parameters.py",
        "federated_task/agent_tool/manifest.json",
        "model_release/manifest.json",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise ValueError(f"required Task files are missing: {', '.join(missing)}")
    for path in PROJECT_ROOT.rglob("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            # uv environments, local data, caches, and Studio state are allowed
            # beside source but are never part of a release manifest/archive.
            continue
        if path.is_symlink():
            raise ValueError(f"release source contains a symlink: {relative}")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    missing_headings = sorted(heading for heading in REQUIRED_README_HEADINGS if heading not in readme)
    if missing_headings:
        raise ValueError(f"README is missing sections: {', '.join(missing_headings)}")
    return {"fileCount": len(required), "readmeSections": len(REQUIRED_README_HEADINGS)}


def _load_model_manifest() -> dict[str, Any]:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise ValueError("model_release/manifest.json is not ready")
    if manifest.get("artifact") != MODEL_PATH.name or not MODEL_PATH.is_file():
        raise ValueError("model release artifact is missing")
    if manifest.get("size") != MODEL_PATH.stat().st_size or manifest.get("sha256") != _sha256(MODEL_PATH):
        raise ValueError("model release size or checksum does not match")
    model = load_released_model()
    signature = parameter_signature(model)
    if manifest.get("parameterSignature", {}).get("fingerprint") != signature["fingerprint"]:
        raise ValueError("model parameter signature does not match its manifest")
    return manifest


def check_readiness(
    *,
    mode: str,
    data_root: str | None = None,
    expected_parameter_signature: str | None = None,
    sample_count: int = 32,
    max_batches: int = 2,
    allow_synthetic_participation: bool = False,
) -> dict[str, Any]:
    if mode not in {"release", "participation"}:
        raise ValueError("readiness mode must be release or participation")
    config = load_config()
    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    project = _check_project_files()
    model_manifest = _load_model_manifest()

    if mode == "participation" and not data_root and not allow_synthetic_participation:
        raise ValueError("participation readiness requires a local data binding")
    if mode == "participation" and data_root:
        train_loader, validation_loader, _ = load_partition(
            dataset=str(config["dataset"]["name"]),
            validation_split=float(config["dataset"]["validation_split"]),
            batch_size=int(config["batch_size"]),
            data_root=data_root,
            seed=seed,
            download=False,
        )
        data_source = "local-binding"
    else:
        train_loader, validation_loader = build_smoke_loaders(
            sample_count=sample_count,
            batch_size=min(int(config["batch_size"]), 8),
            seed=seed,
        )
        data_source = "synthetic-contract-smoke"

    contract = describe_input_features()
    model = load_released_model()
    probe = torch.zeros(2, *contract["features"][0]["shape"])
    output = model(probe)
    expected_output = int(config["model"]["output_size"])
    if list(output.shape) != [2, expected_output]:
        raise ValueError("model output does not match the Task output contract")
    before = serialize_parameters(model)
    device = torch.device("cpu")
    train_loss = train_model(
        model,
        train_loader,
        epochs=1,
        learning_rate=float(config["learning_rate"]),
        device=device,
        max_batches=max_batches,
    )
    validation_loss, accuracy, metrics = evaluate_model(
        model,
        validation_loader,
        device=device,
        max_batches=max_batches,
    )
    values = [train_loss, validation_loss, accuracy, metrics["f1_score"]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("readiness local training produced a non-finite value")
    after = serialize_parameters(model)
    update = parameter_update(before, after)
    if not update or not all(np.isfinite(value).all() for value in update):
        raise ValueError("federated parameter update is empty or non-finite")
    round_trip = verify_round_trip(model, lambda: build_model(config["model"]))
    signature = round_trip["signature"]
    expected = expected_parameter_signature or model_manifest["parameterSignature"]["fingerprint"]
    if signature["fingerprint"] != expected:
        raise ValueError("local parameter signature does not match the Published Task")
    tool_result = predict({"image": [[0 for _ in range(28)] for _ in range(28)]})

    return {
        "schemaVersion": 1,
        "ok": True,
        "mode": mode,
        "checkerVersion": CHECKER_VERSION,
        "sourceFingerprint": source_fingerprint(),
        "parameterSignatureFingerprint": signature["fingerprint"],
        "taskId": os.environ.get("FEDOPS_TASK_ID") or None,
        "releaseId": os.environ.get("FEDOPS_RELEASE_ID") or None,
        "modelVersionId": os.environ.get("FEDOPS_MODEL_VERSION_ID") or None,
        "checks": {
            "project": project,
            "modelArtifactSha256": model_manifest["sha256"],
            "dataSource": data_source,
            "inputShape": contract["features"][0]["shape"],
            "outputShape": list(output.shape),
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "accuracy": accuracy,
            "weightedF1": metrics["f1_score"],
            "updateTensorCount": len(update),
            "updateBytes": sum(value.nbytes for value in update),
            "roundTripPayloadBytes": round_trip["payloadBytes"],
            "toolOutput": tool_result,
        },
        "privacy": {
            "rawDataUploaded": False,
            "localPathReported": False,
            "parameterValuesReported": False,
        },
    }
