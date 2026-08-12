"""FEDOPS RUNTIME FILE - fixed Release/Participation Readiness implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch

from fedops.client.client_fl import FLClient
from fedops.client.parameter_contract import (
    get_parameters,
    parameter_signature,
    verify_parameter_round_trip,
)

from .config import load_config
from .data_preparation import (
    build_contract_probe,
    build_smoke_loaders,
    describe_input_features,
    load_partition,
)
from .model import build_model, run_model, validate_model_output
from .tool import build_tool_smoke_payload, predict
from .training import (
    MODEL_MANIFEST_PATH,
    MODEL_PATH,
    evaluate_model,
    load_released_model,
    normalize_evaluation,
    test_torch,
    train_model,
    train_torch,
)


CHECKER_VERSION = "2.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        "federated_task/conf/config.yaml",
        "federated_task/model.py",
        "federated_task/data_preparation.py",
        "federated_task/training.py",
        "federated_task/client_main.py",
        "federated_task/client_manager_main.py",
        "federated_task/server_main.py",
        "federated_task/task_check.py",
        "federated_task/manifest.json",
        "federated_task/tool.py",
        "model_release/manifest.json",
    ]
    missing = [value for value in required if not (PROJECT_ROOT / value).is_file()]
    if missing:
        raise ValueError(f"required Task files are missing: {', '.join(missing)}")
    for path in PROJECT_ROOT.rglob("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"release source contains a symlink: {relative}")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    missing_headings = sorted(value for value in REQUIRED_README_HEADINGS if value not in readme)
    if missing_headings:
        raise ValueError(f"README is missing sections: {', '.join(missing_headings)}")
    return {"fileCount": len(required), "readmeSections": len(REQUIRED_README_HEADINGS)}


def _load_model_manifest(model_type: str) -> dict[str, Any]:
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise ValueError("model_release/manifest.json is not ready")
    if manifest.get("artifact") != MODEL_PATH.name or not MODEL_PATH.is_file():
        raise ValueError("model release artifact is missing")
    if manifest.get("size") != MODEL_PATH.stat().st_size or manifest.get("sha256") != _sha256(MODEL_PATH):
        raise ValueError("model release size or checksum does not match")
    model = load_released_model()
    signature = parameter_signature(model, model_type)
    if manifest.get("parameterSignature", {}).get("fingerprint") != signature["fingerprint"]:
        raise ValueError("model parameter signature does not match FedOps transport")
    return manifest


def _construct_fedops_client(model, config, train_loader, validation_loader) -> int:
    """Construct the actual 1.2 client and verify its exported payload."""
    client = FLClient(
        model=model,
        validation_split=float(config["dataset"]["validation_split"]),
        fl_task_id="readiness",
        client_mac="00:00:00:00:00:00",
        client_name="readiness",
        fl_round=1,
        gl_model=0,
        wandb_use=False,
        wandb_name="",
        model_name=type(model).__name__,
        model_type=str(config["model_type"]),
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=validation_loader,
        cfg=OmegaConf.create(config),
        train_torch=train_torch(),
        test_torch=test_torch(),
    )
    payload = client.get_parameters()
    authoritative = get_parameters(model, str(config["model_type"]))
    if len(payload) != len(authoritative) or any(
        left.shape != right.shape or left.dtype != right.dtype
        for left, right in zip(payload, authoritative)
    ):
        raise ValueError("FLClient parameter payload differs from the FedOps contract")
    return sum(value.nbytes for value in payload)


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
    model_type = str(config["model_type"])
    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    project = _check_project_files()
    model_manifest = _load_model_manifest(model_type)

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
    if not isinstance(contract, dict) or contract.get("raw_data_upload") is not False:
        raise ValueError("describe_input_features() must declare raw_data_upload=False")
    try:
        json.dumps(contract, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("input feature contract must be JSON-serializable") from error
    model = load_released_model()
    probe = build_contract_probe(batch_size=2)
    output = run_model(model, probe)
    output_summary = validate_model_output(output, config)
    if not isinstance(output_summary, dict):
        raise ValueError("validate_model_output() must return a dictionary")
    try:
        json.dumps(output_summary, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("model output summary must be JSON-serializable") from error

    before = [value.copy() for value in get_parameters(model, model_type)]
    payload_bytes = _construct_fedops_client(model, config, train_loader, validation_loader)
    train_loss = train_model(
        model,
        train_loader,
        epochs=1,
        learning_rate=float(config["learning_rate"]),
        device=torch.device("cpu"),
        max_batches=max_batches,
    )
    validation_loss, primary_metric, metrics = normalize_evaluation(
        evaluate_model(
            model,
            validation_loader,
            device=torch.device("cpu"),
            max_batches=max_batches,
        )
    )
    values = [float(train_loss), validation_loss, primary_metric, *metrics.values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("readiness local training produced a non-finite value")
    after = get_parameters(model, model_type)
    changed = sum(not np.array_equal(old, new) for old, new in zip(before, after))
    if not after or changed == 0 or not all(np.isfinite(value).all() for value in after):
        raise ValueError("FedOps client parameter payload was not updated by local training")

    round_trip = verify_parameter_round_trip(
        model,
        lambda: build_model(config["model"]),
        model_type,
    )
    signature = round_trip["signature"]
    expected = expected_parameter_signature or model_manifest["parameterSignature"]["fingerprint"]
    if signature["fingerprint"] != expected:
        raise ValueError("local parameter signature does not match the Published Task")
    tool_result = predict(build_tool_smoke_payload())
    if not isinstance(tool_result, dict):
        raise ValueError("Tool predict() must return a dictionary")
    try:
        json.dumps(tool_result, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Tool output must be JSON-serializable") from error

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
            "inputContract": contract,
            "outputContract": output_summary,
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "primaryMetric": primary_metric,
            "additionalMetrics": metrics,
            "fedopsClientConstructed": True,
            "parameterTensorCount": len(after),
            "changedParameterTensorCount": changed,
            "parameterPayloadBytes": payload_bytes,
            "roundTripPayloadBytes": round_trip["payloadBytes"],
            "toolOutput": tool_result,
        },
        "privacy": {
            "rawDataUploaded": False,
            "localPathReported": False,
            "parameterValuesReported": False,
        },
    }
