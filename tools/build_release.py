"""Export the exact user Baseline and its immutable verification manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "federated-task-baseline"
DEFAULT_OUTPUT = ROOT / "dist" / "federated-task-baseline-0.3.0"
FORBIDDEN_PARTS = {".git", ".venv", "__pycache__", ".fedops-studio", "dataset", "datasets"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(relative: str) -> dict[str, Any]:
    if relative == "README.md":
        return {"role": "documentation", "editable": True, "content_type": "text/markdown"}
    if relative == "LICENSE":
        return {"role": "license", "editable": False, "content_type": "text/plain"}
    if relative == "pyproject.toml":
        return {"role": "task_definition", "editable": False, "content_type": "application/toml"}
    if relative == "uv.lock":
        return {"role": "dependency_lock", "editable": False, "content_type": "application/toml"}
    if relative == "federated_task/config.toml":
        return {"role": "task_config", "editable": True, "content_type": "application/toml"}
    if relative.endswith("local_training/model.py"):
        return {"role": "model_code", "editable": True, "content_type": "text/x-python"}
    if relative.endswith("local_training/data_preparation.py"):
        return {"role": "data_preparation", "editable": True, "content_type": "text/x-python"}
    if relative.endswith("local_training/train.py"):
        return {"role": "local_training", "editable": True, "content_type": "text/x-python"}
    if relative == "federated_task/agent_tool/manifest.json":
        return {"role": "tool_manifest", "editable": True, "content_type": "application/json"}
    if relative == "federated_task/agent_tool/inference.py":
        return {"role": "tool_inference", "editable": True, "content_type": "text/x-python"}
    if relative == "model_release/manifest.json":
        return {"role": "model_release_manifest", "editable": False, "content_type": "application/json"}
    content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
    return {"role": "runtime", "editable": False, "content_type": content_type}


def source_files() -> list[Path]:
    files = []
    for path in SOURCE.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(SOURCE)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            continue
        if relative.as_posix() == "model_release/model.safetensors":
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(SOURCE).as_posix())


def build_manifest() -> dict[str, Any]:
    entries = []
    for path in source_files():
        relative = path.relative_to(SOURCE).as_posix()
        entries.append({
            "path": relative,
            **_metadata(relative),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        })
    if not entries or not any(entry["path"] == "uv.lock" for entry in entries):
        raise RuntimeError("Baseline source is incomplete; uv.lock is required")
    return {
        "schema_version": 2,
        "baseline": {
            "name": "federated-task-baseline",
            "release_version": "0.3.0",
            "template_revision": 1,
        },
        "compatibility": {
            "python": ">=3.10,<3.13",
            "fedops_participation": "==1.1.30.13",
            "agent_studio_task_schema": 2,
        },
        "entrypoints": {
            "task_cli": "fedops-task",
            "task_runtime": "federated_task.main:main",
            "readiness": "federated_task.task_readiness.check:check_readiness",
        },
        "run_modes": [
            "local-train",
            "release-readiness",
            "participation-readiness",
            "participate",
            "tool-test",
        ],
        "data_boundary": {
            "raw_data_upload": False,
            "local_path_upload": False,
            "model_update_values_upload": False,
        },
        "files": entries,
    }


def export_release(output: Path = DEFAULT_OUTPUT) -> Path:
    if output.exists():
        shutil.rmtree(output)
    files_root = output / "files"
    for source in source_files():
        relative = source.relative_to(SOURCE)
        destination = files_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline-manifest.json").write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = export_release(args.output.resolve())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
