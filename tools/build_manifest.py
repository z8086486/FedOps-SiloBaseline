"""Generate the immutable Baseline file manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "baseline-manifest.json"

FILE_METADATA: Dict[str, Dict[str, object]] = {
    "LICENSE": {"role": "license", "content_type": "text/plain", "editable": False},
    "README.md": {"role": "documentation", "content_type": "text/markdown", "editable": True},
    "pyproject.toml": {"role": "flower_config", "content_type": "application/toml", "editable": False},
    "fedops_silo_baseline/__init__.py": {"role": "runtime", "content_type": "text/x-python", "editable": False},
    "fedops_silo_baseline/client_app.py": {"role": "runtime", "content_type": "text/x-python", "editable": False},
    "fedops_silo_baseline/config.py": {"role": "runtime", "content_type": "text/x-python", "editable": False},
    "fedops_silo_baseline/launcher_app.py": {"role": "runtime", "content_type": "text/x-python", "editable": False},
    "fedops_silo_baseline/client_main.py": {"role": "runtime", "content_type": "text/x-python", "editable": False},
    "fedops_silo_baseline/client_manager_main.py": {"role": "runtime", "content_type": "text/x-python", "editable": False},
    "fedops_silo_baseline/model.py": {"role": "model_code", "content_type": "text/x-python", "editable": True},
    "fedops_silo_baseline/data_preparation.py": {"role": "data_preparation", "content_type": "text/x-python", "editable": True},
    "fedops_silo_baseline/validation.py": {"role": "runtime", "content_type": "text/x-python", "editable": False},
    "fedops_silo_baseline/conf/config.toml": {"role": "task_config", "content_type": "application/toml", "editable": True},
    "tests/test_config_contract.py": {"role": "test", "content_type": "text/x-python", "editable": False},
    "tests/test_data_contract.py": {"role": "test", "content_type": "text/x-python", "editable": False},
    "tests/test_model_contract.py": {"role": "test", "content_type": "text/x-python", "editable": False},
    "tests/test_validation.py": {"role": "test", "content_type": "text/x-python", "editable": False},
    "tests/test_launcher_lifecycle.py": {"role": "test", "content_type": "text/x-python", "editable": False},
    "tests/test_manifest.py": {"role": "test", "content_type": "text/x-python", "editable": False},
    "tools/__init__.py": {"role": "build_tool", "content_type": "text/x-python", "editable": False},
    "tools/build_manifest.py": {"role": "build_tool", "content_type": "text/x-python", "editable": False},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> Dict[str, object]:
    files = []
    for relative_path, metadata in sorted(FILE_METADATA.items()):
        file_path = ROOT / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Manifest input is missing: {relative_path}")
        files.append({
            "path": relative_path,
            **metadata,
            "size": file_path.stat().st_size,
            "sha256": _sha256(file_path),
        })

    return {
        "schema_version": 1,
        "baseline": {
            "name": "fedops-silo-baseline",
            "release_version": "0.1.0",
            "template_revision": 2,
        },
        "source": {
            "reference_app": "@gfedops/fedops-mnist",
            "reference_version": "0.0.3",
            "reference_url": "https://flower.ai/apps/gfedops/fedops-mnist/",
            "license": "Apache-2.0",
        },
        "compatibility": {
            "python": ">=3.10,<3.13",
            "flower": "==1.26.1",
            "fedops_participation": "==1.1.30.13",
            "launcher": ">=1.0.0",
        },
        "entrypoints": {
            "flower_serverapp": "fedops_silo_baseline.launcher_app:app",
            "flower_clientapp": "fedops_silo_baseline.client_app:app",
            "validation_cli": "fedops-baseline-validate",
        },
        "run_modes": {
            "default": "validate",
            "allowed": ["validate", "participate"],
            "participate_requires": ["task_id", "runtime_key", "participate extra"],
            "public_run_config_keys": [
                "mode",
                "task_id",
                "runtime_key",
                "validation_samples",
                "validation_batches",
                "manager_port",
                "manager_startup_timeout",
                "server_manager_url",
                "fl_server_host",
            ],
        },
        "task_binding": {
            "stable_identifier_key": "task_id",
            "runtime_identifier_key": "runtime_key",
            "identifier_source": "FedOps-Launcher authorized Task API",
            "display_name_is_identifier": False,
        },
        "data_boundary": {
            "raw_data_upload": False,
            "validation_requires_network": False,
        },
        "manifest_checksum_policy": "baseline-manifest.json is excluded to avoid a self-referential checksum",
        "files": files,
    }


def serialized_manifest() -> str:
    return json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = serialized_manifest()
    if args.check:
        current = MANIFEST_PATH.read_text(encoding="utf-8")
        if current != expected:
            print("baseline-manifest.json is out of date")
            return 1
        print("baseline-manifest.json is current")
        return 0
    MANIFEST_PATH.write_text(expected, encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
