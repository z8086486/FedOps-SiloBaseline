"""Command-line launcher for a FedOps Federated Task.

This module is the stable boundary used by FedOps Agent Studio. It validates an
owner-edited Task locally or starts the FedOps participation processes. It does
not expose a third-party application runtime as the Task contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .config import load_config
from .validation import validate_baseline


def _read_str(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key)
    return default if value is None else str(value)


def _read_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    return default if value is None else int(value)


def _terminate_process(
    process: subprocess.Popen,
    name: str,
    timeout_seconds: float = 10.0,
) -> None:
    if process.poll() is not None:
        return
    print(f"[fedops-task] stopping {name} (pid={process.pid})", flush=True)
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"[fedops-task] force-killing {name} (pid={process.pid})", flush=True)
        process.kill()
        process.wait(timeout=5.0)


def _wait_for_manager(
    process: subprocess.Popen,
    port: int,
    timeout_seconds: int,
) -> None:
    health_url = f"http://127.0.0.1:{port}/healthz"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"communication manager exited with code {exit_code}")
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise RuntimeError(
        f"communication manager did not become ready within {timeout_seconds} seconds"
    )


def _run_participation(run_config: Mapping[str, Any]) -> None:
    config = load_config()
    task_id = _read_str(run_config, "task_id", config["task_id"]).strip()
    if not task_id or task_id == "task_id":
        raise ValueError("participate mode requires a real task_id")
    runtime_key = _read_str(run_config, "runtime_key", config["runtime_key"]).strip()
    if not runtime_key or runtime_key == "task_id":
        raise ValueError("participate mode requires a real runtime_key")

    runtime = config["runtime"]
    manager_port = _read_int(run_config, "manager_port", int(runtime["manager_port"]))
    startup_timeout = _read_int(run_config, "manager_startup_timeout", 30)
    server_manager_url = _read_str(
        run_config,
        "server_manager_url",
        runtime["server_manager_url"],
    )
    federated_server_host = _read_str(
        run_config,
        "federated_server_host",
        runtime["federated_server_host"],
    )

    process_env = os.environ.copy()
    process_env.update({
        "FEDOPS_TASK_OBJECT_ID": task_id,
        "FEDOPS_TASK_ID": runtime_key,
        "FEDOPS_MANAGER_PORT": str(manager_port),
        "FEDOPS_SERVER_MANAGER_URL": server_manager_url.rstrip("/"),
        "FEDOPS_SERVER_HOST": federated_server_host,
    })
    manager_command = [sys.executable, "-m", "fedops_silo_baseline.client_manager_main"]
    client_command = [sys.executable, "-m", "fedops_silo_baseline.client_app"]

    print(
        "[fedops-task] mode=participate "
        f"task_id={task_id} runtime_key={runtime_key}",
        flush=True,
    )
    manager_process = subprocess.Popen(manager_command, env=process_env)
    client_process: Optional[subprocess.Popen] = None
    try:
        _wait_for_manager(manager_process, manager_port, startup_timeout)
        client_process = subprocess.Popen(client_command, env=process_env)
        while True:
            manager_code = manager_process.poll()
            client_code = client_process.poll()
            if manager_code is not None:
                raise RuntimeError(
                    f"communication manager exited with code {manager_code}"
                )
            if client_code is not None:
                if client_code == 0:
                    print("[fedops-task] client completed", flush=True)
                    return
                raise RuntimeError(f"FedOps client exited with code {client_code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[fedops-task] participation interrupted", flush=True)
    finally:
        if client_process is not None:
            _terminate_process(client_process, "FedOps client")
        _terminate_process(manager_process, "communication manager")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedOps Federated Task")
    actions = parser.add_subparsers(dest="action", required=True)

    validate = actions.add_parser("validate", help="Validate model and data contracts locally")
    validate.add_argument("--config", type=Path)
    validate.add_argument("--samples", type=int, default=32)
    validate.add_argument("--max-batches", type=int, default=2)

    participate = actions.add_parser("participate", help="Participate in an authorized FedOps Task")
    participate.add_argument("--task-id", required=True)
    participate.add_argument("--runtime-key", required=True)
    participate.add_argument("--manager-port", type=int)
    participate.add_argument("--manager-startup-timeout", type=int)
    participate.add_argument("--server-manager-url")
    participate.add_argument("--federated-server-host")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "validate":
            result = validate_baseline(
                config_path=args.config,
                sample_count=args.samples,
                max_batches=args.max_batches,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        run_config = {
            key: value
            for key, value in vars(args).items()
            if key != "action" and value is not None
        }
        _run_participation(run_config)
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
