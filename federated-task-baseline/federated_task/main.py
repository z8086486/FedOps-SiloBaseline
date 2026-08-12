"""Single FedOps Federated Task command-line entrypoint."""

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
from typing import Any, Mapping, Sequence

from .tool import predict
from .config import load_config
from .training import export_initial_model
from .task_check import check_readiness


def _terminate_process(process: subprocess.Popen, name: str, timeout_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    print(f"[fedops-task] stopping {name} (pid={process.pid})", flush=True)
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _wait_for_manager(process: subprocess.Popen, port: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"communication manager exited with code {exit_code}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise RuntimeError("communication manager did not become ready")


def _required(value: str | None, name: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValueError(f"participate mode requires {name}")
    return selected


def _run_participation(run_config: Mapping[str, Any]) -> None:
    config = load_config()
    task_id = _required(run_config.get("task_id"), "task_id")
    runtime_key = _required(run_config.get("runtime_key"), "runtime_key")
    data_root = _required(run_config.get("data_root"), "data_root")
    manager_url = _required(run_config.get("server_manager_url"), "server_manager_url")
    server_host = _required(run_config.get("federated_server_host"), "federated_server_host")
    manager_port = int(run_config.get("manager_port") or config["runtime"]["manager_port"])
    startup_timeout = int(run_config.get("manager_startup_timeout") or 30)
    process_env = os.environ.copy()
    process_env.update({
        "FEDOPS_TASK_ID": task_id,
        "FEDOPS_RUNTIME_KEY": runtime_key,
        "FEDOPS_LOCAL_DATA_DIR": data_root,
        "FEDOPS_MANAGER_PORT": str(manager_port),
        "FEDOPS_SERVER_MANAGER_URL": manager_url.rstrip("/"),
        "FEDOPS_SERVER_HOST": server_host,
    })
    manager = subprocess.Popen(
        [sys.executable, "-m", "federated_task.client_manager_main"],
        env=process_env,
    )
    client = None
    try:
        _wait_for_manager(manager, manager_port, startup_timeout)
        client = subprocess.Popen(
            [sys.executable, "-m", "federated_task.client_main"],
            env=process_env,
        )
        while True:
            if manager.poll() is not None:
                raise RuntimeError(f"communication manager exited with code {manager.returncode}")
            if client.poll() is not None:
                if client.returncode == 0:
                    return
                raise RuntimeError(f"FedOps client exited with code {client.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[fedops-task] participation interrupted", flush=True)
    finally:
        if client is not None:
            _terminate_process(client, "FedOps client")
        _terminate_process(manager, "communication manager")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedOps Federated Task")
    actions = parser.add_subparsers(dest="action", required=True)

    local_train = actions.add_parser("local-train", help="Train and export an Initial Model")
    local_train.add_argument("--data-root")
    local_train.add_argument("--download", action="store_true")
    local_train.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    local_train.add_argument("--max-batches", type=int)

    readiness = actions.add_parser("check-readiness", help="Check release or participation readiness")
    readiness.add_argument("--mode", required=True, choices=["release", "participation"])
    readiness.add_argument("--data-root")
    readiness.add_argument("--expected-parameter-signature")
    readiness.add_argument("--samples", type=int, default=32)
    readiness.add_argument("--max-batches", type=int, default=2)
    readiness.add_argument("--allow-synthetic-participation", action="store_true", help=argparse.SUPPRESS)

    participate = actions.add_parser("participate", help="Join an authorized FedOps run")
    participate.add_argument("--task-id", required=True)
    participate.add_argument("--runtime-key", required=True)
    participate.add_argument("--data-root", required=True)
    participate.add_argument("--server-manager-url", required=True)
    participate.add_argument("--federated-server-host", required=True)
    participate.add_argument("--manager-port", type=int)
    participate.add_argument("--manager-startup-timeout", type=int)

    tool_test = actions.add_parser("tool-test", help="Run one Agent Tool inference smoke test")
    tool_test.add_argument("--model-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "local-train":
            if not args.synthetic and not args.data_root:
                raise ValueError("local-train requires --data-root unless using the test-only synthetic mode")
            result = export_initial_model(
                data_root=args.data_root,
                synthetic=args.synthetic,
                download=args.download,
                max_batches=args.max_batches,
            )
        elif args.action == "check-readiness":
            result = check_readiness(
                mode=args.mode,
                data_root=args.data_root,
                expected_parameter_signature=args.expected_parameter_signature,
                sample_count=args.samples,
                max_batches=args.max_batches,
                allow_synthetic_participation=args.allow_synthetic_participation,
            )
        elif args.action == "tool-test":
            result = predict(
                {"image": [[0 for _ in range(28)] for _ in range(28)]},
                model_path=args.model_path,
            )
        else:
            _run_participation({key: value for key, value in vars(args).items() if value is not None})
            result = {"ok": True, "mode": "participate"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
