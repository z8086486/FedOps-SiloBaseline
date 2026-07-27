"""Flower ServerApp entrypoint for validation and FedOps participation."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from .config import load_config
from .validation import validate_baseline


app = ServerApp()


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
    print(f"[fedops-baseline] stopping {name} (pid={process.pid})", flush=True)
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"[fedops-baseline] force-killing {name} (pid={process.pid})", flush=True)
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
    runtime_key = _read_str(
        run_config,
        "runtime_key",
        config["runtime_key"],
    ).strip()
    if not runtime_key or runtime_key == "task_id":
        raise ValueError("participate mode requires a real runtime_key")

    runtime = config["runtime"]
    manager_port = _read_int(run_config, "manager_port", 8004)
    startup_timeout = _read_int(run_config, "manager_startup_timeout", 30)
    server_manager_url = _read_str(
        run_config,
        "server_manager_url",
        runtime["server_manager_url"],
    )
    fl_server_host = _read_str(
        run_config,
        "fl_server_host",
        runtime["fl_server_host"],
    )

    process_env = os.environ.copy()
    process_env.update({
        "FEDOPS_TASK_OBJECT_ID": task_id,
        "FEDOPS_TASK_ID": runtime_key,
        "FEDOPS_MANAGER_PORT": str(manager_port),
        "FEDOPS_SERVER_MANAGER_URL": server_manager_url.rstrip("/"),
        "FEDOPS_FL_SERVER_HOST": fl_server_host,
    })
    manager_command = [
        sys.executable,
        "-m",
        "fedops_silo_baseline.client_manager_main",
    ]
    client_command = [
        sys.executable,
        "-m",
        "fedops_silo_baseline.client_main",
    ]

    print(
        "[fedops-baseline] mode=participate "
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
                    print("[fedops-baseline] client completed", flush=True)
                    return
                raise RuntimeError(f"FedOps client exited with code {client_code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[fedops-baseline] participation interrupted", flush=True)
    finally:
        if client_process is not None:
            _terminate_process(client_process, "FedOps client")
        _terminate_process(manager_process, "communication manager")


@app.main()
def main(grid: Grid, context: Context) -> None:
    del grid
    run_config = context.run_config
    mode = _read_str(run_config, "mode", "validate").strip().lower()
    if mode == "validate":
        result = validate_baseline(
            sample_count=_read_int(run_config, "validation_samples", 32),
            max_batches=_read_int(run_config, "validation_batches", 2),
        )
        print(
            "[fedops-baseline] validation passed "
            f"(model={result['model']}, samples={result['samples']}, "
            f"train_loss={result['train_loss']:.6f})",
            flush=True,
        )
        return
    if mode == "participate":
        _run_participation(run_config)
        return
    raise ValueError(f"Unsupported mode {mode!r}; use 'validate' or 'participate'")
