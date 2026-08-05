"""Communication manager used by FedOps participation mode."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from contextlib import suppress
from typing import Any, Dict, Optional

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


LOGGER = logging.getLogger(__name__)
app = FastAPI()


def _mac_address() -> str:
    value = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join(value[index:index + 2] for index in range(0, 12, 2))


class FLTask(BaseModel):
    FL_task_ID: Optional[str] = None
    Device_mac: Optional[str] = None
    Device_hostname: Optional[str] = None
    Device_online: Optional[bool] = None
    Device_training: Optional[bool] = None


class ManagerState(BaseModel):
    client_online: bool = False
    client_training: bool = False
    fl_ready: bool = False
    global_model_version: int = 0
    task_id: str = ""
    task_status: Optional[FLTask] = None
    client_mac: str = _mac_address()
    client_name: str = socket.gethostname()


STATE = ManagerState(task_id=os.environ.get("FEDOPS_TASK_ID", ""))
BACKGROUND_TASKS = []


def _manager_base_url() -> str:
    return os.environ.get(
        "FEDOPS_SERVER_MANAGER_URL",
        "http://ccl.gachon.ac.kr:40019",
    ).rstrip("/")


def _client_base_url() -> str:
    return os.environ.get("FEDOPS_CLIENT_API_URL", "http://127.0.0.1:8003").rstrip("/")


def _server_url(path: str) -> str:
    return f"{_manager_base_url()}/FLSe/{path.lstrip('/')}"


def _request(method: str, url: str, **kwargs):
    return requests.request(method, url, timeout=5.0, **kwargs)


async def _to_thread(function, *args, **kwargs):
    return await asyncio.to_thread(function, *args, **kwargs)


async def _client_presence_loop() -> None:
    while True:
        try:
            response = await _to_thread(_request, "GET", f"{_client_base_url()}/online")
            if response.ok:
                payload = response.json()
                STATE.client_online = bool(payload.get("client_online"))
                STATE.client_training = bool(payload.get("client_start"))
                STATE.task_id = str(payload.get("task_id") or STATE.task_id)
            else:
                STATE.client_online = False

            if STATE.task_id:
                registration: Dict[str, Any] = {
                    "FL_task_ID": STATE.task_id,
                    "Device_mac": STATE.client_mac,
                    "Device_hostname": STATE.client_name,
                    "Device_online": STATE.client_online,
                    "Device_training": STATE.client_training,
                }
                await _to_thread(
                    _request,
                    "PUT",
                    _server_url("RegisterFLTask"),
                    json=registration,
                )
        except Exception as error:
            STATE.client_online = False
            LOGGER.info("FedOps client presence unavailable: %s", error)
        await asyncio.sleep(6)


async def _server_status_loop() -> None:
    while True:
        try:
            if STATE.task_id and STATE.client_online and not STATE.client_training:
                response = await _to_thread(
                    _request,
                    "GET",
                    _server_url(f"info/{STATE.task_id}/{STATE.client_mac}"),
                )
                if response.ok:
                    server_status = response.json().get("Server_Status", {})
                    STATE.fl_ready = bool(server_status.get("FLSeReady"))
                    STATE.global_model_version = int(server_status.get("GL_Model_V") or 0)
                    task_status = server_status.get("Task_status")
                    STATE.task_status = FLTask(**task_status) if task_status else None
        except Exception as error:
            LOGGER.info("FedOps server status unavailable: %s", error)
        await asyncio.sleep(8)


async def _training_loop() -> None:
    while True:
        try:
            should_start = (
                STATE.task_status is not None
                and STATE.client_online
                and not STATE.client_training
                and STATE.fl_ready
            )
            if should_start:
                response = await _to_thread(
                    _request,
                    "POST",
                    f"{_client_base_url()}/start",
                    json={
                        "server_ip": os.environ.get(
                            "FEDOPS_SERVER_HOST",
                            "ccl.gachon.ac.kr",
                        ),
                        "client_mac": STATE.client_mac,
                    },
                )
                if response.ok and response.json().get("FL_client_start"):
                    STATE.client_training = True
        except Exception as error:
            LOGGER.info("FedOps training start unavailable: %s", error)
        await asyncio.sleep(8)


@app.on_event("startup")
async def startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    BACKGROUND_TASKS.extend([
        asyncio.create_task(_client_presence_loop()),
        asyncio.create_task(_server_status_loop()),
        asyncio.create_task(_training_loop()),
    ])


@app.on_event("shutdown")
async def shutdown() -> None:
    for task in BACKGROUND_TASKS:
        task.cancel()
    for task in BACKGROUND_TASKS:
        with suppress(asyncio.CancelledError):
            await task
    BACKGROUND_TASKS.clear()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "task_id": STATE.task_id}


@app.get("/info")
def info():
    return STATE


@app.get("/trainFin")
def training_finished():
    STATE.client_training = False
    STATE.fl_ready = False
    return STATE


@app.get("/trainFail")
def training_failed():
    STATE.client_training = False
    STATE.fl_ready = False
    return STATE


@app.get("/flclient_out")
def client_stopped():
    STATE.client_online = False
    STATE.client_training = False
    return STATE


def main() -> None:
    uvicorn.run(
        app,
        host=os.environ.get("FEDOPS_MANAGER_HOST", "127.0.0.1"),
        port=int(os.environ.get("FEDOPS_MANAGER_PORT", "8004")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
