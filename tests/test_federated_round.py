"""One real Flower round with two FedOps clients and local-only smoke data."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import flwr as fl
from omegaconf import OmegaConf

from fedops.client import client_api
from fedops.client.client_fl import FLClient
from fedops.client.parameter_contract import get_parameters
from federated_task.data_preparation import build_smoke_loaders
from federated_task.model import build_model
from federated_task.training import test_torch, train_torch


class _NoopClientServerAPI:
    def __init__(self, *_args, **_kwargs):
        pass

    def put_train_result(self, *_args, **_kwargs):
        return None

    def put_test_result(self, *_args, **_kwargs):
        return None

    def put_cluster_assign(self, *_args, **_kwargs):
        return None


class _CaptureFedAvg(fl.server.strategy.FedAvg):
    final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated[0]
        return aggregated


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _run_client(address: str, client_number: int) -> int:
    client_api.ClientServerAPI = _NoopClientServerAPI
    Path("local_model/task").mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = build_smoke_loaders(
        sample_count=16,
        batch_size=4,
        seed=40 + client_number,
    )
    config = OmegaConf.create({
        "batch_size": 4,
        "num_epochs": 1,
        "num_rounds": 1,
        "learning_rate": 0.001,
    })
    client = FLClient(
        model=build_model({"output_size": 10}),
        validation_split=0.2,
        fl_task_id="task",
        client_mac=f"smoke-{client_number}",
        client_name=f"client-{client_number}",
        fl_round=1,
        gl_model=0,
        wandb_use=False,
        wandb_name="",
        model_name=f"mnist-{client_number}",
        model_type="Pytorch",
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=validation_loader,
        cfg=config,
        train_torch=train_torch(),
        test_torch=test_torch(),
    )
    fl.client.start_client(server_address=address, client=client.to_client())
    return 0


class FederatedRoundTest(unittest.TestCase):
    def test_two_fedops_clients_complete_one_aggregate_round(self):
        port = _available_port()
        address = f"127.0.0.1:{port}"
        initial_model = build_model({"output_size": 10})
        strategy = _CaptureFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_parameters(initial_model, "Pytorch")
            ),
            on_fit_config_fn=lambda _round: {
                "batch_size": 4,
                "local_epochs": 1,
                "num_rounds": 1,
            },
            on_evaluate_config_fn=lambda _round: {"batch_size": 4},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            clients = []
            for client_number in (1, 2):
                working_directory = Path(temporary_directory) / f"client-{client_number}"
                working_directory.mkdir()
                clients.append(subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "client", address, str(client_number)],
                    cwd=working_directory,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ))
            fl.server.start_server(
                server_address=address,
                config=fl.server.ServerConfig(num_rounds=1),
                strategy=strategy,
            )
            outputs = []
            for process in clients:
                output, _ = process.communicate(timeout=30)
                outputs.append(output)
                self.assertEqual(process.returncode, 0, output)
        self.assertIsNotNone(strategy.final_parameters, "Flower did not aggregate client updates")


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "client":
    raise SystemExit(_run_client(sys.argv[2], int(sys.argv[3])))
