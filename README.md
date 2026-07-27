# FedOps Silo Baseline

FedOps Silo Baseline is an owner starter project for defining the model and
local-data contract of a FedOps Task. It is based on the structure of
[`@gfedops/fedops-mnist:0.0.3`](https://flower.ai/apps/gfedops/fedops-mnist/)
and is designed to run inside FedOps-Launcher.

The default example uses MNIST and PyTorch. It has two explicit modes:

- `validate`: verify the project locally without connecting to a FedOps server.
- `participate`: start the FedOps local client and communication manager for a
  real Task.

`validate` is the default. Creating or opening the project never starts
federated participation by itself.

## Owner editing guide

Task owners normally edit these files:

| File | Purpose |
| --- | --- |
| `fedops_silo_baseline/model.py` | Model, local training, and evaluation |
| `fedops_silo_baseline/data_preparation.py` | Input-feature and local-data contract |
| `fedops_silo_baseline/conf/config.toml` | Public training defaults |
| `README.md` | Task-specific description and usage |

The remaining Python files are runtime adapters used by Flower and FedOps.

## Requirements

- Python 3.10, 3.11, or 3.12
- CPU is sufficient for validation
- Network access is required only when downloading MNIST or participating in a
  real Task

## Install and validate

Create an isolated environment and install the base project:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the direct validator:

```bash
fedops-baseline-validate
```

Run the same validation through Flower:

```bash
flwr run .
```

Validation uses deterministic synthetic MNIST-shaped samples. It checks the
input contract, model output shape, one short local training pass, and
evaluation without downloading data or contacting a FedOps service.

## Participate in a FedOps Task

Install the participation dependencies:

```bash
python -m pip install -e ".[participate]"
```

Then run with the stable Task identifier and its FedOps runtime key:

```bash
flwr run . --run-config \
  'mode="participate" task_id="<TASK_OBJECT_ID>" runtime_key="<RUNTIME_KEY>"'
```

Connection values can be overridden without changing Python code:

```bash
flwr run . --run-config \
  'mode="participate" task_id="<TASK_OBJECT_ID>" runtime_key="<RUNTIME_KEY>" server_manager_url="http://HOST:PORT"'
```

`task_id` is the stable database identifier. `runtime_key` is the existing
FedOps server, S3, and Kubernetes lookup key and must not be replaced with the
Task display name in Launcher code. FedOps-Launcher supplies both values and
the connection configuration in the integrated workflow.

## Data boundary

- Raw local records remain on the participant device.
- Validation does not upload model parameters or data.
- The default MNIST loader downloads public MNIST data only in participation
  mode.
- Do not place credentials, private Task YAML, or Kubernetes information in
  this project.

## Project structure

```text
FedOps-SiloBaseline/
├── baseline-manifest.json
├── pyproject.toml
├── fedops_silo_baseline/
│   ├── client_app.py
│   ├── launcher_app.py
│   ├── client_main.py
│   ├── client_manager_main.py
│   ├── model.py
│   ├── data_preparation.py
│   ├── validation.py
│   └── conf/config.toml
├── tests/
└── tools/build_manifest.py
```

## Baseline provenance

- Baseline release: `0.1.0`
- Template revision: `2`
- Flower compatibility target: `1.26.1`
- FedOps participation package: `1.1.30.13`
- Reference App: `@gfedops/fedops-mnist:0.0.3`

The reference App is Apache-2.0 licensed. This project keeps that provenance in
`baseline-manifest.json` and contains a restructured implementation intended
for the FedOps Task Hub workflow.
