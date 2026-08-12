# FedOps Federated Task Baseline

This repository develops and verifies the starter used to create a FedOps Federated
Task. Runtime products do not clone this repository. A verified release is vendored
into the FedOps Web backend, and Agent Studio receives it through an authenticated
FedOps Web API.

## Repository boundary

```text
federated-task-baseline/  exact user Workspace starter
tests/                    Baseline maintainer tests; not published
tools/                    release exporter; not published
```

The starter supports:

- local model training and Initial Model export
- FedOps federated participation using the same training implementation
- Release Readiness for Owner publication
- Participation Readiness for participant data and parameter-update preflight
- Agent Builder Tool inference with an Initial or Global Model

## Verify

```bash
cd federated-task-baseline
uv sync --locked
cd ..
PYTHONPATH=federated-task-baseline \
  federated-task-baseline/.venv/bin/python -m unittest discover -s tests
federated-task-baseline/.venv/bin/python tools/build_release.py
```

## Release policy

- Current release: `federated-task-baseline@0.3.0`
- Existing `0.2.0` remains available through Git history and existing Web/S3 tasks.
- A release is immutable. Changes require a new version.
- Raw datasets, `.venv`, local artifacts, credentials, and readiness run details are
  excluded from the distributed starter.
