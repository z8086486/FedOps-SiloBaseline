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

The current starter is a functionally organized implementation-contract template rather than an MNIST example.
It supports:

- fixed, documented user hooks for model, data, training, readiness probes, and Tool AI
- local model training and Initial Model export after those hooks are implemented
- live Local Train percentage, epoch/batch position, loss, and evaluation metrics in Agent Studio
- FedOps federated participation using the same training implementation
- Release Readiness for Owner publication
- Participation Readiness for participant data and parameter-update preflight
- Agent Builder Tool inference with an Initial or Global Model

Owner-editable code is grouped under `local_training/`, `tool_ai/`, and `conf/`.
FedOps-managed integration is grouped under `federated_learning/`, `task_readiness/`,
and `runtime/`. The same model definition is shared across local training, federation,
and Tool AI.

Runnable domain examples are kept separately in
`../FedOps-AgentStudio-TestRunCases/`; they are not shipped as the default Baseline.

## Verify

```bash
cd federated-task-baseline
uv sync --locked --extra participate --link-mode copy
cd ..
federated-task-baseline/.venv/bin/python -m unittest discover -s tests
federated-task-baseline/.venv/bin/python tools/build_release.py
```

## Release policy

- Current development release: `federated-task-baseline@0.12.0`
- Existing releases remain available through Git history and existing Web/S3 tasks.
- A release is immutable. Changes require a new version.
- Raw datasets, `.venv`, local artifacts, credentials, and readiness run details are
  excluded from the distributed starter.
- Python build output (`build/`, `dist/`) and hidden/cache files are never included in
  a Baseline manifest, even when a maintainer builds a wheel before exporting it.
