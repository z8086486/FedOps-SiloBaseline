# Federated Task: MNIST classifier

This starter preserves the FedOps 1.2 client/server execution contract and adds the
FedOps 1.3 local-development, Registry Release, and Tool AI contracts. It can be
developed locally, published as one immutable Release, and opened by another user in
FedOps Agent Studio.

## Intended use

Classify normalized 28×28 grayscale handwritten digit images into labels 0–9. Replace
the starter model and data adapter when creating a different Federated Task.

## Local data setup

Raw data remains on the Agent Studio device. Bind a local MNIST directory in Agent
Studio or place it in the account-local data area. The release packager always excludes
raw datasets and local paths.

Expected input:

- feature `image`: `float32`, shape `[1, 28, 28]`, normalized to `[-1, 1]`
- label `digit`: `int64`, values `0`–`9`

## Local training

```bash
uv sync --locked --extra participate --link-mode copy
uv run --locked --no-sync fedops-task local-train \
  --data-root "$FEDOPS_LOCAL_DATA_DIR"
uv run --locked --no-sync fedops-task check-readiness --mode release
```

Local training writes the versioned initial model to `model_release/`. It never uploads
the dataset.

## Federated participation

After downloading a Published Release, connect local data and run:

```bash
uv run --locked --no-sync fedops-task check-readiness \
  --mode participation \
  --data-root "$FEDOPS_LOCAL_DATA_DIR"
```

The readiness check constructs the actual FedOps client and uses the same parameter
contract as `fedops.client.client_fl.FLClient`. Agent Studio enables participation only
when local training changes that payload and its structure is compatible with the
Published Release.

The existing FedOps 1.2 entrypoints remain explicit:

```bash
uv run --locked --no-sync fedops-task-client
uv run --locked --no-sync fedops-task-client-manager
uv run --locked --no-sync fedops-task-server
```

FedOps Web and Agent Studio inject Task identity, local-data binding, server-manager
address, and federated-server endpoint at runtime. Do not hard-code those values in the
Release.

## Model use

`federated_task.tool:predict` loads the selected Initial or Global
Model and applies the same input normalization used by local training.

## Limitations

The bundled architecture is an MNIST example, not a general-purpose vision model.
Owners must document task-specific data quality, bias, safety, and evaluation limits
before publishing a derived task.

## Privacy

Raw samples, local filesystem paths, model updates, credentials, and signed download
URLs are never included in a Federated Task Release. Only readiness status and contract
fingerprints may be sent to FedOps Web.
