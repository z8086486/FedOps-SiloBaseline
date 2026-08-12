# Federated Task: MNIST classifier

This starter defines a complete FedOps Federated Task that can be developed locally,
published as an immutable release, and opened by an approved participant in FedOps
Agent Studio.

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
uv sync
uv run --locked --no-sync fedops-task local-train
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

Agent Studio enables participation only when local training produces a serializable
parameter update compatible with the Published Release and current server run.

## Model use

`federated_task.agent_tool.inference:predict` loads the selected Initial or Global
Model and applies the same input normalization used by local training.

## Limitations

The bundled architecture is an MNIST example, not a general-purpose vision model.
Owners must document task-specific data quality, bias, safety, and evaluation limits
before publishing a derived task.

## Privacy

Raw samples, local filesystem paths, model updates, credentials, and signed download
URLs are never included in a Federated Task Release. Only readiness status and contract
fingerprints may be sent to FedOps Web.
