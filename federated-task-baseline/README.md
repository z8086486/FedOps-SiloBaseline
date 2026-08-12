# Federated Task: replace with Task name

Replace this starter text with the Registry Task Card. The Baseline intentionally
contains no example model or dataset. Implement the marked contracts below while
keeping their public function names, arguments, and return values.

## Implementation contracts

| File | Function | Fixed output |
| --- | --- | --- |
| `federated_task/model.py` | `build_model(config)` | new `torch.nn.Module` |
|  | `run_model(model, inputs)` | raw model output |
|  | `validate_model_output(output, config)` | JSON-serializable output summary |
| `federated_task/data_preparation.py` | `describe_input_features()` | JSON-serializable feature/label contract |
|  | `preprocess(sample)` | model input tensor structure |
|  | `load_partition(...)` | `(train_loader, validation_loader, test_loader)` |
|  | `build_smoke_loaders(...)` | `(train_loader, validation_loader)` |
|  | `build_contract_probe(batch_size)` | one non-sensitive batched model input |
|  | `gl_model_torch_validation(...)` | server validation `DataLoader` |
| `federated_task/training.py` | `train_model(...)` | finite mean training-loss `float` |
|  | `evaluate_model(...)` | `(loss, primary_metric, metrics)` |
| `federated_task/tool.py` | `predict(payload, model_path)` | Tool manifest-compatible JSON object |
|  | `build_tool_smoke_payload()` | non-sensitive Tool input JSON object |

Each function contains its argument and return contract in its docstring. Do not
duplicate federated parameter serialization: the FedOps library transports the model
parameters created by `build_model()`.

## Intended use

TODO: Describe the problem, intended users, Initial/Global Model behavior, and the
primary evaluation metric.

## Local data setup

TODO: Document the expected local files, columns/features, labels, shapes, dtypes,
preprocessing, and train/validation/test split. Raw data must remain on the Agent
Studio device and is supplied through `--data-root` or `FEDOPS_LOCAL_DATA_DIR`.

Do not include raw data, credentials, or a user-specific absolute path in a Registry
Release.

## Local development

1. Implement the user contracts in `model.py`, `data_preparation.py`, `training.py`,
   and `tool.py`.
2. Replace `conf/config.yaml` placeholders and update the Tool `manifest.json`.
3. Add Task dependencies to `pyproject.toml`, then let Agent Studio Environment Sync
   update `uv.lock`; do not edit `uv.lock` by hand.
4. Bind local data and create the Initial Model:

```bash
uv sync --extra participate --link-mode copy
uv run fedops-task local-train --data-root "$FEDOPS_LOCAL_DATA_DIR"
uv run fedops-task tool-test
uv run fedops-task check-readiness --mode release
```

Unimplemented contracts stop with a specific `NotImplementedError`; a blank starter
cannot accidentally pass Release Readiness.

## Federated participation

After opening a Published Release, a participant binds their own local data and runs:

```bash
uv run --locked --no-sync fedops-task check-readiness \
  --mode participation \
  --data-root "$FEDOPS_LOCAL_DATA_DIR"
```

Participation Readiness verifies the actual local loader, local training update,
FedOps parameter signature, model input/output contract, and Tool inference without
uploading raw data or parameter values.

## Model and Tool release

`local-train` writes `model_release/model.safetensors` and replaces the draft model
manifest with checksum, size, parameter signature, provenance, and evaluation metrics.
Update `federated_task/manifest.json` and `tool.py` together so Agent Builder receives
the exact JSON input/output contract.

## Limitations

TODO: Document known model, population, data-quality, bias, safety, latency, hardware,
and out-of-distribution limitations.

## Privacy

Raw samples, local filesystem paths, credentials, signed download URLs, and model
parameter values must not be included in the Federated Task Release or readiness
metadata. Only source/model checksums, parameter-structure fingerprints, metrics, and
pass/fail status may leave the local device.
