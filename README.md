# FedOps Silo Baseline

FedOps Silo Baseline은 FedOps Agent Studio에서 새 Federated Task를 만들 때 사용하는
기본 프로젝트입니다. Task Owner는 모델과 로컬 데이터 계약을 수정하고, Agent Studio의
uv Python 환경에서 검증한 뒤 FedOps Registry workflow로 연결할 수 있습니다.

공개 실행 계약은 `fedops-task` 하나이며 두 action을 제공합니다.

- `validate`: FedOps 서버에 연결하지 않고 모델·입력 feature·학습 코드를 로컬 검증
- `participate`: 승인된 Task ID와 runtime key로 FedOps 연합학습에 참여

프로젝트를 생성하거나 여는 것만으로 연합학습이 시작되지는 않습니다.

## Owner editing guide

| File | Purpose |
| --- | --- |
| `fedops_silo_baseline/model.py` | 모델 정의, 로컬 학습, 평가 |
| `fedops_silo_baseline/data_preparation.py` | 입력 feature와 로컬 데이터 전처리 계약 |
| `fedops_silo_baseline/conf/config.toml` | Task 학습 기본 설정 |
| `README.md` | Task 설명과 사용법 |

`launcher_app.py`, `client_app.py`, `client_main.py`, `client_manager_main.py`는 Agent Studio와 FedOps
참여 workflow를 연결하는 runtime adapter입니다.

## Requirements

- Python 3.10, 3.11, 3.12
- uv
- 로컬 검증은 CPU만으로 가능
- 공개 MNIST 다운로드 또는 실제 Task 참여 시에만 네트워크 필요

## Install and validate

```bash
uv sync
uv run --locked --no-sync fedops-task validate
```

검증은 deterministic synthetic MNIST-shaped sample을 사용합니다. 입력 계약, 모델 출력
shape, 짧은 로컬 학습, 평가를 확인하며 FedOps 서비스에 데이터를 보내지 않습니다.

## Participate in a FedOps Task

참여 dependency를 동기화합니다.

```bash
uv sync --extra participate
```

Agent Studio가 승인된 Task 정보로 다음 계약을 호출합니다.

```bash
uv run --locked --no-sync fedops-task participate \
  --task-id "<TASK_OBJECT_ID>" \
  --runtime-key "<RUNTIME_KEY>"
```

연결 주소가 Task 응답으로 제공된 경우 CLI option으로 덮어쓸 수 있습니다.

```bash
uv run --locked --no-sync fedops-task participate \
  --task-id "<TASK_OBJECT_ID>" \
  --runtime-key "<RUNTIME_KEY>" \
  --server-manager-url "http://HOST:PORT" \
  --federated-server-host "HOST"
```

`task_id`는 안정적인 데이터베이스 식별자이고 `runtime_key`는 현재 FedOps 서버, S3,
Kubernetes resource lookup에 사용되는 값입니다. 화면의 Task display name으로 대체하면
안 됩니다.

## Data boundary

- 원본 로컬 데이터는 참여자 장치에 남습니다.
- 로컬 검증은 데이터나 모델 parameter를 업로드하지 않습니다.
- 기본 MNIST loader는 참여 action에서만 공개 MNIST 데이터를 다운로드합니다.
- credential, private Task YAML, Kubernetes 정보는 프로젝트 파일에 저장하지 않습니다.

## Project structure

```text
FedOps-SiloBaseline/
├── baseline-manifest.json
├── manifest.json
├── pyproject.toml
├── uv.lock
├── fedops_silo_baseline/
│   ├── launcher_app.py
│   ├── client_app.py
│   ├── client_main.py
│   ├── client_manager_main.py
│   ├── model.py
│   ├── data_preparation.py
│   ├── validation.py
│   └── conf/config.toml
├── tests/
├── tools/build_manifest.py
└── .gitignore
```

## Baseline metadata

- Baseline release: `0.2.0`
- Template revision: `3`
- Python: `>=3.10,<3.13`
- FedOps participation package: `1.1.30.13`
- Task contract: `[tool.fedops.task]`, schema version `1`

`baseline-manifest.json`은 Files & versions 업로드 및 무결성 확인을 위한 파일 목록과
checksum을 제공합니다.
