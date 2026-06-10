# Integration Test Guide (Real GPUs)

This project includes an integration test that exercises the CLI against real NVIDIA hardware.

- Test file: `tests/integration/test_multi_gpu_integration.py`
- Purpose: run the tool end-to-end on a machine with **2+ GPUs** and verify JSON output shape and profiler labels.

## Prerequisites

- Linux host with NVIDIA driver installed
- `nvidia-smi` available in `PATH`
- 2 or more visible GPUs
- Optional: `all_reduce_perf` from `nccl-tests` for NCCL checks

## Run unit tests only

```bash
pytest -q
```

## Run integration test on GPU machine

```bash
RUN_GPU_INTEGRATION=1 pytest -q -m integration
```

The test auto-skips if:

- `nvidia-smi` is missing,
- fewer than 2 GPUs are available,
- or `RUN_GPU_INTEGRATION` is not set.

## Run a synthetic workload while monitoring

Open terminal 1:

```bash
stallscope --interval-seconds 5 --json
```

Open terminal 2 and run any GPU workload (examples):

```bash
# PyTorch example if torch is installed
python - <<'PY'
import torch
x = torch.randn(8192, 8192, device='cuda')
for _ in range(200):
    x = x @ x
torch.cuda.synchronize()
print('done')
PY
```

or NCCL benchmark if available:

```bash
stallscope --nccl-test --json
```
