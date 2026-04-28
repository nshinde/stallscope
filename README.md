# Monitoring Tool

A starter implementation of an all-in-one NVIDIA GPU + network monitoring utility with an early-signal profiler that labels a job as:

- `FAST`
- `SLOW`
- `FAIL_RISK`
- `UNKNOWN` (if GPU telemetry is unavailable)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
monitoring-tool --json
```

## What it collects now

- GPU telemetry via `nvidia-smi` (utilization, memory usage, temperature, power, throttle reasons)
- Network telemetry via `/proc/net/dev` (traffic, drops, errors)
- System signals (CPU load average and memory availability from `/proc/meminfo`)
- Optional NCCL benchmark (`all_reduce_perf`) via `--nccl-test`

## Early-signal profiler

The profiler identifies likely slowdown/failure factors based on:

- Low GPU utilization (pipeline stalls / under-utilization)
- GPU memory pressure
- Thermal risk and clock throttling
- Network errors and drops
- High host CPU load and low available system memory

When risk flags are detected, alerts are emitted:

- to stdout (`Alerts:` section)
- as Prometheus metrics (textfile exporter hook)
- as webhook events (Grafana/Alertmanager webhook integration)

## Periodic mode + alert hooks

Run continuously every 30 seconds, write Prometheus metrics, and send webhooks:

```bash
monitoring-tool \
  --interval-seconds 30 \
  --prometheus-textfile /var/lib/node_exporter/textfile_collector/monitoring_tool.prom \
  --alert-webhook-url http://alertmanager:9093/api/v2/alerts
```

Key CLI arguments:

- `--interval-seconds`: run loop interval (0 = one-shot)
- `--prometheus-textfile`: writes latest gauges each cycle
- `--alert-webhook-url`: posts active alerts as JSON payload
- `--nccl-test`: optionally run NCCL all-reduce benchmark
- `--json`: structured output for automation



## Docker Compose (GPU-enabled)

Use the provided `Dockerfile` + `docker-compose.yml` to run this periodically with NVIDIA GPUs:

```bash
docker compose up --build -d
```

Then check:

```bash
docker compose logs -f monitoring-tool
```

Full instructions (including NVIDIA Container Toolkit install) are in `docs/docker_compose_gpu_guide.md`.

## Testing

### Local/unit tests

```bash
pytest -q
```

### Real GPU integration test (2+ GPUs)

```bash
RUN_GPU_INTEGRATION=1 pytest -q -m integration
```

Detailed instructions are in `docs/integration_test_guide.md`.

## Run NCCL benchmark integration

If [`nccl-tests`](https://github.com/NVIDIA/nccl-tests) is installed and `all_reduce_perf` is in `PATH`:

```bash
monitoring-tool --nccl-test --json
```

The output includes an `nccl` section with status, parsed time, and bandwidth metrics.

## Next steps

See architecture and expansion plan in:

- `docs/all_in_one_gpu_network_monitoring_blueprint.md`
