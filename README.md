# stallscope

Correlate GPU training job health with RDMA fabric counters on the same node with shared job context.

`stallscope` is a Python 3.10+ observability utility for GPU training nodes. It collects local GPU, host, network, RDMA/RoCEv2, PFC pause, and scheduler context signals, then applies rule-based profiling to label the current job as `FAST`, `SLOW`, `FAIL_RISK`, or `UNKNOWN`.

The implementation is stdlib-only at runtime and is designed to keep parsing testable without GPU, RDMA, or Slurm/Kubernetes hardware by using pure parser functions and fixture-backed tests.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
stallscope --json
```

Run continuously every 30 seconds:

```bash
stallscope --interval-seconds 30 --json
```

Write Prometheus textfile metrics for node_exporter:

```bash
stallscope \
  --interval-seconds 30 \
  --prometheus-textfile /var/lib/node_exporter/textfile_collector/stallscope.prom
```

Send alert webhook payloads when alerts are active:

```bash
stallscope \
  --interval-seconds 30 \
  --alert-webhook-url http://alertmanager:9093/api/v2/alerts
```

Run the optional NCCL all-reduce benchmark if `all_reduce_perf` is installed and available in `PATH`:

```bash
stallscope --nccl-test --json
```

## What It Collects

| Metric family | Source | What a spike means |
| --- | --- | --- |
| GPU utilization, memory, temperature, power, throttle reasons | `nvidia-smi --query-gpu=...` | Low utilization can indicate stalls; high memory use, thermal risk, or throttling can contribute to slow or risky jobs. |
| Interface bytes, packets, errors, drops | `/proc/net/dev` | Errors and drops indicate local network interface health problems visible to the host. |
| Host load and memory availability | `os.getloadavg()`, `/proc/meminfo` | High CPU load or low available memory can indicate host-side pressure. |
| RDMA port traffic and basic link counters | `/sys/class/infiniband/<dev>/ports/<port>/counters/port_rcv_data`, `port_xmit_data`, `port_rcv_errors`, `symbol_error`, `link_downed`, `port_rcv_remote_physical_errors`, `port_xmit_discards` | Receive/transmit data tracks RDMA activity; errors, link-down events, remote physical errors, and transmit discards indicate fabric or link health issues. |
| RoCEv2 congestion counters | `/sys/class/infiniband/<dev>/ports/<port>/hw_counters/np_cnp_sent`, `rp_cnp_handled`, `np_ecn_marked_roce_packets` | Positive per-interval deltas indicate CNP/ECN congestion signaling on the fabric. |
| RDMA transport error counters | `/sys/class/infiniband/<dev>/ports/<port>/hw_counters/rnr_nak_retry_err`, `out_of_sequence`, `packet_seq_err`, `local_ack_timeout_err`, `duplicate_request` | Positive deltas on retry, ordering, sequence, or timeout counters indicate transport-level reliability problems. |
| PFC pause counters | `ethtool -S <interface>` fields `rx_prio*_pause`, `tx_prio*_pause`, `rx_pause`, `tx_pause` | High per-interval pause-frame deltas indicate pause storms or severe lossless fabric congestion. |
| Slurm job context | `SLURM_JOB_ID` environment or `/proc/*/environ` scan | Adds scheduler, job id, job name, user, and nodelist context when visible on the node. |
| Kubernetes pod context | `/var/run/secrets/kubernetes.io` and hostname | Adds Kubernetes scheduler context and pod name when running inside a pod. |

All RDMA sysfs counters are treated as optional because availability varies by NIC, driver, and firmware. Missing counters produce warnings and do not stop collection.

For fixture-based tests on non-Linux or non-RDMA systems, `MONITORING_PROCFS_ROOT` overrides `/proc` and `MONITORING_SYSFS_ROOT` overrides `/sys`.

## Profiler Output

The profiler emits:

- `label`: `FAST`, `SLOW`, `FAIL_RISK`, or `UNKNOWN`
- `confidence`: rule-derived confidence score
- `reasons`: the signals that fired
- `bottleneck_hint`: `compute`, `network`, `memory`, `mixed`, or `unknown`

Fabric rules use deltas against the previous snapshot in periodic mode:

- CNP or ECN-marked RoCE packet deltas produce a fabric congestion reason and can classify the job as `SLOW`.
- RNR NAK retry, out-of-sequence, or local ACK timeout deltas produce a fabric error reason and contribute to `FAIL_RISK`.
- Sustained PFC pause-frame rates produce a fabric congestion reason.

One-shot mode has no previous snapshot, so lifetime RDMA/PFC counters are collected and reported in JSON, but delta-based fabric profiler rules only apply after a second periodic sample exists.

## Prometheus And Alerts

Prometheus textfile output includes core profile, alert, GPU, network, and bottleneck gauges. When Slurm or Kubernetes context is detected, metric samples include `scheduler` and `job_id` labels.

Alert generation categorizes fabric congestion and RDMA transport reasons as `category="fabric"`. Existing GPU, network, system, profiler, and telemetry alerts remain rule-based and local to the node.

## Docker Compose

Use the provided `Dockerfile` and `docker-compose.yml` to run the collector periodically with NVIDIA GPU access:

```bash
docker compose up --build -d
docker compose logs -f stallscope
```

Full Docker and NVIDIA Container Toolkit setup notes are in `docs/docker_compose_gpu_guide.md`.

## Testing

Run unit tests:

```bash
pytest -q
```

Run real GPU integration tests on a node with 2+ GPUs:

```bash
RUN_GPU_INTEGRATION=1 pytest -q -m integration
```

Integration test details are in `docs/integration_test_guide.md`.

## Roadmap

Future work, not implemented in this repository today:

- DCGM profiling fields for SM occupancy, tensor activity, PCIe/NVLink throughput, and memory bandwidth.
- NCCL timing ingestion from training runs, separate from the optional `all_reduce_perf` smoke benchmark.
- PyTorch Profiler trace classification for correlating framework-level stalls with node and fabric counters.
