# All-in-One NVIDIA GPU + Network Monitoring & Job Profiler Blueprint

This document gives you a practical way to start building a single tool that:
1. Collects GPU and network telemetry,
2. Correlates those metrics with jobs,
3. Profiles runs in near-real-time, and
4. Predicts whether a job is likely to be **FAST**, **SLOW**, or **AT-RISK/FAIL**.

---

## 1) Define the outcomes first

Before writing code, define the outputs your tool must provide:

- **Real-time dashboard** (cluster/node/job-level).
- **Per-job timeline** with bottleneck annotations.
- **Classifier**: FAST / SLOW / FAIL-RISK.
- **Root-cause hints** (e.g., network congestion, GPU memory pressure, thermal throttling).
- **Action suggestions** (reduce batch size, tune NCCL, re-place job, etc.).

If these outputs are clear, architecture decisions become straightforward.

---

## 2) What to observe (metric checklist)

### A) GPU-level (NVIDIA)
Collect from NVML / DCGM (prefer DCGM exporter for production):

- Utilization: `sm`, `tensor`, `memory`, `encoder/decoder`.
- Memory: used/free, alloc failures, ECC errors, page retirement.
- Clocks: graphics/SM/memory clocks and throttle reasons.
- Power + thermals: power draw, power cap, temperature, thermal slowdowns.
- Reliability: XID errors, PCIe replay/errors, NVLink errors.
- Fabric/interconnect:
  - PCIe throughput and link width/speed.
  - NVLink bandwidth per link.

### B) Network-level
Collect host + NIC + transport metrics:

- Interface throughput (bytes/packets in/out).
- Drops, errors, retransmits, queue depth, ring-buffer pressure.
- Latency/jitter (especially for distributed training).
- TCP/UDP stats, socket pressure.
- RDMA/InfiniBand (if applicable): CQE errors, retries, link utilization, congestion.
- NCCL-visible effects: all-reduce latency, bandwidth, timeout events.

### C) System + runtime context
Without context, telemetry is hard to explain:

- CPU utilization, load, steal time.
- RAM/swap pressure and OOM events.
- Disk I/O and checkpoint throughput.
- Container/Kubernetes metadata: pod, node, namespace, image, cgroup limits.
- Scheduler metadata: job ID, user, queue, requested GPUs.
- Process-level metadata: PID, command, start time, environment knobs (NCCL/CUDA).

### D) Job-level derived features (important for profiler)
These are often more predictive than raw counters:

- GPU utilization stability (mean + variance).
- Step-time trend slope (improving/stable/degrading).
- Communication/computation ratio.
- Memory headroom percentage.
- Error-rate trend (ECC/XID/retries).
- Thermal throttle duty-cycle.

---

## 3) Reference architecture (MVP -> production)

### Data plane
1. **Node Agent** (DaemonSet or systemd):
   - Pull GPU metrics (DCGM/NVML), network metrics (ethtool + /proc + exporter), and process metadata.
2. **Ingestion Bus**:
   - Prometheus scrape for metrics; optional Kafka for events/logs.
3. **Time-series storage**:
   - Prometheus for short-term, optionally Thanos/Cortex/Mimir for long retention.
4. **Event store**:
   - Keep XID/NCCL/OOM events as structured records.

### Control/analysis plane
1. **Feature builder**:
   - Convert windows of metrics into features (30s/2m/10m windows).
2. **Profiler service**:
   - Rule engine first, ML model second.
3. **API service**:
   - Expose `/job/{id}/health`, `/job/{id}/prediction`, `/node/{id}/diagnostics`.
4. **UI**:
   - Grafana + custom panel for FAST/SLOW/FAIL state and confidence.

---

## 4) Profiler design (FAST/SLOW/FAIL)

Start simple and explainable.

### Phase 1: Rule-based baseline
Define interpretable rules with severity points:

- High GPU util + low network retransmits + stable step time -> **FAST**.
- GPU util oscillating + high comm wait + rising retransmits -> **SLOW**.
- Repeated XID, ECC growth, OOM, NCCL timeout, sustained thermal throttle -> **FAIL-RISK**.

Generate:
- Label: FAST/SLOW/FAIL-RISK
- Confidence (0-1)
- Top contributing factors

### Phase 2: Supervised model
After collecting historical job outcomes:

- Model candidates: Gradient Boosted Trees, Random Forest, LightGBM/XGBoost.
- Inputs: rolling-window features + job metadata.
- Outputs:
  - Class probabilities,
  - Time-to-failure estimate (optional).

Keep the rule-engine as fallback so behavior remains robust when model confidence is low.

---

## 5) Data model you should define early

- `job_run`: job id, start/end, framework, model, batch size, node set.
- `sample`: timestamp, entity (gpu_id, nic_id, pod_id), metric/value.
- `event`: timestamp, type (XID/OOM/NCCL_TIMEOUT), severity, details.
- `prediction`: timestamp, label, confidence, feature snapshot.
- `diagnosis`: probable cause + recommendation.

This schema discipline prevents chaos later.

---

## 6) Alerting and SLO strategy

Separate **noise** from actionable alerts:

- Warning: short anomalies (e.g., transient packet drops).
- Critical: persistent degradations + job impact signals.

Example SLOs:

- p95 step time degradation < 10% from baseline.
- GPU duty-cycle > 85% for compute-bound workloads.
- NCCL timeout rate = 0 for production jobs.

---

## 7) Security, multitenancy, and reliability

- RBAC: users only view their jobs unless cluster-admin.
- Redact sensitive command/env values.
- Backpressure and buffering in agents (avoid data loss).
- Version all metric contracts; avoid breaking dashboards.
- Handle missing sensors gracefully (degraded mode).

---

## 8) MVP build order (practical 6-step plan)

1. **Collector MVP**: pull DCGM + node exporter + job metadata.
2. **Store + dashboard MVP**: Prometheus + Grafana panels per node/GPU/job.
3. **Correlation MVP**: join metrics with job ids and process ids.
4. **Rule profiler MVP**: FAST/SLOW/FAIL-RISK with reason codes.
5. **Alerting MVP**: threshold + trend-based alerts.
6. **Historical learning**: store outcomes, train first model, compare vs rules.

---

## 9) Technology choices (pragmatic defaults)

- Agent language: Go or Rust (low-overhead systems integration).
- Metrics: Prometheus format, OpenTelemetry where useful.
- GPU integration: DCGM exporter first, NVML direct only when needed.
- UI: Grafana for speed + optional React app for investigation workflows.
- Model serving: Python microservice (FastAPI) or embedded model runtime.

---

## 10) Pitfalls to avoid

- Overfitting to one workload (train/serve mismatch).
- Ignoring per-GPU topology (PCIe/NVLink asymmetry matters).
- Alert floods (no cooldown/hysteresis).
- Missing clock sync across nodes (breaks correlation).
- No baseline profiles per workload type.

---

## 11) Suggested repository starter layout

```text
/agent
  /collectors (gpu, network, process, kube)
  /enrichers
/api
  /routes
  /services
/profiler
  /rules
  /models
  /features
/ui
/deploy
  /k8s
/docs
```

---

## 12) "Day 1" acceptance criteria

You can call v1 successful when you can:

- See per-job GPU/network timelines.
- Detect at least 3 common failure modes automatically.
- Produce FAST/SLOW/FAIL-RISK every minute with reasons.
- Show fewer than 5% false critical alerts on stable workloads.

