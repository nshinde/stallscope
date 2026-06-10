from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request

from .models import Snapshot
from .profiler import JobProfile


@dataclass
class AlertEvent:
    severity: str
    category: str
    message: str


def build_alerts(snapshot: Snapshot, profile: JobProfile) -> list[AlertEvent]:
    alerts: list[AlertEvent] = []

    for reason in profile.reasons:
        if "Fabric" in reason or "RDMA" in reason or "CNP" in reason or "ECN" in reason or "PFC" in reason:
            alerts.append(AlertEvent(severity="warning", category="fabric", message=reason))
        elif "Network" in reason:
            alerts.append(AlertEvent(severity="warning", category="network", message=reason))
        elif "GPU" in reason or "Thermal" in reason or "Clock throttling" in reason:
            alerts.append(AlertEvent(severity="critical", category="gpu", message=reason))
        elif "memory" in reason.lower() or "CPU" in reason:
            alerts.append(AlertEvent(severity="warning", category="system", message=reason))

    if profile.label == "FAIL_RISK":
        alerts.append(AlertEvent(severity="critical", category="profiler", message="Job has FAIL_RISK profile"))
    elif profile.label == "SLOW":
        alerts.append(AlertEvent(severity="warning", category="profiler", message="Job has SLOW profile"))

    if not snapshot.gpus:
        alerts.append(AlertEvent(severity="warning", category="telemetry", message="GPU telemetry unavailable"))

    return alerts


def send_alert_webhook(url: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    request.urlopen(req, timeout=5).read()


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _job_labels(snapshot: Snapshot) -> dict[str, str]:
    if snapshot.job.scheduler == "UNKNOWN":
        return {}
    labels = {"scheduler": snapshot.job.scheduler}
    if snapshot.job.job_id:
        labels["job_id"] = snapshot.job.job_id
    return labels


def _metric_line(name: str, value: str | int | float, labels: dict[str, str]) -> str:
    if not labels:
        return f"{name} {value}"
    rendered = ",".join(f'{key}="{_escape_label_value(label_value)}"' for key, label_value in labels.items())
    return f"{name}{{{rendered}}} {value}"


def render_prometheus_metrics(snapshot: Snapshot, profile: JobProfile, alerts: list[AlertEvent]) -> str:
    label_value = {"FAST": 0, "SLOW": 1, "FAIL_RISK": 2, "UNKNOWN": 3}.get(profile.label, 3)
    bottleneck_value = {"compute": 0, "network": 1, "memory": 2, "mixed": 3, "unknown": 4}.get(
        profile.bottleneck_hint, 4
    )
    job_labels = _job_labels(snapshot)
    lines = [
        "# HELP monitoring_profile_label Encoded profile label (0=FAST,1=SLOW,2=FAIL_RISK,3=UNKNOWN)",
        "# TYPE monitoring_profile_label gauge",
        _metric_line("monitoring_profile_label", label_value, job_labels),
        "# HELP monitoring_profile_confidence Confidence score for current profile",
        "# TYPE monitoring_profile_confidence gauge",
        _metric_line("monitoring_profile_confidence", profile.confidence, job_labels),
        "# HELP monitoring_bottleneck_hint Encoded bottleneck hint (0=compute,1=network,2=memory,3=mixed,4=unknown)",
        "# TYPE monitoring_bottleneck_hint gauge",
        _metric_line("monitoring_bottleneck_hint", bottleneck_value, job_labels),
        "# HELP monitoring_active_alerts Number of active alerts",
        "# TYPE monitoring_active_alerts gauge",
        _metric_line("monitoring_active_alerts", len(alerts), job_labels),
    ]

    if snapshot.gpus:
        avg_gpu_util = sum(g.utilization_gpu_pct for g in snapshot.gpus) / len(snapshot.gpus)
        lines += [
            "# HELP monitoring_gpu_utilization_avg Average GPU utilization percent",
            "# TYPE monitoring_gpu_utilization_avg gauge",
            _metric_line("monitoring_gpu_utilization_avg", f"{avg_gpu_util:.2f}", job_labels),
        ]

    total_net_err = sum(n.rx_errs + n.tx_errs + n.rx_drop + n.tx_drop for n in snapshot.net)
    lines += [
        "# HELP monitoring_network_errors_total Sum of rx/tx errors and drops",
        "# TYPE monitoring_network_errors_total gauge",
        _metric_line("monitoring_network_errors_total", total_net_err, job_labels),
    ]

    return "\n".join(lines) + "\n"
