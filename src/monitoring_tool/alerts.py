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
        if "Network" in reason:
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


def render_prometheus_metrics(snapshot: Snapshot, profile: JobProfile, alerts: list[AlertEvent]) -> str:
    label_value = {"FAST": 0, "SLOW": 1, "FAIL_RISK": 2, "UNKNOWN": 3}.get(profile.label, 3)
    lines = [
        "# HELP monitoring_profile_label Encoded profile label (0=FAST,1=SLOW,2=FAIL_RISK,3=UNKNOWN)",
        "# TYPE monitoring_profile_label gauge",
        f"monitoring_profile_label {label_value}",
        "# HELP monitoring_profile_confidence Confidence score for current profile",
        "# TYPE monitoring_profile_confidence gauge",
        f"monitoring_profile_confidence {profile.confidence}",
        "# HELP monitoring_active_alerts Number of active alerts",
        "# TYPE monitoring_active_alerts gauge",
        f"monitoring_active_alerts {len(alerts)}",
    ]

    if snapshot.gpus:
        avg_gpu_util = sum(g.utilization_gpu_pct for g in snapshot.gpus) / len(snapshot.gpus)
        lines += [
            "# HELP monitoring_gpu_utilization_avg Average GPU utilization percent",
            "# TYPE monitoring_gpu_utilization_avg gauge",
            f"monitoring_gpu_utilization_avg {avg_gpu_util:.2f}",
        ]

    total_net_err = sum(n.rx_errs + n.tx_errs + n.rx_drop + n.tx_drop for n in snapshot.net)
    lines += [
        "# HELP monitoring_network_errors_total Sum of rx/tx errors and drops",
        "# TYPE monitoring_network_errors_total gauge",
        f"monitoring_network_errors_total {total_net_err}",
    ]

    return "\n".join(lines) + "\n"
