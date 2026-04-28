from __future__ import annotations

from dataclasses import dataclass

from .models import Snapshot


@dataclass
class JobProfile:
    label: str
    confidence: float
    reasons: list[str]


def classify_job(snapshot: Snapshot) -> JobProfile:
    if not snapshot.gpus:
        return JobProfile(
            label="UNKNOWN",
            confidence=0.2,
            reasons=["No GPU telemetry available"],
        )

    reasons: list[str] = []
    risk_score = 0

    avg_gpu_util = sum(g.utilization_gpu_pct for g in snapshot.gpus) / len(snapshot.gpus)
    avg_mem_util = sum(g.utilization_mem_pct for g in snapshot.gpus) / len(snapshot.gpus)
    min_headroom = min(g.memory_headroom_pct for g in snapshot.gpus)
    hot_gpus = [g.index for g in snapshot.gpus if g.temperature_c >= 85.0]
    throttled = [
        g.index
        for g in snapshot.gpus
        if g.clocks_throttle_reasons and g.clocks_throttle_reasons.lower() != "not active"
    ]

    if avg_gpu_util < 35:
        risk_score += 2
        reasons.append(f"Low average GPU utilization ({avg_gpu_util:.1f}%)")
    elif avg_gpu_util >= 80:
        reasons.append(f"High average GPU utilization ({avg_gpu_util:.1f}%)")

    if avg_mem_util > 90 or min_headroom < 8:
        risk_score += 2
        reasons.append("GPU memory pressure is high")

    if hot_gpus:
        risk_score += 2
        reasons.append(f"Thermal risk on GPUs {hot_gpus}")

    if throttled:
        risk_score += 2
        reasons.append(f"Clock throttling detected on GPUs {throttled}")

    total_net_err = sum(n.rx_errs + n.tx_errs + n.rx_drop + n.tx_drop for n in snapshot.net)
    if total_net_err > 0:
        risk_score += 2
        reasons.append(f"Network errors/drops detected ({total_net_err})")

    if snapshot.system is not None:
        if snapshot.system.load_1m > 16:
            risk_score += 1
            reasons.append(f"High CPU load average (1m={snapshot.system.load_1m:.2f})")
        if snapshot.system.mem_available_pct < 10:
            risk_score += 2
            reasons.append("System memory availability is critically low")

    if risk_score >= 6:
        label = "FAIL_RISK"
    elif risk_score >= 3:
        label = "SLOW"
    else:
        label = "FAST"

    if not reasons:
        reasons = ["No immediate bottleneck signals detected"]

    confidence = min(0.95, 0.45 + risk_score * 0.08)
    return JobProfile(label=label, confidence=round(confidence, 2), reasons=reasons)
