from __future__ import annotations

from dataclasses import dataclass

from .models import PFCPauseMetrics, RDMAPortMetrics, Snapshot


FABRIC_CONGESTION_COUNTERS = ("np_cnp_sent", "rp_cnp_handled", "np_ecn_marked_roce_packets")
FABRIC_ERROR_COUNTERS = ("rnr_nak_retry_err", "out_of_sequence", "local_ack_timeout_err")
PFC_PAUSE_STORM_RATE = 1000.0


@dataclass
class JobProfile:
    label: str
    confidence: float
    reasons: list[str]
    bottleneck_hint: str = "unknown"


def _elapsed_seconds(snapshot: Snapshot, previous_snapshot: Snapshot) -> float:
    elapsed = (snapshot.timestamp - previous_snapshot.timestamp).total_seconds()
    return max(elapsed, 1.0)


def _rdma_key(metric: RDMAPortMetrics) -> tuple[str, str]:
    return metric.device, metric.port


def _pfc_key(metric: PFCPauseMetrics) -> str:
    return metric.interface


def _counter_delta(current: int | None, previous: int | None) -> int:
    if current is None or previous is None:
        return 0
    return max(0, current - previous)


def _total_pfc_pause(metric: PFCPauseMetrics) -> int:
    total = 0
    if metric.rx_pause is not None:
        total += metric.rx_pause
    if metric.tx_pause is not None:
        total += metric.tx_pause
    total += sum(metric.rx_prio_pause.values())
    total += sum(metric.tx_prio_pause.values())
    return total


def _bottleneck_hint(categories: set[str]) -> str:
    if not categories:
        return "unknown"
    if len(categories) > 1:
        return "mixed"
    return next(iter(categories))


def classify_job(snapshot: Snapshot, previous_snapshot: Snapshot | None = None) -> JobProfile:
    if not snapshot.gpus:
        return JobProfile(
            label="UNKNOWN",
            confidence=0.2,
            reasons=["No GPU telemetry available"],
        )

    reasons: list[str] = []
    categories: set[str] = set()
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
        categories.add("compute")
        reasons.append(f"Low average GPU utilization ({avg_gpu_util:.1f}%)")
    elif avg_gpu_util >= 80:
        reasons.append(f"High average GPU utilization ({avg_gpu_util:.1f}%)")

    if avg_mem_util > 90 or min_headroom < 8:
        risk_score += 2
        categories.add("memory")
        reasons.append("GPU memory pressure is high")

    if hot_gpus:
        risk_score += 2
        categories.add("compute")
        reasons.append(f"Thermal risk on GPUs {hot_gpus}")

    if throttled:
        risk_score += 2
        categories.add("compute")
        reasons.append(f"Clock throttling detected on GPUs {throttled}")

    total_net_err = sum(n.rx_errs + n.tx_errs + n.rx_drop + n.tx_drop for n in snapshot.net)
    if total_net_err > 0:
        risk_score += 2
        categories.add("network")
        reasons.append(f"Network errors/drops detected ({total_net_err})")

    if previous_snapshot is not None:
        elapsed = _elapsed_seconds(snapshot, previous_snapshot)
        previous_rdma = {_rdma_key(metric): metric for metric in previous_snapshot.rdma}
        cnp_ecn_delta = 0
        fabric_error_delta = 0
        for metric in snapshot.rdma:
            previous = previous_rdma.get(_rdma_key(metric))
            if previous is None:
                continue
            for name in FABRIC_CONGESTION_COUNTERS:
                cnp_ecn_delta += _counter_delta(metric.hw_counters.get(name), previous.hw_counters.get(name))
            for name in FABRIC_ERROR_COUNTERS:
                fabric_error_delta += _counter_delta(metric.hw_counters.get(name), previous.hw_counters.get(name))

        cnp_ecn_rate = cnp_ecn_delta / elapsed
        if cnp_ecn_rate > 0:
            risk_score += 3
            categories.add("network")
            reasons.append(f"Fabric congestion detected: CNP/ECN rate {cnp_ecn_rate:.2f}/s")

        if fabric_error_delta > 0:
            risk_score += 6
            categories.add("network")
            reasons.append(f"Fabric transport errors increased ({fabric_error_delta} RDMA retry/ordering/timeout events)")

        previous_pfc = {_pfc_key(metric): metric for metric in previous_snapshot.pfc}
        pfc_delta = 0
        for metric in snapshot.pfc:
            previous = previous_pfc.get(_pfc_key(metric))
            if previous is None:
                continue
            pfc_delta += _counter_delta(_total_pfc_pause(metric), _total_pfc_pause(previous))

        pfc_rate = pfc_delta / elapsed
        if pfc_rate >= PFC_PAUSE_STORM_RATE:
            risk_score += 3
            categories.add("network")
            reasons.append(f"Fabric congestion detected: PFC pause storm rate {pfc_rate:.2f}/s")

    if snapshot.system is not None:
        if snapshot.system.load_1m > 16:
            risk_score += 1
            categories.add("compute")
            reasons.append(f"High CPU load average (1m={snapshot.system.load_1m:.2f})")
        if snapshot.system.mem_available_pct < 10:
            risk_score += 2
            categories.add("memory")
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
    return JobProfile(
        label=label,
        confidence=round(confidence, 2),
        reasons=reasons,
        bottleneck_hint=_bottleneck_hint(categories),
    )
