from datetime import datetime, timedelta, timezone

from monitoring_tool.models import GPUMetrics, NetworkMetrics, PFCPauseMetrics, RDMAPortMetrics, Snapshot
from monitoring_tool.profiler import classify_job


def gpu(**overrides):
    base = dict(
        index=0,
        utilization_gpu_pct=90.0,
        utilization_mem_pct=65.0,
        memory_used_mb=13000,
        memory_total_mb=24000,
        temperature_c=70.0,
        power_draw_w=210.0,
        clocks_throttle_reasons="Not Active",
    )
    base.update(overrides)
    return GPUMetrics(**base)


def test_fast_profile_when_healthy():
    snapshot = Snapshot(
        gpus=[gpu(index=0), gpu(index=1, utilization_gpu_pct=85.0)],
        net=[
            NetworkMetrics("eth0", 100, 50, 10, 5, 0, 0, 0, 0),
        ],
    )

    profile = classify_job(snapshot)
    assert profile.label == "FAST"
    assert profile.bottleneck_hint == "unknown"


def test_fail_risk_profile_when_multiple_signals():
    snapshot = Snapshot(
        gpus=[
            gpu(utilization_gpu_pct=20.0, temperature_c=89.0, clocks_throttle_reasons="Thermal", memory_used_mb=23800),
        ],
        net=[
            NetworkMetrics("eth0", 100, 50, 10, 5, 2, 1, 3, 2),
        ],
    )

    profile = classify_job(snapshot)
    assert profile.label == "FAIL_RISK"
    assert profile.bottleneck_hint == "mixed"
    assert any("Network" in reason for reason in profile.reasons)


def test_unknown_when_no_gpu():
    profile = classify_job(Snapshot(gpus=[], net=[]))
    assert profile.label == "UNKNOWN"


def test_fabric_congestion_from_rdma_cnp_ecn_delta():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    previous = Snapshot(
        timestamp=start,
        gpus=[gpu()],
        rdma=[
            RDMAPortMetrics(
                device="mlx5_0",
                port="1",
                hw_counters={"np_cnp_sent": 10, "rp_cnp_handled": 20, "np_ecn_marked_roce_packets": 30},
            )
        ],
    )
    current = Snapshot(
        timestamp=start + timedelta(seconds=10),
        gpus=[gpu()],
        rdma=[
            RDMAPortMetrics(
                device="mlx5_0",
                port="1",
                hw_counters={"np_cnp_sent": 15, "rp_cnp_handled": 25, "np_ecn_marked_roce_packets": 40},
            )
        ],
    )

    profile = classify_job(current, previous)

    assert profile.label == "SLOW"
    assert profile.bottleneck_hint == "network"
    assert any("CNP/ECN rate 2.00/s" in reason for reason in profile.reasons)


def test_fabric_errors_from_rdma_error_delta_drive_fail_risk():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    previous = Snapshot(
        timestamp=start,
        gpus=[gpu()],
        rdma=[
            RDMAPortMetrics(
                device="mlx5_0",
                port="1",
                hw_counters={"rnr_nak_retry_err": 0, "out_of_sequence": 0, "local_ack_timeout_err": 0},
            )
        ],
    )
    current = Snapshot(
        timestamp=start + timedelta(seconds=10),
        gpus=[gpu()],
        rdma=[
            RDMAPortMetrics(
                device="mlx5_0",
                port="1",
                hw_counters={"rnr_nak_retry_err": 1, "out_of_sequence": 1, "local_ack_timeout_err": 1},
            )
        ],
    )

    profile = classify_job(current, previous)

    assert profile.label == "FAIL_RISK"
    assert profile.bottleneck_hint == "network"
    assert any("RDMA retry/ordering/timeout" in reason for reason in profile.reasons)


def test_pfc_pause_storm_from_delta():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    previous = Snapshot(
        timestamp=start,
        gpus=[gpu()],
        pfc=[PFCPauseMetrics(interface="ens3f0np0", rx_prio_pause={3: 100}, tx_prio_pause={3: 100})],
    )
    current = Snapshot(
        timestamp=start + timedelta(seconds=1),
        gpus=[gpu()],
        pfc=[PFCPauseMetrics(interface="ens3f0np0", rx_prio_pause={3: 700}, tx_prio_pause={3: 600})],
    )

    profile = classify_job(current, previous)

    assert profile.label == "SLOW"
    assert profile.bottleneck_hint == "network"
    assert any("PFC pause storm rate 1100.00/s" in reason for reason in profile.reasons)
