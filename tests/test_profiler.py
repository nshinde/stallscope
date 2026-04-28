from monitoring_tool.models import GPUMetrics, NetworkMetrics, Snapshot
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
    assert any("Network" in reason for reason in profile.reasons)


def test_unknown_when_no_gpu():
    profile = classify_job(Snapshot(gpus=[], net=[]))
    assert profile.label == "UNKNOWN"
