from monitoring_tool.alerts import build_alerts, render_prometheus_metrics
from monitoring_tool.models import GPUMetrics, JobContext, NetworkMetrics, Snapshot, SystemMetrics
from monitoring_tool.profiler import JobProfile, classify_job


def test_build_alerts_for_fail_risk_contains_profiler_alert():
    snapshot = Snapshot(
        gpus=[
            GPUMetrics(
                index=0,
                utilization_gpu_pct=20,
                utilization_mem_pct=95,
                memory_used_mb=23000,
                memory_total_mb=24000,
                temperature_c=90,
                power_draw_w=250,
                clocks_throttle_reasons="Thermal",
            )
        ],
        net=[NetworkMetrics("eth0", 100, 100, 10, 10, 1, 1, 1, 1)],
        system=SystemMetrics(load_1m=20.0, load_5m=18.0, mem_total_kb=100000, mem_available_kb=5000),
    )
    profile = classify_job(snapshot)
    alerts = build_alerts(snapshot, profile)
    assert profile.label == "FAIL_RISK"
    assert any(a.category == "profiler" for a in alerts)


def test_render_prometheus_metrics_contains_core_series():
    snapshot = Snapshot(gpus=[], net=[NetworkMetrics("eth0", 1, 1, 1, 1, 0, 0, 0, 0)])
    profile = JobProfile(label="UNKNOWN", confidence=0.2, reasons=["No GPU telemetry available"])
    alerts = build_alerts(snapshot, profile)
    metrics = render_prometheus_metrics(snapshot, profile, alerts)
    assert "monitoring_profile_label" in metrics
    assert "monitoring_active_alerts" in metrics
    assert "monitoring_network_errors_total" in metrics


def test_render_prometheus_metrics_adds_job_labels():
    snapshot = Snapshot(
        job=JobContext(scheduler="SLURM", job_id="123"),
        gpus=[],
        net=[NetworkMetrics("eth0", 1, 1, 1, 1, 0, 0, 0, 0)],
    )
    profile = JobProfile(label="UNKNOWN", confidence=0.2, reasons=["No GPU telemetry available"])
    alerts = build_alerts(snapshot, profile)

    metrics = render_prometheus_metrics(snapshot, profile, alerts)

    assert 'monitoring_profile_label{scheduler="SLURM",job_id="123"}' in metrics
    assert 'monitoring_network_errors_total{scheduler="SLURM",job_id="123"}' in metrics
