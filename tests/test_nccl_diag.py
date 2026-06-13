"""Tests for the NCCL configuration diagnostic module."""
from monitoring_tool.nccl_diag import (
    NcclDiagReport,
    NcclEnvSnapshot,
    NcclHardwareContext,
    SEVERITY_ERROR,
    SEVERITY_WARN,
    SEVERITY_INFO,
    analyze_nccl_config,
    collect_nccl_env,
    format_nccl_report,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def env(**kwargs: str) -> NcclEnvSnapshot:
    return NcclEnvSnapshot(vars=dict(kwargs))


def hw(**kwargs) -> NcclHardwareContext:
    defaults = dict(
        gpu_count=8,
        ib_devices=["mlx5_0", "mlx5_1"],
        net_interfaces=["eth0", "eth1"],
        gpudirect_rdma_available=True,
        nvlink_available=True,
        nccl_version="2.19.3",
    )
    defaults.update(kwargs)
    return NcclHardwareContext(**defaults)


def findings_for(e: NcclEnvSnapshot, h: NcclHardwareContext) -> list:
    return analyze_nccl_config(env=e, hw=h).findings


def severities(report: NcclDiagReport) -> list[str]:
    return [f.severity for f in report.findings]


def env_vars(report: NcclDiagReport) -> list[str]:
    return [f.env_var for f in report.findings]


# ── collect_nccl_env ──────────────────────────────────────────────────────────

def test_collect_nccl_env_filters_nccl_prefix(monkeypatch):
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    monkeypatch.setenv("UNRELATED_VAR", "ignored")
    snapshot = collect_nccl_env()
    assert "NCCL_DEBUG" in snapshot.vars
    assert "NCCL_IB_DISABLE" in snapshot.vars
    assert "UNRELATED_VAR" not in snapshot.vars


def test_collect_nccl_env_includes_tracked_non_nccl(monkeypatch):
    monkeypatch.setenv("FI_EFA_USE_DEVICE_RDMA", "1")
    monkeypatch.setenv("FI_PROVIDER", "efa")
    snapshot = collect_nccl_env()
    assert "FI_EFA_USE_DEVICE_RDMA" in snapshot.vars
    assert "FI_PROVIDER" in snapshot.vars


def test_collect_nccl_env_accepts_explicit_dict():
    src = {"NCCL_DEBUG": "INFO", "PATH": "/usr/bin"}
    snapshot = collect_nccl_env(environ=src)
    assert snapshot.vars == {"NCCL_DEBUG": "INFO"}


# ── NcclEnvSnapshot helpers ───────────────────────────────────────────────────

def test_env_snapshot_is_set_and_get():
    s = NcclEnvSnapshot(vars={"NCCL_DEBUG": "WARN"})
    assert s.is_set("NCCL_DEBUG")
    assert not s.is_set("NCCL_IB_DISABLE")
    assert s.get("NCCL_DEBUG") == "WARN"
    assert s.get("NCCL_MISSING", "default") == "default"


def test_env_snapshot_is_enabled_disabled():
    s = NcclEnvSnapshot(vars={"NCCL_IB_DISABLE": "1", "NCCL_SHM_DISABLE": "0"})
    assert s.is_enabled("NCCL_IB_DISABLE")
    assert s.is_disabled("NCCL_SHM_DISABLE")
    assert not s.is_enabled("NCCL_MISSING")


# ── transport: IB disabled with devices present ───────────────────────────────

def test_error_when_ib_disabled_but_ib_devices_present():
    report = analyze_nccl_config(
        env=env(NCCL_IB_DISABLE="1"),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    errors = [f for f in report.findings if f.env_var == "NCCL_IB_DISABLE" and f.severity == SEVERITY_ERROR]
    assert errors, "Expected ERROR for IB_DISABLE=1 with devices present"
    assert "TCP" in errors[0].description or "slower" in errors[0].description


def test_no_error_when_ib_disabled_and_no_devices():
    report = analyze_nccl_config(
        env=env(NCCL_IB_DISABLE="1"),
        hw=hw(gpu_count=0, ib_devices=[]),
    )
    ib_errors = [f for f in report.findings if f.env_var == "NCCL_IB_DISABLE" and f.severity == SEVERITY_ERROR]
    assert not ib_errors


# ── transport: no IB devices but GPUs present ─────────────────────────────────

def test_warn_when_gpus_present_but_no_rdma_devices():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(gpu_count=8, ib_devices=[]),
    )
    warns = [f for f in report.findings if f.env_var == "NCCL_IB_DISABLE" and f.severity == SEVERITY_WARN]
    assert warns


def test_no_warn_when_ib_explicitly_disabled_and_no_devices():
    report = analyze_nccl_config(
        env=env(NCCL_IB_DISABLE="1"),
        hw=hw(gpu_count=8, ib_devices=[]),
    )
    # NCCL_IB_DISABLE=1 suppresses the no-RDMA warning
    no_rdma_warns = [
        f for f in report.findings
        if f.env_var == "NCCL_IB_DISABLE" and f.severity == SEVERITY_WARN
    ]
    assert not no_rdma_warns


# ── transport: HCA selection ──────────────────────────────────────────────────

def test_warn_when_multiple_hcas_and_no_hca_pin():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=["mlx5_0", "mlx5_1"]),
    )
    hca_warns = [f for f in report.findings if f.env_var == "NCCL_IB_HCA"]
    assert hca_warns
    assert hca_warns[0].severity == SEVERITY_WARN
    assert "mlx5_0" in hca_warns[0].suggestion


def test_no_hca_warn_when_single_device():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    hca_warns = [f for f in report.findings if f.env_var == "NCCL_IB_HCA"]
    assert not hca_warns


def test_no_hca_warn_when_hca_pinned():
    report = analyze_nccl_config(
        env=env(NCCL_IB_HCA="mlx5_0"),
        hw=hw(ib_devices=["mlx5_0", "mlx5_1"]),
    )
    hca_warns = [f for f in report.findings if f.env_var == "NCCL_IB_HCA"]
    assert not hca_warns


# ── transport: GID index ──────────────────────────────────────────────────────

def test_warn_when_gid_index_unset_with_ib_devices():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    gid_findings = [f for f in report.findings if f.env_var == "NCCL_IB_GID_INDEX"]
    assert gid_findings
    assert gid_findings[0].severity == SEVERITY_WARN
    assert "RoCEv2" in gid_findings[0].description


def test_no_gid_warn_when_no_ib_devices():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=[], gpu_count=0),
    )
    gid_findings = [f for f in report.findings if f.env_var == "NCCL_IB_GID_INDEX"]
    assert not gid_findings


def test_no_gid_warn_when_gid_index_set():
    report = analyze_nccl_config(
        env=env(NCCL_IB_GID_INDEX="3"),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    gid_findings = [f for f in report.findings if f.env_var == "NCCL_IB_GID_INDEX"]
    assert not gid_findings


# ── transport: GPUDirect RDMA ─────────────────────────────────────────────────

def test_warn_when_gdr_module_loaded_but_not_configured():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=["mlx5_0"], gpudirect_rdma_available=True),
    )
    gdr_findings = [f for f in report.findings if f.env_var == "NCCL_NET_GDR_LEVEL"]
    assert gdr_findings
    assert gdr_findings[0].severity == SEVERITY_WARN
    assert "zero-copy" in gdr_findings[0].suggestion or "GDR" in gdr_findings[0].suggestion


def test_info_when_ib_present_but_gdr_module_not_loaded():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=["mlx5_0"], gpudirect_rdma_available=False),
    )
    gdr_findings = [f for f in report.findings if f.env_var == "NCCL_NET_GDR_LEVEL"]
    assert gdr_findings
    assert gdr_findings[0].severity == SEVERITY_INFO


def test_no_gdr_finding_when_gdr_level_set():
    report = analyze_nccl_config(
        env=env(NCCL_NET_GDR_LEVEL="5", NCCL_IB_CUDA_SUPPORT="1"),
        hw=hw(ib_devices=["mlx5_0"], gpudirect_rdma_available=True),
    )
    gdr_findings = [f for f in report.findings if f.env_var == "NCCL_NET_GDR_LEVEL"]
    assert not gdr_findings


# ── transport: socket interface ───────────────────────────────────────────────

def test_warn_when_multiple_interfaces_and_no_socket_ifname():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(net_interfaces=["eth0", "eth1", "ens3f0"]),
    )
    sock_findings = [f for f in report.findings if f.env_var == "NCCL_SOCKET_IFNAME"]
    assert sock_findings


def test_no_socket_warn_when_single_interface():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(net_interfaces=["eth0"]),
    )
    sock_findings = [f for f in report.findings if f.env_var == "NCCL_SOCKET_IFNAME"]
    assert not sock_findings


def test_no_socket_warn_when_ifname_set():
    report = analyze_nccl_config(
        env=env(NCCL_SOCKET_IFNAME="eth0"),
        hw=hw(net_interfaces=["eth0", "eth1"]),
    )
    sock_findings = [f for f in report.findings if f.env_var == "NCCL_SOCKET_IFNAME"]
    assert not sock_findings


def test_docker_interfaces_excluded_from_multiple_iface_check():
    report = analyze_nccl_config(
        env=env(),
        # Only one real iface; docker0 and br-xxx should be excluded
        hw=hw(net_interfaces=["eth0", "docker0", "br-abc123"]),
    )
    sock_findings = [f for f in report.findings if f.env_var == "NCCL_SOCKET_IFNAME"]
    assert not sock_findings


# ── performance: buffer size ──────────────────────────────────────────────────

def test_warn_when_buffsize_too_small():
    report = analyze_nccl_config(
        env=env(NCCL_BUFFSIZE="1048576"),  # 1 MB < 2 MB threshold
        hw=hw(),
    )
    buf_findings = [f for f in report.findings if f.env_var == "NCCL_BUFFSIZE"]
    assert buf_findings
    assert buf_findings[0].severity == SEVERITY_WARN


def test_error_when_buffsize_not_integer():
    report = analyze_nccl_config(
        env=env(NCCL_BUFFSIZE="8M"),
        hw=hw(),
    )
    buf_findings = [f for f in report.findings if f.env_var == "NCCL_BUFFSIZE"]
    assert buf_findings
    assert buf_findings[0].severity == SEVERITY_ERROR


def test_no_buffsize_finding_when_unset():
    report = analyze_nccl_config(env=env(), hw=hw())
    buf_findings = [f for f in report.findings if f.env_var == "NCCL_BUFFSIZE"]
    assert not buf_findings


def test_no_buffsize_finding_when_adequate():
    report = analyze_nccl_config(
        env=env(NCCL_BUFFSIZE="8388608"),  # 8 MB
        hw=hw(),
    )
    buf_findings = [f for f in report.findings if f.env_var == "NCCL_BUFFSIZE"]
    assert not buf_findings


# ── performance: channel count ────────────────────────────────────────────────

def test_info_when_multi_nic_and_min_channels_unset():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=["mlx5_0", "mlx5_1"]),
    )
    ch_findings = [f for f in report.findings if f.env_var == "NCCL_MIN_NCHANNELS"]
    assert ch_findings
    assert ch_findings[0].severity == SEVERITY_INFO


def test_no_channel_info_when_single_ib_device():
    report = analyze_nccl_config(
        env=env(),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    ch_findings = [f for f in report.findings if f.env_var == "NCCL_MIN_NCHANNELS"]
    assert not ch_findings


# ── performance: cross-NIC ────────────────────────────────────────────────────

def test_warn_when_cross_nic_disabled_with_multi_nic():
    report = analyze_nccl_config(
        env=env(NCCL_CROSS_NIC="0"),
        hw=hw(ib_devices=["mlx5_0", "mlx5_1"]),
    )
    cross_findings = [f for f in report.findings if f.env_var == "NCCL_CROSS_NIC"]
    assert cross_findings
    assert cross_findings[0].severity == SEVERITY_WARN


def test_no_cross_nic_warn_when_single_nic():
    report = analyze_nccl_config(
        env=env(NCCL_CROSS_NIC="0"),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    cross_findings = [f for f in report.findings if f.env_var == "NCCL_CROSS_NIC"]
    assert not cross_findings


# ── congestion: traffic class and service level ───────────────────────────────

def test_info_when_ib_tc_unset_with_ib_devices():
    report = analyze_nccl_config(env=env(), hw=hw(ib_devices=["mlx5_0"]))
    tc_findings = [f for f in report.findings if f.env_var == "NCCL_IB_TC"]
    assert tc_findings
    assert tc_findings[0].severity == SEVERITY_INFO


def test_no_tc_finding_when_no_ib_devices():
    report = analyze_nccl_config(env=env(), hw=hw(ib_devices=[], gpu_count=0))
    tc_findings = [f for f in report.findings if f.env_var == "NCCL_IB_TC"]
    assert not tc_findings


def test_info_when_ib_sl_unset_with_ib_devices():
    report = analyze_nccl_config(env=env(), hw=hw(ib_devices=["mlx5_0"]))
    sl_findings = [f for f in report.findings if f.env_var == "NCCL_IB_SL"]
    assert sl_findings
    assert sl_findings[0].severity == SEVERITY_INFO


# ── debug: NCCL_DEBUG level ───────────────────────────────────────────────────

def test_info_when_nccl_debug_unset():
    report = analyze_nccl_config(env=env(), hw=hw())
    debug_findings = [f for f in report.findings if f.env_var == "NCCL_DEBUG"]
    assert debug_findings
    assert debug_findings[0].severity == SEVERITY_INFO


def test_warn_when_nccl_debug_is_info():
    report = analyze_nccl_config(env=env(NCCL_DEBUG="INFO"), hw=hw())
    debug_findings = [f for f in report.findings if f.env_var == "NCCL_DEBUG"]
    assert debug_findings
    assert debug_findings[0].severity == SEVERITY_WARN


def test_no_debug_finding_when_debug_warn():
    report = analyze_nccl_config(env=env(NCCL_DEBUG="WARN"), hw=hw())
    debug_findings = [f for f in report.findings if f.env_var == "NCCL_DEBUG"]
    assert not debug_findings


# ── transport: P2P and SHM ───────────────────────────────────────────────────

def test_error_when_p2p_disabled_with_nvlink():
    report = analyze_nccl_config(
        env=env(NCCL_P2P_DISABLE="1"),
        hw=hw(nvlink_available=True),
    )
    p2p_findings = [f for f in report.findings if f.env_var == "NCCL_P2P_DISABLE"]
    assert p2p_findings
    assert p2p_findings[0].severity == SEVERITY_ERROR
    assert "NVLink" in p2p_findings[0].description


def test_no_p2p_error_without_nvlink():
    report = analyze_nccl_config(
        env=env(NCCL_P2P_DISABLE="1"),
        hw=hw(nvlink_available=False),
    )
    p2p_errors = [
        f for f in report.findings
        if f.env_var == "NCCL_P2P_DISABLE" and f.severity == SEVERITY_ERROR
    ]
    assert not p2p_errors


def test_warn_when_shm_disabled_with_multiple_gpus():
    report = analyze_nccl_config(
        env=env(NCCL_SHM_DISABLE="1"),
        hw=hw(gpu_count=8),
    )
    shm_findings = [f for f in report.findings if f.env_var == "NCCL_SHM_DISABLE"]
    assert shm_findings
    assert shm_findings[0].severity == SEVERITY_WARN


def test_no_shm_warn_when_single_gpu():
    report = analyze_nccl_config(
        env=env(NCCL_SHM_DISABLE="1"),
        hw=hw(gpu_count=1),
    )
    shm_findings = [f for f in report.findings if f.env_var == "NCCL_SHM_DISABLE"]
    assert not shm_findings


# ── transport: IB timeout ─────────────────────────────────────────────────────

def test_warn_when_ib_timeout_too_low():
    report = analyze_nccl_config(
        env=env(NCCL_IB_TIMEOUT="10"),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    to_findings = [f for f in report.findings if f.env_var == "NCCL_IB_TIMEOUT"]
    assert to_findings
    assert to_findings[0].severity == SEVERITY_WARN
    assert "ms" in to_findings[0].description


def test_no_timeout_warn_when_safe_value():
    report = analyze_nccl_config(
        env=env(NCCL_IB_TIMEOUT="22"),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    to_findings = [f for f in report.findings if f.env_var == "NCCL_IB_TIMEOUT"]
    assert not to_findings


# ── protocol: NCCL_ALGO ───────────────────────────────────────────────────────

def test_error_when_nccl_algo_unknown():
    report = analyze_nccl_config(
        env=env(NCCL_ALGO="SuperRing"),
        hw=hw(),
    )
    algo_findings = [f for f in report.findings if f.env_var == "NCCL_ALGO"]
    assert algo_findings
    assert algo_findings[0].severity == SEVERITY_ERROR


def test_info_when_tree_forced_with_nvlink():
    report = analyze_nccl_config(
        env=env(NCCL_ALGO="Tree"),
        hw=hw(nvlink_available=True),
    )
    algo_findings = [f for f in report.findings if f.env_var == "NCCL_ALGO"]
    assert algo_findings
    assert algo_findings[0].severity == SEVERITY_INFO


def test_no_algo_finding_for_valid_algo():
    report = analyze_nccl_config(
        env=env(NCCL_ALGO="Ring"),
        hw=hw(nvlink_available=False),
    )
    algo_findings = [f for f in report.findings if f.env_var == "NCCL_ALGO"]
    assert not algo_findings


# ── report aggregation ────────────────────────────────────────────────────────

def test_overall_health_degraded_on_error():
    report = analyze_nccl_config(
        env=env(NCCL_IB_DISABLE="1"),
        hw=hw(ib_devices=["mlx5_0"]),
    )
    assert report.overall_health == "DEGRADED"
    assert report.error_count >= 1


def test_overall_health_optimal_when_all_configured():
    # Provide a maximally-configured env to eliminate all findings
    report = analyze_nccl_config(
        env=env(
            NCCL_IB_DISABLE="0",
            NCCL_IB_HCA="mlx5_0",
            NCCL_IB_GID_INDEX="3",
            NCCL_NET_GDR_LEVEL="5",
            NCCL_IB_CUDA_SUPPORT="1",
            NCCL_SOCKET_IFNAME="eth0",
            NCCL_MIN_NCHANNELS="4",
            NCCL_IB_TC="106",
            NCCL_IB_SL="0",
            NCCL_DEBUG="WARN",
        ),
        hw=hw(
            gpu_count=8,
            ib_devices=["mlx5_0"],   # single device → no HCA/cross-NIC/multi-port warnings
            net_interfaces=["eth0"],
            gpudirect_rdma_available=True,
            nvlink_available=False,
        ),
    )
    # With a single IB device and single interface, most multi-device warnings go away.
    # Remaining findings should be INFO at worst.
    errors = [f for f in report.findings if f.severity == SEVERITY_ERROR]
    warnings = [f for f in report.findings if f.severity == SEVERITY_WARN]
    assert not errors, f"Unexpected errors: {[f.description for f in errors]}"
    assert not warnings, f"Unexpected warnings: {[f.description for f in warnings]}"


# ── format_nccl_report ────────────────────────────────────────────────────────

def test_format_report_contains_health_and_counts():
    report = analyze_nccl_config(env=env(NCCL_DEBUG="INFO"), hw=hw())
    text = format_nccl_report(report)
    assert "Overall health" in text
    assert "NCCL_DEBUG" in text
    assert "Fix" in text


def test_format_report_no_findings_message():
    # Build an empty report manually
    report = NcclDiagReport(
        findings=[],
        env=NcclEnvSnapshot(),
        hw=NcclHardwareContext(),
    )
    text = format_nccl_report(report)
    assert "No issues found" in text
    assert "OPTIMAL" in text


def test_format_report_verbose_shows_env_vars():
    report = analyze_nccl_config(
        env=env(NCCL_DEBUG="WARN"),
        hw=hw(
            ib_devices=["mlx5_0"],
            net_interfaces=["eth0"],
        ),
    )
    text = format_nccl_report(report, verbose=True)
    assert "Captured NCCL environment" in text
    assert "NCCL_DEBUG=WARN" in text
