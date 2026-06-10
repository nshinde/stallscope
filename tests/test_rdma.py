from monitoring_tool.rdma import collect_rdma_metrics, parse_rdma_counter


def test_parse_rdma_counter_strips_newline():
    assert parse_rdma_counter("42\n") == 42


def test_collect_rdma_metrics_from_fixture(monkeypatch):
    monkeypatch.setenv("MONITORING_SYSFS_ROOT", "tests/fixtures/sysfs_rdma")

    metrics, warnings = collect_rdma_metrics()

    assert warnings == []
    assert len(metrics) == 1
    port = metrics[0]
    assert port.device == "mlx5_0"
    assert port.port == "1"
    assert port.counters["port_rcv_data"] == 1024
    assert port.counters["port_xmit_discards"] == 2
    assert port.hw_counters["np_cnp_sent"] == 7
    assert port.hw_counters["local_ack_timeout_err"] == 1


def test_collect_rdma_metrics_warns_for_missing_counters(monkeypatch):
    monkeypatch.setenv("MONITORING_SYSFS_ROOT", "tests/fixtures/sysfs_rdma_missing")

    metrics, warnings = collect_rdma_metrics()

    assert len(metrics) == 1
    assert metrics[0].counters["port_rcv_data"] == 11
    assert "port_xmit_data" not in metrics[0].counters
    assert any("port_xmit_data missing" in warning for warning in warnings)
    assert any("np_cnp_sent missing" in warning for warning in warnings)


def test_collect_rdma_metrics_handles_no_rdma_device(monkeypatch):
    monkeypatch.setenv("MONITORING_SYSFS_ROOT", "tests/fixtures/sysfs_no_rdma")

    metrics, warnings = collect_rdma_metrics()

    assert metrics == []
    assert warnings
    assert "RDMA metrics not collected" in warnings[0]
