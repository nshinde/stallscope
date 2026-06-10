import subprocess

from monitoring_tool.pfc import collect_pfc_metrics, parse_ethtool_pause_stats


def test_parse_mlx5_per_priority_pause_stats():
    content = open("tests/fixtures/ethtool_mlx5.txt").read()

    metrics = parse_ethtool_pause_stats("ens3f0np0", content)

    assert metrics is not None
    assert metrics.interface == "ens3f0np0"
    assert metrics.rx_pause is None
    assert metrics.tx_pause is None
    assert metrics.rx_prio_pause[3] == 17
    assert metrics.tx_prio_pause[4] == 23


def test_parse_generic_pause_stats():
    content = open("tests/fixtures/ethtool_generic.txt").read()

    metrics = parse_ethtool_pause_stats("eth0", content)

    assert metrics is not None
    assert metrics.rx_pause == 101
    assert metrics.tx_pause == 202
    assert metrics.rx_prio_pause == {}
    assert metrics.tx_prio_pause == {}


def test_parse_returns_none_without_pause_stats():
    assert parse_ethtool_pause_stats("eth0", "rx_packets: 100\n") is None


def test_collect_pfc_metrics_handles_missing_ethtool(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "check_output", raise_missing)

    metrics, warnings = collect_pfc_metrics(["eth0"])

    assert metrics == []
    assert warnings == ["ethtool unavailable; PFC pause metrics not collected"]


def test_collect_pfc_metrics_skips_failed_interface(monkeypatch):
    def raise_failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "check_output", raise_failed)

    metrics, warnings = collect_pfc_metrics(["eth0"])

    assert metrics == []
    assert "ethtool -S eth0 failed" in warnings[0]
