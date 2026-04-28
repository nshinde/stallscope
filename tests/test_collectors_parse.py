from monitoring_tool.collectors import parse_meminfo, parse_proc_net_dev


def test_parse_proc_net_dev_parses_loopback_and_eth():
    sample = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 100 2 0 0 0 0 0 0 100 2 0 0 0 0 0 0
  eth0: 200 3 1 2 0 0 0 0 300 4 5 6 0 0 0 0
"""
    parsed = parse_proc_net_dev(sample)
    assert len(parsed) == 2
    assert parsed[1].interface == "eth0"
    assert parsed[1].rx_errs == 1
    assert parsed[1].tx_drop == 6


def test_parse_meminfo_extracts_total_and_available():
    sample = """MemTotal:       16000000 kB
MemFree:         1000000 kB
MemAvailable:    4000000 kB
"""
    total, available = parse_meminfo(sample)
    assert total == 16000000
    assert available == 4000000
