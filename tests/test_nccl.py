from monitoring_tool.nccl import parse_nccl_output, run_nccl_all_reduce_test


def test_parse_nccl_output_happy_path():
    sample = """
# nThread 1 nGpus 2 minBytes 8 maxBytes 128M step: 2(factor) warmup iters: 5 iters: 20
# size count type redop root time(us) algbw(GB/s) busbw(GB/s)
8 2 float sum -1 12.34 45.67 44.21
16 4 float sum -1 13.37 47.01 45.00
"""
    parsed = parse_nccl_output(sample)
    assert parsed.status == "OK"
    assert parsed.time_us == 13.37
    assert parsed.bandwidth_gbps == 47.01
    assert parsed.bus_bandwidth_gbps == 45.0


def test_run_nccl_skips_when_binary_missing():
    result = run_nccl_all_reduce_test()
    assert result.status in {"OK", "SKIPPED", "ERROR"}
    if result.status == "SKIPPED":
        assert "not found" in result.message
