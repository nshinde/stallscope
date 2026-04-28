import json
import os
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.integration


def _gpu_count() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return len([line for line in out.splitlines() if line.strip()])


@pytest.mark.skipif(os.getenv("RUN_GPU_INTEGRATION") != "1", reason="set RUN_GPU_INTEGRATION=1")
def test_monitoring_cli_integration_multi_gpu():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not available")

    gpus = _gpu_count()
    if gpus < 2:
        pytest.skip(f"need >=2 GPUs, found {gpus}")

    cmd = [sys.executable, "-m", "monitoring_tool.cli", "--json"]
    if shutil.which("all_reduce_perf") is not None:
        cmd.append("--nccl-test")

    out = subprocess.check_output(cmd, text=True, env={**os.environ, "PYTHONPATH": "src"})
    payload = json.loads(out)

    assert "snapshot" in payload
    assert "profile" in payload
    assert payload["profile"]["label"] in {"FAST", "SLOW", "FAIL_RISK", "UNKNOWN"}
    assert len(payload["snapshot"]["gpus"]) >= 2

    if "nccl" in payload:
        assert payload["nccl"]["status"] in {"OK", "SKIPPED", "ERROR"}
