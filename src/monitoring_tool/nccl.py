from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class NcclResult:
    status: str
    binary: str | None
    command: list[str]
    bandwidth_gbps: float | None
    bus_bandwidth_gbps: float | None
    time_us: float | None
    raw_tail: list[str]
    message: str


def parse_nccl_output(output: str) -> NcclResult:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    data_lines = [line for line in lines if re.match(r"^\s*\d+\s+\d+", line)]

    if not data_lines:
        return NcclResult(
            status="ERROR",
            binary=None,
            command=[],
            bandwidth_gbps=None,
            bus_bandwidth_gbps=None,
            time_us=None,
            raw_tail=lines[-15:],
            message="Could not parse NCCL output",
        )

    # Typical nccl-tests data rows include:
    # size count type redop root time(us) algbw busbw ...
    last = data_lines[-1]
    parts = last.split()

    # time/us and bandwidth columns are generally at fixed positions.
    # Keep this parser resilient by checking minimum expected width.
    if len(parts) < 8:
        return NcclResult(
            status="ERROR",
            binary=None,
            command=[],
            bandwidth_gbps=None,
            bus_bandwidth_gbps=None,
            time_us=None,
            raw_tail=lines[-15:],
            message="NCCL output row had unexpected format",
        )

    # Known stable positions across nccl-tests versions.
    # 0:size 1:count 2:type 3:redop 4:root 5:time(us) 6:algbw 7:busbw
    time_us = float(parts[5])
    algbw = float(parts[6])
    busbw = float(parts[7])

    return NcclResult(
        status="OK",
        binary=None,
        command=[],
        bandwidth_gbps=algbw,
        bus_bandwidth_gbps=busbw,
        time_us=time_us,
        raw_tail=lines[-15:],
        message="Parsed NCCL performance output",
    )


def run_nccl_all_reduce_test(
    min_bytes: str = "8",
    max_bytes: str = "128M",
    step_factor: str = "2",
    gpus: int = 1,
    iters: int = 20,
    warmup_iters: int = 5,
) -> NcclResult:
    binary = shutil.which("all_reduce_perf")
    if binary is None:
        return NcclResult(
            status="SKIPPED",
            binary=None,
            command=[],
            bandwidth_gbps=None,
            bus_bandwidth_gbps=None,
            time_us=None,
            raw_tail=[],
            message="all_reduce_perf not found in PATH",
        )

    command = [
        binary,
        "-b",
        min_bytes,
        "-e",
        max_bytes,
        "-f",
        step_factor,
        "-g",
        str(gpus),
        "-n",
        str(iters),
        "-w",
        str(warmup_iters),
    ]

    try:
        proc = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        return NcclResult(
            status="ERROR",
            binary=binary,
            command=command,
            bandwidth_gbps=None,
            bus_bandwidth_gbps=None,
            time_us=None,
            raw_tail=(exc.stdout or "").splitlines()[-15:] + (exc.stderr or "").splitlines()[-15:],
            message=f"NCCL test failed with code {exc.returncode}",
        )

    parsed = parse_nccl_output(proc.stdout)
    parsed.binary = binary
    parsed.command = command
    return parsed
