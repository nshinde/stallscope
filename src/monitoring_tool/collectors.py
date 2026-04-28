from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .models import GPUMetrics, NetworkMetrics, Snapshot, SystemMetrics


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _proc_path(*parts: str) -> Path:
    root = os.getenv("MONITORING_PROCFS_ROOT", "/proc")
    return Path(root).joinpath(*parts)


def collect_gpu_metrics() -> tuple[list[GPUMetrics], list[str]]:
    warnings: list[str] = []
    query = (
        "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw,clocks_throttle_reasons.active"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]

    try:
        output = _run(cmd)
    except (FileNotFoundError, subprocess.CalledProcessError):
        warnings.append("nvidia-smi unavailable; GPU metrics not collected")
        return [], warnings

    gpus: list[GPUMetrics] = []
    for line in output.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 8:
            continue
        gpus.append(
            GPUMetrics(
                index=int(fields[0]),
                utilization_gpu_pct=float(fields[1]),
                utilization_mem_pct=float(fields[2]),
                memory_used_mb=int(fields[3]),
                memory_total_mb=int(fields[4]),
                temperature_c=float(fields[5]),
                power_draw_w=float(fields[6]),
                clocks_throttle_reasons=fields[7],
            )
        )

    return gpus, warnings


def parse_proc_net_dev(content: str) -> list[NetworkMetrics]:
    metrics: list[NetworkMetrics] = []
    lines = content.splitlines()[2:]
    for line in lines:
        if ":" not in line:
            continue
        iface, data = line.split(":", 1)
        iface = iface.strip()
        parts = data.split()
        if len(parts) < 16:
            continue
        metrics.append(
            NetworkMetrics(
                interface=iface,
                rx_bytes=int(parts[0]),
                rx_packets=int(parts[1]),
                rx_errs=int(parts[2]),
                rx_drop=int(parts[3]),
                tx_bytes=int(parts[8]),
                tx_packets=int(parts[9]),
                tx_errs=int(parts[10]),
                tx_drop=int(parts[11]),
            )
        )
    return metrics


def collect_network_metrics() -> tuple[list[NetworkMetrics], list[str]]:
    warnings: list[str] = []
    path = _proc_path("net", "dev")
    if not path.exists():
        warnings.append(f"{path} unavailable; network metrics not collected")
        return [], warnings

    return parse_proc_net_dev(path.read_text()), warnings


def parse_meminfo(content: str) -> tuple[int, int]:
    mem_total = 0
    mem_available = 0
    for line in content.splitlines():
        if line.startswith("MemTotal:"):
            mem_total = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1])
    return mem_total, mem_available


def collect_system_metrics() -> tuple[SystemMetrics | None, list[str]]:
    warnings: list[str] = []
    try:
        load_1m, load_5m, _ = os.getloadavg()
    except OSError:
        warnings.append("load average unavailable")
        return None, warnings

    mem_total = 0
    mem_available = 0
    meminfo = _proc_path("meminfo")
    if meminfo.exists():
        mem_total, mem_available = parse_meminfo(meminfo.read_text())
    else:
        warnings.append(f"{meminfo} unavailable")

    return (
        SystemMetrics(
            load_1m=load_1m,
            load_5m=load_5m,
            mem_total_kb=mem_total,
            mem_available_kb=mem_available,
        ),
        warnings,
    )


def collect_snapshot() -> Snapshot:
    gpus, gpu_warnings = collect_gpu_metrics()
    net, net_warnings = collect_network_metrics()
    system, system_warnings = collect_system_metrics()
    return Snapshot(
        gpus=gpus,
        net=net,
        system=system,
        warnings=[*gpu_warnings, *net_warnings, *system_warnings],
    )
