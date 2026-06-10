from __future__ import annotations

import os
from pathlib import Path

from .models import RDMAPortMetrics


PORT_COUNTERS = (
    "port_rcv_data",
    "port_xmit_data",
    "port_rcv_errors",
    "symbol_error",
    "link_downed",
    "port_rcv_remote_physical_errors",
    "port_xmit_discards",
)

HW_COUNTERS = (
    "np_cnp_sent",
    "rp_cnp_handled",
    "np_ecn_marked_roce_packets",
    "rnr_nak_retry_err",
    "out_of_sequence",
    "packet_seq_err",
    "local_ack_timeout_err",
    "duplicate_request",
)


def _sysfs_path(*parts: str) -> Path:
    root = os.getenv("MONITORING_SYSFS_ROOT", "/sys")
    return Path(root).joinpath(*parts)


def parse_rdma_counter(content: str) -> int:
    return int(content.strip())


def _read_counter(path: Path, warnings: list[str]) -> int | None:
    if not path.exists():
        warnings.append(f"{path} missing; RDMA counter skipped")
        return None
    try:
        return parse_rdma_counter(path.read_text())
    except (OSError, ValueError) as exc:
        warnings.append(f"{path} unreadable; RDMA counter skipped: {exc}")
        return None


def collect_rdma_metrics() -> tuple[list[RDMAPortMetrics], list[str]]:
    warnings: list[str] = []
    root = _sysfs_path("class", "infiniband")
    if not root.exists():
        warnings.append(f"{root} unavailable; RDMA metrics not collected")
        return [], warnings

    metrics: list[RDMAPortMetrics] = []
    devices = sorted(path for path in root.iterdir() if path.is_dir())
    if not devices:
        warnings.append(f"{root} has no RDMA devices; RDMA metrics not collected")
        return [], warnings

    for device_path in devices:
        ports_path = device_path / "ports"
        if not ports_path.exists():
            warnings.append(f"{ports_path} unavailable; RDMA device skipped")
            continue

        for port_path in sorted(path for path in ports_path.iterdir() if path.is_dir()):
            counters: dict[str, int] = {}
            hw_counters: dict[str, int] = {}
            counters_path = port_path / "counters"
            hw_counters_path = port_path / "hw_counters"

            for name in PORT_COUNTERS:
                value = _read_counter(counters_path / name, warnings)
                if value is not None:
                    counters[name] = value

            for name in HW_COUNTERS:
                value = _read_counter(hw_counters_path / name, warnings)
                if value is not None:
                    hw_counters[name] = value

            metrics.append(
                RDMAPortMetrics(
                    device=device_path.name,
                    port=port_path.name,
                    counters=counters,
                    hw_counters=hw_counters,
                )
            )

    return metrics, warnings
