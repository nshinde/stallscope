from __future__ import annotations

import re
import subprocess

from .models import PFCPauseMetrics


_PRIO_PAUSE_RE = re.compile(r"^(?P<direction>rx|tx)_prio(?P<priority>\d+)_pause$")


def parse_ethtool_pause_stats(interface: str, content: str) -> PFCPauseMetrics | None:
    rx_pause: int | None = None
    tx_pause: int | None = None
    rx_prio_pause: dict[int, int] = {}
    tx_prio_pause: dict[int, int] = {}

    for line in content.splitlines():
        if ":" not in line:
            continue
        name, raw_value = line.split(":", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue

        match = _PRIO_PAUSE_RE.match(name)
        if match:
            priority = int(match.group("priority"))
            if match.group("direction") == "rx":
                rx_prio_pause[priority] = value
            else:
                tx_prio_pause[priority] = value
        elif name == "rx_pause":
            rx_pause = value
        elif name == "tx_pause":
            tx_pause = value

    if rx_pause is None and tx_pause is None and not rx_prio_pause and not tx_prio_pause:
        return None

    return PFCPauseMetrics(
        interface=interface,
        rx_pause=rx_pause,
        tx_pause=tx_pause,
        rx_prio_pause=rx_prio_pause,
        tx_prio_pause=tx_prio_pause,
    )


def collect_pfc_metrics(interfaces: list[str]) -> tuple[list[PFCPauseMetrics], list[str]]:
    warnings: list[str] = []
    metrics: list[PFCPauseMetrics] = []
    if not interfaces:
        return metrics, warnings

    for interface in interfaces:
        try:
            output = subprocess.check_output(["ethtool", "-S", interface], text=True).strip()
        except FileNotFoundError:
            warnings.append("ethtool unavailable; PFC pause metrics not collected")
            return [], warnings
        except subprocess.CalledProcessError as exc:
            warnings.append(f"ethtool -S {interface} failed; PFC pause metrics skipped: {exc}")
            continue

        parsed = parse_ethtool_pause_stats(interface, output)
        if parsed is not None:
            metrics.append(parsed)

    return metrics, warnings
