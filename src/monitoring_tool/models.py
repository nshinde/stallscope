from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class GPUMetrics:
    index: int
    utilization_gpu_pct: float
    utilization_mem_pct: float
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: float
    power_draw_w: float
    clocks_throttle_reasons: str = ""

    @property
    def memory_headroom_pct(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return max(0.0, 100.0 - (self.memory_used_mb / self.memory_total_mb * 100.0))


@dataclass
class NetworkMetrics:
    interface: str
    rx_bytes: int
    tx_bytes: int
    rx_packets: int
    tx_packets: int
    rx_errs: int
    tx_errs: int
    rx_drop: int
    tx_drop: int


@dataclass
class RDMAPortMetrics:
    device: str
    port: str
    counters: dict[str, int] = field(default_factory=dict)
    hw_counters: dict[str, int] = field(default_factory=dict)


@dataclass
class PFCPauseMetrics:
    interface: str
    rx_pause: int | None = None
    tx_pause: int | None = None
    rx_prio_pause: dict[int, int] = field(default_factory=dict)
    tx_prio_pause: dict[int, int] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    load_1m: float
    load_5m: float
    mem_total_kb: int
    mem_available_kb: int

    @property
    def mem_available_pct(self) -> float:
        if self.mem_total_kb <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.mem_available_kb / self.mem_total_kb) * 100.0))


@dataclass
class Snapshot:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    gpus: list[GPUMetrics] = field(default_factory=list)
    net: list[NetworkMetrics] = field(default_factory=list)
    rdma: list[RDMAPortMetrics] = field(default_factory=list)
    pfc: list[PFCPauseMetrics] = field(default_factory=list)
    system: SystemMetrics | None = None
    warnings: list[str] = field(default_factory=list)
