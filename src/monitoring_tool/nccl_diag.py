"""NCCL configuration diagnostic: detects common misconfigurations and suggests flags.

Why NCCL tuning is hard
───────────────────────
1. 60+ interacting environment variables with hardware-specific optimal values.
2. Wrong settings cause silent fallbacks (e.g. RDMA→TCP) with no error, just low bandwidth.
3. GID index, traffic class, and service level must match the physical fabric config.
4. Optimal channel/buffer counts depend on message size, GPU count, and NIC count.
5. Variables changed meaning or were renamed across NCCL versions (2.12 → 2.19+).
6. No built-in "check" mode: NCCL runs and is just slow.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Severity ──────────────────────────────────────────────────────────────────

SEVERITY_ERROR = "ERROR"   # Definite misconfiguration; will hurt performance or correctness
SEVERITY_WARN  = "WARN"    # Likely suboptimal; verify topology before changing
SEVERITY_INFO  = "INFO"    # Best-practice suggestion; low-risk to apply


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class NcclEnvSnapshot:
    """NCCL-related environment variables captured from the process environment."""

    vars: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.vars.get(key, default)

    def is_set(self, key: str) -> bool:
        return key in self.vars

    def is_disabled(self, key: str) -> bool:
        """Return True when the variable is explicitly set to a falsy value."""
        return self.vars.get(key, "").strip() in ("0", "false", "no", "off")

    def is_enabled(self, key: str) -> bool:
        """Return True when the variable is explicitly set to a truthy value."""
        return self.vars.get(key, "").strip() in ("1", "true", "yes", "on")


@dataclass
class NcclHardwareContext:
    """Hardware context inferred from the local node, relevant to NCCL tuning."""

    gpu_count: int = 0
    ib_devices: list[str] = field(default_factory=list)   # e.g. ["mlx5_0", "mlx5_1"]
    net_interfaces: list[str] = field(default_factory=list)  # non-loopback interfaces
    gpudirect_rdma_available: bool = False   # nv_peer_mem / nvidia_peermem loaded
    nvlink_available: bool = False           # NVLink topology detected
    nccl_version: str | None = None          # e.g. "2.19.3"


@dataclass
class NcclDiagFinding:
    """A single diagnostic finding from the NCCL config analysis."""

    severity: str        # SEVERITY_ERROR | SEVERITY_WARN | SEVERITY_INFO
    category: str        # transport | performance | congestion | debug | protocol
    env_var: str         # primary env var involved (empty string for hardware-only findings)
    description: str     # what is wrong and why it matters
    suggestion: str      # exact remediation with the flag to set
    current_value: str | None = None   # current value when the var is set


@dataclass
class NcclDiagReport:
    """Aggregated result of the NCCL diagnostic run."""

    findings: list[NcclDiagFinding] = field(default_factory=list)
    env: NcclEnvSnapshot = field(default_factory=NcclEnvSnapshot)
    hw: NcclHardwareContext = field(default_factory=NcclHardwareContext)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_ERROR)

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_WARN)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_INFO)

    @property
    def overall_health(self) -> str:
        if self.error_count >= 1:
            return "DEGRADED"
        if self.warn_count >= 2:
            return "SUBOPTIMAL"
        if self.findings:
            return "OK_WITH_SUGGESTIONS"
        return "OPTIMAL"


# ── Environment collection ────────────────────────────────────────────────────

_TRACKED_NON_NCCL_VARS = frozenset({
    "FI_EFA_USE_DEVICE_RDMA",   # AWS EFA GPUDirect gate
    "FI_PROVIDER",               # libfabric provider override
    "LD_LIBRARY_PATH",           # plugin discovery
    "OMPI_MCA_btl",              # OpenMPI transport selector
})


def collect_nccl_env(environ: dict[str, str] | None = None) -> NcclEnvSnapshot:
    """Capture all NCCL_* and related variables from the environment."""
    src = environ if environ is not None else dict(os.environ)
    captured = {
        k: v
        for k, v in src.items()
        if k.startswith("NCCL_") or k in _TRACKED_NON_NCCL_VARS
    }
    return NcclEnvSnapshot(vars=captured)


# ── Hardware detection ────────────────────────────────────────────────────────

def detect_hardware(
    sysfs_root: str | None = None,
    proc_root: str | None = None,
) -> NcclHardwareContext:
    """Detect local hardware context relevant to NCCL tuning."""
    sysfs = sysfs_root or os.environ.get("MONITORING_SYSFS_ROOT", "/sys")
    proc  = proc_root  or os.environ.get("MONITORING_PROCFS_ROOT", "/proc")
    hw = NcclHardwareContext()

    hw.gpu_count = _detect_gpu_count()

    ib_path = Path(sysfs) / "class" / "infiniband"
    if ib_path.exists():
        hw.ib_devices = sorted(
            p.name for p in ib_path.iterdir() if p.is_symlink() or p.is_dir()
        )

    net_dev = Path(proc) / "net" / "dev"
    if net_dev.exists():
        for line in net_dev.read_text().splitlines()[2:]:
            iface = line.split(":")[0].strip()
            if iface and iface != "lo":
                hw.net_interfaces.append(iface)

    modules_path = Path(proc) / "modules"
    if modules_path.exists():
        mods = modules_path.read_text()
        hw.gpudirect_rdma_available = bool(
            re.search(r"^(nv_peer_mem|nvidia_peermem)\s", mods, re.MULTILINE)
        )

    hw.nvlink_available = _detect_nvlink()
    hw.nccl_version = _detect_nccl_version()

    return hw


def _detect_gpu_count() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=10,
        )
        if out.returncode == 0:
            return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        pass
    return 0


def _detect_nvlink() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "nvlink", "-s"],
            text=True, capture_output=True, timeout=10,
        )
        return out.returncode == 0 and "Link" in out.stdout
    except Exception:
        return False


def _detect_nccl_version() -> str | None:
    binary = shutil.which("all_reduce_perf")
    if not binary:
        return None
    try:
        out = subprocess.run(
            [binary, "--help"],
            text=True, capture_output=True, timeout=5,
        )
        m = re.search(r"NCCL version\s+([\d.]+)", out.stdout + out.stderr, re.IGNORECASE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


# ── Diagnostic rules ──────────────────────────────────────────────────────────

_RuleFunc = Callable[[NcclEnvSnapshot, NcclHardwareContext], NcclDiagFinding | None]


def _check_ib_disabled_with_devices(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """NCCL_IB_DISABLE=1 kills RDMA when IB devices are present."""
    if hw.ib_devices and env.get("NCCL_IB_DISABLE") == "1":
        return NcclDiagFinding(
            severity=SEVERITY_ERROR,
            category="transport",
            env_var="NCCL_IB_DISABLE",
            current_value="1",
            description=(
                f"NCCL_IB_DISABLE=1 is set but {len(hw.ib_devices)} InfiniBand/RoCE device(s) "
                f"are present ({', '.join(hw.ib_devices)}). "
                "NCCL will fall back to TCP sockets, which are typically 10–20× slower."
            ),
            suggestion="Unset NCCL_IB_DISABLE or set NCCL_IB_DISABLE=0 to re-enable RDMA transport.",
        )
    return None


def _check_no_ib_devices(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """GPUs present but no RDMA devices found and IB is not explicitly disabled."""
    if hw.gpu_count > 0 and not hw.ib_devices and env.get("NCCL_IB_DISABLE") != "1":
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="transport",
            env_var="NCCL_IB_DISABLE",
            current_value=env.get("NCCL_IB_DISABLE"),
            description=(
                f"{hw.gpu_count} GPU(s) detected but no RDMA/InfiniBand devices found. "
                "NCCL will use TCP sockets for inter-node transfers (~10–20× slower than RDMA). "
                "This is expected for single-node runs but degrades multi-node training."
            ),
            suggestion=(
                "For multi-node: install RDMA NICs and drivers, then verify with "
                "'ls /sys/class/infiniband/'. "
                "For single-node / intentional TCP: set NCCL_IB_DISABLE=1 to suppress this warning."
            ),
        )
    return None


def _check_ib_hca_unset(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Multiple HCAs but NCCL_IB_HCA not pinned — NCCL may pick suboptimal NIC."""
    if len(hw.ib_devices) > 1 and not env.is_set("NCCL_IB_HCA"):
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="transport",
            env_var="NCCL_IB_HCA",
            current_value=None,
            description=(
                f"{len(hw.ib_devices)} RDMA devices found ({', '.join(hw.ib_devices)}) but "
                "NCCL_IB_HCA is not set. NCCL auto-selects HCAs, which may not follow GPU "
                "NUMA affinity, causing unnecessary PCIe crossings on every RDMA operation."
            ),
            suggestion=(
                "Run 'nvidia-smi topo -m' to map GPU-to-NIC affinity, then set "
                f"NCCL_IB_HCA to the closest device, e.g. NCCL_IB_HCA={hw.ib_devices[0]}. "
                "Comma-separate multiple devices to use all, e.g. "
                f"NCCL_IB_HCA={','.join(hw.ib_devices[:2])}."
            ),
        )
    return None


def _check_gid_index(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Missing NCCL_IB_GID_INDEX — default 0 uses RoCEv1, most fabrics need RoCEv2."""
    if not hw.ib_devices:
        return None
    if not env.is_set("NCCL_IB_GID_INDEX"):
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="transport",
            env_var="NCCL_IB_GID_INDEX",
            current_value=None,
            description=(
                "NCCL_IB_GID_INDEX is not set. The default GID index 0 maps to RoCEv1 "
                "(link-local addressing). Most modern datacenter fabrics are RoCEv2 (routable), "
                "requiring GID index 3. Wrong GID causes silent fallback to slower paths."
            ),
            suggestion=(
                "Check available GIDs: 'show_gids' (rdma-core) or "
                "'cat /sys/class/infiniband/<dev>/ports/1/gids/*'. "
                "For RoCEv2 fabrics set NCCL_IB_GID_INDEX=3. "
                "Confirm with your fabric admin which index carries the routable address."
            ),
        )
    return None


def _check_gpudirect_rdma(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """GPUDirect RDMA module loaded but NCCL not configured to use it."""
    if not hw.ib_devices:
        return None
    gdr_level = env.get("NCCL_NET_GDR_LEVEL")
    cuda_support = env.get("NCCL_IB_CUDA_SUPPORT")
    gdr_disabled = gdr_level in (None, "0") or cuda_support == "0"
    if not gdr_disabled:
        return None
    if hw.gpudirect_rdma_available:
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="transport",
            env_var="NCCL_NET_GDR_LEVEL",
            current_value=gdr_level,
            description=(
                "GPUDirect RDMA kernel module (nv_peer_mem/nvidia_peermem) is loaded but "
                "NCCL_NET_GDR_LEVEL is not enabled. All RDMA transfers will bounce through "
                "host CPU memory, doubling PCIe bandwidth consumption and increasing latency."
            ),
            suggestion=(
                "Set NCCL_NET_GDR_LEVEL=5 (enable GDR for all distance peers) "
                "and NCCL_IB_CUDA_SUPPORT=1 to use zero-copy GPUDirect RDMA transfers."
            ),
        )
    return NcclDiagFinding(
        severity=SEVERITY_INFO,
        category="transport",
        env_var="NCCL_NET_GDR_LEVEL",
        current_value=gdr_level,
        description=(
            "RDMA devices detected but the GPUDirect RDMA kernel module "
            "(nv_peer_mem or nvidia_peermem) does not appear to be loaded. "
            "Without it NCCL cannot perform zero-copy GPU–NIC transfers."
        ),
        suggestion=(
            "Load the GPUDirect RDMA module: 'modprobe nvidia_peermem' (kernel 5.12+) "
            "or install nv_peer_mem from MLNX_OFED. "
            "Then set NCCL_NET_GDR_LEVEL=5 NCCL_IB_CUDA_SUPPORT=1."
        ),
    )


def _check_socket_ifname(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Multiple interfaces present but NCCL_SOCKET_IFNAME not set."""
    data_ifaces = [
        i for i in hw.net_interfaces
        if not i.startswith(("docker", "br-", "virbr", "veth"))
    ]
    if len(data_ifaces) > 1 and not env.is_set("NCCL_SOCKET_IFNAME"):
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="transport",
            env_var="NCCL_SOCKET_IFNAME",
            current_value=None,
            description=(
                f"Multiple non-virtual interfaces detected ({', '.join(data_ifaces[:6])}) "
                "but NCCL_SOCKET_IFNAME is not set. When NCCL falls back to TCP/IP sockets "
                "it picks an interface arbitrarily, which may be the management NIC instead "
                "of the high-bandwidth data-plane NIC."
            ),
            suggestion=(
                "Set NCCL_SOCKET_IFNAME to the data-plane interface, e.g. "
                "NCCL_SOCKET_IFNAME=ens or NCCL_SOCKET_IFNAME=eth0. "
                "Prefix matching is supported (e.g. 'ens' matches 'ens3f0')."
            ),
        )
    return None


def _check_buffer_size(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """NCCL_BUFFSIZE set too small for large all-reduce workloads."""
    val = env.get("NCCL_BUFFSIZE")
    if val is None:
        return None
    try:
        size = int(val)
    except ValueError:
        return NcclDiagFinding(
            severity=SEVERITY_ERROR,
            category="performance",
            env_var="NCCL_BUFFSIZE",
            current_value=val,
            description=f"NCCL_BUFFSIZE='{val}' is not a valid integer byte count.",
            suggestion="Set NCCL_BUFFSIZE to a power-of-two byte count, e.g. 8388608 (8 MB).",
        )
    if size < 2 * 1024 * 1024:
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="performance",
            env_var="NCCL_BUFFSIZE",
            current_value=val,
            description=(
                f"NCCL_BUFFSIZE={size:,} B ({size // 1024} KB) is below the 4 MB default. "
                "A small ring buffer stalls the pipeline between RDMA put operations, "
                "reducing effective bandwidth especially for large all-reduce messages."
            ),
            suggestion=(
                "For LLM training use NCCL_BUFFSIZE=8388608 (8 MB) or 16777216 (16 MB). "
                "Larger buffers trade memory for pipeline efficiency."
            ),
        )
    return None


def _check_channel_count(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Multi-NIC node with NCCL_MIN_NCHANNELS unset leaves bandwidth under-utilized."""
    if len(hw.ib_devices) > 1 and not env.is_set("NCCL_MIN_NCHANNELS"):
        return NcclDiagFinding(
            severity=SEVERITY_INFO,
            category="performance",
            env_var="NCCL_MIN_NCHANNELS",
            current_value=None,
            description=(
                f"{len(hw.ib_devices)} RDMA devices detected but NCCL_MIN_NCHANNELS is not set. "
                "NCCL may create as few as 2 channels (QPs), leaving multi-port HCA bandwidth "
                "under-utilized. Each NCCL channel maps to one queue pair per peer."
            ),
            suggestion=(
                "Set NCCL_MIN_NCHANNELS=4 (or up to 16 for many NICs) to force more channels. "
                "Pair with NCCL_NCHANNELS_PER_NET_PEER=2 for multi-NIC nodes."
            ),
        )
    return None


def _check_cross_nic(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """NCCL_CROSS_NIC=0 on a multi-NIC node caps ring bandwidth to one NIC."""
    if len(hw.ib_devices) < 2:
        return None
    if env.get("NCCL_CROSS_NIC") == "0":
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="performance",
            env_var="NCCL_CROSS_NIC",
            current_value="0",
            description=(
                f"NCCL_CROSS_NIC=0 disables cross-NIC ring formation but {len(hw.ib_devices)} "
                "RDMA NICs are present. Each ring is confined to a single NIC, leaving "
                f"{len(hw.ib_devices) - 1} NIC(s) idle during collective operations."
            ),
            suggestion=(
                "Set NCCL_CROSS_NIC=1 (allow cross-NIC rings) or "
                "NCCL_CROSS_NIC=2 (force cross-NIC rings) to aggregate multi-NIC bandwidth."
            ),
        )
    return None


def _check_ib_tc(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Missing NCCL_IB_TC means NCCL traffic may bypass ECN-managed queues."""
    if not hw.ib_devices:
        return None
    if not env.is_set("NCCL_IB_TC"):
        return NcclDiagFinding(
            severity=SEVERITY_INFO,
            category="congestion",
            env_var="NCCL_IB_TC",
            current_value=None,
            description=(
                "NCCL_IB_TC (IPv4 traffic class / DSCP byte) is not set. "
                "On fabrics using DCQCN-based ECN congestion control, NCCL traffic must "
                "carry the correct DSCP marking to enter the feedback loop. "
                "Default TC=0 typically routes traffic to a best-effort queue, "
                "bypassing lossless QoS and ECN marking entirely."
            ),
            suggestion=(
                "Confirm the DSCP value your fabric uses for lossless RDMA traffic. "
                "Common: NCCL_IB_TC=106 (DSCP 26, AF31) or NCCL_IB_TC=136 (DSCP 34, AF41). "
                "Align with the switch QoS policy and set NCCL_IB_SL to the matching service level."
            ),
        )
    return None


def _check_ib_sl(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Missing NCCL_IB_SL may land IB traffic on a non-lossless virtual lane."""
    if not hw.ib_devices:
        return None
    if not env.is_set("NCCL_IB_SL"):
        return NcclDiagFinding(
            severity=SEVERITY_INFO,
            category="congestion",
            env_var="NCCL_IB_SL",
            current_value=None,
            description=(
                "NCCL_IB_SL (InfiniBand service level / virtual lane selector) is not set. "
                "On IB fabrics with QoS partitioning, traffic landed on the wrong service level "
                "may encounter a lossy virtual lane, causing RNR NAKs and throughput collapse."
            ),
            suggestion=(
                "Confirm the lossless service level with your fabric admin. "
                "NCCL_IB_SL=0 (default) is usually fine for simple flat fabrics; "
                "set to the SL mapped to the lossless VL on partitioned fabrics."
            ),
        )
    return None


def _check_debug_level(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Flag missing NCCL_DEBUG (no visibility) or INFO in production (too noisy)."""
    val = env.get("NCCL_DEBUG")
    if val is None:
        return NcclDiagFinding(
            severity=SEVERITY_INFO,
            category="debug",
            env_var="NCCL_DEBUG",
            current_value=None,
            description=(
                "NCCL_DEBUG is not set. Without it you get no visibility into which transport "
                "NCCL negotiated, channel count, algorithm selected, or ring topology. "
                "Startup diagnostics are emitted only once and have negligible runtime cost."
            ),
            suggestion=(
                "Set NCCL_DEBUG=WARN for one-time startup diagnostics. "
                "Use NCCL_DEBUG=INFO for full per-operation tracing (high volume — avoid in production). "
                "Add NCCL_DEBUG_SUBSYS=INIT,NET to filter to initialization and network selection only."
            ),
        )
    if val.upper() == "INFO":
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="debug",
            env_var="NCCL_DEBUG",
            current_value=val,
            description=(
                "NCCL_DEBUG=INFO logs a line for every collective operation and can generate "
                "gigabytes of output per hour during training. This adds measurable I/O overhead "
                "and can mask real performance issues behind log-write pressure."
            ),
            suggestion=(
                "Lower to NCCL_DEBUG=WARN for production. "
                "To investigate a specific issue, use NCCL_DEBUG=INFO with "
                "NCCL_DEBUG_SUBSYS=NET,INIT for a targeted log window."
            ),
        )
    return None


def _check_p2p_disabled_with_nvlink(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """NCCL_P2P_DISABLE=1 kills NVLink P2P and forces PCIe/CPU paths."""
    if hw.nvlink_available and env.get("NCCL_P2P_DISABLE") == "1":
        return NcclDiagFinding(
            severity=SEVERITY_ERROR,
            category="transport",
            env_var="NCCL_P2P_DISABLE",
            current_value="1",
            description=(
                "NCCL_P2P_DISABLE=1 is set but NVLink is detected on this node. "
                "All intra-node GPU-to-GPU transfers are forced through PCIe or host CPU memory, "
                "typically 12× slower than direct NVLink transfers."
            ),
            suggestion="Unset NCCL_P2P_DISABLE or set NCCL_P2P_DISABLE=0 to re-enable NVLink P2P.",
        )
    return None


def _check_shm_disabled(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """NCCL_SHM_DISABLE=1 on multi-GPU node removes fastest intra-node fallback."""
    if hw.gpu_count > 1 and env.get("NCCL_SHM_DISABLE") == "1":
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="transport",
            env_var="NCCL_SHM_DISABLE",
            current_value="1",
            description=(
                f"NCCL_SHM_DISABLE=1 is set with {hw.gpu_count} GPUs on this node. "
                "Shared-memory transport is the fastest intra-node fallback when NVLink is "
                "unavailable. Disabling it forces slower socket-based transfers even within the node."
            ),
            suggestion=(
                "Unset NCCL_SHM_DISABLE unless debugging a specific SHM issue. "
                "SHM is safe to use on any multi-GPU single-host system."
            ),
        )
    return None


def _check_ib_timeout(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Very low NCCL_IB_TIMEOUT causes spurious retransmits under PFC/ECN events."""
    if not hw.ib_devices:
        return None
    val = env.get("NCCL_IB_TIMEOUT")
    if val is None:
        return None
    try:
        timeout = int(val)
    except ValueError:
        return None
    # IB timeout = 4.096us × 2^timeout; below 14 → 67ms, very aggressive
    if timeout < 14:
        timeout_ms = 4.096 * (2 ** timeout) / 1000
        return NcclDiagFinding(
            severity=SEVERITY_WARN,
            category="transport",
            env_var="NCCL_IB_TIMEOUT",
            current_value=val,
            description=(
                f"NCCL_IB_TIMEOUT={timeout} gives a retransmission timeout of "
                f"~{timeout_ms:.0f} ms (4.096 µs × 2^{timeout}). "
                "Under PFC or ECN congestion events the fabric may pause longer than this, "
                "triggering spurious retransmits that cascade into RNR NAK storms."
            ),
            suggestion=(
                "For typical datacenter fabrics use NCCL_IB_TIMEOUT=22 (~17 s). "
                "Aggressive values like 14 (~67 ms) work only on zero-congestion fabrics."
            ),
        )
    return None


def _check_nccl_algo(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Flag unknown NCCL_ALGO values or suboptimal choices given the hardware."""
    val = env.get("NCCL_ALGO")
    if val is None:
        return None
    valid = frozenset({"Ring", "Tree", "CollNetChain", "CollNetDirect", "NVLS", "NVLSTree"})
    if val not in valid:
        return NcclDiagFinding(
            severity=SEVERITY_ERROR,
            category="protocol",
            env_var="NCCL_ALGO",
            current_value=val,
            description=(
                f"NCCL_ALGO='{val}' is not a recognized algorithm name. "
                f"Valid values: {', '.join(sorted(valid))}. "
                "An invalid value is silently ignored by some NCCL versions and causes "
                "unexpected fallback behavior in others."
            ),
            suggestion=(
                "Unset NCCL_ALGO to let NCCL auto-select the best algorithm for each "
                "message size, or use one of the valid values above."
            ),
        )
    if val == "Tree" and hw.nvlink_available:
        return NcclDiagFinding(
            severity=SEVERITY_INFO,
            category="protocol",
            env_var="NCCL_ALGO",
            current_value=val,
            description=(
                "NCCL_ALGO=Tree is forced while NVLink is available. "
                "Tree has higher latency than Ring for small messages on NVLink topologies; "
                "NCCL auto-selection typically prefers Ring for small messages and Tree for "
                "large messages when bandwidth-delay product is the bottleneck."
            ),
            suggestion=(
                "Consider unsetting NCCL_ALGO to restore auto-selection. "
                "If forcing Tree, verify with NCCL_DEBUG=INFO that Tree actually improves "
                "your specific workload's message-size distribution."
            ),
        )
    return None


def _check_qps_per_connection(
    env: NcclEnvSnapshot, hw: NcclHardwareContext
) -> NcclDiagFinding | None:
    """Suggest NCCL_IB_QPS_PER_CONNECTION > 1 for multi-NIC high-bandwidth nodes."""
    if len(hw.ib_devices) < 2:
        return None
    val = env.get("NCCL_IB_QPS_PER_CONNECTION")
    if val is None or int(val) < 2:
        return NcclDiagFinding(
            severity=SEVERITY_INFO,
            category="performance",
            env_var="NCCL_IB_QPS_PER_CONNECTION",
            current_value=val,
            description=(
                f"NCCL_IB_QPS_PER_CONNECTION is {val or 'unset (default 1)'}. "
                "With multiple RDMA NICs and high per-connection bandwidth, a single QP per "
                "connection can become a bottleneck due to PCIe or HCA send-queue depth limits."
            ),
            suggestion=(
                "Try NCCL_IB_QPS_PER_CONNECTION=4 on 400 Gb/s+ HDR/NDR nodes. "
                "Higher values increase HCA queue depth but may increase memory usage."
            ),
        )
    return None


# ── Rule registry ─────────────────────────────────────────────────────────────

_ALL_RULES: list[_RuleFunc] = [
    _check_ib_disabled_with_devices,
    _check_no_ib_devices,
    _check_ib_hca_unset,
    _check_gid_index,
    _check_gpudirect_rdma,
    _check_socket_ifname,
    _check_buffer_size,
    _check_channel_count,
    _check_cross_nic,
    _check_ib_tc,
    _check_ib_sl,
    _check_debug_level,
    _check_p2p_disabled_with_nvlink,
    _check_shm_disabled,
    _check_ib_timeout,
    _check_nccl_algo,
    _check_qps_per_connection,
]


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_nccl_config(
    env: NcclEnvSnapshot | None = None,
    hw: NcclHardwareContext | None = None,
) -> NcclDiagReport:
    """Run all diagnostic rules and return the consolidated report.

    Accepts pre-built *env* and *hw* for testing; detects from the live system when omitted.
    """
    env = env or collect_nccl_env()
    hw = hw or detect_hardware()
    findings = [f for rule in _ALL_RULES for f in [rule(env, hw)] if f is not None]
    return NcclDiagReport(findings=findings, env=env, hw=hw)


def format_nccl_report(report: NcclDiagReport, *, verbose: bool = False) -> str:
    """Render a NcclDiagReport as a human-readable console string."""
    lines: list[str] = []
    lines.append("═══ NCCL Configuration Diagnostic ═══")
    lines.append(
        f"Hardware: {report.hw.gpu_count} GPU(s)  "
        f"{len(report.hw.ib_devices)} RDMA device(s)  "
        f"{len(report.hw.net_interfaces)} network interface(s)"
    )
    if report.hw.ib_devices:
        lines.append(f"  RDMA devices   : {', '.join(report.hw.ib_devices)}")
    if report.hw.nccl_version:
        lines.append(f"  NCCL version   : {report.hw.nccl_version}")
    lines.append(f"  GPUDirect RDMA : {'available' if report.hw.gpudirect_rdma_available else 'not detected'}")
    lines.append(f"  NVLink         : {'detected' if report.hw.nvlink_available else 'not detected'}")

    lines.append("")
    lines.append(
        f"Overall health: {report.overall_health}"
        f"  ({report.error_count} error(s)  {report.warn_count} warning(s)  {report.info_count} suggestion(s))"
    )

    if not report.findings:
        lines.append("")
        lines.append("No issues found. NCCL configuration looks good.")
        return "\n".join(lines)

    lines.append("")
    for i, f in enumerate(report.findings, 1):
        lines.append(f"[{f.severity:<5}] #{i:02d}  {f.category.upper():<12} {f.env_var or '(hardware)'}")
        lines.append(f"  Issue      : {f.description}")
        if f.current_value is not None:
            lines.append(f"  Current    : {f.env_var}={f.current_value}")
        lines.append(f"  Fix        : {f.suggestion}")
        if i < len(report.findings):
            lines.append("")

    if verbose and report.env.vars:
        lines.append("")
        lines.append("── Captured NCCL environment ──")
        for k, v in sorted(report.env.vars.items()):
            lines.append(f"  {k}={v}")

    return "\n".join(lines)
