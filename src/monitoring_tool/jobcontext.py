from __future__ import annotations

import os
import socket
from pathlib import Path

from .models import JobContext


KUBERNETES_SERVICE_ACCOUNT_PATH = Path("/var/run/secrets/kubernetes.io")


def _proc_path(*parts: str) -> Path:
    root = os.getenv("MONITORING_PROCFS_ROOT", "/proc")
    return Path(root).joinpath(*parts)


def parse_environ(content: bytes | str) -> dict[str, str]:
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    values: dict[str, str] = {}
    for entry in content.split("\0"):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        values[key] = value
    return values


def slurm_context_from_environ(environ: dict[str, str]) -> JobContext | None:
    job_id = environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    return JobContext(
        scheduler="SLURM",
        job_id=job_id,
        job_name=environ.get("SLURM_JOB_NAME", ""),
        user=environ.get("SLURM_JOB_USER", environ.get("USER", "")),
        nodelist=environ.get("SLURM_JOB_NODELIST", ""),
    )


def _scan_proc_for_slurm(warnings: list[str]) -> JobContext | None:
    proc_root = _proc_path()
    if not proc_root.exists():
        warnings.append(f"{proc_root} unavailable; Slurm process scan skipped")
        return None

    for pid_path in sorted(path for path in proc_root.iterdir() if path.name.isdigit()):
        environ_path = pid_path / "environ"
        if not environ_path.exists():
            continue
        try:
            context = slurm_context_from_environ(parse_environ(environ_path.read_bytes()))
        except OSError:
            continue
        if context is not None:
            return context
    return None


def kubernetes_context() -> JobContext | None:
    if not KUBERNETES_SERVICE_ACCOUNT_PATH.exists():
        return None
    pod_name = socket.gethostname()
    return JobContext(scheduler="KUBERNETES", job_id=pod_name, job_name=pod_name)


def collect_job_context() -> tuple[JobContext, list[str]]:
    warnings: list[str] = []

    context = slurm_context_from_environ(dict(os.environ))
    if context is not None:
        return context, warnings

    context = _scan_proc_for_slurm(warnings)
    if context is not None:
        return context, warnings

    context = kubernetes_context()
    if context is not None:
        return context, warnings

    return JobContext(), warnings
