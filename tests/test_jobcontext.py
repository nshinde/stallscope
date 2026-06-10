from pathlib import Path

from monitoring_tool import jobcontext
from monitoring_tool.jobcontext import collect_job_context, parse_environ, slurm_context_from_environ


def test_parse_environ_handles_null_separated_content():
    parsed = parse_environ(b"SLURM_JOB_ID=123\0USER=alice\0")

    assert parsed == {"SLURM_JOB_ID": "123", "USER": "alice"}


def test_slurm_context_from_environment():
    context = slurm_context_from_environ(
        {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NAME": "train",
            "SLURM_JOB_USER": "alice",
            "SLURM_JOB_NODELIST": "node[01-02]",
        }
    )

    assert context is not None
    assert context.scheduler == "SLURM"
    assert context.job_id == "123"
    assert context.job_name == "train"
    assert context.user == "alice"
    assert context.nodelist == "node[01-02]"


def test_collect_job_context_scans_proc_environ(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    environ = proc / "100" / "environ"
    environ.parent.mkdir(parents=True)
    environ.write_bytes(b"SLURM_JOB_ID=456\0SLURM_JOB_NAME=scan\0USER=bob\0")
    monkeypatch.setenv("MONITORING_PROCFS_ROOT", str(proc))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(jobcontext, "KUBERNETES_SERVICE_ACCOUNT_PATH", Path("tests/fixtures/no-kubernetes"))

    context, warnings = collect_job_context()

    assert warnings == []
    assert context.scheduler == "SLURM"
    assert context.job_id == "456"
    assert context.job_name == "scan"


def test_collect_job_context_detects_kubernetes(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("MONITORING_PROCFS_ROOT", "tests/fixtures/sysfs_no_rdma")
    monkeypatch.setattr(jobcontext, "KUBERNETES_SERVICE_ACCOUNT_PATH", Path("tests/fixtures/kubernetes.io"))
    monkeypatch.setattr(jobcontext.socket, "gethostname", lambda: "trainer-pod-0")

    context, warnings = collect_job_context()

    assert warnings == []
    assert context.scheduler == "KUBERNETES"
    assert context.job_id == "trainer-pod-0"


def test_collect_job_context_unknown(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("MONITORING_PROCFS_ROOT", "tests/fixtures/sysfs_no_rdma")
    monkeypatch.setattr(jobcontext, "KUBERNETES_SERVICE_ACCOUNT_PATH", Path("tests/fixtures/no-kubernetes"))

    context, warnings = collect_job_context()

    assert warnings == []
    assert context.scheduler == "UNKNOWN"
    assert context.job_id == ""
