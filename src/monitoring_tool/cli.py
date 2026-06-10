from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .alerts import build_alerts, render_prometheus_metrics, send_alert_webhook
from .collectors import collect_snapshot
from .nccl import run_nccl_all_reduce_test
from .profiler import classify_job


def _to_json(snapshot, profile, alerts, nccl_result=None) -> str:
    payload = {
        "snapshot": asdict(snapshot),
        "profile": asdict(profile),
        "alerts": [asdict(a) for a in alerts],
    }
    if nccl_result is not None:
        payload["nccl"] = asdict(nccl_result)
    return json.dumps(payload, default=str, indent=2)


def _run_once(args, previous_snapshot=None):
    snapshot = collect_snapshot()
    profile = classify_job(snapshot, previous_snapshot)
    nccl_result = run_nccl_all_reduce_test() if args.nccl_test else None
    alerts = build_alerts(snapshot, profile)

    if args.prometheus_textfile:
        metrics = render_prometheus_metrics(snapshot, profile, alerts)
        path = Path(args.prometheus_textfile)
        path.write_text(metrics)

    if args.alert_webhook_url and alerts:
        payload = {
            "profile": asdict(profile),
            "alerts": [asdict(a) for a in alerts],
        }
        try:
            send_alert_webhook(args.alert_webhook_url, payload)
        except Exception as exc:
            print(f"Warning: failed to send alert webhook: {exc}")

    if args.json:
        print(_to_json(snapshot, profile, alerts, nccl_result))
    else:
        print(f"Prediction: {profile.label} (confidence={profile.confidence})")
        print(f"Bottleneck hint: {profile.bottleneck_hint}")
        for reason in profile.reasons:
            print(f"- {reason}")

        if nccl_result is not None:
            print("NCCL test:")
            print(f"- status: {nccl_result.status}")
            print(f"- message: {nccl_result.message}")
            if nccl_result.bandwidth_gbps is not None:
                print(f"- algbw(GB/s): {nccl_result.bandwidth_gbps}")
                print(f"- busbw(GB/s): {nccl_result.bus_bandwidth_gbps}")

        if alerts:
            print("Alerts:")
            for alert in alerts:
                print(f"- [{alert.severity}] ({alert.category}) {alert.message}")

        if snapshot.warnings:
            print("Warnings:")
            for warning in snapshot.warnings:
                print(f"- {warning}")

    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="monitoring-tool",
        description="Collect GPU/network stats and predict job risk.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    parser.add_argument("--nccl-test", action="store_true", help="Run NCCL all_reduce_perf benchmark if installed")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=0,
        help="Run continuously at this interval in seconds (0 runs once)",
    )
    parser.add_argument(
        "--prometheus-textfile",
        type=str,
        default="",
        help="Write Prometheus metrics to this file path each cycle",
    )
    parser.add_argument(
        "--alert-webhook-url",
        type=str,
        default="",
        help="POST alert payloads to this webhook URL (Alertmanager/Grafana webhook)",
    )
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        _run_once(args)
        return

    print(f"Running periodically every {args.interval_seconds} seconds. Press Ctrl+C to stop.")
    previous_snapshot = None
    while True:
        previous_snapshot = _run_once(args, previous_snapshot)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
