from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.kube_client import KubectlClient
from controller.state_collector import StateCollector
from experiments.runtime import append_jsonl, wait_http
from workload.replay_driver import replay


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _image_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}|{{.Id}}"],
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _plan(rate: float, duration: float, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    count = int(round(rate * duration))
    offsets = np.arange(count, dtype=np.float64) / rate
    offsets += rng.uniform(0.0, min(0.25 / rate, 0.0005), size=count)
    return [
        {
            "request_id": index, "step": int(offset // 1),
            "scheduled_offset_seconds": float(offset), "target_rps": rate,
            "payload": f"smoke:{seed}:{index}",
        }
        for index, offset in enumerate(offsets)
    ]


async def _scale_schedule(
    kube: KubectlClient, schedule: list[dict], started: float, output: Path
) -> list[dict]:
    events: list[dict] = []
    for item in schedule:
        delay = float(item["offset_seconds"]) - (time.perf_counter() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        before = await asyncio.to_thread(kube.deployment_status)
        evidence = await asyncio.to_thread(kube.scale, int(item["replicas"]))
        event = {
            "event": "scale_requested", "offset_seconds": time.perf_counter() - started,
            "target_replicas": int(item["replicas"]), "before": before,
            "api_latency_seconds": evidence.latency_seconds,
            "wall_time": datetime.now(timezone.utc).isoformat(),
        }
        events.append(event)
        append_jsonl(output, event)
    return events


async def _monitor(
    collector: StateCollector, kube: KubectlClient, *, duration: float,
    started: float, interval: float, metrics_path: Path, events_path: Path,
) -> list[dict]:
    rows: list[dict] = []
    known: dict[str, bool] = {}
    while time.perf_counter() - started < duration + 2.0:
        offset = time.perf_counter() - started
        snapshot = await asyncio.to_thread(
            collector.collect, remaining_budget_ratio=1.0,
            remaining_horizon_ratio=max(1.0 - offset / duration, 0.0),
        )
        row = snapshot.as_dict()
        row["offset_seconds"] = offset
        append_jsonl(metrics_path, row)
        rows.append(row)
        pods = await asyncio.to_thread(kube.worker_pods)
        for pod in pods:
            state = bool(pod["ready"])
            if known.get(pod["uid"]) != state:
                append_jsonl(
                    events_path,
                    {
                        "event": "pod_ready_transition", "offset_seconds": offset,
                        "pod": pod, "wall_time": datetime.now(timezone.utc).isoformat(),
                    },
                )
                known[pod["uid"]] = state
        await asyncio.sleep(max(interval - snapshot.collection_latency_seconds, 0.05))
    return rows


def _phase_metrics(events: list[dict], schedule: list[dict], duration: float) -> list[dict]:
    boundaries = [(float(row["offset_seconds"]), int(row["replicas"])) for row in schedule]
    phases: list[dict] = []
    for index, (start, replicas) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else duration
        scheduled = [
            row for row in events
            if start <= float(row["sent_offset_seconds"]) < end
        ]
        selected = [
            row for row in events
            if start <= float(row["completed_offset_seconds"]) < end
        ]
        completed_rows = [row for row in selected if 200 <= int(row["http_status"]) < 300]
        latencies = np.asarray([row["client_latency_seconds"] for row in completed_rows], dtype=float)
        completed = len(completed_rows)
        phases.append(
            {
                "start_seconds": start, "end_seconds": end, "target_replicas": replicas,
                "requests_scheduled": len(scheduled), "responses_observed": len(selected),
                "completed": completed,
                "completion_rate_rps": completed / max(end - start, 1.0e-9),
                "mean_latency_seconds": float(latencies.mean()) if len(latencies) else float("nan"),
                "p95_latency_seconds": float(np.quantile(latencies, 0.95)) if len(latencies) else float("nan"),
            }
        )
    return phases


def _write_report(result: dict, path: Path) -> None:
    checks = result["checks"]
    lines = [
        "# Smoke Test Report", "", f"Status: **{result['status'].upper()}**", "",
        "This report records a real Kubernetes run. Request, metric, scale, and Pod Ready evidence is retained under the linked run directory.", "",
        "## Executed System", "",
        f"- Kubernetes context: `{result['context']}`",
        f"- Namespace: `{result['namespace']}`",
        f"- Image evidence: `{result['image_digest']}`",
        f"- Real HTTP requests scheduled: {result['replay']['scheduled']}",
        f"- Real HTTP requests completed: {result['replay']['completed']}",
        f"- Maximum observed Ready Pods: {result['max_ready_replicas']}",
        f"- Maximum observed application queue: {result['max_queue_depth']:.0f}", "",
        "## Gates", "",
        "| Gate | Result |", "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    lines += ["", "## Replica Phases", "", "| Target Pods | Completed rps | Mean latency (s) | P95 latency (s) |", "|---:|---:|---:|---:|"]
    for phase in result["phases"]:
        lines.append(
            f"| {phase['target_replicas']} | {phase['completion_rate_rps']:.3f} | "
            f"{phase['mean_latency_seconds']:.4f} | {phase['p95_latency_seconds']:.4f} |"
        )
    lines += ["", f"Raw run: `{result['run_directory']}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


async def run_smoke(config: dict) -> dict:
    context = str(config["kubernetes"]["context"])
    namespace = str(config["kubernetes"]["namespace"])
    deployment = str(config["kubernetes"].get("deployment", "dap-worker"))
    image = str(config["kubernetes"]["image"])
    kube = KubectlClient(context=context, namespace=namespace, deployment=deployment)
    conflicts = kube.autoscaler_conflicts()
    if conflicts:
        raise RuntimeError(f"autoscaler conflict before smoke: {conflicts}")
    kube.set_profile(str(config["service"]["profile"]))
    kube.scale(1)
    kube.rollout_status()

    duration = float(config["workload"]["duration_seconds"])
    schedule = list(config["scaling_schedule"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "smoke" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "system_events.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    request_path = run_dir / "requests.jsonl"
    replay_summary_path = run_dir / "replay_summary.json"
    plan = _plan(float(config["workload"]["rps"]), duration, int(config["seed"]))
    (run_dir / "request_plan.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in plan), encoding="utf-8"
    )
    collector = StateCollector(
        kube, capacity_per_pod_rps=float(config["service"]["nominal_capacity_per_pod_rps"])
    )
    node_ip = kube.node_internal_ip()
    port = int(config["kubernetes"].get("node_port", 30080))
    service_url = f"http://{node_ip}:{port}"
    # Use NodePort instead of kubectl port-forward: port-forward selects one
    # backend pod, whereas every NodePort TCP connection exercises Service
    # endpoint selection and therefore real replica scaling.
    health = wait_http(f"{service_url}/healthz")
    started = time.perf_counter()
    replay_task = asyncio.create_task(
        replay(
            plan, url=f"{service_url}/infer", output=request_path,
            timeout_seconds=float(config["workload"]["client_timeout_seconds"]),
            connection_limit=int(config["workload"]["connection_limit"]),
            force_close_connections=bool(config["workload"].get("force_close_connections", True)),
        )
    )
    scale_task = asyncio.create_task(_scale_schedule(kube, schedule, started, events_path))
    monitor_task = asyncio.create_task(
        _monitor(
            collector, kube, duration=duration, started=started,
            interval=float(config["metrics"]["poll_interval_seconds"]),
            metrics_path=metrics_path, events_path=events_path,
        )
    )
    replay_summary, scale_events, metric_rows = await asyncio.gather(
        replay_task, scale_task, monitor_task
    )
    replay_summary_path.write_text(json.dumps(replay_summary, indent=2, sort_keys=True) + "\n")
    requests = [json.loads(line) for line in request_path.read_text().splitlines() if line]
    phases = _phase_metrics(requests, schedule, duration)
    max_ready = max((int(row["ready_replicas"]) for row in metric_rows), default=0)
    max_queue = max((float(row["fields"]["queue_depth"]["raw"]) for row in metric_rows), default=0.0)
    missing_samples = sum(bool(row["missing_pods"]) for row in metric_rows)
    checks = {
        "real_http_requests_completed": replay_summary["completed"] > 0,
        "manual_scale_reached_five_ready_pods": max_ready >= 5,
        "insufficient_capacity_formed_queue": max_queue > 0,
        "pod_metrics_missing_below_one_percent": missing_samples / max(len(metric_rows), 1) < 0.01,
        "all_scale_api_calls_succeeded": len(scale_events) == len(schedule),
        "request_plan_fully_sent": replay_summary["sent"] == replay_summary["scheduled"],
        "replica_scaling_increases_steady_completion_capacity": (
            len(phases) >= 3
            and phases[2]["completion_rate_rps"] > phases[0]["completion_rate_rps"] * 1.10
        ),
    }
    result = {
        "schema": "dap.k8s.smoke_result.v1", "status": "pass" if all(checks.values()) else "fail",
        "context": context, "namespace": namespace, "health": health,
        "image_digest": _image_digest(image), "run_directory": str(run_dir.relative_to(ROOT)),
        "config_sha256": _sha256(Path(config["_config_path"])),
        "request_plan_sha256": _sha256(run_dir / "request_plan.jsonl"),
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "replay": replay_summary, "max_ready_replicas": max_ready,
        "max_queue_depth": max_queue, "phases": phases, "checks": checks,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_report(result, ROOT / "docs" / "SMOKE_TEST_REPORT.md")
    kube.scale(1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["_config_path"] = str(args.config.resolve())
    result = asyncio.run(run_smoke(config))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
