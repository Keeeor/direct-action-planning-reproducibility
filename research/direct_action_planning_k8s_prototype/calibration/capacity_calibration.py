from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.kube_client import KubectlClient
from controller.state_collector import StateCollector
from workload.replay_driver import replay

from calibration.common import ROOT, fixed_rate_plan, load_config, node_url, reset_profile, timestamp_id, write_json


async def _monitor(collector: StateCollector, stop: asyncio.Event, output: Path) -> list[dict]:
    rows: list[dict] = []
    while not stop.is_set():
        snapshot = await asyncio.to_thread(
            collector.collect, remaining_budget_ratio=1.0, remaining_horizon_ratio=1.0
        )
        row = snapshot.as_dict()
        rows.append(row)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        await asyncio.sleep(0.5)
    return rows


async def run(config: dict) -> Path:
    kube_cfg = config["kubernetes"]
    kube = KubectlClient(
        context=str(kube_cfg["context"]), namespace=str(kube_cfg["namespace"]),
        deployment=str(kube_cfg.get("deployment", "dap-worker")),
    )
    if kube.autoscaler_conflicts():
        raise RuntimeError("capacity calibration requires all autoscalers disabled")
    run_dir = ROOT / "results" / "calibration" / "capacity" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    url = node_url(kube, int(kube_cfg.get("node_port", 30080)))
    rows: list[dict] = []
    seed = int(config["seed"])
    for profile, profile_cfg in config["capacity_profiles"].items():
        for replicas in profile_cfg["replicas"]:
            for rate in profile_cfg["rates_rps"]:
                reset_profile(kube, profile, int(replicas))
                collector = StateCollector(
                    kube, capacity_per_pod_rps=float(profile_cfg["nominal_capacity_per_pod_rps"])
                )
                collector.collect(remaining_budget_ratio=1.0, remaining_horizon_ratio=1.0)
                duration = float(profile_cfg["duration_seconds"])
                plan = fixed_rate_plan(
                    rate=float(rate), duration_seconds=duration,
                    seed=seed, prefix=f"capacity:{profile}:{replicas}:{rate}",
                )
                case_id = f"{profile}__r{replicas}__q{float(rate):g}"
                case_dir = run_dir / case_id
                case_dir.mkdir()
                plan_path = case_dir / "request_plan.jsonl"
                plan_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in plan))
                started = time.perf_counter()
                stop = asyncio.Event()
                monitor_task = asyncio.create_task(_monitor(collector, stop, case_dir / "metrics.jsonl"))
                try:
                    replay_summary = await replay(
                        plan, url=url + "/infer", output=case_dir / "requests.jsonl",
                        timeout_seconds=float(profile_cfg["client_timeout_seconds"]),
                        connection_limit=int(profile_cfg["connection_limit"]), force_close_connections=True,
                    )
                finally:
                    stop.set()
                samples = await monitor_task
                elapsed = time.perf_counter() - started
                state = collector.collect(remaining_budget_ratio=1.0, remaining_horizon_ratio=1.0)
                all_samples = samples + [state.as_dict()]
                raw = lambda name: [float(item["fields"][name]["raw"]) for item in all_samples]
                completed_rates = [value for value in raw("completed_request_rate") if value > 0]
                row = {
                    "schema": "dap.k8s.capacity_point.v1", "case_id": case_id,
                    "profile": profile, "replicas": int(replicas), "offered_rps": float(rate),
                    "duration_seconds": duration, "observed_wall_seconds": elapsed,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "replay": replay_summary, "state": state.as_dict(),
                    "ready_replicas": state.ready_replicas,
                    "application_completed_rps": float(np.median(completed_rates)) if completed_rates else 0.0,
                    "application_peak_completed_rps": max(completed_rates, default=0.0),
                    "application_queue_depth": max(raw("queue_depth"), default=0.0),
                    "application_p95_seconds": float(np.quantile(raw("p95_latency_seconds"), 0.95)),
                    "application_cpu_utilization": float(np.median(raw("cpu_utilization"))),
                    "metric_samples": len(all_samples),
                }
                write_json(case_dir / "result.json", row)
                rows.append(row)
                seed += 1
    write_json(run_dir / "manifest.json", {
        "schema": "dap.k8s.capacity_calibration.v1", "status": "completed",
        "calibration_protocol_revision": 3,
        "config_path": config["_config_path"], "points": len(rows), "rows": rows,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    kube.scale(1)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = asyncio.run(run(load_config(args.config)))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
