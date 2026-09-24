from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.kube_client import KubectlClient

from calibration.common import ROOT, load_config, reset_profile, timestamp_id, write_json


def run(config: dict) -> Path:
    kube_cfg = config["kubernetes"]
    kube = KubectlClient(
        context=str(kube_cfg["context"]), namespace=str(kube_cfg["namespace"]),
        deployment=str(kube_cfg.get("deployment", "dap-worker")),
    )
    if kube.autoscaler_conflicts():
        raise RuntimeError("startup calibration requires all autoscalers disabled")
    run_dir = ROOT / "results" / "calibration" / "startup" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for profile in config["startup_profiles"]:
        for repeat in range(int(config["startup_repeats"])):
            reset_profile(kube, profile, 1)
            before = {pod["uid"]: pod for pod in kube.worker_pods()}
            for target in config["startup_targets"]:
                target = int(target)
                started = time.perf_counter()
                evidence = kube.scale(target)
                timeline = []
                deadline = started + float(config["startup_timeout_seconds"])
                while time.perf_counter() < deadline:
                    status = kube.deployment_status()
                    pods = kube.worker_pods()
                    timeline.append({
                        "offset_seconds": time.perf_counter() - started,
                        "status": status, "pods": pods,
                    })
                    if status["ready_replicas"] >= target:
                        break
                    time.sleep(float(config["startup_poll_seconds"]))
                status = kube.deployment_status()
                if status["ready_replicas"] < target:
                    raise TimeoutError(f"target {target} not Ready during startup calibration")
                new_pods = [pod for pod in kube.worker_pods() if pod["uid"] not in before]
                rows.append({
                    "schema": "dap.k8s.startup_delay_point.v1", "profile": profile,
                    "repeat": repeat, "target_replicas": target,
                    "from_replicas": len(before), "scale_api_latency_seconds": evidence.latency_seconds,
                    "all_ready_latency_seconds": time.perf_counter() - started,
                    "new_pods": new_pods, "timeline": timeline,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                before = {pod["uid"]: pod for pod in kube.worker_pods()}
            kube.scale(1)
            kube.wait_ready(1)
    write_json(run_dir / "manifest.json", {
        "schema": "dap.k8s.startup_delay_calibration.v1", "status": "completed",
        "calibration_protocol_revision": 3,
        "config_path": config["_config_path"], "rows": rows,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = run(load_config(args.config))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
