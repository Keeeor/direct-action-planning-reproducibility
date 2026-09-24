from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.common import fixed_rate_plan, node_url
from controller.config import ControllerConfig
from controller.dap_controller import DAPController
from controller.kube_client import KubectlClient
from experiments.runtime import wait_http
from workload.replay_driver import replay


async def run(config: dict, config_path: Path) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / "dap_smoke" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    control_values = {
        "kubernetes": config["kubernetes"], "controller": config["controller"],
        "paths": {
            "checkpoint": config["paths"]["checkpoint"],
            "system_model": config["paths"]["system_model"],
            "result_directory": str((run_dir / "controller").resolve()),
        },
    }
    controller_config = ControllerConfig.from_mapping(control_values, base_directory=config_path.parent)
    kube = KubectlClient(
        context=controller_config.context, namespace=controller_config.namespace,
        deployment=controller_config.deployment,
    )
    if kube.autoscaler_conflicts():
        raise RuntimeError("DAP smoke requires no autoscaler conflict")
    kube.set_profile(controller_config.profile)
    kube.scale(1)
    kube.rollout_status()
    url = node_url(kube, int(config["kubernetes"].get("node_port", 30080)))
    wait_http(url + "/healthz")
    duration = controller_config.horizon_steps * controller_config.control_interval_seconds
    plan = fixed_rate_plan(
        rate=float(config["workload"]["rps"]), duration_seconds=duration,
        seed=int(config["seed"]), prefix="dap-smoke",
    )
    plan_path = run_dir / "request_plan.jsonl"
    plan_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in plan))
    controller = DAPController(controller_config)
    controller_task = asyncio.create_task(asyncio.to_thread(controller.run))
    # The controller has no startup subscription; use a fixed trace plan and
    # let the first step observe the real initially sparse load window.
    replay_summary = await replay(
        plan, url=url + "/infer", output=run_dir / "requests.jsonl",
        timeout_seconds=float(config["workload"]["client_timeout_seconds"]),
        connection_limit=int(config["workload"]["connection_limit"]), force_close_connections=True,
    )
    controller_result = await controller_task
    result = {
        "schema": "dap.k8s.dap_smoke.v1", "status": "pass" if controller_result["budget_violation_seconds"] <= 1.0e-9 else "fail",
        "run_directory": str(run_dir.relative_to(ROOT)), "replay": replay_summary,
        "controller": controller_result,
        "config_sha256": "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = asyncio.run(run(config, args.config.resolve()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
