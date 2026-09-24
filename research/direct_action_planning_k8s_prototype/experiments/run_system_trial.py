from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.autoscaler_runtime import NativeAutoscalerRuntime
from baselines.policies import MPCPolicy, StaticPolicy, ThresholdPolicy
from baselines.runtime_controller import BaselineRuntimeController
from controller.checkpoint_loader import load_checkpoint
from controller.config import ControllerConfig
from controller.dap_controller import DAPController
from controller.kube_client import KubectlClient
from controller.system_model import StructuredSystemModel
from experiments.cleanup import cleanup_target
from experiments.monitor import LiveMetricMonitor
from experiments.runtime import wait_http
from workload.replay_driver import load_plan, replay


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _image_digest(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _tree_sha256(root: Path) -> str:
    """Content hash for a source snapshot when the surrounding project has no Git repo."""

    digest = hashlib.sha256()
    roots = (
        "app", "controller", "baselines", "workload", "calibration", "experiments",
        "analysis", "kubernetes", "tests",
    )
    for relative_root in roots:
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    for path in (root / "Makefile",):
        if path.exists():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _kubernetes_manifest_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted((root / "kubernetes").rglob("*.yaml"))
        if path.is_file()
    }


def _kubectl_version(context: str) -> str:
    result = subprocess.run(
        ["kubectl", "--context", context, "version", "--client", "-o", "yaml"],
        text=True,
        capture_output=True,
        check=False,
    )
    return (result.stdout or result.stderr).strip()[:4000]


def _controller_config(
    config: dict, config_path: Path, run_dir: Path, profile: str, budget: float
) -> ControllerConfig:
    data = {
        "kubernetes": config["kubernetes"],
        "controller": {
            **config["controller_defaults"],
            **config["profiles"][profile],
            "profile": profile,
            "total_budget_seconds": budget,
        },
        "paths": {
            "checkpoint": config["profiles"][profile]["checkpoint"],
            "system_model": config["paths"]["system_model"],
            "result_directory": str((run_dir / "controller").resolve()),
        },
        "actions": config.get("actions"),
    }
    return ControllerConfig.from_mapping(data, base_directory=config_path.parent)


def _make_controller(method: str, controller_config: ControllerConfig, config: dict):
    if method == "dap":
        return DAPController(controller_config)
    kube = KubectlClient(
        context=controller_config.context,
        namespace=controller_config.namespace,
        deployment=controller_config.deployment,
    )
    model = StructuredSystemModel.load(
        controller_config.system_model_path,
        controller_config.profile,
        slo_seconds=controller_config.slo_seconds,
    )
    if method in {"hpa", "keda"}:
        return NativeAutoscalerRuntime(
            kind=method,
            kube=kube,
            system_model=model,
            result_directory=controller_config.result_directory,
            total_budget_seconds=controller_config.total_budget_seconds,
            horizon_steps=controller_config.horizon_steps,
            control_interval_seconds=controller_config.control_interval_seconds,
            capacity_per_pod_rps=controller_config.capacity_per_pod_rps,
            native_scale_down_reserve_seconds=float(
                config["controller_defaults"].get("native_scale_down_reserve_seconds", 45.0)
            ),
        )
    if method == "static":
        policy = StaticPolicy(
            mapper=controller_config.action_mapper,
            replicas=int(config["baseline_parameters"]["static_replicas"]),
            control_interval_seconds=controller_config.control_interval_seconds,
            guard_seconds=model.scale_down_guard_seconds,
        )
    elif method == "threshold":
        values = config["baseline_parameters"]["threshold"]
        policy = ThresholdPolicy(
            mapper=controller_config.action_mapper,
            control_interval_seconds=controller_config.control_interval_seconds,
            guard_seconds=model.scale_down_guard_seconds,
            slo_seconds=controller_config.slo_seconds,
            queue_small=float(values["queue_small"]),
            queue_medium=float(values["queue_medium"]),
            queue_large=float(values["queue_large"]),
        )
    elif method == "mpc_4":
        policy = MPCPolicy(
            checkpoint=load_checkpoint(controller_config.checkpoint_path),
            system_model=model,
            mapper=controller_config.action_mapper,
            total_budget_seconds=controller_config.total_budget_seconds,
            control_interval_seconds=controller_config.control_interval_seconds,
            horizon=4,
        )
    else:
        raise ValueError(f"unsupported external controller method: {method}")
    return BaselineRuntimeController(
        kube=kube,
        policy=policy,
        system_model=model,
        profile=controller_config.profile,
        result_directory=controller_config.result_directory,
        total_budget_seconds=controller_config.total_budget_seconds,
        horizon_steps=controller_config.horizon_steps,
        control_interval_seconds=controller_config.control_interval_seconds,
        capacity_per_pod_rps=controller_config.capacity_per_pod_rps,
    )


def _prepare(kube: KubectlClient, profile: str) -> str:
    cleanup_target(kube)
    kube.set_profile(profile)
    kube.scale(1)
    kube.rollout_restart()
    kube.rollout_status()
    return f"http://{kube.node_internal_ip()}:30080"


async def _finish_monitor(
    task: asyncio.Task | None, stop: threading.Event
) -> dict[str, Any] | None:
    if task is None:
        return None
    stop.set()
    return (await task).as_dict()


async def run_trial(
    *,
    config: dict,
    config_path: Path,
    method: str,
    profile: str,
    budget: float,
    plan_path: Path,
    run_dir: Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one append-only, real-system trial and retain its raw evidence."""

    if run_dir.exists():
        existing = {path.name for path in run_dir.iterdir()}
        if existing - {"stdout.log", "stderr.log"}:
            raise FileExistsError(f"append-only run directory is not empty: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": "dap.k8s.system_trial.v2",
        "status": "running",
        "method": method,
        "profile": profile,
        "budget_seconds": budget,
        "config_sha256": _sha256(config_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "software": {"platform": platform.platform()},
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "source_tree_sha256": _tree_sha256(ROOT),
        "kubernetes_manifest_hashes": _kubernetes_manifest_hashes(ROOT),
        **(run_metadata or {}),
    }
    kube: KubectlClient | None = None
    monitor_stop = threading.Event()
    monitor_task: asyncio.Task | None = None
    monitor_summary: dict[str, Any] | None = None
    cleanup_summary: dict[str, Any] | None = None
    try:
        controller_config = _controller_config(config, config_path, run_dir, profile, budget)
        kube = KubectlClient(
            context=controller_config.context,
            namespace=controller_config.namespace,
            deployment=controller_config.deployment,
        )
        url = _prepare(kube, profile)
        wait_http(url + "/healthz")
        controller = _make_controller(method, controller_config, config)
        system_model = StructuredSystemModel.load(
            controller_config.system_model_path,
            profile,
            slo_seconds=controller_config.slo_seconds,
        )
        plan = load_plan(plan_path)
        plan_manifest_path = plan_path.with_suffix(plan_path.suffix + ".manifest.json")
        plan_manifest = (
            json.loads(plan_manifest_path.read_text(encoding="utf-8"))
            if plan_manifest_path.exists()
            else {}
        )
        shutil.copy2(config_path, run_dir / "config_snapshot.yaml")
        if plan_manifest_path.exists():
            shutil.copy2(plan_manifest_path, run_dir / "request_plan.manifest.json")
        manifest.update({
            "plan_path": str(plan_path),
            "plan_sha256": _sha256(plan_path),
            "plan_manifest_sha256": _sha256(plan_manifest_path) if plan_manifest_path.exists() else None,
            "plan_seed": plan_manifest.get("seed"),
            "plan_split": plan_manifest.get("split"),
            "checkpoint_sha256": _sha256(controller_config.checkpoint_path),
            "system_model_sha256": _sha256(controller_config.system_model_path),
            "image_digest": _image_digest(str(config["kubernetes"]["image"])),
            "kubectl_client_version": _kubectl_version(controller_config.context),
            "slo_seconds": controller_config.slo_seconds,
        })
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        monitor = LiveMetricMonitor(
            kube=kube,
            system_model=system_model,
            capacity_per_pod_rps=controller_config.capacity_per_pod_rps,
            output=run_dir / "live_metrics.jsonl",
            interval_seconds=float(config.get("monitor", {}).get("interval_seconds", 1.0)),
        )
        monitor_task = asyncio.create_task(asyncio.to_thread(monitor.run, monitor_stop))
        controller_task = asyncio.create_task(asyncio.to_thread(controller.run, prepare=False))
        replay_task = asyncio.create_task(replay(
            plan,
            url=url + "/infer",
            output=run_dir / "requests.jsonl",
            timeout_seconds=float(config["workload"][profile]["client_timeout_seconds"]),
            connection_limit=int(config["workload"][profile]["connection_limit"]),
            force_close_connections=True,
        ))
        done, _ = await asyncio.wait(
            {controller_task, replay_task}, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            error = task.exception()
            if error is not None:
                for pending in (controller_task, replay_task):
                    if pending is not task:
                        pending.cancel()
                await asyncio.gather(controller_task, replay_task, return_exceptions=True)
                raise error
        replay_summary = await replay_task
        controller_result = await controller_task
        monitor_summary = await _finish_monitor(monitor_task, monitor_stop)
        monitor_task = None
        cleanup_summary = cleanup_target(kube).__dict__
        result = {
            "schema": "dap.k8s.system_trial_result.v2",
            "status": "completed",
            "method": method,
            "profile": profile,
            "budget_seconds": budget,
            "replay": replay_summary,
            "controller": controller_result,
            "monitor": monitor_summary or {},
            "cleanup": cleanup_summary,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest.update({
            "status": "completed",
            "ended_at": result["ended_at"],
            "monitor": monitor_summary,
            "cleanup": cleanup_summary,
        })
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "failure.json").write_text(json.dumps({
            "schema": "dap.k8s.system_trial_failure.v1",
            "error": manifest["failure"],
            "at": manifest["ended_at"],
            "formal_result": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        if monitor_task is not None:
            try:
                monitor_summary = await _finish_monitor(monitor_task, monitor_stop)
            except Exception:
                monitor_summary = None
        if kube is not None and cleanup_summary is None:
            try:
                cleanup_summary = cleanup_target(kube).__dict__
            except Exception:
                cleanup_summary = None
        if monitor_summary is not None:
            manifest["monitor"] = monitor_summary
        if cleanup_summary is not None:
            manifest["cleanup"] = cleanup_summary
        manifest.setdefault("ended_at", datetime.now(timezone.utc).isoformat())
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("static", "threshold", "hpa", "keda", "mpc_4", "dap"), required=True
    )
    parser.add_argument("--profile", choices=("azure_http", "gentd_inference"), required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = asyncio.run(run_trial(
        config=config,
        config_path=args.config.resolve(),
        method=args.method,
        profile=args.profile,
        budget=args.budget,
        plan_path=args.plan.resolve(),
        run_dir=args.run_directory.resolve(),
    ))
    print(json.dumps({
        "status": result["status"],
        "method": result["method"],
        "budget_violation_seconds": result["controller"]["budget_violation_seconds"],
        "completed": result["replay"]["completed"],
    }, sort_keys=True))
    return 0 if result["controller"]["budget_violation_seconds"] <= 1.0e-9 else 2


if __name__ == "__main__":
    raise SystemExit(main())
