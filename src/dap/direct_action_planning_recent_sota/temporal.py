from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from dap.direct_action_planning_dataset_benchmark.evaluation import (
    evaluate_agent,
)
from dap.direct_action_planning_dataset_validation.data import (
    TraceDataset,
    load_trace_dataset,
)
from dap.direct_action_planning_paper_closure.baseline_adapter import (
    calibrated_baseline_runtime,
)
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from dap.utils.seed import set_global_seed

from .experiment import load_protocol
from .lcpo import LCPOAgent, LCPOPolicy


def _run_dir(root: Path, tier: str, dataset: str, budget: float, seed: int) -> Path:
    run_id = f"{tier}__{dataset}__b{budget:.0f}__s{seed}"
    return root / "results/direct_action_planning_recent_sota" / tier / dataset / run_id


def _cells(config: dict[str, Any]) -> set[tuple[str, float, int]]:
    return {
        (str(dataset), float(budget), int(seed))
        for dataset in config["datasets"]
        for budget in config["budgets"]
        for seed in config["seeds"]
    }


def _development_inventory(root: Path, development_config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    tier = str(development_config["tier"])
    for dataset, budget, seed in sorted(_cells(development_config)):
        directory = _run_dir(root, tier, dataset, budget, seed)
        manifest = directory / "manifest.json"
        checkpoint = directory / "models.pt"
        if not manifest.exists() or not checkpoint.exists():
            raise ValueError(f"missing LCPO development bundle: {directory}")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("status") != "completed" or data.get("formal_test_accessed") is not False:
            raise ValueError(f"invalid LCPO development manifest: {manifest}")
        paths.extend((manifest, checkpoint))
    return paths


def _inventory_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n".encode()
        )
    return "sha256:" + digest.hexdigest()


def freeze_contract(
    project_root: str | Path,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(project_root).resolve()
    template_path = Path(template_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"LCPO temporal contract is append-only: {output_path}")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    development_config_path = root / str(template["development_config"])
    development = load_protocol(development_config_path)
    for key in ("datasets", "budgets", "seeds", "horizon", "gamma", "capacity_training_quantile"):
        if template[key] != development[key]:
            raise ValueError(f"temporal template differs from development config: {key}")
    paths = _development_inventory(root, development)
    code_root = root / "src/dap/direct_action_planning_recent_sota"
    contract = {
        "schema": "dap.dap_recent_sota.temporal_contract.v1",
        "status": "frozen_before_lcpo_test_access",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "template_path": str(template_path.relative_to(root)),
        "template_sha256": sha256_file(template_path),
        "config": template,
        "development_config_sha256": sha256_file(development_config_path),
        "code_tree_sha256": sha256_tree(code_root),
        "source_inventory_sha256": _inventory_hash(paths, root),
        "source_artifact_count": len(paths),
        "test_arrays_loaded": False,
        "selection_performed": False,
        "post_hoc_extension": True,
        "project_wide_test_previously_accessed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return output_path


def _test_hashes(dataset: TraceDataset) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for domain in dataset.domain_names:
        values = dataset.domains[domain]["test"]
        digest = hashlib.sha256()
        digest.update(f"{dataset.name}/{domain}/test".encode())
        digest.update(values.astype(np.float64).tobytes(order="C"))
        hashes[domain] = "sha256:" + digest.hexdigest()
    return hashes


def _load_agent(checkpoint_path: Path, *, budget: float) -> LCPOAgent:
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    policy = LCPOPolicy(hidden_dim=int(checkpoint["hidden_dim"]))
    policy.load_state_dict(checkpoint["policy"], strict=True)
    return LCPOAgent(policy, budget=float(budget))


def run_test_unit(
    project_root: str | Path,
    contract_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_lcpo_test_access":
        raise ValueError("invalid LCPO temporal contract")
    config = contract["config"]
    if (dataset_name, float(budget), int(seed)) not in _cells(config):
        raise ValueError("unregistered LCPO temporal cell")
    development_config = load_protocol(root / str(config["development_config"]))
    if _inventory_hash(_development_inventory(root, development_config), root) != contract["source_inventory_sha256"]:
        raise ValueError("LCPO checkpoint inventory changed after freeze")
    code_root = root / "src/dap/direct_action_planning_recent_sota"
    if sha256_tree(code_root) != contract["code_tree_sha256"]:
        raise ValueError("LCPO source changed after freeze")
    tier = str(config["tier"])
    run_id = f"{tier}__{dataset_name}__b{budget:.0f}__s{seed}"
    if attempt:
        run_id += f"__a{attempt}"
    run_dir = root / "results/direct_action_planning_recent_sota" / tier / dataset_name / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"LCPO test run is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.json", {**config, "dataset": dataset_name, "budget": budget, "seed": seed})
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("LCPO temporal evaluation started\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    test_loaded = False
    started = time.perf_counter()
    try:
        source = _run_dir(root, str(development_config["tier"]), dataset_name, budget, seed)
        agent = _load_agent(source / "models.pt", budget=budget)
        set_global_seed(int(seed), torch_threads=1)
        dataset = load_trace_dataset(root, dataset_name)
        test_loaded = True
        with calibrated_baseline_runtime(
            quantile=float(config["capacity_training_quantile"])
        ):
            episodes, steps = evaluate_agent(
                dataset,
                agent,
                split="test",
                horizon=int(config["horizon"]),
                budget=float(budget),
                seed=int(seed) + int(config["test_seed_offset"]),
                episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
                gamma=float(config["gamma"]),
            )
        metrics = pd.DataFrame(episodes)
        step_frame = pd.DataFrame(steps)
        metrics["training_seed"] = int(seed)
        step_frame["training_seed"] = int(seed)
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise FloatingPointError("LCPO test metrics are not finite")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        step_frame.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        metrics.drop(columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore").to_json(
            run_dir / "raw_metrics.jsonl", orient="records", lines=True
        )
        step_frame.drop(columns=["decision_ms"], errors="ignore").to_csv(
            run_dir / "predictions.csv", index=False
        )
        write_json(run_dir / "input_hashes.json", _test_hashes(dataset))
        write_json(
            run_dir / "source_checkpoint.json",
            {
                "models_sha256": sha256_file(source / "models.pt"),
                "manifest_sha256": sha256_file(source / "manifest.json"),
            },
        )
        write_json(
            run_dir / "guardrails.json",
            {
                "budget_overspend_max": float(metrics.budget_overspend.max()),
                "budget_safe": bool(metrics.budget_overspend.max() <= 1.0e-8),
                "all_metrics_finite": True,
                "training_or_selection_performed": False,
            },
        )
        write_json(
            run_dir / "test_evidence.json",
            {
                "status": "PASS",
                "evaluation_split": "test",
                "current_checkpoint_first_test": True,
                "project_wide_test_previously_accessed": True,
                "post_hoc_extension": True,
                "episode_rows": len(metrics),
                "step_rows": len(step_frame),
                "method_count": 1,
            },
        )
        write_json(
            run_dir / "runtime.json",
            {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        )
        (run_dir / "stdout.log").write_text("LCPO temporal evaluation completed\n", encoding="utf-8")
        ended_at = datetime.now(timezone.utc).isoformat()
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(
            run_dir / "manifest.json",
            {
                "schema": str(config["manifest_schema"]),
                "status": "completed",
                "dataset": dataset_name,
                "budget": float(budget),
                "training_seed": int(seed),
                "attempt": int(attempt),
                "started_at": started_at,
                "ended_at": ended_at,
                "training_performed": False,
                "selection_performed": False,
                "formal_test_accessed": True,
                "post_hoc_extension": True,
                "contract_sha256": sha256_file(contract_path),
                "artifacts": artifacts,
            },
        )
        return run_dir
    except Exception:
        failure = {
            "status": "failed",
            "test_data_loaded": test_loaded,
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        (run_dir / "stderr.log").write_text(failure["traceback"], encoding="utf-8")
        raise


def run_test_matrix(project_root: str | Path, contract_path: str | Path) -> list[Path]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    return [
        run_test_unit(
            project_root,
            contract_path,
            dataset_name=dataset,
            budget=budget,
            seed=seed,
        )
        for dataset, budget, seed in sorted(_cells(contract["config"]))
    ]
