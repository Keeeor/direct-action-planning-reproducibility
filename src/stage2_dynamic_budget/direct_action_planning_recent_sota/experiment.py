from __future__ import annotations

from datetime import datetime, timezone
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

from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.evaluation import (
    evaluate_agent,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.baseline_adapter import (
    calibrated_baseline_runtime,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.data import (
    development_input_hashes,
    load_development_trace_dataset,
)
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .lcpo import LCPOConfig, train_lcpo


REQUIRED_KEYS = {
    "tier",
    "manifest_schema",
    "matrix_row_id",
    "datasets",
    "horizon",
    "budgets",
    "seeds",
    "seed_role",
    "gamma",
    "training_steps",
    "batch_size",
    "evaluation_split",
    "evaluation_episodes_per_domain",
    "capacity_training_quantile",
    "evaluation_seed_offset",
    "upstream_commit",
    "upstream_archive_sha256",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"missing LCPO protocol keys: {sorted(missing)}")
    if str(config["evaluation_split"]) != "validation_eval":
        raise ValueError("development evaluation must use validation_eval")
    if str(config["upstream_commit"]) != "aeb93563cbd6ede13e145381ec3f90e0ac840c41":
        raise ValueError("LCPO upstream commit changed")
    if str(config["upstream_archive_sha256"]) != "80995fc206277a7d01f7e13d3c6b021b77b2211d507a25fc6217a65c5b342008":
        raise ValueError("LCPO upstream archive hash changed")
    lcpo_config_from_dict(config).validate()
    return config


def lcpo_config_from_dict(config: dict[str, Any]) -> LCPOConfig:
    return LCPOConfig(
        total_steps=int(config["training_steps"]),
        batch_size=int(config["batch_size"]),
        hidden_dim=int(config.get("hidden_dim", 64)),
        gamma=float(config["gamma"]),
        gae_lambda=float(config.get("gae_lambda", 0.9)),
        policy_learning_rate=float(config.get("policy_learning_rate", 4.0e-4)),
        value_learning_rate=float(config.get("value_learning_rate", 1.0e-3)),
        entropy_max=float(config.get("entropy_max", 0.03)),
        auto_target_entropy=float(config.get("auto_target_entropy", 0.1)),
        entropy_learning_rate=float(config.get("entropy_learning_rate", 1.0e-3)),
        recent_kl_limit=float(config.get("recent_kl_limit", 0.1)),
        anchor_kl_limit=float(config.get("anchor_kl_limit", 1.0e-4)),
        damping=float(config.get("damping", 0.1)),
        cg_steps=int(config.get("cg_steps", 15)),
        max_backtracks=int(config.get("max_backtracks", 10)),
        recent_window=int(config.get("recent_window", 200)),
        reservoir_capacity=int(config.get("reservoir_capacity", 1_024)),
        ood_log_likelihood_threshold=float(
            config.get("ood_log_likelihood_threshold", -6.0)
        ),
        context_indices=tuple(int(value) for value in config.get("context_indices", (0, 1, 10))),
        covariance_ridge=float(config.get("covariance_ridge", 1.0e-3)),
        solve_dual=bool(config.get("solve_dual", False)),
        capacity_training_quantile=float(config["capacity_training_quantile"]),
    )


def _run_dir(
    root: Path,
    config: dict[str, Any],
    dataset: str,
    budget: float,
    seed: int,
    attempt: int,
) -> Path:
    run_id = f"{config['tier']}__{dataset}__b{budget:.0f}__s{seed}"
    if attempt:
        run_id += f"__a{attempt}"
    return root / "results/direct_action_planning_recent_sota" / str(config["tier"]) / dataset / run_id


def run_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    attempt: int = 0,
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    if dataset_name not in tuple(str(value) for value in config["datasets"]):
        raise ValueError("unregistered dataset")
    if float(budget) not in tuple(float(value) for value in config["budgets"]):
        raise ValueError("unregistered budget")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise ValueError("unregistered seed")
    run_dir = _run_dir(root, config, dataset_name, float(budget), int(seed), int(attempt))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"LCPO run is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "config.json", {**config, "dataset": dataset_name, "budget": budget, "seed": seed})
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("LCPO training started\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    wall_started = time.perf_counter()
    try:
        set_global_seed(int(seed), torch_threads=1)
        dataset = load_development_trace_dataset(root, dataset_name, horizon=int(config["horizon"]))
        result = train_lcpo(
            dataset,
            horizon=int(config["horizon"]),
            budget=float(budget),
            seed=int(seed),
            config=lcpo_config_from_dict(config),
        )
        with calibrated_baseline_runtime(
            quantile=float(config["capacity_training_quantile"])
        ):
            episodes, steps = evaluate_agent(
                dataset,
                result.agent,
                split=str(config["evaluation_split"]),
                horizon=int(config["horizon"]),
                budget=float(budget),
                seed=int(seed) + int(config["evaluation_seed_offset"]),
                episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
                gamma=float(config["gamma"]),
            )
        metrics = pd.DataFrame(episodes)
        step_frame = pd.DataFrame(steps)
        metrics["training_seed"] = int(seed)
        step_frame["training_seed"] = int(seed)
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise FloatingPointError("LCPO evaluation metrics are not finite")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        step_frame.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        metrics.drop(columns=["decision_ms_mean", "decision_ms_p95"], errors="ignore").to_json(
            run_dir / "raw_metrics.jsonl", orient="records", lines=True
        )
        step_frame.drop(columns=["decision_ms"], errors="ignore").to_csv(
            run_dir / "predictions.csv", index=False
        )
        write_json(
            run_dir / "training.json",
            {
                "elapsed_seconds": result.elapsed_seconds,
                "diagnostics": result.diagnostics,
                "history": result.history,
            },
        )
        torch.save(
            {
                "policy": result.policy_state,
                "value": result.value_state,
                "hidden_dim": int(config.get("hidden_dim", 64)),
                "budget": float(budget),
            },
            run_dir / "models.pt",
        )
        write_json(
            run_dir / "guardrails.json",
            {
                "budget_overspend_max": float(metrics.budget_overspend.max()),
                "budget_safe": bool(metrics.budget_overspend.max() <= 1.0e-8),
                "all_metrics_finite": True,
                "formal_test_accessed": False,
                "future_features_used": False,
            },
        )
        write_json(
            run_dir / "test_evidence.json",
            {
                "status": "PASS",
                "evaluation_split": str(config["evaluation_split"]),
                "episode_rows": int(len(metrics)),
                "step_rows": int(len(step_frame)),
                "method_count": 1,
            },
        )
        write_json(
            run_dir / "runtime.json",
            {
                "wall_seconds": float(time.perf_counter() - wall_started),
                "training_seconds": float(result.elapsed_seconds),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        )
        code_root = root / "src/stage2_dynamic_budget/direct_action_planning_recent_sota"
        write_json(
            run_dir / "code_snapshot.json",
            {
                "code": str(code_root.relative_to(root)),
                "sha256": sha256_tree(code_root),
                "config_sha256": sha256_file(config_path),
            },
        )
        (run_dir / "stdout.log").write_text("LCPO training completed\n", encoding="utf-8")
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
                "development_only": True,
                "formal_test_accessed": False,
                "dataset": dataset_name,
                "budget": float(budget),
                "training_seed": int(seed),
                "attempt": int(attempt),
                "started_at": started_at,
                "ended_at": ended_at,
                "config_sha256": sha256_file(config_path),
                "input_hashes": development_input_hashes(root, dataset_name),
                "artifacts": artifacts,
            },
        )
        return run_dir
    except Exception:
        failure = {
            "status": "failed",
            "formal_test_accessed": False,
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        (run_dir / "stderr.log").write_text(failure["traceback"], encoding="utf-8")
        raise


def run_matrix(project_root: str | Path, config_path: str | Path) -> list[Path]:
    root = Path(project_root).resolve()
    config = load_protocol(config_path)
    outputs: list[Path] = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                existing = _run_dir(root, config, str(dataset), float(budget), int(seed), 0)
                manifest_path = existing / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("status") == "completed" and manifest.get("formal_test_accessed") is False:
                        outputs.append(existing)
                        continue
                outputs.append(
                    run_unit(
                        root,
                        config_path,
                        dataset_name=str(dataset),
                        budget=float(budget),
                        seed=int(seed),
                    )
                )
    return outputs
