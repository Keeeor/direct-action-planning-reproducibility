from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.budgeted import (
    train_budgeted_fitted_q,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.cpo import (
    CPOConfig,
    train_cpo,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.evaluation import (
    evaluate_agent,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.rl import (
    RLTrainConfig,
    train_double_dqn,
    train_policy,
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

from .baseline_adapter import calibrated_baseline_runtime
from .experiment import _write_standard_bundle


SUPPORTED_METHODS = (
    "double_dqn",
    "a2c",
    "ppo",
    "ppo_lagrangian",
    "pid_lagrangian",
    "p3o",
    "cpo",
    "budgeted_fitted_q",
)

CONSTRAINED_POLICY_METHODS = {
    "ppo_lagrangian",
    "pid_lagrangian",
    "p3o",
    "cpo",
}


def _policy_train_config(
    method: str,
    training_steps: int,
    hidden_dim: int,
    config: dict | None = None,
) -> RLTrainConfig:
    options = config or {}
    return RLTrainConfig(
        total_steps=int(training_steps),
        rollout_steps=int(options.get("rollout_steps", 512)),
        update_epochs=int(options.get("update_epochs", 4)),
        minibatch_size=int(options.get("minibatch_size", 256)),
        hidden_dim=int(hidden_dim),
        cost_value_coef=(0.0 if method in {"a2c", "ppo"} else 0.5),
    )


def load_baseline_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
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
        "methods",
        "evaluation_split",
        "evaluation_episodes_per_domain",
        "capacity_training_quantile",
        "evaluation_seed_offset",
        "primary_methods",
        "supplementary_methods",
        "slo_constraint_rate",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing baseline protocol keys: {sorted(missing)}")
    methods = tuple(str(value) for value in config["methods"])
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("baseline methods must be non-empty and unique")
    unsupported = set(methods) - set(SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(f"unsupported baseline methods: {sorted(unsupported)}")
    primary = tuple(str(value) for value in config["primary_methods"])
    supplementary = tuple(str(value) for value in config["supplementary_methods"])
    if set(primary) & set(supplementary):
        raise ValueError("primary and supplementary methods must be disjoint")
    if tuple(primary + supplementary) != methods:
        raise ValueError(
            "methods must equal primary_methods followed by supplementary_methods"
        )
    if str(config["evaluation_split"]) != "validation_eval":
        raise ValueError("baseline evaluation_split must be validation_eval")
    if str(config["seed_role"]) not in {"fixed_repro", "randomness_estimation"}:
        raise ValueError("seed_role must be fixed_repro or randomness_estimation")
    quantile = float(config["capacity_training_quantile"])
    if not np.isfinite(quantile) or not 0.0 < quantile <= 1.0:
        raise ValueError("capacity_training_quantile must be in (0, 1]")
    risk_rate = float(config["slo_constraint_rate"])
    if not np.isfinite(risk_rate) or not 0.0 < risk_rate <= 1.0:
        raise ValueError("slo_constraint_rate must be in (0, 1]")
    return config


def _run_dir(root: Path, config: dict, dataset: str, budget: float, seed: int, attempt: int) -> Path:
    base = f"{config['tier']}__{dataset}__b{budget:.0f}__s{seed}"
    run_id = base if attempt == 0 else f"{base}__a{attempt}"
    return (
        root
        / "results/direct_action_planning_dataset_specific_stabilization"
        / str(config["tier"])
        / dataset
        / run_id
    )


def _train_method(method: str, dataset, config: dict, budget: float, seed: int):
    training_steps = int(config["training_steps"])
    common = _policy_train_config(
        method,
        training_steps,
        int(config.get("hidden_dim", 64)),
        config,
    )
    if method == "double_dqn":
        return train_double_dqn(
            dataset,
            horizon=int(config["horizon"]),
            budget=budget,
            seed=seed,
            total_steps=training_steps,
            hidden_dim=int(config.get("hidden_dim", 64)),
        )
    if method in {"a2c", "ppo", "ppo_lagrangian", "pid_lagrangian", "p3o"}:
        return train_policy(
            dataset,
            horizon=int(config["horizon"]),
            budget=budget,
            seed=seed,
            variant=method,
            config=common,
        )
    if method == "cpo":
        return train_cpo(
            dataset,
            horizon=int(config["horizon"]),
            budget=budget,
            seed=seed,
            config=CPOConfig(
                total_steps=training_steps,
                rollout_steps=int(config.get("rollout_steps", 512)),
                critic_epochs=int(config.get("update_epochs", 4)),
                hidden_dim=int(config.get("hidden_dim", 64)),
            ),
        )
    if method == "budgeted_fitted_q":
        agent, history, elapsed = train_budgeted_fitted_q(
            dataset,
            horizon=int(config["horizon"]),
            budget=budget,
            seed=seed,
            episodes_per_domain=int(config.get("branch_episodes_per_domain", 16)),
            iterations=int(config.get("bftq_iterations", 8)),
            epochs_per_iteration=int(config.get("bftq_epochs_per_iteration", 2)),
            hidden_dim=int(config.get("hidden_dim", 64)),
        )
        return type(
            "FittedQResult",
            (),
            {
                "agent": agent,
                "history": history,
                "elapsed_seconds": elapsed,
                "diagnostics": {},
            },
        )()
    raise AssertionError(method)


def run_baseline_unit(
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
    config = load_baseline_protocol(config_path)
    if dataset_name not in tuple(str(value) for value in config["datasets"]):
        raise ValueError(f"dataset is not registered: {dataset_name}")
    if not any(np.isclose(float(budget), float(value)) for value in config["budgets"]):
        raise ValueError(f"budget is not registered: {budget}")
    if int(seed) not in tuple(int(value) for value in config["seeds"]):
        raise ValueError(f"seed is not registered: {seed}")
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    run_dir = _run_dir(root, config, dataset_name, float(budget), int(seed), attempt)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {**config, "dataset": dataset_name, "budget": float(budget), "seed": int(seed)}
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("baseline run started\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(int(seed), torch_threads=1)
    wall_started = time.perf_counter()
    try:
        dataset = load_development_trace_dataset(
            root, dataset_name, horizon=int(config["horizon"])
        )
        training: dict[str, dict] = {}
        agents = {}
        model_states = {}
        for method in config["methods"]:
            method = str(method)
            training_constraint = (
                "slo_violation_rate"
                if method in CONSTRAINED_POLICY_METHODS
                else "resource_cost"
            )
            set_global_seed(int(seed), torch_threads=1)
            with calibrated_baseline_runtime(
                quantile=float(config["capacity_training_quantile"]),
                training_constraint=training_constraint,
                slo_constraint_rate=float(config["slo_constraint_rate"]),
            ):
                result = _train_method(str(method), dataset, config, float(budget), int(seed))
            agents[method] = result.agent
            model_states[method] = result.agent.model.state_dict()
            training[method] = {
                "elapsed_seconds": float(result.elapsed_seconds),
                "diagnostics": result.diagnostics,
                "history": result.history,
                "comparison_family": (
                    "primary"
                    if method in set(config["primary_methods"])
                    else "supplementary"
                ),
                "training_constraint": training_constraint,
                "slo_constraint_rate": (
                    float(config["slo_constraint_rate"])
                    if training_constraint == "slo_violation_rate"
                    else None
                ),
            }
        with calibrated_baseline_runtime(
            quantile=float(config["capacity_training_quantile"])
        ):
            episode_rows: list[dict] = []
            step_rows: list[dict] = []
            for method in config["methods"]:
                episodes, steps = evaluate_agent(
                    dataset,
                    agents[str(method)],
                    split=str(config["evaluation_split"]),
                    horizon=int(config["horizon"]),
                    budget=float(budget),
                    seed=int(seed) + int(config["evaluation_seed_offset"]),
                    episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
                    gamma=float(config["gamma"]),
                )
                for row in episodes:
                    row["training_seed"] = int(seed)
                for row in steps:
                    row["training_seed"] = int(seed)
                episode_rows.extend(episodes)
                step_rows.extend(steps)
        metrics = pd.DataFrame(episode_rows)
        steps = pd.DataFrame(step_rows)
        if not np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("baseline evaluation produced non-finite metrics")
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        steps.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        write_json(run_dir / "training.json", training)
        torch.save({"models": model_states}, run_dir / "models.pt")
        write_json(
            run_dir / "runtime.json",
            {
                "wall_seconds": float(time.perf_counter() - wall_started),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "per_method_training_seconds": {
                    method: float(training[method]["elapsed_seconds"])
                    for method in training
                },
            },
        )
        write_json(
            run_dir / "guardrails.json",
            {
                "budget_overspend_max": float(metrics.budget_overspend.max()),
                "budget_safe": bool(metrics.budget_overspend.max() <= 1.0e-8),
                "all_metrics_finite": True,
                "formal_test_accessed": False,
            },
        )
        (run_dir / "stdout.log").write_text("baseline run completed\n", encoding="utf-8")
        ended_at = datetime.now(timezone.utc).isoformat()
        _write_standard_bundle(
            run_dir,
            root=root,
            config=config,
            dataset_name=dataset_name,
            seed=int(seed),
            started_at=started_at,
            ended_at=ended_at,
            metrics=metrics,
            steps=steps,
            code_paths=(
                root
                / "src/stage2_dynamic_budget/direct_action_planning_dataset_specific_stabilization",
                root / "src/stage2_dynamic_budget/direct_action_planning_dataset_benchmark",
                root / "src/stage2_dynamic_budget/direct_action_planning_dataset_validation",
                root / "src/stage2_dynamic_budget/direct_action_planning_paper_evidence",
                root / "src/stage2_dynamic_budget/envs",
                root / "src/stage2_dynamic_budget/data",
                root / "src/stage2_dynamic_budget/utils",
                root / "scripts/run_direct_action_planning_dataset_specific_baselines.py",
            ),
        )
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
                "code_snapshot_sha256": sha256_file(run_dir / "code_snapshot.json"),
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


def run_baseline_matrix(project_root: str | Path, config_path: str | Path) -> list[Path]:
    config = load_baseline_protocol(config_path)
    outputs = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                outputs.append(
                    run_baseline_unit(
                        project_root,
                        config_path,
                        dataset_name=str(dataset),
                        budget=float(budget),
                        seed=int(seed),
                    )
                )
    return outputs
