from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import pandas as pd
import torch
import yaml

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import load_trace_dataset
from stage2_dynamic_budget.utils.artifacts import environment_record, sha256_file, sha256_tree, write_json
from stage2_dynamic_budget.utils.seed import set_global_seed

from .controllers import (
    CausalMPCController,
    LyapunovDPPController,
    PIDBudgetController,
    ReactiveThresholdController,
)
from .budgeted import train_budgeted_fitted_q
from .cpo import CPOConfig, train_cpo
from .evaluation import evaluate_agent
from .rl import RLTrainConfig, train_double_dqn, train_policy


def load_smoke_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"tier", "datasets", "horizon", "budgets", "seeds", "training_steps", "methods"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing smoke protocol keys: {sorted(missing)}")
    return config


def run_smoke_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_smoke_protocol(config_path)
    dataset = load_trace_dataset(project_root, dataset_name)
    run_id = f"{config['tier']}__{dataset_name}__b{budget:.0f}__s{seed}"
    run_dir = project_root / "results/direct_action_planning_dataset_benchmark" / config["tier"] / dataset_name / run_id
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(f"completed run is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {**config, "dataset": dataset_name, "budget": budget, "seed": seed, "run_id": run_id})
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=1)
    started = time.perf_counter()
    agents = {
        "reactive_threshold": ReactiveThresholdController(),
        "pid_budget_autoscaler": PIDBudgetController(),
        "causal_mpc": CausalMPCController(horizon=3),
        "lyapunov_dpp": LyapunovDPPController(),
    }
    training_records: dict[str, dict] = {}
    train_steps = int(config["training_steps"])
    common = RLTrainConfig(
        total_steps=train_steps,
        rollout_steps=min(int(config.get("rollout_steps", 256)), train_steps),
        update_epochs=int(config.get("update_epochs", 2)),
        minibatch_size=int(config.get("minibatch_size", 128)),
        hidden_dim=int(config.get("hidden_dim", 64)),
    )
    for method in config["methods"]:
        if method in agents:
            continue
        if method == "double_dqn":
            result = train_double_dqn(dataset, horizon=int(config["horizon"]), budget=budget, seed=seed, total_steps=train_steps)
        elif method == "budgeted_fitted_q":
            agent, history, elapsed = train_budgeted_fitted_q(
                dataset,
                horizon=int(config["horizon"]),
                budget=budget,
                seed=seed,
                episodes_per_domain=int(config.get("branch_episodes_per_domain", 4)),
                iterations=int(config.get("bftq_iterations", 3)),
                epochs_per_iteration=int(config.get("bftq_epochs_per_iteration", 1)),
                hidden_dim=int(config.get("hidden_dim", 64)),
            )
            result = type("FittedQResult", (), {
                "agent": agent,
                "elapsed_seconds": elapsed,
                "diagnostics": {"branches": float(sum(row.get("branches", 0.0) for row in history))},
                "history": history,
            })()
        elif method == "cpo":
            result = train_cpo(
                dataset,
                horizon=int(config["horizon"]),
                budget=budget,
                seed=seed,
                config=CPOConfig(
                    total_steps=train_steps,
                    rollout_steps=min(int(config.get("rollout_steps", 256)), train_steps),
                    critic_epochs=int(config.get("update_epochs", 2)),
                    hidden_dim=int(config.get("hidden_dim", 64)),
                    cg_iterations=int(config.get("cpo_cg_iterations", 6)),
                    backtracks=int(config.get("cpo_backtracks", 8)),
                ),
            )
        elif method in {"a2c", "ppo", "ppo_lagrangian", "pid_lagrangian", "p3o"}:
            result = train_policy(dataset, horizon=int(config["horizon"]), budget=budget, seed=seed, variant=method, config=common)
        else:
            raise ValueError(f"smoke protocol contains unsupported method: {method}")
        agents[method] = result.agent
        training_records[method] = {
            "elapsed_seconds": result.elapsed_seconds,
            "diagnostics": result.diagnostics,
            "history": result.history,
        }
    episode_rows = []
    step_rows = []
    for method in config["methods"]:
        episodes, steps = evaluate_agent(
            dataset,
            agents[method],
            split="validation",
            horizon=int(config["horizon"]),
            budget=budget,
            seed=seed + int(config.get("evaluation_seed_offset", 50_000_003)),
            episodes_per_domain=int(config.get("evaluation_episodes_per_domain", 1)),
        )
        episode_rows.extend(episodes)
        step_rows.extend(steps)
    pd.DataFrame(episode_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(step_rows).to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
    write_json(run_dir / "training.json", training_records)
    write_json(run_dir / "runtime.json", {"elapsed_seconds": time.perf_counter() - started})
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(run_dir / "manifest.json", {
        "schema": "stage2.dap_dataset_benchmark.smoke.v1",
        "status": "completed",
        "run_id": run_id,
        "development_only": True,
        "formal_test_accessed": False,
        "config_sha256": sha256_file(config_path),
        "code_sha256": sha256_tree(Path(__file__).parent),
        "artifacts": artifacts,
    })
    return run_dir


def run_smoke_matrix(project_root: str | Path, config_path: str | Path) -> list[Path]:
    config = load_smoke_protocol(config_path)
    return [
        run_smoke_unit(project_root, config_path, dataset_name=dataset, budget=float(budget), seed=int(seed))
        for dataset in config["datasets"]
        for budget in config["budgets"]
        for seed in config["seeds"]
    ]
