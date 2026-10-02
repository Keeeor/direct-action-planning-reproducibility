from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import torch
import yaml

from dap.direct_action_planning_dataset_validation.data import load_trace_dataset
from dap.direct_action_planning_dataset_validation.evaluation import evaluate_methods as evaluate_legacy_methods
from dap.direct_action_planning_dataset_validation.experiment import select_refresh_candidate
from dap.direct_action_planning_dataset_validation.planning import make_planner
from dap.direct_action_planning_dataset_validation.training import train_dap_components
from dap.utils.artifacts import environment_record, sha256_file, sha256_tree, write_json
from dap.utils.seed import set_global_seed

from .budgeted import train_budgeted_fitted_q
from .cpo import CPOConfig, train_cpo
from .evaluation import evaluate_agent
from .rl import RLTrainConfig, train_double_dqn, train_policy
from .tuning import tune_controllers


class PlannerAgent:
    name = "structured_dap_selected"

    def __init__(self, planner):
        self.planner = planner

    def reset(self) -> None:
        pass

    def select(self, env, observation):
        return self.planner(env, observation)


def load_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"tier", "datasets", "horizon", "budgets", "seeds", "training_steps", "methods"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing protocol keys: {sorted(missing)}")
    return config


def _train_dap(dataset, config: dict, budget: float, seed: int):
    dap = train_dap_components(
        dataset,
        horizon=int(config["horizon"]),
        budget=budget,
        seed=seed,
        gamma=float(config.get("gamma", 0.99)),
        collection_episodes=int(config.get("dap_collection_episodes_per_domain", 24)),
        validation_episodes=int(config.get("dap_validation_episodes_per_domain", 6)),
        fvi_iterations=int(config.get("dap_fvi_iterations", 18)),
        refresh_iterations=int(config.get("dap_refresh_iterations", 8)),
        model_epochs=int(config.get("dap_model_epochs", 24)),
        hidden_dim=int(config.get("hidden_dim", 64)),
    )
    base = make_planner("structured_dap", dap.base_value, dap.load_forecaster, dap.full_transition, gamma=float(config.get("gamma", 0.99)))
    refreshed = make_planner("structured_dap_refresh", dap.refreshed_value, dap.load_forecaster, dap.full_transition, gamma=float(config.get("gamma", 0.99)))
    selection_rows, _ = evaluate_legacy_methods(
        dataset,
        {"structured_dap": base, "structured_dap_refresh": refreshed},
        split="validation",
        horizon=int(config["horizon"]),
        budget=budget,
        seed=seed + int(config.get("selection_seed_offset", 40_000_003)),
        episodes_per_domain=int(config.get("selection_episodes_per_domain", 2)),
        gamma=float(config.get("gamma", 0.99)),
    )
    selection = select_refresh_candidate(selection_rows)
    value = dap.refreshed_value if selection == "refreshed_value" else dap.base_value
    planner = make_planner("structured_dap_refresh", value, dap.load_forecaster, dap.full_transition, gamma=float(config.get("gamma", 0.99)))
    return PlannerAgent(planner), selection, dap


def run_development_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
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
    controller_families = {
        "reactive_threshold", "pid_budget_autoscaler", "causal_mpc", "lyapunov_dpp"
    } & set(config["methods"])
    agents, tuning_records = tune_controllers(
        dataset,
        horizon=int(config["horizon"]),
        budget=budget,
        seed=seed + int(config.get("selection_seed_offset", 40_000_003)),
        episodes_per_domain=int(config.get("selection_episodes_per_domain", 2)),
        families=controller_families,
    )
    training: dict[str, dict] = {method: {"kind": "validation_tuned_controller"} for method in agents}
    train_steps = int(config["training_steps"])
    policy_config = RLTrainConfig(
        total_steps=train_steps,
        rollout_steps=int(config.get("rollout_steps", 512)),
        update_epochs=int(config.get("update_epochs", 4)),
        minibatch_size=int(config.get("minibatch_size", 256)),
        hidden_dim=int(config.get("hidden_dim", 64)),
    )
    for method in config["methods"]:
        if method in agents or method == "structured_dap_selected":
            continue
        if method == "double_dqn":
            result = train_double_dqn(dataset, horizon=int(config["horizon"]), budget=budget, seed=seed, total_steps=train_steps)
        elif method in {"a2c", "ppo", "ppo_lagrangian", "pid_lagrangian", "p3o"}:
            result = train_policy(dataset, horizon=int(config["horizon"]), budget=budget, seed=seed, variant=method, config=policy_config)
        elif method == "budgeted_fitted_q":
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
            result = type("FittedQResult", (), {"agent": agent, "elapsed_seconds": elapsed, "diagnostics": {}, "history": history})()
        elif method == "cpo":
            result = train_cpo(
                dataset,
                horizon=int(config["horizon"]),
                budget=budget,
                seed=seed,
                config=CPOConfig(
                    total_steps=train_steps,
                    rollout_steps=int(config.get("rollout_steps", 512)),
                    critic_epochs=int(config.get("update_epochs", 4)),
                    hidden_dim=int(config.get("hidden_dim", 64)),
                ),
            )
        else:
            raise ValueError(f"unsupported development method: {method}")
        agents[method] = result.agent
        training[method] = {"elapsed_seconds": result.elapsed_seconds, "diagnostics": result.diagnostics, "history": result.history}
    if "structured_dap_selected" in config["methods"]:
        dap_agent, selection, dap = _train_dap(dataset, config, budget, seed)
        agents["structured_dap_selected"] = dap_agent
        training["structured_dap_selected"] = {"selection": selection, "training_seconds": dap.training_seconds, "histories": dap.histories, "collection": dap.collection}
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
            episodes_per_domain=int(config.get("evaluation_episodes_per_domain", 5)),
            gamma=float(config.get("gamma", 0.99)),
        )
        episode_rows.extend(episodes)
        step_rows.extend(steps)
    pd.DataFrame(episode_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(step_rows).to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
    pd.DataFrame(tuning_records).to_json(run_dir / "controller_tuning.jsonl", orient="records", lines=True)
    write_json(run_dir / "training.json", training)
    write_json(run_dir / "runtime.json", {"elapsed_seconds": time.perf_counter() - started})
    artifacts = {path.name: sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "manifest.json"}
    write_json(run_dir / "manifest.json", {
        "schema": "dap.dap_dataset_benchmark.development.v1",
        "status": "completed",
        "run_id": run_id,
        "development_only": True,
        "formal_test_accessed": False,
        "config_sha256": sha256_file(config_path),
        "code_sha256": sha256_tree(Path(__file__).parent),
        "artifacts": artifacts,
    })
    return run_dir
