from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from stage2_dynamic_budget.agents.ppo import PPOConfig, PPOTrainer
from stage2_dynamic_budget.models.policy import PolicyConfig
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .data import DATASET_SPECS, load_trace_dataset, make_trace_env
from .evaluation import evaluate_methods
from .models import MaskedBudgetStatePolicy
from .planning import assert_structured_planner_prefix_invariant, make_planner
from .training import train_dap_components


METHODS = (
    "b4_budget_state",
    "learned_value_true_transition",
    "original_learned_model",
    "structured_dap",
    "structured_dap_refresh",
)


def load_protocol(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "tier",
        "datasets",
        "horizon",
        "budgets",
        "seeds",
        "gamma",
        "b4",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing protocol keys: {sorted(missing)}")
    return config


def _train_b4(dataset, config: dict, budget: float, seed: int):
    horizon = int(config["horizon"])
    policy_config = PolicyConfig(
        method="budget_state",
        action_dim=4,
        hidden_dim=int(config["hidden_dim"]),
        episode_budget=budget,
        horizon=horizon,
    )
    action_costs = np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    policy = MaskedBudgetStatePolicy(policy_config, action_costs)
    trainer = PPOTrainer(
        policy,
        PPOConfig(**config["b4"]),
        device=torch.device("cpu"),
        budget=budget,
        seed=seed,
    )
    domains = dataset.domain_names

    def factory(env_seed: int):
        domain = domains[abs(env_seed) % len(domains)]
        env, _ = make_trace_env(
            dataset,
            domain,
            "train",
            horizon=horizon,
            budget=budget,
            window_seed=env_seed,
        )
        return env

    result = trainer.train(factory)
    return policy.eval(), result


def _input_hashes(project_root: Path, dataset: str) -> dict[str, str]:
    spec = DATASET_SPECS[dataset]
    processed = project_root / str(spec["processed_dir"])
    hashes = {
        f"processed_{domain}": sha256_file(processed / f"{domain}.npz")
        for domain in spec["domains"]
    }
    if dataset == "azure2019":
        hashes["raw_archive"] = "aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89"
    else:
        hashes["metadata"] = sha256_file(project_root / "data/metadata/gentd26/dataset.json")
    return hashes


def select_refresh_candidate(validation_rows: list[dict]) -> str:
    frame = pd.DataFrame(validation_rows)
    required = {"structured_dap", "structured_dap_refresh"}
    if not required.issubset(set(frame.method)):
        raise ValueError("validation rows lack DAP refresh candidates")
    summary = frame.groupby("method").agg(
        discounted_return=("discounted_return", "mean"),
        completion_ratio=("completion_ratio", "mean"),
        slo_violation_rate=("slo_violation_rate", "mean"),
        total_cost=("total_cost", "mean"),
    )
    base = summary.loc["structured_dap"]
    refresh = summary.loc["structured_dap_refresh"]
    checks = (
        refresh.discounted_return >= base.discounted_return - 1.0e-8,
        refresh.completion_ratio >= base.completion_ratio - 0.01,
        refresh.slo_violation_rate <= base.slo_violation_rate + 0.01,
        refresh.total_cost <= base.total_cost + 0.05 * max(base.total_cost, 1.0),
    )
    return "refreshed_value" if all(checks) else "base_value"


def run_unit(
    project_root: str | Path,
    config_path: str | Path,
    *,
    dataset_name: str,
    budget: float,
    seed: int,
    include_test: bool,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    dataset = load_trace_dataset(project_root, dataset_name)
    budget_label = f"{budget:.6g}".replace(".", "p")
    run_id = f"{config['tier']}__{dataset_name}__b{budget_label}__s{seed}"
    run_dir = (
        project_root
        / "results/direct_action_planning_dataset_validation"
        / str(config["tier"])
        / dataset_name
        / run_id
    )
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(f"completed run is append-only: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {
        **config,
        "dataset": dataset_name,
        "budget": budget,
        "seed": seed,
        "include_test": include_test,
        "run_id": run_id,
    }
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text(f"started {started_at}\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(seed, torch_threads=1)
    wall_started = time.perf_counter()
    try:
        dap = train_dap_components(
            dataset,
            horizon=int(config["horizon"]),
            budget=budget,
            seed=seed,
            gamma=float(config["gamma"]),
            collection_episodes=int(config["collection_episodes_per_domain"]),
            validation_episodes=int(config["validation_collection_episodes_per_domain"]),
            fvi_iterations=int(config["fvi_iterations"]),
            refresh_iterations=int(config["refresh_iterations"]),
            model_epochs=int(config["model_epochs"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        b4_started = time.perf_counter()
        b4, b4_result = _train_b4(dataset, config, budget, seed)
        b4_seconds = time.perf_counter() - b4_started
        planners = {
            "b4_budget_state": b4,
            "learned_value_true_transition": make_planner(
                "learned_value_true_transition",
                dap.base_value,
                dap.load_forecaster,
                dap.full_transition,
                gamma=float(config["gamma"]),
            ),
            "original_learned_model": make_planner(
                "original_learned_model",
                dap.base_value,
                dap.load_forecaster,
                dap.full_transition,
                gamma=float(config["gamma"]),
            ),
            "structured_dap": make_planner(
                "structured_dap",
                dap.base_value,
                dap.load_forecaster,
                dap.full_transition,
                gamma=float(config["gamma"]),
            ),
            "structured_dap_refresh": make_planner(
                "structured_dap_refresh",
                dap.refreshed_value,
                dap.load_forecaster,
                dap.full_transition,
                gamma=float(config["gamma"]),
            ),
        }
        first_domain = dataset.domain_names[0]
        prefix = dataset.domains[first_domain]["validation"][: int(config["horizon"])].copy()
        trace_a = prefix.copy()
        trace_b = prefix.copy()
        trace_a[1:] = 0.0
        trace_b[1:] = 100.0
        from stage2_dynamic_budget.envs.synthetic_queue_env import SyntheticQueueConfig
        from stage2_dynamic_budget.envs.trace_driven_env import TraceDrivenQueueEnv

        leak_config = SyntheticQueueConfig(horizon=len(prefix), budget=budget)
        assert_structured_planner_prefix_invariant(
            planners["structured_dap"],
            TraceDrivenQueueEnv(trace_a, leak_config),
            TraceDrivenQueueEnv(trace_b, leak_config),
        )
        episode_rows, step_rows = evaluate_methods(
            dataset,
            planners,
            split="validation",
            horizon=int(config["horizon"]),
            budget=budget,
            seed=seed + 50_000_003,
            episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
            gamma=float(config["gamma"]),
        )
        selected_refresh = select_refresh_candidate(episode_rows)
        write_json(
            run_dir / "selection.json",
            {
                "rule": "validation discounted return primary; completion, SLO, and <=5% cost increase guardrails",
                "selected": selected_refresh,
                "test_selection_forbidden": True,
            },
        )
        if include_test:
            selected_value = (
                dap.refreshed_value if selected_refresh == "refreshed_value" else dap.base_value
            )
            test_planners = dict(planners)
            test_planners["structured_dap_refresh"] = make_planner(
                "structured_dap_refresh",
                selected_value,
                dap.load_forecaster,
                dap.full_transition,
                gamma=float(config["gamma"]),
            )
            test_episodes, test_steps = evaluate_methods(
                dataset,
                test_planners,
                split="test",
                horizon=int(config["horizon"]),
                budget=budget,
                seed=seed + 70_000_001,
                episodes_per_domain=int(config["evaluation_episodes_per_domain"]),
                gamma=float(config["gamma"]),
            )
            for row in test_episodes:
                row["selected_refresh_variant"] = selected_refresh
            for row in test_steps:
                row["selected_refresh_variant"] = selected_refresh
            episode_rows.extend(test_episodes)
            step_rows.extend(test_steps)
        pd.DataFrame(episode_rows).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(step_rows).to_csv(
            run_dir / "steps.csv.gz", index=False, compression="gzip"
        )
        write_json(run_dir / "training_history.json", dap.histories)
        write_json(
            run_dir / "runtime.json",
            {
                "dap_training_seconds": dap.training_seconds,
                "b4_training_seconds": b4_seconds,
                "total_seconds": time.perf_counter() - wall_started,
                "collection": dap.collection,
                "b4_global_lambda": b4_result.global_lambda,
                "b4_parameter_count": b4.parameter_count(),
                "value_parameter_count": sum(p.numel() for p in dap.base_value.parameters()),
                "forecaster_parameter_count": sum(
                    p.numel() for p in dap.load_forecaster.parameters()
                ),
                "full_transition_parameter_count": sum(
                    p.numel() for p in dap.full_transition.parameters()
                ),
            },
        )
        torch.save(
            {
                "b4": b4.state_dict(),
                "base_value": dap.base_value.state_dict(),
                "refreshed_value": dap.refreshed_value.state_dict(),
                "load_forecaster": dap.load_forecaster.state_dict(),
                "full_transition": dap.full_transition.state_dict(),
                "policy_config": asdict(b4.config),
                "normalizer": {
                    "mean": dap.base_value.normalizer.mean,
                    "scale": dap.base_value.normalizer.scale,
                },
            },
            run_dir / "models.pt",
        )
        ended_at = datetime.now(timezone.utc).isoformat()
        artifacts = [
            "config.json",
            "environment.json",
            "metrics.csv",
            "steps.csv.gz",
            "training_history.json",
            "runtime.json",
            "models.pt",
            "selection.json",
        ]
        manifest = {
            "schema": "light.run_manifest.v3",
            "run_id": run_id,
            "status": "completed",
            "started_at": started_at,
            "ended_at": ended_at,
            "formal_test_included": include_test,
            "seed": {"role": "paired_training_and_window_sampling", "value": seed},
            "split_contract": dataset.split_contract,
            "future_leakage_gate": "PASS",
            "budget_accounting_gate": "PASS",
            "methods": list(METHODS),
            "validation_selected_refresh": selected_refresh,
            "config_sha256": sha256_file(run_dir / "config.json"),
            "code_sha256": sha256_tree(
                project_root
                / "src/stage2_dynamic_budget/direct_action_planning_dataset_validation"
            ),
            "input_sha256": _input_hashes(project_root, dataset_name),
            "artifacts": {name: sha256_file(run_dir / name) for name in artifacts},
        }
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "stdout.log").write_text(
            (run_dir / "stdout.log").read_text(encoding="utf-8")
            + f"completed {ended_at}\n",
            encoding="utf-8",
        )
        return run_dir
    except Exception as exc:
        import traceback

        failure = {
            "run_id": run_id,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        (run_dir / "stderr.log").write_text(failure["traceback"], encoding="utf-8")
        raise


def _run_matrix(project_root: Path, config_path: Path, *, include_test: bool) -> list[Path]:
    config = load_protocol(config_path)
    outputs = []
    for dataset in config["datasets"]:
        for budget in config["budgets"]:
            for seed in config["seeds"]:
                outputs.append(
                    run_unit(
                        project_root,
                        config_path,
                        dataset_name=str(dataset),
                        budget=float(budget),
                        seed=int(seed),
                        include_test=include_test,
                    )
                )
    return outputs


def run_smoke(project_root: str | Path, config_path: str | Path) -> list[Path]:
    return _run_matrix(Path(project_root).resolve(), Path(config_path).resolve(), include_test=False)


def run_gate_matrix(project_root: str | Path, config_path: str | Path) -> list[Path]:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_protocol(config_path)
    ledger_path = (
        project_root
        / "results/direct_action_planning_dataset_validation"
        / str(config["tier"])
        / "FINAL_TEST_LEDGER.json"
    )
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("status") == "finalized":
            raise RuntimeError("final test ledger is finalized; rerun is forbidden")
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            ledger_path,
            {
                "status": "started",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256_file(config_path),
                "test_execution_count": 1,
            },
        )
    outputs = _run_matrix(project_root, config_path, include_test=True)
    write_json(
        ledger_path,
        {
            "status": "finalized",
            "started_at": json.loads(ledger_path.read_text(encoding="utf-8"))["started_at"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_path),
            "test_execution_count": 1,
            "completed_units": len(outputs),
        },
    )
    return outputs
