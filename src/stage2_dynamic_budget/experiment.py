from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

import pandas as pd
import torch
import yaml

from .agents.ppo import PPOConfig, PPOTrainer
from .agents.rules import AggressiveRule, ConservativeRule, NoIntervention
from .envs.synthetic_queue_env import ActionSpec, DynamicBudgetSchedulingEnv, SyntheticQueueConfig
from .evaluation.rollout import evaluate_agent
from .models.policy import ConstrainedSchedulingPolicy, PolicyConfig
from .utils.artifacts import environment_record, sha256_file, sha256_tree, write_json
from .utils.seed import set_global_seed


RULES = {
    "b0_no_intervention": NoIntervention,
    "b1_conservative": ConservativeRule,
    "b1_aggressive": AggressiveRule,
}
LEARNING_METHODS = {
    "b2_ppo": "ppo",
    "b3_lagrangian": "lagrangian",
    "b4_budget_state": "budget_state",
    "b5_fixed_local": "fixed_local",
    "cdba": "cdba",
    "cdba_discrete": "cdba_discrete",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a mapping")
    return data


def _synthetic_config(
    config: dict,
    scenario: str,
    budget: float,
    environment_override: dict | None = None,
) -> SyntheticQueueConfig:
    env = {**config["environment"], **(environment_override or {})}
    costs = list(env.get("action_costs", [0.0, 1.0, 2.0, 4.0]))
    capacities = list(env.get("capacity_deltas", [0.0, 1.0, 2.2, 4.8]))
    if len(costs) != 4 or len(capacities) != 4:
        raise ValueError("action_costs and capacity_deltas must each contain four entries")
    names = ("no_op", "scale_small", "scale_medium", "scale_large")
    actions = tuple(
        ActionSpec(name, float(cost), float(capacity))
        for name, cost, capacity in zip(names, costs, capacities)
    )
    return SyntheticQueueConfig(
        horizon=int(env["horizon"]),
        budget=float(budget),
        scenario=scenario,
        base_arrival_rate=float(env.get("base_arrival_rate", 6.0)),
        base_capacity=float(env.get("base_capacity", 5.0)),
        slo_latency=float(env.get("slo_latency", 3.0)),
        burst_scale=float(env.get("burst_scale", 2.4)),
        actions=actions,
    )


def run_synthetic_experiment(
    project_root: str | Path,
    config_path: str | Path,
    method: str,
    budget: float,
    seed: int,
    variant: str = "main",
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    tier = str(config["tier"])
    budget_label = f"{budget:.6g}".replace(".", "p")
    run_id = f"{tier}__{variant}__{method}__b{budget_label}__s{seed}"
    run_dir = project_root / "results" / "raw_logs" / tier / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    stdout_path, stderr_path = run_dir / "stdout.log", run_dir / "stderr.log"
    stdout_path.write_text(f"start {run_id} at {started_at}\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    resolved = {
        **config,
        "method": method,
        "budget": float(budget),
        "seed": int(seed),
        "variant": variant,
        "run_id": run_id,
    }
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=int(config.get("torch_threads", 1)))
    device_name = str(config.get("device", "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    train_scenarios = list(config["train_scenarios"])
    eval_scenarios = list(config["eval_scenarios"])
    eval_episodes = int(config["evaluation_episodes"])
    variant_settings = dict(config.get("variant_options", {}).get(variant, {}))
    environment_override = dict(variant_settings.get("environment", {}))
    try:
        training_result = None
        global_lambda = 0.0
        if method in RULES:
            agent = RULES[method]()
            parameter_count = 0
            training_seconds = 0.0
        else:
            internal_method = LEARNING_METHODS[method]
            method_options = dict(config.get("method_options", {}).get(method, {}))
            method_options.update(
                variant_settings.get(
                    "method_options",
                    {key: value for key, value in variant_settings.items() if key != "environment"},
                )
            )
            policy_config = PolicyConfig(
                method=internal_method,
                action_dim=4,
                hidden_dim=int(config["training"].get("hidden_dim", 64)),
                episode_budget=float(budget),
                horizon=int(config["environment"]["horizon"]),
                eta=float(method_options.get("eta", 1.0)),
                min_multiplier=float(method_options.get("min_multiplier", 0.25)),
                max_multiplier=float(method_options.get("max_multiplier", 4.0)),
                use_remaining_budget=bool(method_options.get("use_remaining_budget", True)),
                use_remaining_horizon=bool(method_options.get("use_remaining_horizon", True)),
                use_local_lambda=bool(method_options.get("use_local_lambda", True)),
                budget_update_period=int(method_options.get("budget_update_period", 1)),
            )
            agent = ConstrainedSchedulingPolicy(policy_config)
            ppo_fields = dict(config["training"])
            ppo_fields.pop("hidden_dim", None)
            ppo_fields.update({key: value for key, value in method_options.items() if key in PPOConfig.__dataclass_fields__})
            trainer = PPOTrainer(
                agent,
                PPOConfig(**ppo_fields),
                device=device,
                budget=budget,
                seed=seed,
            )

            def train_factory(env_seed: int):
                scenario = train_scenarios[env_seed % len(train_scenarios)]
                return DynamicBudgetSchedulingEnv(
                    _synthetic_config(config, scenario, budget, environment_override)
                )

            training_result = trainer.train(train_factory)
            global_lambda = training_result.global_lambda
            parameter_count = agent.parameter_count()
            training_seconds = training_result.elapsed_seconds
            torch.save(
                {
                    "state_dict": agent.state_dict(),
                    "policy_config": asdict(policy_config),
                    "global_lambda": global_lambda,
                },
                run_dir / "model.pt",
            )
            write_json(run_dir / "training_history.json", [dict(row) for row in training_result.update_history])

        all_episode_rows, all_step_rows = [], []
        latency_rows = []
        for scenario in eval_scenarios:
            def eval_factory(eval_seed: int, scenario=scenario):
                return DynamicBudgetSchedulingEnv(
                    _synthetic_config(config, scenario, budget, environment_override)
                )

            episodes, steps, latency = evaluate_agent(
                agent,
                eval_factory,
                budget,
                seed + 500_009,
                eval_episodes,
                device,
                global_lambda,
                deterministic=bool(config.get("deterministic_evaluation", True)),
            )
            for row in episodes:
                row.update({"scenario": scenario, "method": method, "budget": budget, "seed": seed, "variant": variant})
            for row in steps:
                row.update({"scenario": scenario, "method": method, "budget": budget, "seed": seed, "variant": variant})
            latency_rows.append({"scenario": scenario, **latency})
            all_episode_rows.extend(episodes)
            all_step_rows.extend(steps)
        metrics_path = run_dir / "metrics.csv"
        steps_path = run_dir / "steps.csv.gz"
        pd.DataFrame(all_episode_rows).to_csv(metrics_path, index=False)
        pd.DataFrame(all_step_rows).to_csv(steps_path, index=False, compression="gzip")
        write_json(
            run_dir / "runtime.json",
            {
                "parameter_count": parameter_count,
                "training_seconds": training_seconds,
                "latency_by_scenario": latency_rows,
                "device": str(device),
                "global_lambda": global_lambda,
            },
        )
        ended_at = datetime.now(timezone.utc).isoformat()
        artifacts = ["config.json", "environment.json", "metrics.csv", "steps.csv.gz", "runtime.json"]
        if (run_dir / "model.pt").exists():
            artifacts.extend(["model.pt", "training_history.json"])
        manifest = {
            "schema": "light.run_manifest.v3",
            "run_id": run_id,
            "matrix_row_id": f"synthetic-{tier}-{method}-{budget_label}",
            "status": "completed",
            "termination": "planned_steps_and_evaluation_complete",
            "completion": {"oracle": "PASS", "formal_claim_eligible": tier == "formal"},
            "seed": {"role": "randomness_estimation", "value": seed},
            "started_at": started_at,
            "ended_at": ended_at,
            "config_sha256": sha256_file(run_dir / "config.json"),
            "code_sha256": sha256_tree(project_root),
            "input_sha256": {
                "research_guidance": sha256_file(project_root.parent / "上下文感知动态预算约束调度_独立研究指导.md")
            },
            "artifacts": {name: sha256_file(run_dir / name) for name in artifacts},
            "failure_tree_refs": {
                "hypothesis_ids": ["H1", "H2", "H3", "H4"],
                "branch_action_ids": ["continue", "diagnose", "negative_evidence"],
                "guardrail_ids": ["cost_accounting", "no_future_leakage", "fair_comparison"],
            },
        }
        write_json(run_dir / "manifest.json", manifest)
        stdout_path.write_text(stdout_path.read_text() + f"completed {ended_at}\n", encoding="utf-8")
        return run_dir
    except Exception as exc:
        ended_at = datetime.now(timezone.utc).isoformat()
        failure = {
            "run_id": run_id,
            "status": "failed",
            "started_at": started_at,
            "ended_at": ended_at,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        stderr_path.write_text(failure["traceback"], encoding="utf-8")
        raise
