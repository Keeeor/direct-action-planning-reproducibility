from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from stage2_dynamic_budget.agents.ppo import PPOConfig, PPOTrainer
from stage2_dynamic_budget.evaluation.metrics import summarize_episode
from stage2_dynamic_budget.experiment import _synthetic_config, load_config
from stage2_dynamic_budget.envs.synthetic_queue_env import DynamicBudgetSchedulingEnv
from stage2_dynamic_budget.models.policy import ConstrainedSchedulingPolicy, PolicyConfig
from stage2_dynamic_budget.utils.artifacts import environment_record, sha256_file, sha256_tree, write_json
from stage2_dynamic_budget.utils.seed import set_global_seed

from .dsp_trainer import DSPTrainer, DSPTrainerConfig
from .hard_coupling import GlobalBudgetMaskedPolicy
from .shadow_policy import DSPPolicyConfig, DynamicShadowPricePolicy
from .synthetic_joint import AbsoluteBudgetObservationWrapper


def _safe_spearman(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return np.nan
    return float(spearmanr(left, right).statistic)


def _make_env(config: dict, scenario: str, budget: float, budget_scale: float):
    base = DynamicBudgetSchedulingEnv(_synthetic_config(config, scenario, budget))
    return AbsoluteBudgetObservationWrapper(base, budget_scale)


def _build_agent(config: dict, method: str, budget_scale: float, seed: int, device):
    training = dict(config["training"])
    hidden = int(training.pop("hidden_dim", 64))
    if method == "b4_joint_hard":
        base = ConstrainedSchedulingPolicy(
            PolicyConfig(
                method="budget_state",
                action_dim=4,
                hidden_dim=hidden,
                episode_budget=budget_scale,
                horizon=int(config["environment"]["horizon"]),
            )
        )
        agent = GlobalBudgetMaskedPolicy(
            base, config["environment"]["action_costs"], budget_scale
        )
        valid = set(PPOConfig.__dataclass_fields__)
        trainer = PPOTrainer(
            agent,
            PPOConfig(**{key: value for key, value in training.items() if key in valid}),
            device,
            budget_scale,
            seed,
        )
        return agent, trainer, base
    dsp = config["dsp"]
    policy = DynamicShadowPricePolicy(
        DSPPolicyConfig(
            variant=method,
            episode_budget=budget_scale,
            horizon=int(config["environment"]["horizon"]),
            action_costs=tuple(float(value) for value in config["environment"]["action_costs"]),
            hidden_dim=hidden,
            alpha=float(dsp["alpha"]),
            delta_budget_ratio=float(dsp["delta_budget_ratio"]),
            monotonic_coef=float(dsp["monotonic_coef"]),
            actor_use_budget_state=bool(dsp.get("actor_use_budget_state", True)),
            actor_use_horizon=bool(dsp.get("actor_use_horizon", True)),
            fixed_shadow_price=dsp.get("fixed_shadow_price"),
            hard_global_budget=True,
        )
    )
    valid = set(DSPTrainerConfig.__dataclass_fields__)
    trainer = DSPTrainer(
        policy,
        DSPTrainerConfig(**{key: value for key, value in training.items() if key in valid}),
        device,
        budget_scale,
        seed,
    )
    return policy, trainer, policy


def evaluate_synthetic_joint(agent, config, budgets, seed, device, budget_scale):
    episodes = []
    steps = []
    latencies = []
    for budget in budgets:
        for scenario in config["eval_scenarios"]:
            scenario_latencies = []
            for episode in range(int(config["evaluation_episodes"])):
                episode_seed = seed + 500_009 + episode * 100_003
                torch.manual_seed(episode_seed)
                env = _make_env(config, scenario, budget, budget_scale)
                obs, _ = env.reset(seed=episode_seed)
                if hasattr(agent, "reset_budget_controller"):
                    agent.reset_budget_controller()
                current = []
                raw_mu = []
                mu = []
                price_kl = []
                while True:
                    started = time.perf_counter_ns()
                    with torch.no_grad():
                        output = agent.act(
                            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                            deterministic=bool(config.get("deterministic_evaluation", False)),
                        )
                    scenario_latencies.append((time.perf_counter_ns() - started) / 1e6)
                    action = int(output.action.item())
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    row = {**info, "reward": reward, "episode": episode}
                    if hasattr(output, "shadow_price"):
                        row.update(
                            {
                                "shadow_price": float(output.shadow_price.item()),
                                "raw_shadow_price": float(output.raw_shadow_price.item()),
                                "price_policy_kl": float(output.policy_kl.item()),
                                "mean_action_price_penalty": float(output.price_penalty.mean().item()),
                            }
                        )
                        raw_mu.append(row["raw_shadow_price"])
                        mu.append(row["shadow_price"])
                        price_kl.append(row["price_policy_kl"])
                    current.append(row)
                    steps.append(row)
                    obs = next_obs
                    if terminated or truncated:
                        break
                summary = summarize_episode(current, budget, int(config["environment"]["horizon"]))
                risk = np.asarray([row["risk_level"] for row in current])
                costs = np.asarray([row["resource_cost"] for row in current])
                low, high = np.quantile(risk, [0.25, 0.75])
                low_cost = float(costs[risk <= low].mean())
                high_cost = float(costs[risk >= high].mean())
                summary.update(
                    {
                        "scenario": scenario,
                        "budget": budget,
                        "eval_episode": episode,
                        "eval_seed": episode_seed,
                        "risk_action_cost_correlation": _safe_spearman(risk, costs),
                        "low_risk_action_cost": low_cost,
                        "high_risk_action_cost": high_cost,
                        "action_reallocation_difference": high_cost - low_cost,
                        "shadow_price_mean": float(np.mean(mu)) if mu else np.nan,
                        "shadow_price_std": float(np.std(mu)) if mu else np.nan,
                        "raw_negative_mu_rate": float(np.mean(np.asarray(raw_mu) < 0)) if raw_mu else np.nan,
                        "price_policy_kl_mean": float(np.mean(price_kl)) if price_kl else np.nan,
                    }
                )
                episodes.append(summary)
            latencies.append(
                {
                    "budget": budget,
                    "scenario": scenario,
                    "decision_latency_ms_mean": float(np.mean(scenario_latencies)),
                    "decision_latency_ms_p95": float(np.quantile(scenario_latencies, 0.95)),
                }
            )
    return episodes, steps, latencies


def run_synthetic_joint_experiment(
    project_root: str | Path,
    config_path: str | Path,
    method: str,
    seed: int,
    variant: str = "formal",
    overrides: dict | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    config = load_config(Path(config_path).resolve())
    overrides = overrides or {}
    for key, value in overrides.items():
        if key in config["dsp"]:
            config["dsp"][key] = value
        elif key in config["training"]:
            config["training"][key] = value
        else:
            raise ValueError(f"unknown override {key}")
    budgets = [float(value) for value in config["budgets"]]
    budget_scale = max(budgets)
    run_id = f"synthetic_joint__{variant}__{method}__s{seed}"
    run_dir = project_root / "results/dynamic_shadow_price/synthetic" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {**config, "method": method, "seed": seed, "variant": variant, "overrides": overrides, "budget_scale": budget_scale, "run_id": run_id}
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    try:
        agent, trainer, checkpoint_agent = _build_agent(config, method, budget_scale, seed, device)
        scenarios = list(config["train_scenarios"])

        def env_factory(env_seed: int):
            scenario = scenarios[env_seed % len(scenarios)]
            budget = budgets[(env_seed // len(scenarios)) % len(budgets)]
            return _make_env(config, scenario, budget, budget_scale)

        result = trainer.train(env_factory)
        torch.save(
            {
                "state_dict": checkpoint_agent.state_dict(),
                "method": method,
                "config": asdict(checkpoint_agent.config),
                "global_lambda": result.global_lambda,
                "joint_training_budgets": budgets,
            },
            run_dir / "model.pt",
        )
        write_json(run_dir / "training_history.json", result.update_history)
        episodes, steps, latency = evaluate_synthetic_joint(
            agent, config, budgets, seed, device, budget_scale
        )
        for row in episodes:
            row.update({"method": method, "seed": seed, "variant": variant})
        for row in steps:
            row.update({"method": method, "seed": seed, "variant": variant})
        pd.DataFrame(episodes).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(steps).to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        write_json(
            run_dir / "runtime.json",
            {
                "parameter_count": agent.parameter_count(),
                "training_seconds": result.elapsed_seconds,
                "decision_latency": latency,
                "device": str(device),
                "global_lambda": result.global_lambda,
            },
        )
        artifacts = ["config.json", "environment.json", "model.pt", "training_history.json", "metrics.csv", "steps.csv.gz", "runtime.json"]
        ended_at = datetime.now(timezone.utc).isoformat()
        write_json(
            run_dir / "manifest.json",
            {
                "schema": "light.run_manifest.v3",
                "run_id": run_id,
                "status": "completed",
                "termination": "joint_budget_synthetic_training_and_evaluation_complete",
                "completion": {"oracle": "PASS", "formal_claim_eligible": variant == "formal"},
                "seed": {"role": "paired_randomness_estimation", "value": seed},
                "started_at": started_at,
                "ended_at": ended_at,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(project_root),
                "artifacts": {name: sha256_file(run_dir / name) for name in artifacts},
                "guardrails": ["same_joint_budget_training_all_methods", "shared_absolute_budget_scale", "hard_global_budget", "paired_evaluation_seeds"],
            },
        )
        return run_dir
    except Exception as exc:
        write_json(run_dir / "failure.json", {"run_id": run_id, "status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
