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

from dap.agents.ppo import PPOConfig, PPOTrainer
from dap.experiment import load_config
from dap.models.policy import ConstrainedSchedulingPolicy, PolicyConfig
from dap.utils.artifacts import environment_record, sha256_file, sha256_tree, write_json
from dap.utils.seed import set_global_seed

from .dp_env import DiscreteBudgetGymEnv
from .dp_reference import DiscreteBudgetMDP, DiscreteDPConfig, solve_backward_dp
from .dsp_trainer import DSPTrainer, DSPTrainerConfig
from .hard_coupling import GlobalBudgetMaskedPolicy
from .shadow_policy import DSPPolicyConfig, DynamicShadowPricePolicy


def _safe_spearman(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.allclose(left[mask], left[mask][0]) or np.allclose(right[mask], right[mask][0]):
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def _dp_config(config: dict, scenario: str) -> DiscreteDPConfig:
    env = config["environment"]
    return DiscreteDPConfig(
        horizon=int(env["horizon"]),
        max_budget=int(env["max_budget"]),
        max_queue=int(env.get("max_queue", 6)),
        scenario=scenario,
        gamma=float(env.get("gamma", 0.99)),
        action_costs=tuple(int(value) for value in env["action_costs"]),
    )


def _build_agent(config: dict, method: str, budget: int, seed: int, device: torch.device):
    training = dict(config["training"])
    hidden = int(training.pop("hidden_dim", 64))
    if method in {"b4_budget_state", "cdba"}:
        internal = "budget_state" if method == "b4_budget_state" else "cdba"
        base = ConstrainedSchedulingPolicy(
            PolicyConfig(
                method=internal,
                action_dim=4,
                hidden_dim=hidden,
                episode_budget=float(budget),
                horizon=int(config["environment"]["horizon"]),
            )
        )
        agent = GlobalBudgetMaskedPolicy(
            base, config["environment"]["action_costs"], budget
        )
        valid = set(PPOConfig.__dataclass_fields__)
        ppo_config = PPOConfig(**{key: value for key, value in training.items() if key in valid})
        trainer = PPOTrainer(agent, ppo_config, device, budget, seed)
        return agent, trainer, base
    if method not in {"dsp_a", "dsp_b"}:
        raise ValueError(f"unknown DP learning method: {method}")
    dsp = config["dsp"]
    policy_config = DSPPolicyConfig(
        variant=method,
        episode_budget=float(budget),
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
    agent = DynamicShadowPricePolicy(policy_config)
    valid = set(DSPTrainerConfig.__dataclass_fields__)
    trainer_config = DSPTrainerConfig(
        **{key: value for key, value in training.items() if key in valid}
    )
    trainer = DSPTrainer(agent, trainer_config, device, budget, seed)
    return agent, trainer, agent


def evaluate_dp_agent(
    agent,
    method: str,
    env_config: DiscreteDPConfig,
    budget: int,
    seed: int,
    episodes: int,
    device: torch.device,
    budget_scale: int | None = None,
):
    mdp = DiscreteBudgetMDP(env_config)
    optimum = solve_backward_dp(mdp)
    episode_rows = []
    step_rows = []
    latency = []
    for episode in range(episodes):
        episode_seed = seed + 700_001 + episode * 100_003
        torch.manual_seed(episode_seed)
        env = DiscreteBudgetGymEnv(env_config, budget, budget_scale=budget_scale)
        optimal_env = DiscreteBudgetGymEnv(
            env_config, budget, budget_scale=budget_scale
        )
        obs, _ = env.reset(seed=episode_seed)
        optimal_env.reset(seed=episode_seed)
        if hasattr(agent, "reset_budget_controller"):
            agent.reset_budget_controller()
        rewards = []
        optimal_rewards = []
        costs = []
        violations = []
        served = []
        arrivals = []
        consistency = []
        budget_gap = []
        predicted_mu = []
        exact_mu = []
        policy_kl = []
        initial_critic_value = np.nan
        while True:
            state = (env.t, env.load, env.queue, env.remaining_budget)
            optimal_action_for_state = int(optimum.actions[state])
            optimal_action = int(
                optimum.actions[
                    optimal_env.t,
                    optimal_env.load,
                    optimal_env.queue,
                    optimal_env.remaining_budget,
                ]
            )
            started = time.perf_counter_ns()
            with torch.no_grad():
                tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                output = agent.act(tensor, deterministic=True)
            latency.append((time.perf_counter_ns() - started) / 1e6)
            action = int(output.action.item())
            if env.t == 0:
                initial_critic_value = float(output.reward_value.item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            _, optimal_reward, optimal_done, _, optimal_info = optimal_env.step(optimal_action)
            rewards.append(float(reward))
            optimal_rewards.append(float(optimal_reward))
            costs.append(float(info["resource_cost"]))
            violations.append(float(info["slo_violation"]))
            served.append(float(info["served"]))
            arrivals.append(float(info["arrivals"]))
            consistency.append(float(action == optimal_action_for_state))
            budget_gap.append(abs(env.cumulative_cost - optimal_env.cumulative_cost))
            mu = np.nan
            kl = np.nan
            if hasattr(output, "shadow_price"):
                mu = float(output.shadow_price.item())
                kl = float(output.policy_kl.item())
            exact = (
                float(optimum.shadow_prices[state]) if state[-1] > 0 else np.nan
            )
            predicted_mu.append(mu)
            exact_mu.append(exact)
            policy_kl.append(kl)
            step_rows.append(
                {
                    "episode": episode,
                    "t": state[0],
                    "load": state[1],
                    "queue": state[2],
                    "budget_before": state[3],
                    "action": action,
                    "optimal_action_same_state": optimal_action_for_state,
                    "action_consistent": float(action == optimal_action_for_state),
                    "reward": reward,
                    "cost": info["resource_cost"],
                    "remaining_budget": info["remaining_budget"],
                    "optimal_remaining_budget": optimal_info["remaining_budget"],
                    "shadow_price": mu,
                    "optimal_shadow_price": exact,
                    "price_policy_kl": kl,
                }
            )
            obs = next_obs
            if terminated or truncated:
                if not optimal_done:
                    raise RuntimeError("paired optimal trajectory ended at a different time")
                break
        discounts = np.power(env_config.gamma, np.arange(env_config.horizon))
        discounted_return = float(np.dot(discounts, rewards))
        optimal_discounted_return = float(np.dot(discounts, optimal_rewards))
        exact_initial_value = float(optimum.values[0, 1, 0, budget])
        episode_rows.append(
            {
                "episode": episode,
                "eval_seed": episode_seed,
                "episode_reward": float(sum(rewards)),
                "discounted_return": discounted_return,
                "optimal_paired_discounted_return": optimal_discounted_return,
                "return_gap_to_paired_optimal": optimal_discounted_return - discounted_return,
                "value_gap_to_exact_expectation": exact_initial_value - discounted_return,
                "critic_initial_value_error": initial_critic_value - exact_initial_value,
                "action_consistency_rate": float(np.mean(consistency)),
                "budget_trajectory_mae": float(np.mean(budget_gap)),
                "total_cost": float(sum(costs)),
                "optimal_total_cost": float(optimal_env.cumulative_cost),
                "service_cost_gap": float(sum(costs) - optimal_env.cumulative_cost),
                "slo_violation_rate": float(np.mean(violations)),
                "completion_rate": float(sum(served) / max(sum(arrivals), 1.0)),
                "mean_queue": float(np.mean([row["queue"] for row in step_rows[-env_config.horizon :]])),
                "mu_optimal_correlation": _safe_spearman(predicted_mu, exact_mu),
                "shadow_price_mean": float(np.nanmean(predicted_mu)) if np.isfinite(predicted_mu).any() else np.nan,
                "price_policy_kl_mean": float(np.nanmean(policy_kl)) if np.isfinite(policy_kl).any() else np.nan,
            }
        )
    latency_stats = {
        "decision_latency_ms_mean": float(np.mean(latency)),
        "decision_latency_ms_p95": float(np.quantile(latency, 0.95)),
    }
    return episode_rows, step_rows, latency_stats


def run_dp_learning_experiment(
    project_root: str | Path,
    config_path: str | Path,
    method: str,
    scenario: str,
    budget: int,
    seed: int,
    variant: str = "default",
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
            raise ValueError(f"unknown override: {key}")
    run_id = f"dp__{variant}__{method}__{scenario}__b{budget}__s{seed}"
    run_dir = project_root / "results/dynamic_shadow_price/dp_learning" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {**config, "method": method, "scenario": scenario, "budget": budget, "seed": seed, "variant": variant, "overrides": overrides, "run_id": run_id}
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    env_config = _dp_config(config, scenario)
    try:
        agent, trainer, checkpoint_agent = _build_agent(config, method, budget, seed, device)
        result = trainer.train(lambda _: DiscreteBudgetGymEnv(env_config, budget))
        torch.save(
            {
                "state_dict": checkpoint_agent.state_dict(),
                "method": method,
                "config": asdict(checkpoint_agent.config),
                "global_lambda": result.global_lambda,
            },
            run_dir / "model.pt",
        )
        write_json(run_dir / "training_history.json", result.update_history)
        episodes, steps, latency = evaluate_dp_agent(
            agent,
            method,
            env_config,
            budget,
            seed,
            int(config["evaluation_episodes"]),
            device,
        )
        for row in episodes:
            row.update({"method": method, "scenario": scenario, "budget": budget, "seed": seed, "variant": variant})
        for row in steps:
            row.update({"method": method, "scenario": scenario, "budget": budget, "seed": seed, "variant": variant})
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
                "termination": "dp_learning_and_optimal_evaluation_complete",
                "completion": {"oracle": "PASS", "formal_claim_eligible": variant == "formal"},
                "seed": {"role": "paired_randomness_estimation", "value": seed},
                "started_at": started_at,
                "ended_at": ended_at,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(project_root),
                "artifacts": {name: sha256_file(run_dir / name) for name in artifacts},
                "guardrails": ["exact_dp_reference", "hard_global_budget_for_all_methods", "paired_load_randomness"],
            },
        )
        return run_dir
    except Exception as exc:
        write_json(run_dir / "failure.json", {"run_id": run_id, "status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise


def run_dp_joint_experiment(
    project_root: str | Path,
    config_path: str | Path,
    method: str,
    scenario: str,
    seed: int,
    variant: str = "joint_default",
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
            raise ValueError(f"unknown override: {key}")
    budgets = [int(value) for value in config["budgets"]]
    budget_scale = int(config["environment"]["max_budget"])
    run_id = f"dp_joint__{variant}__{method}__{scenario}__s{seed}"
    run_dir = project_root / "results/dynamic_shadow_price/dp_learning" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {
        **config,
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "joint_training_budgets": budgets,
        "budget_observation_scale": budget_scale,
        "overrides": overrides,
        "run_id": run_id,
    }
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    env_config = _dp_config(config, scenario)
    try:
        agent, trainer, checkpoint_agent = _build_agent(
            config, method, budget_scale, seed, device
        )

        def env_factory(env_seed: int):
            initial_budget = budgets[env_seed % len(budgets)]
            return DiscreteBudgetGymEnv(
                env_config, initial_budget, budget_scale=budget_scale
            )

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
        all_episodes = []
        all_steps = []
        latency_rows = []
        for budget in budgets:
            episodes, steps, latency = evaluate_dp_agent(
                agent,
                method,
                env_config,
                budget,
                seed,
                int(config["evaluation_episodes"]),
                device,
                budget_scale=budget_scale,
            )
            for row in episodes:
                row.update(
                    {
                        "method": method,
                        "scenario": scenario,
                        "budget": budget,
                        "seed": seed,
                        "variant": variant,
                    }
                )
            for row in steps:
                row.update(
                    {
                        "method": method,
                        "scenario": scenario,
                        "budget": budget,
                        "seed": seed,
                        "variant": variant,
                    }
                )
            all_episodes.extend(episodes)
            all_steps.extend(steps)
            latency_rows.append({"budget": budget, **latency})
        pd.DataFrame(all_episodes).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(all_steps).to_csv(
            run_dir / "steps.csv.gz", index=False, compression="gzip"
        )
        write_json(
            run_dir / "runtime.json",
            {
                "parameter_count": agent.parameter_count(),
                "training_seconds": result.elapsed_seconds,
                "decision_latency": latency_rows,
                "device": str(device),
                "global_lambda": result.global_lambda,
            },
        )
        artifacts = [
            "config.json",
            "environment.json",
            "model.pt",
            "training_history.json",
            "metrics.csv",
            "steps.csv.gz",
            "runtime.json",
        ]
        ended_at = datetime.now(timezone.utc).isoformat()
        write_json(
            run_dir / "manifest.json",
            {
                "schema": "light.run_manifest.v3",
                "run_id": run_id,
                "status": "completed",
                "termination": "joint_budget_dp_learning_and_optimal_evaluation_complete",
                "completion": {
                    "oracle": "PASS",
                    "formal_claim_eligible": variant == "formal",
                },
                "seed": {"role": "paired_randomness_estimation", "value": seed},
                "started_at": started_at,
                "ended_at": ended_at,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(project_root),
                "artifacts": {
                    name: sha256_file(run_dir / name) for name in artifacts
                },
                "guardrails": [
                    "stratified_initial_budgets",
                    "shared_absolute_budget_scale",
                    "same_training_budgets_all_methods",
                    "exact_dp_reference",
                ],
            },
        )
        return run_dir
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            {
                "run_id": run_id,
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
