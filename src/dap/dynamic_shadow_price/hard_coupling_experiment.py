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
from dap.evaluation.metrics import summarize_episode
from dap.experiment import _synthetic_config, load_config
from dap.envs.synthetic_queue_env import DynamicBudgetSchedulingEnv
from dap.models.policy import ConstrainedSchedulingPolicy, PolicyConfig
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from dap.utils.seed import set_global_seed

from .hard_coupling import HardCoupledCDBAPolicy


def _checkpoint_path(project_root: Path, budget: float, seed: int) -> Path:
    label = f"{budget:.6g}".replace(".", "p")
    return (
        project_root
        / "results"
        / "raw_logs"
        / "formal"
        / f"formal__valid_v2__cdba__b{label}__s{seed}"
        / "model.pt"
    )


def load_frozen_cdba(project_root: Path, budget: float, seed: int):
    checkpoint_path = _checkpoint_path(project_root, budget, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy = ConstrainedSchedulingPolicy(PolicyConfig(**checkpoint["policy_config"]))
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    return policy, float(checkpoint["global_lambda"]), checkpoint_path


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _mechanism_summary(rows: list[dict]) -> dict[str, float]:
    risk = np.asarray([row["risk_level"] for row in rows], dtype=np.float64)
    action_cost = np.asarray([row["resource_cost"] for row in rows], dtype=np.float64)
    quota = np.asarray([row["local_budget"] for row in rows], dtype=np.float64)
    low, high = np.quantile(risk, [0.25, 0.75])
    low_cost = float(action_cost[risk <= low].mean())
    high_cost = float(action_cost[risk >= high].mean())
    return {
        "risk_action_cost_correlation": _safe_spearman(risk, action_cost),
        "risk_local_budget_correlation": _safe_spearman(risk, quota),
        "low_risk_action_cost": low_cost,
        "high_risk_action_cost": high_cost,
        "action_reallocation_difference": high_cost - low_cost,
        "action_reallocation_ratio": (
            high_cost / low_cost if low_cost > 1e-12 else float("inf")
        ),
        "invalid_probability_mass_mean": float(
            np.mean([row["invalid_probability_mass"] for row in rows])
        ),
        "mask_policy_kl_mean": float(np.mean([row["mask_policy_kl"] for row in rows])),
    }


def evaluate_hard_coupled(
    agent: HardCoupledCDBAPolicy,
    config: dict,
    budget: float,
    seed: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    episode_rows: list[dict] = []
    step_rows: list[dict] = []
    latency_rows: list[dict] = []
    deterministic = bool(config.get("deterministic_evaluation", False))
    for scenario in config["eval_scenarios"]:
        latencies: list[float] = []
        for episode in range(int(config["evaluation_episodes"])):
            episode_seed = seed + 500_009 + 100_003 * episode
            torch.manual_seed(episode_seed)
            env = DynamicBudgetSchedulingEnv(_synthetic_config(config, scenario, budget))
            obs, _ = env.reset(seed=episode_seed)
            agent.reset_budget_controller()
            current: list[dict] = []
            while True:
                started = time.perf_counter_ns()
                with torch.no_grad():
                    output = agent.act(
                        torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                        deterministic=deterministic,
                    )
                latencies.append((time.perf_counter_ns() - started) / 1e6)
                next_obs, reward, terminated, truncated, info = env.step(
                    int(output.action.item())
                )
                row = {
                    **info,
                    "scenario": scenario,
                    "reward": float(reward),
                    "episode": episode,
                    "local_budget": float(output.local_budget.item()),
                    "budget_multiplier": float(output.budget_multiplier.item()),
                    "invalid_probability_mass": float(
                        agent.last_invalid_probability_mass.item()
                    ),
                    "mask_policy_kl": float(agent.last_policy_kl.item()),
                }
                current.append(row)
                step_rows.append(row)
                obs = next_obs
                if terminated or truncated:
                    break
            summary = summarize_episode(current, budget, env.config.horizon)
            summary.update(_mechanism_summary(current))
            summary.update(
                {
                    "scenario": scenario,
                    "eval_episode": episode,
                    "eval_seed": episode_seed,
                }
            )
            episode_rows.append(summary)
        latency_rows.append(
            {
                "scenario": scenario,
                "decision_latency_ms_mean": float(np.mean(latencies)),
                "decision_latency_ms_p95": float(np.quantile(latencies, 0.95)),
            }
        )
    return episode_rows, step_rows, latency_rows


def run_hard_coupling_experiment(
    project_root: str | Path,
    config_path: str | Path,
    mode: str,
    budget: float,
    seed: int,
    total_steps_override: int | None = None,
) -> Path:
    if mode not in {"d1", "d2"}:
        raise ValueError("mode must be d1 or d2")
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if total_steps_override is not None:
        config["training"]["total_steps"] = int(total_steps_override)
    label = f"{budget:.6g}".replace(".", "p")
    suffix = f"__steps{total_steps_override}" if total_steps_override is not None else ""
    run_id = f"hard_coupling__v3__{mode}__b{label}__s{seed}{suffix}"
    run_dir = project_root / "results" / "dynamic_shadow_price" / "hard_coupling" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {**config, "mode": mode, "budget": budget, "seed": seed, "run_id": run_id}
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    try:
        source_checkpoint = None
        if mode == "d1":
            base, global_lambda, source_checkpoint = load_frozen_cdba(
                project_root, budget, seed
            )
            agent = HardCoupledCDBAPolicy(
                base, config["environment"]["action_costs"], freeze_allocator=True
            ).to(device)
            agent.requires_grad_(False)
            training_seconds = 0.0
            training_history: list[dict] = []
        else:
            policy_config = PolicyConfig(
                method="cdba",
                action_dim=4,
                hidden_dim=int(config["training"].get("hidden_dim", 64)),
                episode_budget=float(budget),
                horizon=int(config["environment"]["horizon"]),
            )
            base = ConstrainedSchedulingPolicy(policy_config)
            agent = HardCoupledCDBAPolicy(
                base, config["environment"]["action_costs"], freeze_allocator=False
            )
            ppo_fields = dict(config["training"])
            ppo_fields.pop("hidden_dim", None)
            trainer = PPOTrainer(agent, PPOConfig(**ppo_fields), device, budget, seed)
            train_scenarios = list(config["train_scenarios"])

            def env_factory(env_seed: int):
                scenario = train_scenarios[env_seed % len(train_scenarios)]
                return DynamicBudgetSchedulingEnv(_synthetic_config(config, scenario, budget))

            result = trainer.train(env_factory)
            global_lambda = result.global_lambda
            training_seconds = result.elapsed_seconds
            training_history = [dict(row) for row in result.update_history]
            torch.save(
                {
                    "state_dict": base.state_dict(),
                    "policy_config": asdict(policy_config),
                    "global_lambda": global_lambda,
                    "hard_coupling": True,
                },
                run_dir / "model.pt",
            )
            write_json(run_dir / "training_history.json", training_history)
        episodes, steps, latency = evaluate_hard_coupled(
            agent, config, budget, seed, device
        )
        for row in episodes:
            row.update({"mode": mode, "budget": budget, "seed": seed})
        for row in steps:
            row.update({"mode": mode, "budget": budget, "seed": seed})
        pd.DataFrame(episodes).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(steps).to_csv(
            run_dir / "steps.csv.gz", index=False, compression="gzip"
        )
        write_json(
            run_dir / "runtime.json",
            {
                "parameter_count": agent.parameter_count(),
                "training_seconds": training_seconds,
                "decision_latency": latency,
                "device": str(device),
                "global_lambda": global_lambda,
            },
        )
        artifacts = ["config.json", "environment.json", "metrics.csv", "steps.csv.gz", "runtime.json"]
        if mode == "d2":
            artifacts += ["model.pt", "training_history.json"]
        ended_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema": "light.run_manifest.v3",
            "run_id": run_id,
            "status": "completed",
            "termination": "hard_coupling_diagnostic_complete",
            "completion": {"oracle": "PASS", "formal_claim_eligible": True},
            "seed": {"role": "paired_randomness_estimation", "value": seed},
            "started_at": started_at,
            "ended_at": ended_at,
            "config_sha256": sha256_file(run_dir / "config.json"),
            "code_sha256": sha256_tree(project_root),
            "input_sha256": {
                "source_checkpoint": (
                    sha256_file(source_checkpoint) if source_checkpoint is not None else None
                )
            },
            "artifacts": {name: sha256_file(run_dir / name) for name in artifacts},
            "guardrails": ["cheapest_action_always_valid", "old_cdba_read_only", "paired_seeds"],
        }
        write_json(run_dir / "manifest.json", manifest)
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
