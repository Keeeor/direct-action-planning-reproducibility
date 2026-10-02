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

from dap.data.trace_windows import select_trace_window
from dap.envs.synthetic_queue_env import SyntheticQueueConfig
from dap.envs.trace_driven_env import TraceDrivenQueueEnv
from dap.evaluation.metrics import summarize_episode
from dap.experiment import load_config
from dap.utils.artifacts import environment_record, sha256_file, sha256_tree, write_json
from dap.utils.seed import set_global_seed

from .synthetic_experiment import _build_agent
from .synthetic_joint import AbsoluteBudgetObservationWrapper


def _load_domains(project_root: Path, config: dict):
    root = project_root / config["trace"]["processed_dir"]
    domains = {}
    for domain in config["trace"]["domains"]:
        with np.load(root / f"{domain}.npz") as bundle:
            domains[domain] = {
                key: bundle[key].copy() for key in ("train", "validation", "test")
            }
    return domains


def _trace_env(config, trace, budget, budget_scale, scenario):
    env = config["environment"]
    base = TraceDrivenQueueEnv(
        trace,
        SyntheticQueueConfig(
            horizon=int(env["horizon"]),
            budget=float(budget),
            scenario=scenario,
            base_capacity=float(env.get("base_capacity", 5.0)),
            slo_latency=float(env.get("slo_latency", 3.0)),
        ),
    )
    return AbsoluteBudgetObservationWrapper(base, budget_scale)


def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(spearmanr(x, y).statistic)


def evaluate_trace(agent, config, domains, budgets, budget_scale, seed, device):
    episodes = []
    steps = []
    latency_rows = []
    horizon = int(config["environment"]["horizon"])
    for budget in budgets:
        for domain in config["trace"]["domains"]:
            for split in ("validation", "test"):
                scenario = f"azure_{domain}_{split}"
                latencies = []
                for episode in range(int(config["evaluation_episodes"])):
                    eval_seed = seed + 700_001 + episode * 100_003
                    trace, start = select_trace_window(
                        domains[domain][split], horizon, eval_seed
                    )
                    env = _trace_env(config, trace, budget, budget_scale, scenario)
                    obs, _ = env.reset(seed=eval_seed)
                    if hasattr(agent, "reset_budget_controller"):
                        agent.reset_budget_controller()
                    current = []
                    mus = []
                    raw = []
                    kls = []
                    while True:
                        started = time.perf_counter_ns()
                        with torch.no_grad():
                            output = agent.act(
                                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                                deterministic=bool(config.get("deterministic_evaluation", False)),
                            )
                        latencies.append((time.perf_counter_ns() - started) / 1e6)
                        next_obs, reward, terminated, truncated, info = env.step(
                            int(output.action.item())
                        )
                        row = {**info, "reward": reward, "episode": episode, "trace_start": start}
                        if hasattr(output, "shadow_price"):
                            row.update(
                                {
                                    "shadow_price": float(output.shadow_price.item()),
                                    "raw_shadow_price": float(output.raw_shadow_price.item()),
                                    "price_policy_kl": float(output.policy_kl.item()),
                                }
                            )
                            mus.append(row["shadow_price"])
                            raw.append(row["raw_shadow_price"])
                            kls.append(row["price_policy_kl"])
                        current.append(row)
                        steps.append(row)
                        obs = next_obs
                        if terminated or truncated:
                            break
                    summary = summarize_episode(current, budget, horizon)
                    risk = np.asarray([row["risk_level"] for row in current])
                    costs = np.asarray([row["resource_cost"] for row in current])
                    low, high = np.quantile(risk, [0.25, 0.75])
                    low_cost = float(costs[risk <= low].mean())
                    high_cost = float(costs[risk >= high].mean())
                    summary.update(
                        {
                            "domain": domain,
                            "split": split,
                            "scenario": scenario,
                            "budget": budget,
                            "eval_episode": episode,
                            "eval_seed": eval_seed,
                            "trace_start": start,
                            "risk_action_cost_correlation": _corr(risk, costs),
                            "low_risk_action_cost": low_cost,
                            "high_risk_action_cost": high_cost,
                            "action_reallocation_difference": high_cost - low_cost,
                            "shadow_price_mean": float(np.mean(mus)) if mus else np.nan,
                            "raw_negative_mu_rate": float(np.mean(np.asarray(raw) < 0)) if raw else np.nan,
                            "price_policy_kl_mean": float(np.mean(kls)) if kls else np.nan,
                        }
                    )
                    episodes.append(summary)
                latency_rows.append(
                    {
                        "budget": budget,
                        "scenario": scenario,
                        "decision_latency_ms_mean": float(np.mean(latencies)),
                        "decision_latency_ms_p95": float(np.quantile(latencies, 0.95)),
                    }
                )
    return episodes, steps, latency_rows


def run_trace_joint_experiment(
    project_root, config_path, method, seed, variant="formal", total_steps=None
):
    project_root = Path(project_root).resolve()
    config = load_config(Path(config_path).resolve())
    if total_steps is not None:
        config["training"]["total_steps"] = int(total_steps)
    budgets = [float(value) for value in config["budgets"]]
    budget_scale = max(budgets)
    domains = _load_domains(project_root, config)
    run_id = f"trace_joint__{variant}__{method}__s{seed}"
    run_dir = project_root / "results/dynamic_shadow_price/trace" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    resolved = {**config, "method": method, "seed": seed, "variant": variant, "budget_scale": budget_scale, "run_id": run_id}
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    try:
        agent, trainer, checkpoint_agent = _build_agent(
            config, method, budget_scale, seed, device
        )
        domain_names = list(config["trace"]["domains"])
        horizon = int(config["environment"]["horizon"])

        def env_factory(env_seed: int):
            domain = domain_names[env_seed % len(domain_names)]
            budget = budgets[(env_seed // len(domain_names)) % len(budgets)]
            trace, _ = select_trace_window(domains[domain]["train"], horizon, env_seed)
            return _trace_env(config, trace, budget, budget_scale, f"azure_{domain}_train")

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
        episodes, steps, latency = evaluate_trace(
            agent, config, domains, budgets, budget_scale, seed, device
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
        processed = project_root / config["trace"]["processed_dir"]
        ended_at = datetime.now(timezone.utc).isoformat()
        write_json(
            run_dir / "manifest.json",
            {
                "schema": "light.run_manifest.v3",
                "run_id": run_id,
                "status": "completed",
                "termination": "joint_budget_chronological_trace_evaluation_complete",
                "completion": {"oracle": "PASS", "formal_claim_eligible": variant == "formal"},
                "seed": {"role": "paired_randomness_estimation", "value": seed},
                "started_at": started_at,
                "ended_at": ended_at,
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(project_root),
                "input_sha256": {domain: sha256_file(processed / f"{domain}.npz") for domain in domain_names},
                "split_contract": "days01-08 train; days09-11 validation; days12-14 test; no validation/test fitting",
                "artifacts": {name: sha256_file(run_dir / name) for name in artifacts},
                "guardrails": ["chronological_split", "same_joint_budget_training_all_methods", "hard_global_budget", "paired_windows"],
            },
        )
        return run_dir
    except Exception as exc:
        write_json(run_dir / "failure.json", {"run_id": run_id, "status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
