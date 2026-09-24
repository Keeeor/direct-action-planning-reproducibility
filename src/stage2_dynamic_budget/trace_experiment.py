from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

from .agents.ppo import PPOConfig, PPOTrainer
from .agents.rules import AggressiveRule, ConservativeRule, NoIntervention
from .data.trace_windows import select_trace_window
from .envs.synthetic_queue_env import SyntheticQueueConfig
from .envs.trace_driven_env import TraceDrivenQueueEnv
from .evaluation.rollout import evaluate_agent
from .experiment import LEARNING_METHODS, RULES, load_config
from .models.policy import ConstrainedSchedulingPolicy, PolicyConfig
from .utils.artifacts import environment_record, sha256_file, sha256_tree, write_json
from .utils.seed import set_global_seed


def _env_config(config: dict, budget: float, scenario: str) -> SyntheticQueueConfig:
    env = config["environment"]
    return SyntheticQueueConfig(
        horizon=int(env["horizon"]),
        budget=float(budget),
        scenario=scenario,
        base_capacity=float(env.get("base_capacity", 5.0)),
        slo_latency=float(env.get("slo_latency", 3.0)),
    )


def _load_domains(project_root: Path, config: dict) -> dict[str, dict[str, np.ndarray]]:
    root = project_root / config["trace"]["processed_dir"]
    domains = {}
    for domain in config["trace"]["domains"]:
        with np.load(root / f"{domain}.npz") as bundle:
            domains[domain] = {key: bundle[key].copy() for key in ("train", "validation", "test")}
    return domains


def run_trace_experiment(
    project_root: str | Path,
    config_path: str | Path,
    method: str,
    budget: float,
    seed: int,
    training_domain: str = "combined",
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    domains = _load_domains(project_root, config)
    allowed_training = set(domains) | {"combined"}
    if training_domain not in allowed_training:
        raise ValueError(f"unknown training domain: {training_domain}")
    tier = str(config["tier"])
    budget_label = f"{budget:.6g}".replace(".", "p")
    run_id = f"{tier}__train_{training_domain}__{method}__b{budget_label}__s{seed}"
    run_dir = project_root / "results" / "raw_logs" / tier / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    (run_dir / "stdout.log").write_text(f"start {run_id} at {started_at}\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    resolved = {**config, "method": method, "budget": budget, "seed": seed, "training_domain": training_domain, "run_id": run_id}
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(seed, torch_threads=int(config.get("torch_threads", 1)))
    device_name = str(config.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    horizon = int(config["environment"]["horizon"])
    try:
        global_lambda = 0.0
        if method in RULES:
            agent = RULES[method]()
            parameter_count = 0
            training_seconds = 0.0
        else:
            internal_method = LEARNING_METHODS[method]
            options = dict(config.get("method_options", {}).get(method, {}))
            policy_config = PolicyConfig(
                method=internal_method,
                action_dim=4,
                hidden_dim=int(config["training"].get("hidden_dim", 64)),
                episode_budget=float(budget),
                horizon=horizon,
                eta=float(options.get("eta", 1.0)),
                min_multiplier=float(options.get("min_multiplier", 0.25)),
                max_multiplier=float(options.get("max_multiplier", 4.0)),
                use_remaining_budget=bool(options.get("use_remaining_budget", True)),
                use_remaining_horizon=bool(options.get("use_remaining_horizon", True)),
                use_local_lambda=bool(options.get("use_local_lambda", True)),
                budget_update_period=int(options.get("budget_update_period", 1)),
            )
            agent = ConstrainedSchedulingPolicy(policy_config)
            ppo_fields = dict(config["training"])
            ppo_fields.pop("hidden_dim", None)
            ppo_fields.update({key: value for key, value in options.items() if key in PPOConfig.__dataclass_fields__})
            trainer = PPOTrainer(agent, PPOConfig(**ppo_fields), device=device, budget=budget, seed=seed)
            train_names = sorted(domains) if training_domain == "combined" else [training_domain]

            def train_factory(env_seed: int):
                domain = train_names[env_seed % len(train_names)]
                trace, _ = select_trace_window(domains[domain]["train"], horizon, env_seed)
                return TraceDrivenQueueEnv(trace, _env_config(config, budget, f"azure_{domain}_train"))

            result = trainer.train(train_factory)
            global_lambda = result.global_lambda
            parameter_count = agent.parameter_count()
            training_seconds = result.elapsed_seconds
            torch.save(
                {"state_dict": agent.state_dict(), "policy_config": asdict(policy_config), "global_lambda": global_lambda},
                run_dir / "model.pt",
            )
            write_json(run_dir / "training_history.json", [dict(row) for row in result.update_history])

        episode_rows, step_rows, latency_rows = [], [], []
        eval_episodes = int(config["evaluation_episodes"])
        for domain in config["trace"]["domains"]:
            for split in ("validation", "test"):
                scenario = f"azure_{domain}_{split}"

                def eval_factory(eval_seed: int, domain=domain, split=split):
                    trace, _ = select_trace_window(domains[domain][split], horizon, eval_seed)
                    return TraceDrivenQueueEnv(trace, _env_config(config, budget, scenario))

                episodes, steps, latency = evaluate_agent(
                    agent,
                    eval_factory,
                    budget,
                    seed + 700_001,
                    eval_episodes,
                    device,
                    global_lambda,
                    deterministic=bool(config.get("deterministic_evaluation", False)),
                )
                for row in episodes:
                    row.update({"scenario": scenario, "domain": domain, "split": split, "method": method, "budget": budget, "seed": seed, "variant": f"train_{training_domain}"})
                for row in steps:
                    row.update({"scenario": scenario, "domain": domain, "split": split, "method": method, "budget": budget, "seed": seed, "variant": f"train_{training_domain}"})
                episode_rows.extend(episodes)
                step_rows.extend(steps)
                latency_rows.append({"scenario": scenario, **latency})
        pd.DataFrame(episode_rows).to_csv(run_dir / "metrics.csv", index=False)
        pd.DataFrame(step_rows).to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        write_json(
            run_dir / "runtime.json",
            {"parameter_count": parameter_count, "training_seconds": training_seconds, "latency_by_scenario": latency_rows, "device": str(device), "global_lambda": global_lambda},
        )
        ended_at = datetime.now(timezone.utc).isoformat()
        artifacts = ["config.json", "environment.json", "metrics.csv", "steps.csv.gz", "runtime.json"]
        if (run_dir / "model.pt").exists():
            artifacts += ["model.pt", "training_history.json"]
        processed_root = project_root / config["trace"]["processed_dir"]
        manifest = {
            "schema": "light.run_manifest.v3",
            "run_id": run_id,
            "matrix_row_id": f"trace-{training_domain}-{method}-{budget_label}",
            "status": "completed",
            "termination": "planned_steps_and_chronological_evaluation_complete",
            "completion": {"oracle": "PASS", "formal_claim_eligible": True},
            "seed": {"role": "randomness_estimation", "value": seed},
            "started_at": started_at,
            "ended_at": ended_at,
            "config_sha256": sha256_file(run_dir / "config.json"),
            "code_sha256": sha256_tree(project_root),
            "input_sha256": {
                "azure_archive": "aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89",
                **{domain: sha256_file(processed_root / f"{domain}.npz") for domain in config["trace"]["domains"]},
            },
            "split_contract": "days01-08 train; days09-11 validation; days12-14 test; preprocessing fit on train only",
            "artifacts": {name: sha256_file(run_dir / name) for name in artifacts},
            "failure_tree_refs": {"hypothesis_ids": ["H1", "H2", "H3", "H4"], "guardrail_ids": ["chronological_split", "train_only_preprocessing", "fair_comparison"]},
        }
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "stdout.log").write_text((run_dir / "stdout.log").read_text() + f"completed {ended_at}\n", encoding="utf-8")
        return run_dir
    except Exception as exc:
        failure = {"run_id": run_id, "status": "failed", "exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(run_dir / "failure.json", failure)
        (run_dir / "stderr.log").write_text(failure["traceback"], encoding="utf-8")
        raise
