from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.dynamic_shadow_price.hard_coupling import GlobalBudgetMaskedPolicy
from stage2_dynamic_budget.experiment import load_config
from stage2_dynamic_budget.models.policy import ConstrainedSchedulingPolicy, PolicyConfig
from stage2_dynamic_budget.utils.artifacts import (
    environment_record,
    sha256_file,
    sha256_tree,
    write_json,
)
from stage2_dynamic_budget.utils.seed import set_global_seed

from .branching import BranchableDiscreteEnv
from .datasets import generate_branch_dataset, summarize_branch_fidelity
from .dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    action_truth_frame,
    solve_action_dp,
)
from .evaluation import (
    ExactDPAgent,
    confusion_table,
    evaluate_full_state_policy,
    evaluate_policy_rollouts,
)
from .figures import generate_all_figures
from .gate import assess_continuation
from .model import ActionAdvantageModel
from .policy import (
    ACBAAPolicy,
    BASELINE_METHODS,
    load_frozen_b4_base,
    load_frozen_baseline,
)
from .training import (
    ACBABTrainer,
    AdvantageFitConfig,
    acba_b_ppo_config,
    fit_advantage_model,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dp_config(config: dict, scenario: str) -> ACBADPConfig:
    env = config["environment"]
    return ACBADPConfig(
        horizon=int(env["horizon"]),
        max_budget=int(env["max_budget"]),
        max_queue=int(env["max_queue"]),
        scenario=scenario,
        gamma=float(env["gamma"]),
        action_costs=tuple(int(value) for value in env["action_costs"]),
        action_capacity=tuple(int(value) for value in env["action_capacity"]),
    )


def _load_advantage_checkpoint(path: Path, device: torch.device) -> ActionAdvantageModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ActionAdvantageModel(
        action_dim=int(checkpoint["action_dim"]), hidden_dim=int(checkpoint["hidden_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _load_acba_b_checkpoint(path: Path, device: torch.device) -> ConstrainedSchedulingPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    base = ConstrainedSchedulingPolicy(PolicyConfig(**dict(checkpoint["config"]))).to(device)
    base.load_state_dict(checkpoint["state_dict"], strict=True)
    base.eval()
    return base


def _resolved_config(config: dict, run_id: str, smoke: bool) -> dict:
    resolved = json.loads(json.dumps(config))
    resolved["run_id"] = run_id
    resolved["smoke"] = bool(smoke)
    if smoke:
        resolved["seeds"] = [0]
        resolved["scenarios"] = ["early_burst"]
        resolved["budgets"] = [4]
        resolved["branch_horizons"] = [1, 5]
        resolved["evaluation_episodes"] = 2
        resolved["acba"]["training_branch_horizon"] = 1
        resolved["acba"]["training_epochs"] = 2
        resolved["acba"]["ppo_finetune_steps"] = 512
        resolved["acba"]["ppo_rollout_steps"] = 128
    return resolved


def _truth_diagnostics(truth: pd.DataFrame, action_costs: tuple[int, ...]) -> dict[str, object]:
    state = truth[truth.action == 0].copy()
    state["risk_score"] = state.load + state.queue
    threshold = float(np.quantile(state.risk_score, 0.75))
    high = state[state.risk_score >= threshold]
    equal_cost_pairs = [
        [left, right]
        for left in range(len(action_costs))
        for right in range(left + 1, len(action_costs))
        if action_costs[left] == action_costs[right]
    ]
    return {
        "rows": len(truth),
        "states": len(state),
        "high_risk_threshold": threshold,
        "high_risk_optimal_lowest_cost_proportion": float((high.optimal_action == 0).mean()),
        "equal_cost_action_pairs": equal_cost_pairs,
        "equal_cost_ordering_status": (
            "identifiable" if equal_cost_pairs else "NA_no_equal_cost_actions_in_primary_environment"
        ),
        "max_absolute_a0_advantage": float(
            truth.loc[truth.action == 0, "A_star"].abs().max()
        ),
    }


def _equal_cost_config(config: dict, scenario: str) -> ACBADPConfig:
    primary = _dp_config(config, scenario)
    return ACBADPConfig(
        horizon=primary.horizon,
        max_budget=primary.max_budget,
        max_queue=primary.max_queue,
        scenario=scenario,
        gamma=primary.gamma,
        action_costs=(0, 1, 2, 2),
        action_capacity=primary.action_capacity,
        action_activation_penalty=(0.0, 0.0, 0.0, 0.75),
        load_arrivals=primary.load_arrivals,
        queue_penalty=primary.queue_penalty,
        severe_queue_penalty=primary.severe_queue_penalty,
    )


def _equal_cost_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    pair = frame[(frame.action.isin([2, 3])) & frame.feasible].pivot(
        index=["scenario", "t", "load", "queue", "remaining_budget"],
        columns="action",
        values="Q_star",
    )
    difference = pair[3] - pair[2]
    strict = difference[np.abs(difference) > 1e-10]
    signs = np.sign(strict)
    modal_sign = float(pd.Series(signs).mode().iloc[0]) if len(signs) else float("nan")
    return {
        "status": "auxiliary_not_used_for_primary_gate",
        "equal_cost_pair": [2, 3],
        "equal_cost": 2,
        "action_2_capacity": 3,
        "action_3_capacity": 4,
        "action_3_activation_penalty": 0.75,
        "feasible_pair_states": int(len(pair)),
        "strict_order_states": int(len(strict)),
        "action_2_preferred_proportion": float((strict < 0).mean()),
        "action_3_preferred_proportion": float((strict > 0).mean()),
        "tie_proportion": float((np.abs(difference) <= 1e-10).mean()),
        "ordering_differs_from_modal_state_proportion": float((signs != modal_sign).mean()),
        "both_strict_orderings_observed": bool((strict < 0).any() and (strict > 0).any()),
    }


def _equal_cost_branch_fidelity(frame: pd.DataFrame) -> pd.DataFrame:
    pair = frame[frame.action.isin([2, 3])]
    rows: list[dict[str, object]] = []
    group_keys = ["scenario", "seed", "k_requested"]
    state_keys = ["t", "load", "queue", "remaining_budget"]
    for key, subset in pair.groupby(group_keys, sort=True):
        correct = total = 0
        for _, state in subset.groupby(state_keys, sort=False):
            if len(state) != 2:
                continue
            state = state.sort_values("action")
            truth = np.sign(state.Q_star.iloc[1] - state.Q_star.iloc[0])
            if truth == 0:
                continue
            prediction = np.sign(state.q_branch.iloc[1] - state.q_branch.iloc[0])
            total += 1
            correct += int(prediction == truth)
        rows.append(
            {
                "scenario": key[0],
                "seed": int(key[1]),
                "K": int(key[2]),
                "strict_equal_cost_pairs": total,
                "equal_cost_pair_ranking_accuracy": correct / max(total, 1),
            }
        )
    return pd.DataFrame(rows)


def _fit_models(
    project_root: Path,
    run_dir: Path,
    config: dict,
    branch_frames: dict[int, pd.DataFrame],
    device: torch.device,
) -> tuple[dict[int, Path], dict[tuple[str, int], Path], dict[str, str]]:
    acba = config["acba"]
    action_costs = tuple(float(value) for value in config["environment"]["action_costs"])
    budget_scale = int(config["environment"]["max_budget"])
    advantage_paths: dict[int, Path] = {}
    acba_b_paths: dict[tuple[str, int], Path] = {}
    lineage: dict[str, str] = {}
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_config = AdvantageFitConfig(
        branch_horizon=int(acba["training_branch_horizon"]),
        epochs=int(acba["training_epochs"]),
        batch_size=int(acba["batch_size"]),
        learning_rate=float(acba["learning_rate"]),
        hidden_dim=int(acba["hidden_dim"]),
    )
    for seed in config["seeds"]:
        model, history = fit_advantage_model(branch_frames[int(seed)], fit_config, int(seed), device)
        model_path = model_dir / f"acba_advantage_s{seed}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "action_dim": len(action_costs),
                "hidden_dim": fit_config.hidden_dim,
                "seed": int(seed),
                "fit_config": asdict(fit_config),
            },
            model_path,
        )
        write_json(model_dir / f"acba_advantage_s{seed}_history.json", history)
        advantage_paths[int(seed)] = model_path

    for scenario in config["scenarios"]:
        mdp_config = _dp_config(config, scenario)
        for seed in config["seeds"]:
            seed = int(seed)
            teacher = _load_advantage_checkpoint(advantage_paths[seed], device)
            base, source_path, source_lambda = load_frozen_b4_base(
                project_root, scenario, seed, device
            )
            lineage[str(source_path.relative_to(project_root))] = sha256_file(source_path)
            source_config = json.loads((source_path.parent / "config.json").read_text(encoding="utf-8"))
            agent = GlobalBudgetMaskedPolicy(base, action_costs, budget_scale).to(device)
            trainer = ACBABTrainer(
                agent,
                acba_b_ppo_config(
                    source_config["training"],
                    int(acba["ppo_finetune_steps"]),
                    int(acba["ppo_rollout_steps"]),
                ),
                device,
                budget_scale,
                seed,
                teacher=teacher,
                ranking_coef=float(acba["ranking_coef"]),
                ranking_margin=float(acba["ranking_margin"]),
            )
            trainer.global_lambda = source_lambda
            budgets = [int(value) for value in config["budgets"]]

            def env_factory(env_seed: int, cfg=mdp_config):
                budget = budgets[env_seed % len(budgets)]
                return BranchableDiscreteEnv(cfg, budget, budget_scale=budget_scale)

            train_result = trainer.train(env_factory)
            model_path = model_dir / f"acba_b__{scenario}__s{seed}.pt"
            torch.save(
                {
                    "state_dict": base.state_dict(),
                    "config": asdict(base.config),
                    "global_lambda": train_result.global_lambda,
                    "source_checkpoint": str(source_path.relative_to(project_root)),
                    "source_checkpoint_sha256": sha256_file(source_path),
                    "scenario": scenario,
                    "seed": seed,
                },
                model_path,
            )
            write_json(
                model_dir / f"acba_b__{scenario}__s{seed}_history.json",
                train_result.update_history,
            )
            acba_b_paths[(scenario, seed)] = model_path
    return advantage_paths, acba_b_paths, lineage


def _method_agents(
    project_root: Path,
    scenario: str,
    seed: int,
    mdp: ActionConditionedBudgetMDP,
    optimum,
    advantage_path: Path,
    acba_b_path: Path,
    config: dict,
    device: torch.device,
):
    costs = tuple(float(value) for value in config["environment"]["action_costs"])
    budget_scale = float(config["environment"]["max_budget"])
    yield "optimal", ExactDPAgent(mdp, optimum), None
    for method in BASELINE_METHODS:
        agent, source = load_frozen_baseline(
            project_root, method, scenario, seed, costs, budget_scale, device
        )
        yield method, agent, source
    teacher = _load_advantage_checkpoint(advantage_path, device)
    base, source, _ = load_frozen_b4_base(project_root, scenario, seed, device)
    base.eval()
    yield (
        "acba_a",
        ACBAAPolicy(
            base,
            teacher,
            costs,
            budget_scale,
            float(config["acba"]["alpha"]),
        ).to(device),
        source,
    )
    acba_b_base = _load_acba_b_checkpoint(acba_b_path, device)
    yield (
        "acba_b",
        GlobalBudgetMaskedPolicy(acba_b_base, costs, budget_scale).to(device),
        acba_b_path,
    )


def run_minimal_validation(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "minimal_v1",
    smoke: bool = False,
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = _resolved_config(load_config(config_path), run_id, smoke)
    run_dir = project_root / "results/action_conditioned_budget_advantage" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("status") == "completed":
                return run_dir
        raise RuntimeError(f"run directory already contains an incomplete run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = _utcnow()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    set_global_seed(0, torch_threads=int(config.get("torch_threads", 1)))
    device = torch.device(str(config.get("device", "cpu")))
    try:
        write_json(run_dir / "stage_status.json", {"stage": "exact_truth", "updated_at": _utcnow()})
        truth_frames: list[pd.DataFrame] = []
        optima: dict[str, tuple[ActionConditionedBudgetMDP, object]] = {}
        for scenario in config["scenarios"]:
            mdp = ActionConditionedBudgetMDP(_dp_config(config, scenario))
            optimum = solve_action_dp(mdp)
            if optimum.max_bellman_residual > 1e-10:
                raise RuntimeError("exact DP Bellman residual exceeded tolerance")
            optima[scenario] = (mdp, optimum)
            truth_frames.append(action_truth_frame(mdp, optimum, scenario))
        truth = pd.concat(truth_frames, ignore_index=True)
        truth.to_csv(run_dir / "action_truth.csv.gz", index=False, compression="gzip")
        truth_diagnostics = _truth_diagnostics(
            truth, tuple(int(value) for value in config["environment"]["action_costs"])
        )
        write_json(run_dir / "truth_diagnostics.json", truth_diagnostics)

        equal_cost_truth_frames: list[pd.DataFrame] = []
        equal_cost_optima: dict[str, tuple[ActionConditionedBudgetMDP, object]] = {}
        for scenario in config["scenarios"]:
            auxiliary_mdp = ActionConditionedBudgetMDP(_equal_cost_config(config, scenario))
            auxiliary_optimum = solve_action_dp(auxiliary_mdp)
            equal_cost_optima[scenario] = (auxiliary_mdp, auxiliary_optimum)
            equal_cost_truth_frames.append(
                action_truth_frame(auxiliary_mdp, auxiliary_optimum, scenario)
            )
        equal_cost_truth = pd.concat(equal_cost_truth_frames, ignore_index=True)
        equal_cost_truth.to_csv(
            run_dir / "equal_cost_action_truth.csv.gz", index=False, compression="gzip"
        )
        write_json(
            run_dir / "equal_cost_diagnostics.json",
            _equal_cost_diagnostics(equal_cost_truth),
        )

        write_json(
            run_dir / "stage_status.json",
            {"stage": "branch_datasets", "seed": None, "updated_at": _utcnow()},
        )
        branch_frames: dict[int, pd.DataFrame] = {}
        fidelity_frames: list[pd.DataFrame] = []
        equal_cost_fidelity_frames: list[pd.DataFrame] = []
        branch_dir = run_dir / "branches"
        branch_dir.mkdir()
        for seed in config["seeds"]:
            write_json(
                run_dir / "stage_status.json",
                {"stage": "branch_datasets", "seed": int(seed), "updated_at": _utcnow()},
            )
            frames = []
            for scenario in config["scenarios"]:
                mdp, optimum = optima[scenario]
                frames.append(
                    generate_branch_dataset(
                        mdp,
                        optimum,
                        tuple(int(value) for value in config["branch_horizons"]),
                        int(seed),
                    )
                )
            branch_frame = pd.concat(frames, ignore_index=True)
            branch_frame.to_csv(
                branch_dir / f"action_branches_s{seed}.csv.gz",
                index=False,
                compression="gzip",
            )
            branch_frames[int(seed)] = branch_frame
            fidelity_frames.append(summarize_branch_fidelity(branch_frame))
            auxiliary_frames = []
            for scenario in config["scenarios"]:
                auxiliary_mdp, auxiliary_optimum = equal_cost_optima[scenario]
                auxiliary_frames.append(
                    generate_branch_dataset(
                        auxiliary_mdp,
                        auxiliary_optimum,
                        tuple(int(value) for value in config["branch_horizons"]),
                        int(seed),
                    )
                )
            auxiliary_frame = pd.concat(auxiliary_frames, ignore_index=True)
            auxiliary_frame.to_csv(
                branch_dir / f"auxiliary_equal_cost_branches_s{seed}.csv.gz",
                index=False,
                compression="gzip",
            )
            equal_cost_fidelity_frames.append(_equal_cost_branch_fidelity(auxiliary_frame))
        branch_fidelity = pd.concat(fidelity_frames, ignore_index=True)
        branch_fidelity.to_csv(run_dir / "branch_fidelity.csv", index=False)
        pd.concat(equal_cost_fidelity_frames, ignore_index=True).to_csv(
            run_dir / "equal_cost_branch_fidelity.csv", index=False
        )

        write_json(run_dir / "stage_status.json", {"stage": "model_training", "updated_at": _utcnow()})
        advantage_paths, acba_b_paths, lineage = _fit_models(
            project_root, run_dir, config, branch_frames, device
        )
        del branch_frames

        state_frames: list[pd.DataFrame] = []
        state_summaries: list[dict[str, object]] = []
        episode_frames: list[pd.DataFrame] = []
        step_frames: list[pd.DataFrame] = []
        runtime_rows: list[dict[str, object]] = []
        write_json(run_dir / "stage_status.json", {"stage": "policy_evaluation", "updated_at": _utcnow()})
        for scenario in config["scenarios"]:
            mdp, optimum = optima[scenario]
            for seed in config["seeds"]:
                seed = int(seed)
                for method, agent, source in _method_agents(
                    project_root,
                    scenario,
                    seed,
                    mdp,
                    optimum,
                    advantage_paths[seed],
                    acba_b_paths[(scenario, seed)],
                    config,
                    device,
                ):
                    if source is not None and source.is_relative_to(project_root):
                        lineage[str(source.relative_to(project_root))] = sha256_file(source)
                    state_frame, state_summary = evaluate_full_state_policy(
                        agent, method, mdp, optimum, scenario, seed, device
                    )
                    episodes, steps, runtime = evaluate_policy_rollouts(
                        agent,
                        method,
                        mdp,
                        optimum,
                        [int(value) for value in config["budgets"]],
                        scenario,
                        seed,
                        int(config["evaluation_episodes"]),
                        device,
                    )
                    state_frames.append(state_frame)
                    state_summaries.append(state_summary)
                    episode_frames.append(episodes)
                    step_frames.append(steps)
                    runtime_rows.append(
                        {"method": method, "scenario": scenario, "seed": seed, **runtime}
                    )
        state_rows = pd.concat(state_frames, ignore_index=True)
        state_summary_frame = pd.DataFrame(state_summaries)
        episodes = pd.concat(episode_frames, ignore_index=True)
        steps = pd.concat(step_frames, ignore_index=True)
        state_rows.to_csv(run_dir / "state_policy_actions.csv.gz", index=False, compression="gzip")
        state_summary_frame.to_csv(run_dir / "state_policy_summary.csv", index=False)
        confusion_table(state_rows).to_csv(run_dir / "action_confusions.csv", index=False)
        episodes.to_csv(run_dir / "metrics.csv", index=False)
        steps.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        pd.DataFrame(runtime_rows).to_csv(run_dir / "runtime.csv", index=False)
        cell_summary = episodes.groupby(
            ["method", "scenario", "budget", "seed"], as_index=False
        ).mean(numeric_only=True)
        cell_summary.to_csv(run_dir / "cell_summary.csv", index=False)

        gate_config = config["continuation_gate"]
        gate = assess_continuation(
            episodes,
            state_summary_frame,
            int(gate_config["paired_budget_seed_wins_required"]),
            int(gate_config["burst_seed_wins_required"]),
            float(gate_config["high_risk_balanced_accuracy_floor"]),
        )
        if smoke:
            gate["decision"] = "NOT_EVALUATED_SMOKE"
            gate["selected_method"] = None
        write_json(run_dir / "continuation_gate.json", gate)
        write_json(
            run_dir / "lineage.json",
            {
                "frozen_predecessor_checkpoints": lineage,
                "periodic_source_rule": "frozen early_burst checkpoint of same method and seed",
                "old_branches_modified": False,
            },
        )
        figure_paths = generate_all_figures(
            truth, state_rows, episodes, steps, run_dir / "figures"
        )
        write_json(run_dir / "stage_status.json", {"stage": "completed", "updated_at": _utcnow()})

        artifact_paths = sorted(
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "failure.json"}
        )
        write_json(
            run_dir / "manifest.json",
            {
                "schema": "light.run_manifest.v3",
                "run_id": run_id,
                "status": "completed",
                "termination": "acba_minimal_validation_complete",
                "completion": {
                    "oracle": "PASS",
                    "formal_claim_eligible": not smoke,
                    "continuation_decision": gate["decision"],
                },
                "started_at": started_at,
                "ended_at": _utcnow(),
                "config_sha256": sha256_file(run_dir / "config.json"),
                "code_sha256": sha256_tree(
                    project_root
                    / "src/stage2_dynamic_budget/action_conditioned_budget_advantage"
                ),
                "artifacts": {
                    str(path.relative_to(run_dir)): sha256_file(path) for path in artifact_paths
                },
                "figure_count": len(figure_paths),
                "guardrails": [
                    "frozen_predecessors_read_only",
                    "hard_global_budget",
                    "paired_common_random_numbers",
                    "exact_dp_action_truth",
                    "no_q_star_training_targets",
                    "preregistered_continuation_gate",
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
                "failed_at": _utcnow(),
                "elapsed_seconds": time.time(),
            },
        )
        raise
