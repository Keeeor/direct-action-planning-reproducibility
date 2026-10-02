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

from dap.action_conditioned_budget_advantage.branching import (
    BranchableDiscreteEnv,
)
from dap.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    action_truth_frame,
    solve_action_dp,
)
from dap.action_conditioned_budget_advantage.evaluation import (
    ExactDPAgent,
    confusion_table,
    evaluate_full_state_policy,
    evaluate_policy_rollouts,
)
from dap.action_conditioned_budget_advantage.policy import (
    load_frozen_baseline,
)
from dap.direct_action_planning.learning import (
    EmpiricalActionModel,
    collect_transition_samples,
    fit_empirical_action_model,
    solve_empirical_value,
)
from dap.direct_action_planning.planning import (
    BudgetValueTable,
    DirectPlanningAgent,
)
from dap.experiment import load_config
from dap.utils.artifacts import (
    environment_record,
    sha256_file,
    write_json,
)
from dap.utils.seed import set_global_seed

from .aggregation import (
    aggregate_branch_labels,
    collect_closed_loop_branch_labels,
    visitation_distribution_distance,
)
from .data import attach_priority_weights, collect_common_random_branch_data
from .gate import assess_repair_gate
from .model import (
    LossWeights,
    TrainingConfig,
    model_checkpoint_payload,
    train_structured_model,
)
from .planning import EnsemblePlanningAgent, StructuredPlanningAgent


ABLATION_WEIGHTS = {
    "structured": (1.0, 0.0, 0.0, 0.0),
    "structured_effect": (1.0, 2.0, 0.0, 0.0),
    "structured_effect_q": (1.0, 2.0, 4.0, 0.0),
    "structured_effect_q_rank": (1.0, 2.0, 4.0, 2.0),
}


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


def _resolved_config(config: dict, run_id: str, smoke: bool) -> dict:
    resolved = json.loads(json.dumps(config))
    resolved["run_id"] = run_id
    resolved["smoke"] = bool(smoke)
    resolved["resolved_at"] = _utcnow()
    if smoke:
        resolved["budgets"] = [4]
        resolved["scenarios"] = ["early_burst"]
        resolved["seeds"] = [5]
        resolved["evaluation_episodes"] = 2
        resolved["repair"]["train_crn_samples_per_state"] = 12
        resolved["repair"]["validation_crn_samples_per_state"] = 4
        resolved["repair"]["max_epochs"] = 30
        resolved["repair"]["patience"] = 10
        resolved["repair"]["ensemble_members"] = 3
        resolved["repair"]["max_aggregation_rounds"] = 1
        resolved["repair"]["aggregation_probe_episodes"] = 2
    return resolved


def _save_empirical_model(path: Path, model: EmpiricalActionModel) -> None:
    np.savez_compressed(
        path,
        next_load_probabilities=model.next_load_probabilities,
        next_queue_probabilities=model.next_queue_probabilities,
        rewards=model.rewards,
        costs=model.costs,
        samples_per_state_action=np.asarray(model.samples_per_state_action),
        smoothing=np.asarray(model.smoothing),
    )


def _save_value(path: Path, value: BudgetValueTable) -> None:
    np.savez_compressed(path, values=value.values, source=np.asarray(value.source))


def _train_variant(
    labels: pd.DataFrame,
    mdp: ActionConditionedBudgetMDP,
    config: dict,
    model_seed: int,
    weights: tuple[float, float, float, float],
    use_priority: bool,
):
    repair = config["repair"]
    return train_structured_model(
        labels,
        mdp.config.horizon,
        mdp.n_loads,
        mdp.n_actions,
        mdp.config.gamma,
        LossWeights(
            state=weights[0],
            effect=weights[1],
            q=weights[2],
            rank=weights[3],
            rank_margin=float(repair["rank_margin"]),
        ),
        TrainingConfig(
            hidden_dim=int(repair["hidden_dim"]),
            learning_rate=float(repair["learning_rate"]),
            max_epochs=int(repair["max_epochs"]),
            patience=int(repair["patience"]),
            seed=int(model_seed),
            use_priority_weights=bool(use_priority),
        ),
        device=config["device"],
    )


def _probe_agent(
    agent,
    mdp,
    optimum,
    budgets,
    scenario,
    seed,
    episodes,
    device,
) -> dict[str, float]:
    _, state = evaluate_full_state_policy(
        agent, "aggregation_probe", mdp, optimum, scenario, seed, device
    )
    rollout, _, _ = evaluate_policy_rollouts(
        agent,
        "aggregation_probe",
        mdp,
        optimum,
        budgets,
        scenario,
        seed,
        episodes,
        device,
    )
    return {
        "full_state_action_consistency": float(state["action_consistency_rate"]),
        "full_state_q_star_regret": float(state["mean_Q_star_regret"]),
        "rollout_q_star_regret": float(rollout.mean_Q_star_regret.mean()),
        "rollout_budget_mae": float(rollout.budget_trajectory_mae.mean()),
        "completion_rate": float(rollout.completion_rate.mean()),
        "slo_violation_rate": float(rollout.slo_violation_rate.mean()),
        "total_cost": float(rollout.total_cost.mean()),
    }


def _planning_q_rows(agent, method, mdp, optimum, scenario, seed, device):
    env = BranchableDiscreteEnv(
        mdp.config, initial_budget=mdp.config.max_budget, budget_scale=mdp.config.max_budget
    )
    env.reset(seed=seed)
    rows = []
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        observation = env.set_markov_state(t, load, queue, budget)
        with torch.no_grad():
            output = agent.act(
                torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),
                deterministic=True,
            )
        if not hasattr(output, "q_values"):
            continue
        q_values = output.q_values.detach().cpu().numpy()[0]
        for action in np.flatnonzero(np.isfinite(q_values)):
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "t": t,
                    "load": load,
                    "queue": queue,
                    "remaining_budget": budget,
                    "remaining_horizon": mdp.config.horizon - t,
                    "action": int(action),
                    "q_plan": float(q_values[action]),
                    "q_star": float(optimum.q_values[t, load, queue, budget, action]),
                }
            )
    return pd.DataFrame(rows)


def _pair_ranking_summary(q_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["method", "scenario", "seed", "t", "load", "queue", "remaining_budget"]
    for key, group in q_rows.groupby(keys, sort=False):
        values = group.set_index("action")
        actions = sorted(values.index)
        for offset, left in enumerate(actions[:-1]):
            for right in actions[offset + 1 :]:
                true_delta = float(values.loc[left, "q_star"] - values.loc[right, "q_star"])
                plan_delta = float(values.loc[left, "q_plan"] - values.loc[right, "q_plan"])
                if abs(true_delta) <= 1.0e-9:
                    continue
                rows.append(
                    {
                        **dict(zip(keys, key)),
                        "action_i": left,
                        "action_j": right,
                        "ranking_correct": float(true_delta * plan_delta > 0.0),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby(["method", "scenario", "seed", "action_i", "action_j"])
        .ranking_correct.agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "pair_ranking_accuracy"})
    )


def _model_component_diagnostics(model, mdp, method, scenario, seed):
    rows = []
    for t, load, action in np.ndindex(mdp.config.horizon, mdp.n_loads, mdp.n_actions):
        predicted = model.predict_probabilities(t, load, action)
        true = mdp.load_probabilities(t, load)
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "seed": seed,
                "t": t,
                "load": load,
                "action": action,
                "next_load_tv": 0.5 * float(np.abs(predicted - true).sum()),
                "next_load_mean_abs_error": abs(
                    float(np.dot(predicted - true, np.arange(mdp.n_loads)))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    spread = (
        frame.assign(
            predicted_mean=[
                float(np.dot(model.predict_probabilities(r.t, r.load, r.action), np.arange(mdp.n_loads)))
                for r in frame.itertuples()
            ]
        )
        .groupby(["t", "load"])
        .predicted_mean.agg(lambda values: float(values.max() - values.min()))
    )
    return {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "next_load_tv": float(frame.next_load_tv.mean()),
        "next_load_mean_abs_error": float(frame.next_load_mean_abs_error.mean()),
        "cross_action_load_mean_spread": float(spread.mean()),
        "queue_reward_cost_mae": 0.0,
    }


def run_minimal_repair(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "minimal_v1",
    smoke: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = _resolved_config(load_config(config_path), run_id, smoke)
    run_dir = root / "results/direct_action_planning_repair" / run_id
    if run_dir.exists():
        manifest = run_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed":
            return run_dir
        raise RuntimeError(f"run directory exists and is not completed: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    (run_dir / "stdout.log").write_text("", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    set_global_seed(0, torch_threads=int(config["torch_threads"]))
    device = torch.device(config["device"])
    try:
        truth_frames = []
        state_frames, state_summaries = [], []
        episode_frames, step_frames, runtime_rows = [], [], []
        q_frames, model_diagnostic_rows, aggregation_rows = [], [], []
        training_rows = []
        loso_payload = {}
        lineage = {}
        write_json(run_dir / "stage_status.json", {"stage": "training", "at": _utcnow()})
        for scenario_index, scenario in enumerate(config["scenarios"]):
            mdp = ActionConditionedBudgetMDP(_dp_config(config, scenario))
            optimum = solve_action_dp(mdp)
            truth_frames.append(action_truth_frame(mdp, optimum, scenario))
            for seed_value in config["seeds"]:
                seed = int(seed_value)
                cell = run_dir / "cells" / f"{scenario}__s{seed}"
                (cell / "models").mkdir(parents=True)
                (cell / "training").mkdir()
                training_seed = seed + scenario_index * 100_003
                old_samples = collect_transition_samples(mdp, 64, training_seed)
                old_model = fit_empirical_action_model(mdp, old_samples, smoothing=0.25)
                value = solve_empirical_value(mdp, old_model)
                old_samples.to_csv(cell / "old_model_samples.csv.gz", index=False, compression="gzip")
                _save_empirical_model(cell / "old_model.npz", old_model)
                _save_value(cell / "learned_value.npz", value)
                data = collect_common_random_branch_data(
                    mdp,
                    value,
                    optimum,
                    scenario,
                    seed=seed + scenario_index * 100_003 + 500_009,
                    train_samples=int(config["repair"]["train_crn_samples_per_state"]),
                    validation_samples=int(config["repair"]["validation_crn_samples_per_state"]),
                )
                data.samples.to_csv(cell / "D0_branch_samples.csv.gz", index=False, compression="gzip")
                labels = attach_priority_weights(
                    data.labels,
                    mdp,
                    old_model,
                    value,
                    optimum,
                    scenario,
                    float(config["repair"]["close_gap_ceiling"]),
                    float(config["repair"]["priority_weight_ceiling"]),
                )
                labels.to_csv(cell / "D0_branch_labels.csv.gz", index=False, compression="gzip")
                loso_payload[(scenario, seed)] = (mdp, optimum, old_model, value, labels)
                trained = {}
                for variant_index, (name, weights) in enumerate(ABLATION_WEIGHTS.items()):
                    result = _train_variant(
                        labels,
                        mdp,
                        config,
                        seed * 10_000 + variant_index,
                        weights,
                        use_priority=False,
                    )
                    trained[name] = result
                    torch.save(model_checkpoint_payload(result), cell / "models" / f"{name}.pt")
                    result.history.to_csv(cell / "training" / f"{name}.csv", index=False)
                    training_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "method": name,
                            "best_epoch": result.best_epoch,
                            **result.selection_metrics,
                        }
                    )

                aggregate_samples = []
                current = trained["structured_effect_q_rank"]
                aggregation_models = {}
                previous_regret = None
                nonimproving = 0
                max_rounds = int(config["repair"]["max_aggregation_rounds"])
                for model_round in range(max_rounds + 1):
                    current_agent = StructuredPlanningAgent(mdp, value, current.model)
                    probe = _probe_agent(
                        current_agent,
                        mdp,
                        optimum,
                        [int(x) for x in config["budgets"]],
                        scenario,
                        seed,
                        int(config["repair"]["aggregation_probe_episodes"]),
                        device,
                    )
                    improvement = (
                        None
                        if previous_regret is None
                        else previous_regret - probe["rollout_q_star_regret"]
                    )
                    aggregation_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "model_round": model_round,
                            "training_data_round": model_round,
                            "improvement_vs_previous": improvement,
                            **probe,
                        }
                    )
                    if previous_regret is not None:
                        if improvement is not None and improvement > 1.0e-6:
                            nonimproving = 0
                        else:
                            nonimproving += 1
                    previous_regret = probe["rollout_q_star_regret"]
                    if model_round >= max_rounds or nonimproving >= int(
                        config["repair"]["stop_after_nonimproving_rounds"]
                    ):
                        break
                    new_samples, visits = collect_closed_loop_branch_labels(
                        current_agent,
                        mdp,
                        value,
                        optimum,
                        [int(x) for x in config["budgets"]],
                        scenario,
                        seed,
                        int(config["repair"]["aggregation_probe_episodes"]),
                        model_round + 1,
                        device,
                    )
                    new_samples.to_csv(
                        cell / f"D{model_round + 1}_branch_samples.csv.gz",
                        index=False,
                        compression="gzip",
                    )
                    visits.to_csv(cell / f"D{model_round + 1}_visits.csv", index=False)
                    aggregate_samples.append(new_samples)
                    aggregated_labels = aggregate_branch_labels(
                        labels, data.samples, aggregate_samples
                    )
                    aggregated_labels.to_csv(
                        cell / f"D{model_round + 1}_labels.csv.gz",
                        index=False,
                        compression="gzip",
                    )
                    next_result = _train_variant(
                        aggregated_labels,
                        mdp,
                        config,
                        seed * 10_000 + 100 + model_round,
                        ABLATION_WEIGHTS["structured_effect_q_rank"],
                        use_priority=False,
                    )
                    name = f"aggregation_round_{model_round + 1}"
                    aggregation_models[name] = next_result
                    torch.save(
                        model_checkpoint_payload(next_result),
                        cell / "models" / f"{name}.pt",
                    )
                    next_result.history.to_csv(
                        cell / "training" / f"{name}.csv", index=False
                    )
                    aggregation_rows[-1]["new_branch_rows"] = len(new_samples)
                    aggregation_rows[-1]["new_unique_grid_states"] = int(
                        len(new_samples[["t", "load", "queue"]].drop_duplicates().merge(
                            data.samples[["t", "load", "queue"]].drop_duplicates(),
                            on=["t", "load", "queue"],
                            how="left",
                            indicator=True,
                        ).query("_merge == 'left_only'"))
                    )
                    aggregation_rows[-1]["visitation_total_variation_from_D0"] = (
                        visitation_distribution_distance(data.samples, visits)
                    )
                    current = next_result
                final_labels = aggregate_branch_labels(labels, data.samples, aggregate_samples)
                full = _train_variant(
                    final_labels,
                    mdp,
                    config,
                    seed * 10_000 + 700,
                    ABLATION_WEIGHTS["structured_effect_q_rank"],
                    use_priority=True,
                )
                trained["full_repair"] = full
                torch.save(model_checkpoint_payload(full), cell / "models" / "full_repair.pt")
                full.history.to_csv(cell / "training" / "full_repair.csv", index=False)
                training_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": "full_repair",
                        "best_epoch": full.best_epoch,
                        **full.selection_metrics,
                    }
                )
                ensemble = []
                for member in range(int(config["repair"]["ensemble_members"])):
                    member_result = _train_variant(
                        final_labels,
                        mdp,
                        config,
                        seed * 10_000 + 1_000 + member,
                        ABLATION_WEIGHTS["structured_effect_q_rank"],
                        use_priority=True,
                    )
                    ensemble.append(member_result.model)
                    torch.save(
                        model_checkpoint_payload(member_result),
                        cell / "models" / f"ensemble_member_{member}.pt",
                    )
                    member_result.history.to_csv(
                        cell / "training" / f"ensemble_member_{member}.csv", index=False
                    )

                checkpoint_seed = int(
                    config["repair"]["frozen_dsp_checkpoint_seed_map"][str(seed)]
                    if str(seed) in config["repair"]["frozen_dsp_checkpoint_seed_map"]
                    else config["repair"]["frozen_dsp_checkpoint_seed_map"][seed]
                )
                dsp, dsp_path = load_frozen_baseline(
                    root,
                    "dsp_b",
                    scenario,
                    checkpoint_seed,
                    tuple(float(x) for x in mdp.action_costs),
                    float(mdp.config.max_budget),
                    device,
                )
                b4, b4_path = load_frozen_baseline(
                    root,
                    "b4_budget_state",
                    scenario,
                    checkpoint_seed,
                    tuple(float(x) for x in mdp.action_costs),
                    float(mdp.config.max_budget),
                    device,
                )
                lineage[str(dsp_path.relative_to(root))] = sha256_file(dsp_path)
                lineage[str(b4_path.relative_to(root))] = sha256_file(b4_path)
                agents = {
                    "optimal": ExactDPAgent(mdp, optimum),
                    "dsp_b": dsp,
                    "learned_value_branch": DirectPlanningAgent(mdp, value),
                    "original_learned_model": DirectPlanningAgent(
                        mdp, value, learned_model=old_model
                    ),
                    **{
                        name: StructuredPlanningAgent(mdp, value, result.model)
                        for name, result in trained.items()
                    },
                }
                if "aggregation_round_1" in aggregation_models:
                    agents["aggregation_round_1"] = StructuredPlanningAgent(
                        mdp, value, aggregation_models["aggregation_round_1"].model
                    )
                agents["ensemble_mean"] = EnsemblePlanningAgent(
                    mdp,
                    value,
                    ensemble,
                    uncertainty_multiplier=float(
                        config["repair"]["ensemble_uncertainty_multiplier"]
                    ),
                    minimum_margin=float(config["repair"]["ensemble_minimum_margin"]),
                )
                agents["conservative_fallback"] = EnsemblePlanningAgent(
                    mdp,
                    value,
                    ensemble,
                    fallback_agent=b4,
                    uncertainty_multiplier=float(
                        config["repair"]["ensemble_uncertainty_multiplier"]
                    ),
                    minimum_margin=float(config["repair"]["ensemble_minimum_margin"]),
                )
                write_json(run_dir / "stage_status.json", {"stage": "evaluation", "at": _utcnow()})
                for method, agent in agents.items():
                    fallback_before = getattr(agent, "fallback_decisions", 0)
                    decisions_before = getattr(agent, "total_decisions", 0)
                    state, summary = evaluate_full_state_policy(
                        agent, method, mdp, optimum, scenario, seed, device
                    )
                    episodes, steps, runtime = evaluate_policy_rollouts(
                        agent,
                        method,
                        mdp,
                        optimum,
                        [int(x) for x in config["budgets"]],
                        scenario,
                        seed,
                        int(config["evaluation_episodes"]),
                        device,
                    )
                    state["dsp_checkpoint_seed"] = checkpoint_seed if method == "dsp_b" else np.nan
                    episodes["dsp_checkpoint_seed"] = checkpoint_seed if method == "dsp_b" else np.nan
                    state_frames.append(state)
                    state_summaries.append(summary)
                    episode_frames.append(episodes)
                    step_frames.append(steps)
                    decisions = getattr(agent, "total_decisions", 0) - decisions_before
                    fallbacks = getattr(agent, "fallback_decisions", 0) - fallback_before
                    runtime_rows.append(
                        {
                            "method": method,
                            "scenario": scenario,
                            "seed": seed,
                            "dsp_checkpoint_seed": checkpoint_seed if method == "dsp_b" else np.nan,
                            "fallback_decisions": fallbacks,
                            "decisions": decisions,
                            "fallback_rate": fallbacks / max(decisions, 1),
                            **runtime,
                        }
                    )
                    if method not in {"optimal", "dsp_b"}:
                        q_frames.append(
                            _planning_q_rows(
                                agent, method, mdp, optimum, scenario, seed, device
                            )
                        )
                for name, result in {
                    **trained,
                    **aggregation_models,
                }.items():
                    model_diagnostic_rows.append(
                        _model_component_diagnostics(
                            result.model, mdp, name, scenario, seed
                        )
                    )
        loso_rows = []
        if not smoke and len(config["scenarios"]) >= 3:
            loso_dir = run_dir / "loso_models"
            loso_dir.mkdir()
            for heldout in config["scenarios"]:
                sources = [scenario for scenario in config["scenarios"] if scenario != heldout]
                for seed_value in config["seeds"]:
                    seed = int(seed_value)
                    source_labels = pd.concat(
                        [loso_payload[(source, seed)][4] for source in sources],
                        ignore_index=True,
                    )
                    held_mdp, held_optimum, _, held_value, _ = loso_payload[(heldout, seed)]
                    result = _train_variant(
                        source_labels,
                        held_mdp,
                        config,
                        seed * 10_000 + 9_000 + list(config["scenarios"]).index(heldout),
                        ABLATION_WEIGHTS["structured_effect_q_rank"],
                        use_priority=True,
                    )
                    torch.save(
                        model_checkpoint_payload(result),
                        loso_dir / f"heldout_{heldout}__s{seed}.pt",
                    )
                    _, summary = evaluate_full_state_policy(
                        StructuredPlanningAgent(held_mdp, held_value, result.model),
                        "loso_full_repair",
                        held_mdp,
                        held_optimum,
                        heldout,
                        seed,
                        device,
                    )
                    old_summary = next(
                        row
                        for row in state_summaries
                        if row["method"] == "original_learned_model"
                        and row["scenario"] == heldout
                        and row["seed"] == seed
                    )
                    old_regret = float(old_summary["mean_Q_star_regret"])
                    repair_regret = float(summary["mean_Q_star_regret"])
                    loso_rows.append(
                        {
                            "heldout_scenario": heldout,
                            "source_scenarios": "+".join(sources),
                            "seed": seed,
                            "old_model_q_star_regret": old_regret,
                            "loso_repair_q_star_regret": repair_regret,
                            "regret_change_fraction": (repair_regret - old_regret)
                            / max(old_regret, 1.0e-12),
                            "regret_reversal": bool(
                                repair_regret > 1.10 * old_regret + 1.0e-12
                            ),
                            "action_consistency_rate": summary[
                                "action_consistency_rate"
                            ],
                        }
                    )

        pd.concat(truth_frames, ignore_index=True).to_csv(
            run_dir / "action_truth.csv.gz", index=False, compression="gzip"
        )
        states = pd.concat(state_frames, ignore_index=True)
        summaries = pd.DataFrame(state_summaries)
        episodes = pd.concat(episode_frames, ignore_index=True)
        steps = pd.concat(step_frames, ignore_index=True)
        runtime = pd.DataFrame(runtime_rows)
        q_values = pd.concat(q_frames, ignore_index=True)
        states.to_csv(run_dir / "state_policy_actions.csv.gz", index=False, compression="gzip")
        summaries.to_csv(run_dir / "state_policy_summary.csv", index=False)
        confusion_table(states).to_csv(run_dir / "action_confusions.csv", index=False)
        episodes.to_csv(run_dir / "metrics.csv", index=False)
        steps.to_csv(run_dir / "steps.csv.gz", index=False, compression="gzip")
        runtime.to_csv(run_dir / "runtime.csv", index=False)
        q_values.to_csv(run_dir / "planning_q_values.csv.gz", index=False, compression="gzip")
        _pair_ranking_summary(q_values).to_csv(
            run_dir / "pair_ranking_summary.csv", index=False
        )
        pd.DataFrame(model_diagnostic_rows).to_csv(
            run_dir / "model_component_diagnostics.csv", index=False
        )
        aggregation_frame = pd.DataFrame(aggregation_rows)
        aggregation_frame.to_csv(
            run_dir / "aggregation_rounds.csv", index=False
        )
        pd.DataFrame(training_rows).to_csv(run_dir / "training_summary.csv", index=False)
        loso_frame = pd.DataFrame(loso_rows)
        loso_frame.to_csv(run_dir / "loso_summary.csv", index=False)
        gate = None
        if not smoke:
            gate = assess_repair_gate(
                states,
                summaries,
                episodes,
                aggregation_frame,
                runtime,
                loso_frame,
                config["continuation_gate"],
            )
            write_json(run_dir / "continuation_gate.json", gate)
        write_json(run_dir / "stage_status.json", {"stage": "completed", "at": _utcnow()})
        output_hashes = {
            str(path.relative_to(run_dir)): sha256_file(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "schema": "direct_action_planning_repair.minimal_run.v1",
            "status": "completed",
            "scientific_status": (
                "SMOKE_NON_CLAIM" if smoke else str(gate["decision"] if gate else "UNKNOWN")
            ),
            "completed_at": _utcnow(),
            "runtime_seconds": time.perf_counter() - started,
            "frozen_predecessor_checkpoints": lineage,
            "output_sha256": output_hashes,
        }
        write_json(run_dir / "manifest.json", manifest)
        return run_dir
    except Exception as error:
        failure = {
            "schema": "direct_action_planning_repair.failure.v1",
            "status": "failed",
            "at": _utcnow(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        (run_dir / "stderr.log").write_text(failure["traceback"], encoding="utf-8")
        raise
