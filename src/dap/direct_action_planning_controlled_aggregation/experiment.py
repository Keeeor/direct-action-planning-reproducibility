from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from dap.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from dap.action_conditioned_budget_advantage.evaluation import (
    evaluate_full_state_policy,
    evaluate_policy_rollouts,
)
from dap.direct_action_planning.learning import (
    collect_transition_samples,
    fit_empirical_action_model,
    solve_empirical_value,
)
from dap.direct_action_planning.planning import DirectPlanningAgent
from dap.direct_action_planning_repair.data import (
    attach_priority_weights,
    collect_common_random_branch_data,
)
from dap.direct_action_planning_repair.experiment import (
    _model_component_diagnostics,
    _pair_ranking_summary,
    _planning_q_rows,
    _probe_agent,
)
from dap.direct_action_planning_repair.model import (
    LossWeights,
    TrainingConfig,
    model_checkpoint_payload,
    selection_metrics,
)
from dap.direct_action_planning_repair.planning import (
    StructuredPlanningAgent,
)
from dap.experiment import load_config
from dap.utils.artifacts import environment_record, sha256_file, write_json
from dap.utils.seed import set_global_seed

from .collection import collect_mixed_policy_branches, selective_samples
from .diagnosis import duplicate_and_coverage, empirical_anchor_labels
from .gate import assess_controlled_gate, validation_guardrail
from .replay import balanced_replay_labels, unconstrained_replay_labels
from .training import train_controlled_model


PROTOCOLS = {
    "original_unconstrained": {
        "balanced": False,
        "selective": False,
        "rho": 1.0,
        "retention": False,
        "priority": False,
    },
    "balanced_replay": {
        "balanced": True,
        "selective": False,
        "rho": 1.0,
        "retention": False,
        "priority": True,
    },
    "selective_aggregation": {
        "balanced": False,
        "selective": True,
        "rho": 1.0,
        "retention": False,
        "priority": True,
    },
    "mixed_policy_rho_0p5": {
        "balanced": False,
        "selective": False,
        "rho": 0.5,
        "retention": False,
        "priority": True,
    },
    "mixed_policy_rho_0p75": {
        "balanced": False,
        "selective": False,
        "rho": 0.75,
        "retention": False,
        "priority": True,
    },
    "mixed_policy_rho_1p0": {
        "balanced": False,
        "selective": False,
        "rho": 1.0,
        "retention": False,
        "priority": True,
    },
    "retention_constraint": {
        "balanced": False,
        "selective": False,
        "rho": 1.0,
        "retention": True,
        "priority": True,
    },
    "controlled_full": {
        "balanced": True,
        "selective": True,
        "rho": 0.75,
        "retention": True,
        "priority": True,
    },
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved(config: dict, run_id: str, smoke: bool) -> dict:
    value = json.loads(json.dumps(config))
    value["run_id"] = run_id
    value["smoke"] = smoke
    value["resolved_at"] = _utcnow()
    if smoke:
        value["budgets"] = [4]
        value["scenarios"] = ["early_burst"]
        value["seed_split"] = {"train": [10], "validation": [15], "test": [20]}
        value["evaluation_episodes"] = 2
        value["aggregation"]["max_rounds"] = 1
        value["aggregation"]["collection_episodes"] = 2
        value["aggregation"]["train_crn_samples_per_state"] = 8
        value["aggregation"]["validation_crn_samples_per_state"] = 4
        value["repair_frozen"]["max_epochs"] = 20
        value["repair_frozen"]["patience"] = 8
    return value


def _mdp(config: dict, scenario: str) -> ActionConditionedBudgetMDP:
    env = config["environment"]
    return ActionConditionedBudgetMDP(
        ACBADPConfig(
            horizon=int(env["horizon"]),
            max_budget=int(env["max_budget"]),
            max_queue=int(env["max_queue"]),
            scenario=scenario,
            gamma=float(env["gamma"]),
            action_costs=tuple(int(x) for x in env["action_costs"]),
            action_capacity=tuple(int(x) for x in env["action_capacity"]),
        )
    )


def _training_config(config: dict, seed: int, priority: bool) -> TrainingConfig:
    frozen = config["repair_frozen"]
    return TrainingConfig(
        hidden_dim=int(frozen["hidden_dim"]),
        learning_rate=float(frozen["learning_rate"]),
        max_epochs=int(frozen["max_epochs"]),
        patience=int(frozen["patience"]),
        seed=seed,
        use_priority_weights=priority,
    )


def _loss_weights(config: dict) -> LossWeights:
    frozen = config["repair_frozen"]
    return LossWeights(
        state=float(frozen["lambda_state"]),
        effect=float(frozen["lambda_effect"]),
        q=float(frozen["lambda_Q"]),
        rank=float(frozen["lambda_rank"]),
        rank_margin=float(frozen["rank_margin"]),
    )


def _train(
    labels: pd.DataFrame,
    mdp: ActionConditionedBudgetMDP,
    config: dict,
    seed: int,
    priority: bool,
    reference=None,
    anchor: pd.DataFrame | None = None,
    state_targets: pd.DataFrame | None = None,
):
    return train_controlled_model(
        labels,
        mdp.config.horizon,
        mdp.n_loads,
        mdp.n_actions,
        mdp.config.gamma,
        _loss_weights(config),
        _training_config(config, seed, priority),
        retention_reference=reference,
        retention_anchor=anchor,
        retention_weight=(
            float(config["aggregation"]["retention_weight"])
            if reference is not None
            else 0.0
        ),
        state_targets=state_targets,
        device=config["device"],
    )


def _independent_d0(
    mdp,
    value,
    optimum,
    scenario,
    train_seed,
    validation_seed,
    config,
):
    train = collect_common_random_branch_data(
        mdp,
        value,
        optimum,
        scenario,
        train_seed + 500_009,
        int(config["aggregation"]["train_crn_samples_per_state"]),
        1,
    )
    validation = collect_common_random_branch_data(
        mdp,
        value,
        optimum,
        scenario,
        validation_seed + 900_001,
        1,
        int(config["aggregation"]["validation_crn_samples_per_state"]),
    )
    samples = pd.concat(
        [train.samples[train.samples.split == "train"], validation.samples[validation.samples.split == "validation"]],
        ignore_index=True,
    )
    labels = pd.concat(
        [train.labels[train.labels.split == "train"], validation.labels[validation.labels.split == "validation"]],
        ignore_index=True,
    )
    return samples, labels


def _retention_anchor(base_labels: pd.DataFrame, d1_samples: pd.DataFrame) -> pd.DataFrame:
    d0 = base_labels[base_labels.split == "validation"].copy()
    if d1_samples.empty:
        return d0
    d1 = empirical_anchor_labels(base_labels, d1_samples)
    return pd.concat([d0, d1], ignore_index=True).drop_duplicates(
        ["t", "load", "queue", "remaining_budget", "action"], keep="last"
    )


def _save_checkpoint(result, path: Path) -> None:
    torch.save(model_checkpoint_payload(result), path)


def _protocol_labels(
    protocol: str,
    spec: dict,
    base_labels: pd.DataFrame,
    base_samples: pd.DataFrame,
    cumulative: list[pd.DataFrame],
    best_history: list[pd.DataFrame],
    current: pd.DataFrame,
    source_weights: dict[str, float],
):
    if spec["balanced"]:
        hard_pool = pd.concat([*cumulative, current], ignore_index=True)
        hard = hard_pool[hard_pool.high_regret | hard_pool.ranking_error]
        return balanced_replay_labels(
            base_labels, best_history, current, hard, source_weights
        )
    return unconstrained_replay_labels(
        base_labels, base_samples, [*cumulative, current], protocol
    )


def _manifest(run_dir: Path, started: float, config: dict, gate: dict) -> dict:
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name != "manifest.json")
    return {
        "schema": "direct_action_planning_controlled_aggregation.minimal.v1",
        "status": "completed",
        "scientific_status": gate["decision"],
        "completed_at": _utcnow(),
        "runtime_seconds": time.perf_counter() - started,
        "test_evaluation_count": 1,
        "test_selection_prohibited": True,
        "protocols": list(PROTOCOLS),
        "config": config,
        "output_sha256": {
            str(path.relative_to(run_dir)): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }


def run_minimal_controlled_aggregation(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "minimal_v1",
    smoke: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    config = _resolved(load_config(Path(config_path).resolve()), run_id, smoke)
    run_dir = root / "results/direct_action_planning_controlled_aggregation" / run_id
    if run_dir.exists():
        manifest = run_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return run_dir
        raise RuntimeError(f"append-only run already exists and is incomplete: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_json(run_dir / "stage_status.json", {"stage": "validation_training", "at": _utcnow()})
    set_global_seed(0, torch_threads=int(config["torch_threads"]))
    device = torch.device(config["device"])
    validation_rows: list[dict[str, object]] = []
    collection_rows: list[dict[str, object]] = []
    source_mix_rows: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    seed_blocks = config["seed_split"]
    triples = list(zip(seed_blocks["train"], seed_blocks["validation"], seed_blocks["test"]))
    for scenario_index, scenario in enumerate(config["scenarios"]):
        mdp = _mdp(config, scenario)
        optimum = solve_action_dp(mdp)
        for unit_index, (train_value, validation_value, test_value) in enumerate(triples):
            train_seed, validation_seed, test_seed = int(train_value), int(validation_value), int(test_value)
            cell = run_dir / "cells" / f"{scenario}__train_s{train_seed}"
            (cell / "models").mkdir(parents=True)
            (cell / "training").mkdir()
            (cell / "collections").mkdir()
            old_seed = train_seed + scenario_index * 100_003
            old_samples = collect_transition_samples(mdp, 64, old_seed)
            old_model = fit_empirical_action_model(mdp, old_samples, smoothing=0.25)
            value = solve_empirical_value(mdp, old_model)
            np.savez_compressed(
                cell / "old_model.npz",
                next_load_probabilities=old_model.next_load_probabilities,
                next_queue_probabilities=old_model.next_queue_probabilities,
                rewards=old_model.rewards,
                costs=old_model.costs,
                samples_per_state_action=old_model.samples_per_state_action,
                smoothing=old_model.smoothing,
            )
            np.savez_compressed(cell / "learned_value.npz", values=value.values, source=np.asarray(value.source))
            base_samples, plain_labels = _independent_d0(
                mdp, value, optimum, scenario, train_seed, validation_seed, config
            )
            labels = attach_priority_weights(
                plain_labels,
                mdp,
                old_model,
                value,
                optimum,
                scenario,
                float(config["repair_frozen"]["close_gap_ceiling"]),
                float(config["repair_frozen"]["priority_weight_ceiling"]),
            )
            base_samples.to_csv(cell / "D0_branch_samples.csv.gz", index=False, compression="gzip")
            labels.to_csv(cell / "D0_labels.csv.gz", index=False, compression="gzip")
            d0 = _train(
                labels,
                mdp,
                config,
                train_seed * 100_000 + scenario_index * 1_000,
                priority=True,
            )
            d0_unweighted = _train(
                labels,
                mdp,
                config,
                train_seed * 100_000 + scenario_index * 1_000 + 1,
                priority=False,
            )
            _save_checkpoint(d0, cell / "models/dap_repair_d0.pt")
            _save_checkpoint(d0_unweighted, cell / "models/original_unconstrained_D0.pt")
            d0.history.to_csv(cell / "training/dap_repair_d0.csv", index=False)
            d0_unweighted.history.to_csv(cell / "training/original_unconstrained_D0.csv", index=False)
            baseline_agents = {
                "learned_value": DirectPlanningAgent(mdp, value),
                "original_learned_model": DirectPlanningAgent(mdp, value, learned_model=old_model),
                "dap_repair_d0": StructuredPlanningAgent(mdp, value, d0.model),
                "original_unconstrained_D0": StructuredPlanningAgent(
                    mdp, value, d0_unweighted.model
                ),
            }
            baseline_probe = {}
            for method, agent in baseline_agents.items():
                probe = _probe_agent(
                    agent,
                    mdp,
                    optimum,
                    [int(x) for x in config["budgets"]],
                    scenario,
                    validation_seed,
                    int(config["evaluation_episodes"]),
                    device,
                )
                baseline_probe[method] = probe
                validation_rows.append(
                    {
                        "scenario": scenario,
                        "train_seed": train_seed,
                        "validation_seed": validation_seed,
                        "protocol": "baseline",
                        "method": method,
                        "round": 0,
                        "selected": method == "dap_repair_d0",
                        **probe,
                    }
                )
            selected_models = {"dap_repair_d0": d0}
            selected_rounds = {"dap_repair_d0": 0}
            for protocol_index, (protocol, spec) in enumerate(PROTOCOLS.items()):
                current = d0_unweighted if protocol == "original_unconstrained" else d0
                best = current
                best_round = 0
                d0_probe_name = (
                    "original_unconstrained_D0"
                    if protocol == "original_unconstrained"
                    else "dap_repair_d0"
                )
                best_regret = float(baseline_probe[d0_probe_name]["rollout_q_star_regret"])
                cumulative: list[pd.DataFrame] = []
                best_history: list[pd.DataFrame] = []
                d1_samples = pd.DataFrame()
                missed = 0
                protocol_round_rows: list[int] = []
                for round_index in range(1, int(config["aggregation"]["max_rounds"]) + 1):
                    repair_agent = StructuredPlanningAgent(mdp, value, current.model)
                    reference_agent = DirectPlanningAgent(mdp, value, learned_model=old_model)
                    raw_samples, visits = collect_mixed_policy_branches(
                        repair_agent,
                        reference_agent,
                        mdp,
                        value,
                        optimum,
                        [int(x) for x in config["budgets"]],
                        scenario,
                        train_seed + scenario_index * 100_003,
                        int(config["aggregation"]["collection_episodes"]),
                        round_index,
                        float(spec["rho"]),
                        cumulative,
                        device,
                    )
                    kept = selective_samples(raw_samples) if spec["selective"] else raw_samples
                    if round_index == 1:
                        d1_samples = kept.copy()
                    replay = _protocol_labels(
                        protocol,
                        spec,
                        labels,
                        base_samples,
                        cumulative,
                        best_history,
                        kept,
                        {key: float(value_) for key, value_ in config["aggregation"]["source_weights"].items()},
                    )
                    anchor = _retention_anchor(labels, d1_samples) if spec["retention"] else None
                    result = _train(
                        replay.labels,
                        mdp,
                        config,
                        train_seed * 100_000
                        + scenario_index * 10_000
                        + protocol_index * 100
                        + round_index,
                        priority=bool(spec["priority"]),
                        reference=best.model if spec["retention"] else None,
                        anchor=anchor,
                        state_targets=replay.state_targets,
                    )
                    model_name = f"{protocol}_D{round_index}"
                    _save_checkpoint(result, cell / "models" / f"{model_name}.pt")
                    result.history.to_csv(cell / "training" / f"{model_name}.csv", index=False)
                    raw_samples.to_csv(
                        cell / "collections" / f"{model_name}_raw.csv.gz",
                        index=False,
                        compression="gzip",
                    )
                    visits.to_csv(cell / "collections" / f"{model_name}_visits.csv", index=False)
                    if spec["selective"]:
                        kept.to_csv(
                            cell / "collections" / f"{model_name}_selected.csv.gz",
                            index=False,
                            compression="gzip",
                        )
                    probe = _probe_agent(
                        StructuredPlanningAgent(mdp, value, result.model),
                        mdp,
                        optimum,
                        [int(x) for x in config["budgets"]],
                        scenario,
                        validation_seed,
                        int(config["evaluation_episodes"]),
                        device,
                    )
                    guards = validation_guardrail(probe, baseline_probe["learned_value"], config["selection"])
                    improved = bool(
                        guards
                        and probe["rollout_q_star_regret"]
                        < best_regret - float(config["selection"]["minimum_regret_improvement"])
                    )
                    if improved:
                        best = result
                        best_round = round_index
                        best_regret = float(probe["rollout_q_star_regret"])
                        best_history = [*cumulative, kept]
                        missed = 0
                    else:
                        missed += 1
                    cumulative.append(kept)
                    validation_rows.append(
                        {
                            "scenario": scenario,
                            "train_seed": train_seed,
                            "validation_seed": validation_seed,
                            "protocol": protocol,
                            "method": model_name,
                            "round": round_index,
                            "selected": False,
                            "guardrails_pass": guards,
                            "improved_vs_validation_best": improved,
                            "best_round_after_evaluation": best_round,
                            **probe,
                        }
                    )
                    coverage = duplicate_and_coverage(raw_samples, base_samples)
                    selected_coverage = duplicate_and_coverage(kept, base_samples)
                    collection_rows.append(
                        {
                            "scenario": scenario,
                            "train_seed": train_seed,
                            "protocol": protocol,
                            "round": round_index,
                            "rho": float(spec["rho"]),
                            **coverage,
                            "selected_rows": len(kept),
                            "effective_sample_fraction": len(kept) / max(len(raw_samples), 1),
                            "selected_grid_coverage": selected_coverage["grid_coverage"],
                            "repair_collection_fraction": float(
                                (visits.collector_component == "repair").mean()
                            ),
                            "mean_collection_q_star_regret": float(visits.q_star_regret.mean()),
                        }
                    )
                    mix = replay.source_mix.copy()
                    mix["scenario"] = scenario
                    mix["train_seed"] = train_seed
                    mix["protocol"] = protocol
                    mix["round"] = round_index
                    source_mix_rows.append(mix)
                    training_rows.append(
                        {
                            "scenario": scenario,
                            "train_seed": train_seed,
                            "protocol": protocol,
                            "round": round_index,
                            "best_epoch": result.best_epoch,
                            **result.selection_metrics,
                        }
                    )
                    protocol_round_rows.append(len(validation_rows) - 1)
                    current = result
                    if missed >= int(config["aggregation"]["stop_after_nonimproving_rounds"]):
                        break
                selected_models[protocol] = best
                selected_rounds[protocol] = best_round
                for index in protocol_round_rows:
                    validation_rows[index]["selected"] = validation_rows[index]["round"] == best_round
                if best_round == 0:
                    validation_rows.append(
                        {
                            "scenario": scenario,
                            "train_seed": train_seed,
                            "validation_seed": validation_seed,
                            "protocol": protocol,
                            "method": "dap_repair_d0" if protocol != "original_unconstrained" else "original_unconstrained_D0",
                            "round": 0,
                            "selected": True,
                            **baseline_probe[d0_probe_name],
                        }
                    )
                anchor_set = _retention_anchor(labels, d1_samples)
                selected_metrics = selection_metrics(
                    best.model, anchor_set, mdp.config.gamma, device
                )
                d0_metrics = selection_metrics(d0.model, anchor_set, mdp.config.gamma, device)
                anchor_rows.append(
                    {
                        "scenario": scenario,
                        "train_seed": train_seed,
                        "method": protocol,
                        "selected_round": best_round,
                        "anchor": "D0_plus_D1_fixed",
                        "lv_action_error_rate": selected_metrics["lv_action_error_rate"],
                        "baseline_action_error_rate": d0_metrics["lv_action_error_rate"],
                        "action_error_increase": selected_metrics["lv_action_error_rate"]
                        - d0_metrics["lv_action_error_rate"],
                        "exact_q_damage": selected_metrics["mean_exact_q_damage_vs_lv"],
                        "baseline_exact_q_damage": d0_metrics["mean_exact_q_damage_vs_lv"],
                    }
                )
            selection_payload = {
                "schema": "controlled_aggregation.cell_selection.v1",
                "scenario": scenario,
                "train_seed": train_seed,
                "validation_seed": validation_seed,
                "test_seed": test_seed,
                "selected_rounds": selected_rounds,
                "selection_source": "validation_only",
                "test_used_for_selection": False,
            }
            write_json(cell / "validation_selection.json", selection_payload)
            cells.append(
                {
                    "scenario": scenario,
                    "train_seed": train_seed,
                    "validation_seed": validation_seed,
                    "test_seed": test_seed,
                    "mdp": mdp,
                    "optimum": optimum,
                    "old_model": old_model,
                    "value": value,
                    "selected_models": selected_models,
                    "cell": cell,
                }
            )
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(run_dir / "validation_rounds.csv", index=False)
    pd.DataFrame(collection_rows).to_csv(run_dir / "collection_metrics.csv", index=False)
    pd.concat(source_mix_rows, ignore_index=True).to_csv(run_dir / "source_mix.csv", index=False)
    pd.DataFrame(training_rows).to_csv(run_dir / "training_summary.csv", index=False)
    anchor_frame = pd.DataFrame(anchor_rows)
    anchor_frame.to_csv(run_dir / "anchor_forgetting.csv", index=False)
    write_json(
        run_dir / "validation_selection.json",
        {
            "schema": "controlled_aggregation.validation_selection.v1",
            "frozen_at": _utcnow(),
            "test_evaluations_completed": 0,
            "cells": [
                json.loads((record["cell"] / "validation_selection.json").read_text())
                for record in cells
            ],
        },
    )

    # The test block begins only after every validation selection is frozen.
    write_json(run_dir / "stage_status.json", {"stage": "single_final_test", "at": _utcnow()})
    state_frames, state_summaries, episode_frames, step_frames = [], [], [], []
    runtime_rows, q_frames, component_rows = [], [], []
    for record in cells:
        scenario = str(record["scenario"])
        train_seed = int(record["train_seed"])
        test_seed = int(record["test_seed"])
        mdp = record["mdp"]
        optimum = record["optimum"]
        value = record["value"]
        agents = {
            "learned_value": DirectPlanningAgent(mdp, value),
            "original_learned_model": DirectPlanningAgent(
                mdp, value, learned_model=record["old_model"]
            ),
            **{
                method: StructuredPlanningAgent(mdp, value, result.model)
                for method, result in record["selected_models"].items()
            },
        }
        for method, agent in agents.items():
            state, summary = evaluate_full_state_policy(
                agent, method, mdp, optimum, scenario, test_seed, device
            )
            episodes, steps, runtime = evaluate_policy_rollouts(
                agent,
                method,
                mdp,
                optimum,
                [int(x) for x in config["budgets"]],
                scenario,
                test_seed,
                int(config["evaluation_episodes"]),
                device,
            )
            for frame in (state, episodes, steps):
                frame["train_seed"] = train_seed
                frame["test_seed"] = test_seed
            summary["train_seed"] = train_seed
            summary["test_seed"] = test_seed
            state_frames.append(state)
            state_summaries.append(summary)
            episode_frames.append(episodes)
            step_frames.append(steps)
            runtime_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "train_seed": train_seed,
                    "test_seed": test_seed,
                    "fallback_rate": 0.0,
                    **runtime,
                }
            )
            if method != "learned_value" or hasattr(agent.act(torch.zeros((1, 14))), "q_values"):
                q_frames.append(_planning_q_rows(agent, method, mdp, optimum, scenario, test_seed, device))
            if method in record["selected_models"]:
                component_rows.append(
                    _model_component_diagnostics(
                        record["selected_models"][method].model,
                        mdp,
                        method,
                        scenario,
                        train_seed,
                    )
                )
    states = pd.concat(state_frames, ignore_index=True)
    summaries = pd.DataFrame(state_summaries)
    episodes = pd.concat(episode_frames, ignore_index=True)
    steps = pd.concat(step_frames, ignore_index=True)
    runtime = pd.DataFrame(runtime_rows)
    q_values = pd.concat(q_frames, ignore_index=True)
    states.to_csv(run_dir / "test_state_policy_actions.csv.gz", index=False, compression="gzip")
    summaries.to_csv(run_dir / "test_state_policy_summary.csv", index=False)
    episodes.to_csv(run_dir / "test_metrics.csv", index=False)
    steps.to_csv(run_dir / "test_steps.csv.gz", index=False, compression="gzip")
    runtime.to_csv(run_dir / "runtime.csv", index=False)
    q_values.to_csv(run_dir / "test_planning_q_values.csv.gz", index=False, compression="gzip")
    _pair_ranking_summary(q_values).to_csv(run_dir / "test_pair_ranking.csv", index=False)
    pd.DataFrame(component_rows).to_csv(run_dir / "model_component_diagnostics.csv", index=False)
    gate = assess_controlled_gate(
        summaries,
        episodes,
        validation_frame,
        anchor_frame,
        pd.DataFrame(collection_rows),
        config["gate"],
    )
    write_json(run_dir / "continuation_gate.json", gate)
    write_json(run_dir / "stage_status.json", {"stage": "completed", "at": _utcnow(), "decision": gate["decision"]})
    write_json(run_dir / "manifest.json", _manifest(run_dir, started, config, gate))
    return run_dir
