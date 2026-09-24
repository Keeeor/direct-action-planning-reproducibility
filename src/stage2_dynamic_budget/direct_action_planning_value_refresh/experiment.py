from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.evaluation import (
    evaluate_full_state_policy,
    evaluate_policy_rollouts,
)
from stage2_dynamic_budget.direct_action_planning.learning import EmpiricalActionModel
from stage2_dynamic_budget.direct_action_planning.planning import (
    BudgetValueTable,
    DirectPlanningAgent,
)
from stage2_dynamic_budget.direct_action_planning_repair.experiment import _probe_agent
from stage2_dynamic_budget.direct_action_planning_repair.experiment import (
    _pair_ranking_summary,
    _planning_q_rows,
)
from stage2_dynamic_budget.direct_action_planning_repair.data import (
    attach_priority_weights,
    collect_common_random_branch_data,
)
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    LossWeights,
    TrainingConfig,
    model_checkpoint_payload,
    train_structured_model,
)
from stage2_dynamic_budget.direct_action_planning_repair.planning import (
    StructuredPlanningAgent,
)
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.diagnosis import (
    load_structured_checkpoint,
)
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.collection import (
    collect_mixed_policy_branches,
)
from stage2_dynamic_budget.experiment import load_config
from stage2_dynamic_budget.utils.artifacts import environment_record, sha256_file, write_json
from stage2_dynamic_budget.utils.seed import set_global_seed

from .causal import (
    causal_decomposition,
    causal_pair_flip_rates,
    summarize_causal_decomposition,
)
from .data import STATE_COLUMNS, build_anchor_states, value_state_targets
from .gate import assess_causal_gate, assess_value_refresh_gate
from .protocol import FinalTestLedger
from .value import (
    ResidualBudgetValueModel,
    ValueLossWeights,
    ValueTrainingConfig,
    train_refreshed_value,
    value_checkpoint_payload,
)


VALUE_ABLATIONS = {
    "aggregate_only": ValueLossWeights(value=1.0),
    "aggregate_anchor": ValueLossWeights(value=1.0, anchor=0.10),
    "aggregate_anchor_mono": ValueLossWeights(value=1.0, anchor=0.10, mono=0.20),
    "aggregate_anchor_mono_rank": ValueLossWeights(
        value=1.0, anchor=0.10, mono=0.20, rank=1.0
    ),
    "full_value_refresh": ValueLossWeights(
        value=1.0, anchor=0.10, mono=0.20, rank=1.0
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mdp(config: dict, scenario: str) -> ActionConditionedBudgetMDP:
    env = config["environment"]
    return ActionConditionedBudgetMDP(
        ACBADPConfig(
            horizon=int(env["horizon"]),
            max_budget=int(env["max_budget"]),
            max_queue=int(env["max_queue"]),
            scenario=scenario,
            gamma=float(env["gamma"]),
            action_costs=tuple(map(int, env["action_costs"])),
            action_capacity=tuple(map(int, env["action_capacity"])),
        )
    )


def load_value_table(path: str | Path) -> BudgetValueTable:
    payload = np.load(Path(path), allow_pickle=False)
    source = str(payload["source"].item()) if "source" in payload else "frozen_learned_value"
    return BudgetValueTable(values=payload["values"], source=source)


def load_empirical_model(path: str | Path) -> EmpiricalActionModel:
    payload = np.load(Path(path), allow_pickle=False)
    return EmpiricalActionModel(
        next_load_probabilities=payload["next_load_probabilities"],
        next_queue_probabilities=payload["next_queue_probabilities"],
        rewards=payload["rewards"],
        costs=payload["costs"],
        samples_per_state_action=int(payload["samples_per_state_action"]),
        smoothing=float(payload["smoothing"]),
    )


def _resolved(config: dict, run_id: str, smoke: bool) -> dict:
    value = json.loads(json.dumps(config))
    value["run_id"] = run_id
    value["smoke"] = bool(smoke)
    value["resolved_at"] = _now()
    if smoke:
        value["scenarios"] = ["early_burst"]
        value["seed_split"] = {"train": [25], "validation": [130], "test": [135]}
        value["evaluation_episodes"] = 2
        value["collection_episodes"] = 2
        value["value_training"]["max_epochs"] = 20
        value["value_training"]["patience"] = 5
        value["alternating"]["max_rounds"] = 1
        value["alternating"]["transition_epochs"] = 8
        value["alternating"]["transition_patience"] = 3
    return value


def _repair_cell(root: Path, scenario: str, train_seed: int, config: dict) -> Path:
    predecessor_seed = train_seed + int(config["predecessor_seed_map"]["repair_offset"])
    return root / "results/direct_action_planning_repair/minimal_v1/cells" / (
        f"{scenario}__s{predecessor_seed}"
    )


def _controlled_cell(root: Path, scenario: str, train_seed: int, config: dict) -> Path:
    predecessor_seed = train_seed + int(config["predecessor_seed_map"]["controlled_offset"])
    return root / "results/direct_action_planning_controlled_aggregation/minimal_v1/cells" / (
        f"{scenario}__train_s{predecessor_seed}"
    )


def _controlled_sources(cell: Path) -> pd.DataFrame:
    files = [
        cell / "collections" / f"controlled_full_D{round_index}_raw.csv.gz"
        for round_index in (1, 2, 3)
    ]
    available = [path for path in files if path.exists()]
    if not available:
        raise FileNotFoundError(f"no frozen controlled states in {cell}")
    return pd.concat(
        [pd.read_csv(path, usecols=STATE_COLUMNS) for path in available],
        ignore_index=True,
    ).drop_duplicates()


def _save_refreshed_value(path: Path, trained, source: str) -> None:
    torch.save(value_checkpoint_payload(trained), path.with_suffix(".pt"))
    table = trained.model.as_value_table(source)
    np.savez_compressed(path.with_suffix(".npz"), values=table.values, source=np.asarray(source))


def _anchor_forgetting(
    anchors: pd.DataFrame,
    fixed: BudgetValueTable,
    refreshed: BudgetValueTable,
    transition,
    mdp,
    optimum,
    scenario: str,
    train_seed: int,
    method: str,
) -> list[dict[str, object]]:
    from stage2_dynamic_budget.direct_action_planning_repair.planning import (
        structured_one_step_plan,
    )

    rows = []
    for anchor_name, mask in {
        "D0": anchors.anchor_reason.str.contains("D0_fixed", regex=False),
        "D1": anchors.anchor_reason.str.contains("D1_high_value", regex=False),
    }.items():
        subset = anchors[mask]
        fixed_errors, refreshed_errors = [], []
        fixed_correct, damaged, retained = 0, 0, 0
        for item in subset.itertuples(index=False):
            state = (int(item.t), int(item.load), int(item.queue), int(item.remaining_budget))
            horizon = mdp.config.horizon - state[0]
            target = float(optimum.values[state])
            fixed_errors.append(
                abs(fixed.predict(state[1], state[2], state[3], horizon) - target)
            )
            refreshed_errors.append(
                abs(refreshed.predict(state[1], state[2], state[3], horizon) - target)
            )
            fixed_action = structured_one_step_plan(mdp, fixed, transition, *state).action
            refreshed_action = structured_one_step_plan(
                mdp, refreshed, transition, *state
            ).action
            optimal_action = int(optimum.actions[state])
            if fixed_action == optimal_action:
                fixed_correct += 1
                damaged += int(refreshed_action != optimal_action)
            retained += int(refreshed_action == fixed_action)
        rows.append(
            {
                "scenario": scenario,
                "train_seed": train_seed,
                "method": method,
                "anchor": anchor_name,
                "states": len(subset),
                "fixed_value_mae": float(np.mean(fixed_errors)),
                "refreshed_value_mae": float(np.mean(refreshed_errors)),
                "value_mae_increase": float(np.mean(refreshed_errors) - np.mean(fixed_errors)),
                "action_ranking_retention": retained / max(len(subset), 1),
                "previously_correct_action_loss": damaged / max(fixed_correct, 1),
            }
        )
    return rows


def verify_frozen_inputs(project_root: str | Path) -> list[dict[str, object]]:
    root = Path(project_root)
    manifest = root / "research/direct_action_planning_value_refresh/FROZEN_INPUTS.sha256"
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        actual = sha256_file(path).removeprefix("sha256:")
        rows.append(
            {
                "path": relative.strip(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    if not all(row["matches"] for row in rows):
        bad = [row["path"] for row in rows if not row["matches"]]
        raise RuntimeError(f"frozen predecessor input changed: {bad}")
    return rows


def _controlled_checkpoint(cell: Path) -> tuple[Path, int]:
    selected = json.loads((cell / "validation_selection.json").read_text(encoding="utf-8"))
    round_index = int(selected["selected_rounds"]["controlled_full"])
    filename = "dap_repair_d0.pt" if round_index == 0 else f"controlled_full_D{round_index}.pt"
    return cell / "models" / filename, round_index


def _frozen_source_states(cell: Path) -> dict[str, pd.DataFrame]:
    sources = {
        "D0": pd.read_csv(cell / "D0_labels.csv.gz", usecols=STATE_COLUMNS).drop_duplicates(),
        "D1": pd.read_csv(
            cell / "collections/controlled_full_D1_raw.csv.gz", usecols=STATE_COLUMNS
        ).drop_duplicates(),
    }
    controlled_files = [
        cell / "collections" / f"controlled_full_D{round_index}_raw.csv.gz"
        for round_index in (2, 3)
    ]
    available = [path for path in controlled_files if path.exists()]
    if available:
        sources["controlled"] = pd.concat(
            [pd.read_csv(path, usecols=STATE_COLUMNS) for path in available],
            ignore_index=True,
        ).drop_duplicates()
    else:
        sources["controlled"] = sources["D1"].copy()
    return sources


def _source_value_errors(
    sources: dict[str, pd.DataFrame],
    fixed: BudgetValueTable,
    optimum,
    scenario: str,
    model_seed: int,
) -> list[dict[str, object]]:
    rows = []
    for source, states in sources.items():
        unique = states[STATE_COLUMNS].drop_duplicates()
        index = tuple(unique[column].to_numpy(np.int64) for column in STATE_COLUMNS)
        horizons = fixed.values.shape[0] - 1 - unique.t.to_numpy(np.int64)
        predicted = fixed.values[
            horizons,
            unique.load.to_numpy(np.int64),
            unique.queue.to_numpy(np.int64),
            unique.remaining_budget.to_numpy(np.int64),
        ]
        target = optimum.values[index]
        rows.append(
            {
                "scenario": scenario,
                "model_seed": model_seed,
                "source": source,
                "states": len(unique),
                "fixed_value_mae": float(np.mean(np.abs(predicted - target))),
                "fixed_value_rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
                "oracle_value_mae": 0.0,
            }
        )
    return rows


def run_frozen_causal_diagnosis(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "causal_v1",
) -> Path:
    root = Path(project_root).resolve()
    config = load_config(Path(config_path).resolve())
    run_dir = root / "results/direct_action_planning_value_refresh" / run_id
    if run_dir.exists():
        status = run_dir / "stage_status.json"
        if status.exists() and json.loads(status.read_text()).get("stage") == "completed":
            return run_dir
        raise RuntimeError(f"append-only causal run exists and is incomplete: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_json(run_dir / "stage_status.json", {"stage": "running", "at": _now()})
    frozen = verify_frozen_inputs(root)
    pd.DataFrame(frozen).to_csv(run_dir / "frozen_input_verification.csv", index=False)
    set_global_seed(0, torch_threads=int(config["torch_threads"]))
    controlled_root = root / "results/direct_action_planning_controlled_aggregation/minimal_v1/cells"
    state_rows, summaries, flips, value_errors, rollout_rows = [], [], [], [], []
    validation_seeds = [int(seed) for seed in config["seed_split"]["validation"]]
    controlled_offset = int(config["predecessor_seed_map"]["controlled_offset"])
    model_seeds = [int(seed) + controlled_offset for seed in config["seed_split"]["train"]]
    for scenario in config["scenarios"]:
        mdp = _mdp(config, scenario)
        optimum = solve_action_dp(mdp)
        oracle = BudgetValueTable.from_exact_dp(optimum.values)
        for model_seed, validation_seed in zip(model_seeds, validation_seeds):
            cell = controlled_root / f"{scenario}__train_s{model_seed}"
            fixed = load_value_table(cell / "learned_value.npz")
            checkpoint, selected_round = _controlled_checkpoint(cell)
            transition = load_structured_checkpoint(checkpoint)
            sources = _frozen_source_states(cell)
            rows = causal_decomposition(
                mdp, optimum, fixed, transition, sources, scenario, model_seed
            )
            rows["selected_controlled_round"] = selected_round
            state_rows.append(rows)
            summaries.append(summarize_causal_decomposition(rows))
            flips.append(causal_pair_flip_rates(rows, optimum, mdp.n_actions))
            value_errors.extend(
                _source_value_errors(sources, fixed, optimum, scenario, model_seed)
            )
            agents = {
                "true_transition_fixed_value": DirectPlanningAgent(mdp, fixed),
                "repaired_transition_fixed_value": StructuredPlanningAgent(
                    mdp, fixed, transition
                ),
                "repaired_transition_oracle_value": StructuredPlanningAgent(
                    mdp, oracle, transition
                ),
                "true_transition_oracle_value": DirectPlanningAgent(mdp, oracle),
            }
            for planner, agent in agents.items():
                probe = _probe_agent(
                    agent,
                    mdp,
                    optimum,
                    [int(value) for value in config["budgets"]],
                    scenario,
                    validation_seed,
                    int(config["evaluation_episodes"]),
                    torch.device(config["device"]),
                )
                rollout_rows.append(
                    {
                        "scenario": scenario,
                        "model_seed": model_seed,
                        "validation_seed": validation_seed,
                        "planner": planner,
                        "selected_controlled_round": selected_round,
                        **probe,
                    }
                )
    states = pd.concat(state_rows, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    pair_flips = pd.concat(flips, ignore_index=True)
    rollouts = pd.DataFrame(rollout_rows)
    states.to_csv(run_dir / "state_causal_decomposition.csv.gz", index=False, compression="gzip")
    summary.to_csv(run_dir / "state_causal_summary.csv", index=False)
    pair_flips.to_csv(run_dir / "action_pair_flip_rates.csv", index=False)
    pd.DataFrame(value_errors).to_csv(run_dir / "source_value_errors.csv", index=False)
    rollouts.to_csv(run_dir / "rollout_causal_summary.csv", index=False)
    gate_cfg = config["causal"]
    gate = assess_causal_gate(
        summary,
        rollouts,
        float(gate_cfg["oracle_relative_gain_floor"]),
        float(gate_cfg["repaired_oracle_regret_ceiling"]),
    )
    write_json(run_dir / "causal_gate.json", gate)
    write_json(
        run_dir / "manifest.json",
        {
            "schema": "direct_action_planning_value_refresh.causal_manifest.v1",
            "status": "completed",
            "decision": gate["decision"],
            "started_at": _now(),
            "elapsed_seconds": time.perf_counter() - started,
            "rows": len(states),
            "predecessor_inputs_verified": True,
        },
    )
    write_json(run_dir / "stage_status.json", {"stage": "completed", "at": _now()})
    return run_dir


def _collect_value_visits(
    agent,
    mdp,
    fixed,
    optimum,
    budgets,
    scenario,
    seed,
    episodes,
    round_index,
    device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples, visits = collect_mixed_policy_branches(
        agent,
        agent,
        mdp,
        fixed,
        optimum,
        budgets,
        scenario,
        seed,
        episodes,
        round_index,
        1.0,
        device=device,
    )
    return samples, visits


def _loso_validation(records: list[dict[str, object]], config: dict) -> pd.DataFrame:
    rows = []
    by_seed = {}
    for record in records:
        by_seed.setdefault(int(record["train_seed"]), []).append(record)
    for train_seed, seed_records in by_seed.items():
        for heldout in seed_records:
            source_records = [
                record for record in seed_records if record["scenario"] != heldout["scenario"]
            ]
            if len(source_records) != 2:
                continue
            corrections = np.stack(
                [
                    record["final_value"].values - record["fixed"].values
                    for record in source_records
                ]
            )
            transferred = BudgetValueTable(
                heldout["fixed"].values + np.mean(corrections, axis=0),
                source="loso_mean_residual_correction",
            )
            mdp, optimum, transition = (
                heldout["mdp"],
                heldout["optimum"],
                heldout["final_transition"],
            )
            d1_probe = _probe_agent(
                StructuredPlanningAgent(mdp, heldout["fixed"], heldout["transition"]),
                mdp,
                optimum,
                [int(value) for value in config["budgets"]],
                heldout["scenario"],
                int(heldout["validation_seed"]),
                int(config["evaluation_episodes"]),
                torch.device(config["device"]),
            )
            loso_probe = _probe_agent(
                StructuredPlanningAgent(mdp, transferred, transition),
                mdp,
                optimum,
                [int(value) for value in config["budgets"]],
                heldout["scenario"],
                int(heldout["validation_seed"]),
                int(config["evaluation_episodes"]),
                torch.device(config["device"]),
            )
            rows.append(
                {
                    "heldout_scenario": heldout["scenario"],
                    "train_seed": train_seed,
                    "validation_seed": heldout["validation_seed"],
                    "source_scenarios": "|".join(
                        sorted(str(record["scenario"]) for record in source_records)
                    ),
                    "frozen_D1_regret": d1_probe["rollout_q_star_regret"],
                    "loso_regret": loso_probe["rollout_q_star_regret"],
                    "regret_delta": loso_probe["rollout_q_star_regret"]
                    - d1_probe["rollout_q_star_regret"],
                    "frozen_D1_completion": d1_probe["completion_rate"],
                    "loso_completion": loso_probe["completion_rate"],
                    "frozen_D1_cost": d1_probe["total_cost"],
                    "loso_cost": loso_probe["total_cost"],
                }
            )
    return pd.DataFrame(rows)


def _run_alternating_training(
    cell: Path,
    mdp,
    optimum,
    scenario: str,
    train_seed: int,
    validation_seed: int,
    old_model,
    initial_transition,
    initial_value,
    initial_probe: dict[str, float],
    all_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    anchors: pd.DataFrame,
    config: dict,
    device: torch.device,
) -> tuple[object, BudgetValueTable, int, list[dict[str, object]], list[dict[str, object]]]:
    alternating = config["alternating"]
    value_cfg = config["value_training"]
    current_transition = initial_transition
    current_value = initial_value
    best_transition = initial_transition
    best_value = initial_value
    best_round = 0
    best_regret = float(initial_probe["full_state_q_star_regret"])
    best_rollout_regret = float(initial_probe["rollout_q_star_regret"])
    missed = 0
    validation_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    transition_loss = alternating["transition_losses"]
    for round_index in range(1, int(alternating["max_rounds"]) + 1):
        branch = collect_common_random_branch_data(
            mdp,
            current_value,
            optimum,
            scenario,
            seed=train_seed + 8_000_009 + round_index * 100_003,
            train_samples=int(alternating["transition_train_crn_samples_per_state"]),
            validation_samples=int(
                alternating["transition_validation_crn_samples_per_state"]
            ),
        )
        labels = attach_priority_weights(
            branch.labels,
            mdp,
            old_model,
            current_value,
            optimum,
            scenario,
            float(alternating["close_gap_ceiling"]),
            float(alternating["priority_weight_ceiling"]),
        )
        labels.to_csv(
            cell / "data" / f"alternating_D{round_index}_transition_labels.csv.gz",
            index=False,
            compression="gzip",
        )
        transition_result = train_structured_model(
            labels,
            mdp.config.horizon,
            mdp.n_loads,
            mdp.n_actions,
            mdp.config.gamma,
            LossWeights(
                state=float(transition_loss["state"]),
                effect=float(transition_loss["effect"]),
                q=float(transition_loss["q"]),
                rank=float(transition_loss["rank"]),
                rank_margin=float(transition_loss["rank_margin"]),
            ),
            TrainingConfig(
                hidden_dim=int(alternating["transition_hidden_dim"]),
                learning_rate=float(alternating["transition_learning_rate"]),
                max_epochs=int(alternating["transition_epochs"]),
                patience=int(alternating["transition_patience"]),
                seed=train_seed * 10_000 + round_index,
                use_priority_weights=True,
            ),
            device=device,
        )
        torch.save(
            model_checkpoint_payload(transition_result),
            cell / "models" / f"alternating_D{round_index}_transition.pt",
        )
        transition_result.history.to_csv(
            cell / "training" / f"alternating_D{round_index}_transition.csv", index=False
        )
        value_result = train_refreshed_value(
            current_value,
            transition_result.model,
            mdp,
            optimum,
            all_targets,
            validation_targets,
            anchors,
            VALUE_ABLATIONS["full_value_refresh"],
            ValueTrainingConfig(
                learning_rate=float(value_cfg["learning_rate"]),
                max_epochs=int(value_cfg["max_epochs"]),
                patience=int(value_cfg["patience"]),
                rank_margin=float(value_cfg["rank_margin"]),
                seed=train_seed * 10_000 + 500 + round_index,
            ),
            source_weights={
                key: float(value) for key, value in value_cfg["source_weights"].items()
            },
            device=device,
        )
        refreshed = value_result.model.as_value_table(f"alternating_D{round_index}")
        _save_refreshed_value(
            cell / "models" / f"alternating_D{round_index}_value",
            value_result,
            f"alternating_D{round_index}",
        )
        value_result.history.to_csv(
            cell / "training" / f"alternating_D{round_index}_value.csv", index=False
        )
        probe = _probe_agent(
            StructuredPlanningAgent(mdp, refreshed, transition_result.model),
            mdp,
            optimum,
            [int(value) for value in config["budgets"]],
            scenario,
            validation_seed,
            int(config["evaluation_episodes"]),
            device,
        )
        improved = (
            probe["full_state_q_star_regret"]
            < best_regret - float(alternating["minimum_validation_improvement"])
            and probe["rollout_q_star_regret"] <= best_rollout_regret + 1.0e-12
        )
        validation_rows.append(
            {
                "scenario": scenario,
                "train_seed": train_seed,
                "validation_seed": validation_seed,
                "method": f"alternating_D{round_index}",
                "selected": False,
                "alternation_round": round_index,
                "improved_vs_best": improved,
                **probe,
            }
        )
        training_rows.append(
            {
                "scenario": scenario,
                "train_seed": train_seed,
                "method": f"alternating_D{round_index}",
                "best_epoch": value_result.best_epoch,
                "transition_best_epoch": transition_result.best_epoch,
                **value_result.selection_metrics,
            }
        )
        if improved:
            best_transition = transition_result.model
            best_value = refreshed
            best_round = round_index
            best_regret = float(probe["full_state_q_star_regret"])
            best_rollout_regret = float(probe["rollout_q_star_regret"])
            missed = 0
        else:
            missed += 1
        current_transition = transition_result.model
        current_value = refreshed
        if missed >= int(alternating["stop_after_nonimproving_rounds"]):
            break
    for row in validation_rows:
        row["selected"] = row["alternation_round"] == best_round
    return best_transition, best_value, best_round, validation_rows, training_rows


def run_minimal_value_refresh(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "minimal_v1",
    smoke: bool = False,
    causal_run_id: str = "causal_v2",
) -> Path:
    root = Path(project_root).resolve()
    config = _resolved(load_config(Path(config_path).resolve()), run_id, smoke)
    causal_path = root / "results/direct_action_planning_value_refresh" / causal_run_id
    causal_gate = json.loads((causal_path / "causal_gate.json").read_text(encoding="utf-8"))
    if causal_gate["decision"] != "CONTINUE":
        raise RuntimeError("causal gate stopped Value Refresh before training")
    verify_frozen_inputs(root)
    run_dir = root / "results/direct_action_planning_value_refresh" / run_id
    if run_dir.exists():
        manifest = run_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return run_dir
        raise RuntimeError(f"append-only value-refresh run exists and is incomplete: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_json(run_dir / "stage_status.json", {"stage": "training", "at": _now()})
    set_global_seed(0, torch_threads=int(config["torch_threads"]))
    device = torch.device(config["device"])
    triples = list(
        zip(
            config["seed_split"]["train"],
            config["seed_split"]["validation"],
            config["seed_split"]["test"],
        )
    )
    records: list[dict[str, object]] = []
    validation_rows, training_rows, anchor_rows, collection_rows = [], [], [], []
    value_cfg = config["value_training"]
    for scenario in config["scenarios"]:
        mdp = _mdp(config, scenario)
        optimum = solve_action_dp(mdp)
        oracle = BudgetValueTable.from_exact_dp(optimum.values)
        for train_value, validation_value, test_value in triples:
            train_seed = int(train_value)
            validation_seed = int(validation_value)
            test_seed = int(test_value)
            cell = run_dir / "cells" / f"{scenario}__train_s{train_seed}"
            (cell / "models").mkdir(parents=True)
            (cell / "training").mkdir()
            (cell / "data").mkdir()
            repair_cell = _repair_cell(root, scenario, train_seed, config)
            controlled_cell = _controlled_cell(root, scenario, train_seed, config)
            fixed = load_value_table(repair_cell / "learned_value.npz")
            old_model = load_empirical_model(repair_cell / "old_model.npz")
            transition = load_structured_checkpoint(
                repair_cell / "models/aggregation_round_1.pt"
            )
            controlled_checkpoint, controlled_round = _controlled_checkpoint(controlled_cell)
            controlled_transition = load_structured_checkpoint(controlled_checkpoint)
            controlled_value = load_value_table(controlled_cell / "learned_value.npz")
            frozen_agent = StructuredPlanningAgent(mdp, fixed, transition)
            train_samples, train_visits = _collect_value_visits(
                frozen_agent,
                mdp,
                fixed,
                optimum,
                [int(value) for value in config["budgets"]],
                scenario,
                train_seed,
                int(config["collection_episodes"]),
                1,
                device,
            )
            validation_samples, validation_visits = _collect_value_visits(
                frozen_agent,
                mdp,
                fixed,
                optimum,
                [int(value) for value in config["budgets"]],
                scenario,
                validation_seed,
                int(config["collection_episodes"]),
                2,
                device,
            )
            train_samples.to_csv(
                cell / "data/fresh_train_branches.csv.gz", index=False, compression="gzip"
            )
            train_visits.to_csv(cell / "data/fresh_train_visits.csv", index=False)
            validation_samples.to_csv(
                cell / "data/fresh_validation_branches.csv.gz",
                index=False,
                compression="gzip",
            )
            validation_visits.to_csv(cell / "data/fresh_validation_visits.csv", index=False)
            d0 = pd.read_csv(
                repair_cell / "D0_branch_labels.csv.gz", usecols=STATE_COLUMNS
            ).drop_duplicates()
            d1 = pd.read_csv(
                repair_cell / "D1_branch_samples.csv.gz", usecols=STATE_COLUMNS
            ).drop_duplicates()
            controlled = _controlled_sources(controlled_cell)
            source_frames = {
                "D0": d0,
                "D1": d1,
                "controlled": controlled,
                "fresh_aggregation": train_samples,
            }
            all_targets = value_state_targets(mdp, optimum, source_frames)
            aggregate_targets = all_targets[all_targets.source != "D0"].reset_index(drop=True)
            validation_targets = value_state_targets(
                mdp, optimum, {"validation": validation_samples}
            )
            anchors = build_anchor_states(
                mdp,
                optimum,
                d0,
                d1,
                scenario,
                float(value_cfg["high_value_quantile"]),
            )
            all_targets.to_csv(cell / "data/value_targets.csv.gz", index=False, compression="gzip")
            validation_targets.to_csv(cell / "data/validation_targets.csv", index=False)
            anchors.to_csv(cell / "data/anchors.csv.gz", index=False, compression="gzip")
            collection_rows.append(
                {
                    "scenario": scenario,
                    "train_seed": train_seed,
                    "validation_seed": validation_seed,
                    "fresh_train_rows": len(train_samples),
                    "fresh_train_states": len(train_samples[STATE_COLUMNS].drop_duplicates()),
                    "fresh_validation_states": len(
                        validation_samples[STATE_COLUMNS].drop_duplicates()
                    ),
                    **{
                        f"{source}_states": len(frame[STATE_COLUMNS].drop_duplicates())
                        for source, frame in source_frames.items()
                    },
                }
            )
            baseline_agents = {
                "learned_value": DirectPlanningAgent(mdp, fixed),
                "original_learned_model": DirectPlanningAgent(
                    mdp, fixed, learned_model=old_model
                ),
                "frozen_best_D1": StructuredPlanningAgent(mdp, fixed, transition),
                "controlled_full": StructuredPlanningAgent(
                    mdp, controlled_value, controlled_transition
                ),
                "repaired_transition_oracle_value": StructuredPlanningAgent(
                    mdp, oracle, transition
                ),
            }
            baseline_probes = {}
            for method, agent in baseline_agents.items():
                probe = _probe_agent(
                    agent,
                    mdp,
                    optimum,
                    [int(value) for value in config["budgets"]],
                    scenario,
                    validation_seed,
                    int(config["evaluation_episodes"]),
                    device,
                )
                baseline_probes[method] = probe
                validation_rows.append(
                    {
                        "scenario": scenario,
                        "train_seed": train_seed,
                        "validation_seed": validation_seed,
                        "method": method,
                        "selected": False,
                        "alternation_round": 0,
                        **probe,
                    }
                )
            trained_variants = {}
            refreshed_tables = {}
            variant_probes = {}
            for variant_index, (method, weights) in enumerate(VALUE_ABLATIONS.items()):
                training_targets = (
                    all_targets if method == "full_value_refresh" else aggregate_targets
                )
                source_weights = (
                    {key: float(value) for key, value in value_cfg["source_weights"].items()}
                    if method == "full_value_refresh"
                    else None
                )
                trained = train_refreshed_value(
                    fixed,
                    transition,
                    mdp,
                    optimum,
                    training_targets,
                    validation_targets,
                    anchors,
                    weights,
                    ValueTrainingConfig(
                        learning_rate=float(value_cfg["learning_rate"]),
                        max_epochs=int(value_cfg["max_epochs"]),
                        patience=int(value_cfg["patience"]),
                        rank_margin=float(value_cfg["rank_margin"]),
                        seed=train_seed * 100 + variant_index,
                    ),
                    source_weights=source_weights,
                    device=device,
                )
                trained_variants[method] = trained
                table = trained.model.as_value_table(method)
                refreshed_tables[method] = table
                _save_refreshed_value(cell / "models" / method, trained, method)
                trained.history.to_csv(cell / "training" / f"{method}.csv", index=False)
                probe = _probe_agent(
                    StructuredPlanningAgent(mdp, table, transition),
                    mdp,
                    optimum,
                    [int(value) for value in config["budgets"]],
                    scenario,
                    validation_seed,
                    int(config["evaluation_episodes"]),
                    device,
                )
                variant_probes[method] = probe
                validation_rows.append(
                    {
                        "scenario": scenario,
                        "train_seed": train_seed,
                        "validation_seed": validation_seed,
                        "method": method,
                        "selected": method == "full_value_refresh",
                        "alternation_round": 0,
                        **probe,
                    }
                )
                training_rows.append(
                    {
                        "scenario": scenario,
                        "train_seed": train_seed,
                        "method": method,
                        "best_epoch": trained.best_epoch,
                        **trained.selection_metrics,
                    }
                )
                anchor_rows.extend(
                    _anchor_forgetting(
                        anchors,
                        fixed,
                        table,
                        transition,
                        mdp,
                        optimum,
                        scenario,
                        train_seed,
                        method,
                    )
                )
            full_value = refreshed_tables["full_value_refresh"]
            final_transition = transition
            final_value = full_value
            selected_alternation_round = 0
            full_improves = (
                variant_probes["full_value_refresh"]["full_state_q_star_regret"]
                < baseline_probes["frozen_best_D1"]["full_state_q_star_regret"]
                - float(config["alternating"]["minimum_validation_improvement"])
                and variant_probes["full_value_refresh"]["rollout_q_star_regret"]
                <= baseline_probes["frozen_best_D1"]["rollout_q_star_regret"] + 1.0e-12
            )
            if full_improves and int(config["alternating"]["max_rounds"]) > 0:
                (
                    final_transition,
                    final_value,
                    selected_alternation_round,
                    alternating_validation,
                    alternating_training,
                ) = _run_alternating_training(
                    cell,
                    mdp,
                    optimum,
                    scenario,
                    train_seed,
                    validation_seed,
                    old_model,
                    transition,
                    full_value,
                    variant_probes["full_value_refresh"],
                    all_targets,
                    validation_targets,
                    anchors,
                    config,
                    device,
                )
                validation_rows.extend(alternating_validation)
                training_rows.extend(alternating_training)
                if selected_alternation_round > 0:
                    for row in validation_rows:
                        if (
                            row["scenario"] == scenario
                            and row["train_seed"] == train_seed
                            and row["method"] == "full_value_refresh"
                        ):
                            row["selected"] = False
            anchor_rows.extend(
                _anchor_forgetting(
                    anchors,
                    fixed,
                    final_value,
                    final_transition,
                    mdp,
                    optimum,
                    scenario,
                    train_seed,
                    "final_value_refresh",
                )
            )
            records.append(
                {
                    "scenario": scenario,
                    "train_seed": train_seed,
                    "validation_seed": validation_seed,
                    "test_seed": test_seed,
                    "mdp": mdp,
                    "optimum": optimum,
                    "oracle": oracle,
                    "fixed": fixed,
                    "old_model": old_model,
                    "transition": transition,
                    "controlled_transition": controlled_transition,
                    "controlled_value": controlled_value,
                    "controlled_round": controlled_round,
                    "refreshed_tables": refreshed_tables,
                    "full_value": full_value,
                    "final_value": final_value,
                    "final_transition": final_transition,
                    "anchors": anchors,
                    "cell": cell,
                    "selected_alternation_round": selected_alternation_round,
                }
            )

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(run_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(run_dir / "training_summary.csv", index=False)
    anchor_frame = pd.DataFrame(anchor_rows)
    anchor_frame.to_csv(run_dir / "anchor_forgetting.csv", index=False)
    pd.DataFrame(collection_rows).to_csv(run_dir / "collection_summary.csv", index=False)
    loso = _loso_validation(records, config) if len(config["scenarios"]) == 3 else pd.DataFrame()
    loso.to_csv(run_dir / "loso_validation.csv", index=False)
    ledger_cells = [
        {
            "scenario": record["scenario"],
            "train_seed": record["train_seed"],
            "validation_seed": record["validation_seed"],
            "test_seed": record["test_seed"],
            "selected_method": (
                f"alternating_D{record['selected_alternation_round']}"
                if int(record["selected_alternation_round"]) > 0
                else "full_value_refresh"
            ),
            "selected_alternation_round": record["selected_alternation_round"],
        }
        for record in records
    ]
    ledger = FinalTestLedger.create(run_dir / "validation_selection.json", ledger_cells)
    ledger.mark_test_started()
    write_json(run_dir / "stage_status.json", {"stage": "single_final_test", "at": _now()})
    state_frames, state_summaries, episode_frames, step_frames = [], [], [], []
    runtime_rows, q_frames = [], []
    for record in records:
        mdp, optimum = record["mdp"], record["optimum"]
        scenario, train_seed, test_seed = (
            str(record["scenario"]),
            int(record["train_seed"]),
            int(record["test_seed"]),
        )
        agents = {
            "exact_dp": DirectPlanningAgent(mdp, record["oracle"]),
            "learned_value": DirectPlanningAgent(mdp, record["fixed"]),
            "original_learned_model": DirectPlanningAgent(
                mdp, record["fixed"], learned_model=record["old_model"]
            ),
            "frozen_best_D1": StructuredPlanningAgent(
                mdp, record["fixed"], record["transition"]
            ),
            "controlled_full": StructuredPlanningAgent(
                mdp, record["controlled_value"], record["controlled_transition"]
            ),
            "repaired_transition_oracle_value": StructuredPlanningAgent(
                mdp, record["oracle"], record["transition"]
            ),
            **{
                method: StructuredPlanningAgent(mdp, table, record["transition"])
                for method, table in record["refreshed_tables"].items()
            },
            "final_value_refresh": StructuredPlanningAgent(
                mdp, record["final_value"], record["final_transition"]
            ),
        }
        for method, agent in agents.items():
            states, summary = evaluate_full_state_policy(
                agent, method, mdp, optimum, scenario, test_seed, device
            )
            episodes, steps, runtime = evaluate_policy_rollouts(
                agent,
                method,
                mdp,
                optimum,
                [int(value) for value in config["budgets"]],
                scenario,
                test_seed,
                int(config["evaluation_episodes"]),
                device,
            )
            for frame in (states, episodes, steps):
                frame["train_seed"] = train_seed
                frame["test_seed"] = test_seed
            summary["train_seed"] = train_seed
            summary["test_seed"] = test_seed
            state_frames.append(states)
            state_summaries.append(summary)
            episode_frames.append(episodes)
            step_frames.append(steps)
            runtime_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "train_seed": train_seed,
                    "test_seed": test_seed,
                    **runtime,
                }
            )
            q_frames.append(
                _planning_q_rows(agent, method, mdp, optimum, scenario, test_seed, device)
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
    ledger.mark_test_completed()
    if smoke:
        gate = {"schema": "value_refresh.smoke.v1", "decision": "SMOKE_ONLY"}
    else:
        gate = assess_value_refresh_gate(
            summaries, episodes, anchor_frame, loso, causal_gate, config["gate"]
        )
    write_json(run_dir / "continuation_gate.json", gate)
    write_json(
        run_dir / "manifest.json",
        {
            "schema": "direct_action_planning_value_refresh.manifest.v1",
            "status": "completed",
            "decision": gate["decision"],
            "elapsed_seconds": time.perf_counter() - started,
            "causal_run_id": causal_run_id,
            "predecessor_inputs_verified": True,
            "final_test_evaluations": 1,
        },
    )
    write_json(
        run_dir / "stage_status.json",
        {"stage": "completed", "at": _now(), "decision": gate["decision"]},
    )
    return run_dir
