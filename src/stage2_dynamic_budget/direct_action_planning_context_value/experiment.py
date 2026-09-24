from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
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
from stage2_dynamic_budget.direct_action_planning.planning import (
    BudgetValueTable,
    DirectPlanningAgent,
)
from stage2_dynamic_budget.direct_action_planning_repair.experiment import (
    _pair_ranking_summary,
    _planning_q_rows,
)
from stage2_dynamic_budget.direct_action_planning_repair.planning import StructuredPlanningAgent
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.diagnosis import (
    load_structured_checkpoint,
)
from stage2_dynamic_budget.direct_action_planning_value_refresh.experiment import (
    load_value_table,
)
from stage2_dynamic_budget.experiment import load_config
from stage2_dynamic_budget.utils.artifacts import environment_record, sha256_file, write_json
from stage2_dynamic_budget.utils.seed import set_global_seed

from .aliasing import alias_summary, build_alias_table, build_near_alias_pairs
from .data import collect_context_trajectories
from .evaluation import evaluate_context_state_grid
from .gate import assess_context_gate, assess_oracle_context_gate
from .history import leakage_audit
from .model import ContextValuePredictor
from .planning import ContextPlanningAgent
from .protocol import FinalTestLedger
from .training import (
    ContextLossWeights,
    ContextTrainingConfig,
    context_checkpoint_payload,
    train_context_value,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decision_latency_mean(runtime: dict[str, float]) -> float:
    if "decision_latency_ms_mean" not in runtime:
        raise KeyError("frozen evaluator did not return decision_latency_ms_mean")
    return float(runtime["decision_latency_ms_mean"])


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


def verify_frozen_inputs(project_root: str | Path) -> list[dict[str, object]]:
    root = Path(project_root).resolve()
    manifest = root / "research/direct_action_planning_context_value/FROZEN_INPUTS.sha256"
    rows: list[dict[str, object]] = []
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
                "matches": expected == actual,
            }
        )
    if not rows or not all(bool(row["matches"]) for row in rows):
        changed = [str(row["path"]) for row in rows if not row["matches"]]
        raise RuntimeError(f"frozen Context Value input changed: {changed}")
    return rows


def _repair_cell(root: Path, scenario: str, model_seed: int) -> Path:
    return root / "results/direct_action_planning_repair/minimal_v1/cells" / (
        f"{scenario}__s{model_seed}"
    )


def _old_selection(root: Path) -> dict[tuple[str, int], dict[str, object]]:
    payload = json.loads(
        (root / "results/direct_action_planning_value_refresh/minimal_v1/validation_selection.json")
        .read_text(encoding="utf-8")
    )
    return {
        (str(row["scenario"]), int(row["train_seed"])): row for row in payload["cells"]
    }


def _selected_old_refresh(
    root: Path,
    scenario: str,
    old_train_seed: int,
    model_seed: int,
    selection: dict[tuple[str, int], dict[str, object]],
) -> tuple[BudgetValueTable, object]:
    cell = root / "results/direct_action_planning_value_refresh/minimal_v1/cells" / (
        f"{scenario}__train_s{old_train_seed}"
    )
    selected = selection[(scenario, old_train_seed)]
    round_index = int(selected["selected_alternation_round"])
    if round_index:
        value_path = cell / "models" / f"alternating_D{round_index}_value.npz"
        transition_path = cell / "models" / f"alternating_D{round_index}_transition.pt"
    else:
        value_path = cell / "models/full_value_refresh.npz"
        transition_path = _repair_cell(root, scenario, model_seed) / "models/aggregation_round_1.pt"
    return load_value_table(value_path), load_structured_checkpoint(transition_path)


def _reconstruct_old_loso_errors(
    root: Path,
    config: dict,
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    selection = _old_selection(root)
    scenarios = [str(value) for value in config["scenarios"]]
    rows: list[pd.DataFrame] = []
    for slot, model_seed in enumerate(config["predecessor_model_seeds"]):
        old_train_seed = 25 + slot
        values: dict[str, BudgetValueTable] = {}
        fixed: dict[str, BudgetValueTable] = {}
        transitions = {}
        for scenario in scenarios:
            repair = _repair_cell(root, scenario, int(model_seed))
            fixed[scenario] = load_value_table(repair / "learned_value.npz")
            values[scenario], transitions[scenario] = _selected_old_refresh(
                root, scenario, old_train_seed, int(model_seed), selection
            )
        for heldout in scenarios:
            source = [name for name in scenarios if name != heldout]
            correction = np.mean(
                np.stack([values[name].values - fixed[name].values for name in source]), axis=0
            )
            transferred = BudgetValueTable(
                fixed[heldout].values + correction, source="reconstructed_old_loso"
            )
            mdp = _mdp(config, heldout)
            optimum = solve_action_dp(mdp)
            agent = StructuredPlanningAgent(mdp, transferred, transitions[heldout])
            state_rows, _ = evaluate_full_state_policy(
                agent,
                "old_loso_refresh",
                mdp,
                optimum,
                heldout,
                old_train_seed,
                torch.device(config["device"]),
            )
            state_rows["model_seed"] = int(model_seed)
            state_rows["old_train_seed"] = old_train_seed
            rows.append(state_rows)
    frame = pd.concat(rows, ignore_index=True)
    return frame.merge(
        aliases[
            [
                "t",
                "load",
                "queue",
                "remaining_budget",
                "optimal_action_conflict",
                "material_action_conflict",
                "cross_scenario_value_variance",
                "max_future_window_difference",
            ]
        ],
        on=["t", "load", "queue", "remaining_budget"],
        validate="many_to_one",
    )


def _plot_alias_heatmap(aliases: pd.DataFrame, output: Path) -> None:
    heat = aliases.pivot_table(
        index="t", columns="remaining_budget", values="material_action_conflict", aggfunc="mean"
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    image = axis.imshow(heat.to_numpy(), aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axis.set_xlabel("Remaining budget")
    axis.set_ylabel("Decision time")
    axis.set_title("Cross-scenario material optimal-action conflicts")
    axis.set_xticks(range(len(heat.columns)), labels=[str(value) for value in heat.columns])
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Conflict rate across load/queue states")
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run_alias_oracle_diagnosis(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "alias_oracle_v1",
) -> Path:
    root = Path(project_root).resolve()
    config = load_config(Path(config_path).resolve())
    run_dir = root / "results/direct_action_planning_context_value" / run_id
    if run_dir.exists():
        status = run_dir / "stage_status.json"
        if status.exists() and json.loads(status.read_text()).get("stage") == "completed":
            return run_dir
        raise RuntimeError(f"append-only alias/oracle run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started_at = _now()
    started = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_json(run_dir / "stage_status.json", {"stage": "alias_audit", "at": _now()})
    frozen = verify_frozen_inputs(root)
    pd.DataFrame(frozen).to_csv(run_dir / "frozen_input_verification.csv", index=False)
    set_global_seed(0, torch_threads=int(config["torch_threads"]))
    scenarios = [str(value) for value in config["scenarios"]]
    mdps = {scenario: _mdp(config, scenario) for scenario in scenarios}
    optima = {scenario: solve_action_dp(mdp) for scenario, mdp in mdps.items()}
    alias_cfg = config["aliasing"]
    aliases = build_alias_table(
        mdps,
        optima,
        future_steps=int(alias_cfg["future_window"]),
        material_q_gap=float(alias_cfg["material_q_gap"]),
    )
    aliases.to_csv(run_dir / "exact_alias_states.csv.gz", index=False, compression="gzip")
    aliases[aliases.optimal_action_conflict].to_csv(
        run_dir / "conflict_states.csv.gz", index=False, compression="gzip"
    )
    near = build_near_alias_pairs(mdps, optima, float(alias_cfg["near_linf_radius"]))
    near.to_csv(run_dir / "near_alias_pairs.csv.gz", index=False, compression="gzip")
    summary = alias_summary(aliases)
    summary["near_alias_pairs"] = int(len(near))
    summary["near_alias_action_conflict_rate"] = float(near.optimal_action_conflict.mean())
    write_json(run_dir / "alias_summary.json", summary)
    _plot_alias_heatmap(aliases, run_dir / "conflict_heatmap")
    old_loso = _reconstruct_old_loso_errors(root, config, aliases)
    old_loso.to_csv(
        run_dir / "old_loso_state_error_attribution.csv.gz", index=False, compression="gzip"
    )
    attribution = (
        old_loso.groupby(["scenario", "material_action_conflict"], as_index=False)
        .agg(
            states=("action_consistent", "size"),
            action_error_rate=("action_consistent", lambda values: 1.0 - float(np.mean(values))),
            mean_q_star_regret=("Q_star_regret", "mean"),
            total_q_star_regret=("Q_star_regret", "sum"),
            mean_value_variance=("cross_scenario_value_variance", "mean"),
            mean_future_difference=("max_future_window_difference", "mean"),
        )
    )
    attribution.to_csv(run_dir / "old_loso_error_attribution.csv", index=False)

    write_json(run_dir / "stage_status.json", {"stage": "oracle_context", "at": _now()})
    state_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    cell_rows: list[dict[str, object]] = []
    validation_seeds = [int(value) for value in config["seed_split"]["validation"]]
    for slot, (model_seed, validation_seed) in enumerate(
        zip(config["predecessor_model_seeds"], validation_seeds)
    ):
        for scenario in scenarios:
            mdp, optimum = mdps[scenario], optima[scenario]
            repair = _repair_cell(root, scenario, int(model_seed))
            fixed = load_value_table(repair / "learned_value.npz")
            transition = load_structured_checkpoint(repair / "models/aggregation_round_1.pt")
            source_scenarios = [name for name in scenarios if name != scenario]
            pooled = BudgetValueTable(
                np.mean(
                    np.stack(
                        [BudgetValueTable.from_exact_dp(optima[name].values).values for name in source_scenarios]
                    ),
                    axis=0,
                ),
                source="pooled_current_state_loso",
            )
            oracle = BudgetValueTable.from_exact_dp(optimum.values)
            agents = {
                "frozen_D1": StructuredPlanningAgent(mdp, fixed, transition),
                "current_state_value": StructuredPlanningAgent(mdp, pooled, transition),
                "oracle_scenario_value": StructuredPlanningAgent(mdp, oracle, transition),
                "oracle_phase_value": StructuredPlanningAgent(mdp, oracle, transition),
            }
            for method, agent in agents.items():
                states, state_summary = evaluate_full_state_policy(
                    agent, method, mdp, optimum, scenario, validation_seed, torch.device(config["device"])
                )
                episodes, _, runtime = evaluate_policy_rollouts(
                    agent,
                    method,
                    mdp,
                    optimum,
                    [int(value) for value in config["budgets"]],
                    scenario,
                    validation_seed,
                    int(config["evaluation_episodes"]),
                    torch.device(config["device"]),
                )
                states["model_seed"] = int(model_seed)
                states["validation_seed"] = validation_seed
                episodes["model_seed"] = int(model_seed)
                episodes["validation_seed"] = validation_seed
                state_frames.append(states)
                episode_frames.append(episodes)
                cell_rows.append(
                    {
                        "scenario": scenario,
                        "model_seed": int(model_seed),
                        "validation_seed": validation_seed,
                        "method": method,
                        "regret": float(episodes.mean_Q_star_regret.mean()),
                        "action_consistency": float(episodes.action_consistency_rate.mean()),
                        "full_state_regret": float(state_summary["mean_Q_star_regret"]),
                        "full_state_action_consistency": float(state_summary["action_consistency_rate"]),
                        "budget_trajectory_mae": float(episodes.budget_trajectory_mae.mean()),
                        "completion_rate": float(episodes.completion_rate.mean()),
                        "slo_violation_rate": float(episodes.slo_violation_rate.mean()),
                        "total_cost": float(episodes.total_cost.mean()),
                        "planning_ms_per_decision": _decision_latency_mean(runtime),
                        "privileged_context": method.startswith("oracle_"),
                    }
                )
    states = pd.concat(state_frames, ignore_index=True).merge(
        aliases[["t", "load", "queue", "remaining_budget", "material_action_conflict"]],
        on=["t", "load", "queue", "remaining_budget"],
        validate="many_to_one",
    )
    episodes = pd.concat(episode_frames, ignore_index=True)
    cells = pd.DataFrame(cell_rows)
    states.to_csv(run_dir / "oracle_state_metrics.csv.gz", index=False, compression="gzip")
    episodes.to_csv(run_dir / "oracle_episode_metrics.csv.gz", index=False, compression="gzip")
    cells.to_csv(run_dir / "oracle_cell_metrics.csv", index=False)
    gate = assess_oracle_context_gate(cells)
    write_json(run_dir / "oracle_gate.json", gate)
    write_json(
        run_dir / "manifest.json",
        {
            "schema": "direct_action_planning_context_value.alias_oracle_manifest.v1",
            "status": "completed",
            "decision": gate["decision"],
            "started_at": started_at,
            "completed_at": _now(),
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_inputs_verified": len(frozen),
            "final_test_evaluations": 0,
            "oracle_context_is_diagnostic_only": True,
        },
    )
    write_json(
        run_dir / "stage_status.json",
        {"stage": "completed", "at": _now(), "decision": gate["decision"]},
    )
    return run_dir


def _resolved_context(config: dict, run_id: str, smoke: bool) -> dict:
    value = json.loads(json.dumps(config))
    value["run_id"] = run_id
    value["smoke"] = bool(smoke)
    value["resolved_at"] = _now()
    if smoke:
        value["seed_split"] = {"train": [40], "validation": [45], "test": [50]}
        value["predecessor_model_seeds"] = [5]
        value["history_windows"] = [4]
        value["collection"]["episodes_per_source"] = 3
        value["evaluation_episodes"] = 2
        value["training"]["max_epochs"] = 8
        value["training"]["patience"] = 2
        value["training"]["batch_size"] = 32
        value["training"]["rank_states"] = 12
        value["training"]["validation_rank_states"] = 16
        value["training"]["validation_interval"] = 2
    return value


def _controlled_policy(
    root: Path,
    scenario: str,
    model_seed: int,
    mdp: ActionConditionedBudgetMDP,
) -> StructuredPlanningAgent:
    controlled_seed = model_seed + 5
    cell = root / "results/direct_action_planning_controlled_aggregation/minimal_v1/cells" / (
        f"{scenario}__train_s{controlled_seed}"
    )
    selection = json.loads((cell / "validation_selection.json").read_text(encoding="utf-8"))
    round_index = int(selection["selected_rounds"]["controlled_full"])
    checkpoint = (
        cell / "models/dap_repair_d0.pt"
        if round_index == 0
        else cell / "models" / f"controlled_full_D{round_index}.pt"
    )
    return StructuredPlanningAgent(
        mdp,
        load_value_table(cell / "learned_value.npz"),
        load_structured_checkpoint(checkpoint),
    )


def _context_source_policies(
    root: Path,
    scenario: str,
    model_seed: int,
    old_train_seed: int,
    mdp: ActionConditionedBudgetMDP,
    selection: dict[tuple[str, int], dict[str, object]],
) -> tuple[dict[str, object | None], BudgetValueTable, object, BudgetValueTable, object]:
    repair = _repair_cell(root, scenario, model_seed)
    fixed = load_value_table(repair / "learned_value.npz")
    d1_transition = load_structured_checkpoint(repair / "models/aggregation_round_1.pt")
    fresh_value, fresh_transition = _selected_old_refresh(
        root, scenario, old_train_seed, model_seed, selection
    )
    policies: dict[str, object | None] = {
        "D0": None,
        "D1": StructuredPlanningAgent(mdp, fixed, d1_transition),
        "controlled": _controlled_policy(root, scenario, model_seed, mdp),
        "fresh_aggregation": StructuredPlanningAgent(mdp, fresh_value, fresh_transition),
    }
    return policies, fixed, d1_transition, fresh_value, fresh_transition


def _training_config(config: dict, seed: int) -> ContextTrainingConfig:
    train = config["training"]
    return ContextTrainingConfig(
        hidden_dim=int(train["hidden_dim"]),
        gru_hidden_dim=int(train["gru_hidden_dim"]),
        learning_rate=float(train["learning_rate"]),
        max_epochs=int(train["max_epochs"]),
        patience=int(train["patience"]),
        batch_size=int(train["batch_size"]),
        rank_states=int(train["rank_states"]),
        validation_rank_states=int(train["validation_rank_states"]),
        validation_interval=int(train["validation_interval"]),
        rank_margin=float(train["rank_margin"]),
        seed=seed,
    )


def _loss_weights(config: dict) -> ContextLossWeights:
    losses = config["training"]["losses"]
    return ContextLossWeights(
        value=float(losses["value"]),
        anchor=float(losses["anchor"]),
        mono=float(losses["mono"]),
        rank=float(losses["rank"]),
    )


def _select_validation_configuration(validation: pd.DataFrame) -> dict[str, object]:
    summary = (
        validation.groupby(["mode", "window"], as_index=False)
        .agg(
            mean_regret=("mean_Q_star_regret", "mean"),
            action_consistency=("action_consistency_rate", "mean"),
            completion_rate=("completion_rate", "mean"),
            slo_violation_rate=("slo_violation_rate", "mean"),
            total_cost=("total_cost", "mean"),
        )
    )
    family: dict[str, dict[str, object]] = {}
    for mode in ("feature", "gru"):
        candidates = summary[summary["mode"] == mode].sort_values(
            ["mean_regret", "action_consistency", "window"],
            ascending=[True, False, True],
            kind="stable",
        )
        if candidates.empty:
            raise RuntimeError(f"no validation candidate for {mode}")
        family[mode] = candidates.iloc[0].to_dict()
    selected = sorted(
        family.values(),
        key=lambda row: (
            float(row["mean_regret"]),
            -float(row["action_consistency"]),
            int(row["window"]),
        ),
    )[0]
    return {
        "schema": "direct_action_planning_context_value.validation_choice.v1",
        "selection_source": "validation_only",
        "family_choices": family,
        "selected_mode": str(selected["mode"]),
        "selected_window": int(selected["window"]),
        "tie_break_order": ["mean_regret", "action_consistency", "shorter_window"],
        "test_used_for_selection": False,
    }


def _noncontext_alias_summary(
    states: pd.DataFrame,
    aliases: pd.DataFrame,
    model_seed: int,
) -> pd.DataFrame:
    joined = states.merge(
        aliases[["t", "load", "queue", "remaining_budget", "material_action_conflict"]],
        on=["t", "load", "queue", "remaining_budget"],
        validate="many_to_one",
    )
    rows = []
    for region in ("all", False, True):
        subset = joined if region == "all" else joined[joined.material_action_conflict == region]
        rows.append(
            {
                "method": str(subset.method.iloc[0]),
                "scenario": str(subset.scenario.iloc[0]),
                "model_seed": model_seed,
                "alias_region": region,
                "states": len(subset),
                "action_consistency": float(subset.action_consistent.mean()),
                "mean_Q_star_regret": float(subset.Q_star_regret.mean()),
                "high_risk_low_cost_balanced_accuracy": float("nan"),
                "planning_ms_per_decision": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run_minimal_context_value(
    project_root: str | Path,
    config_path: str | Path,
    run_id: str = "minimal_v1",
    smoke: bool = False,
    oracle_run_id: str = "alias_oracle_v2",
) -> Path:
    root = Path(project_root).resolve()
    config = _resolved_context(load_config(Path(config_path).resolve()), run_id, smoke)
    oracle_gate = json.loads(
        (root / "results/direct_action_planning_context_value" / oracle_run_id / "oracle_gate.json")
        .read_text(encoding="utf-8")
    )
    if oracle_gate["decision"] != "CONTINUE":
        raise RuntimeError("Oracle context gate did not authorize history training")
    run_dir = root / "results/direct_action_planning_context_value" / run_id
    if run_dir.exists():
        manifest = run_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return run_dir
        raise RuntimeError(f"append-only context run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "data").mkdir()
    (run_dir / "models").mkdir()
    (run_dir / "training").mkdir()
    started = time.perf_counter()
    started_at = _now()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_json(run_dir / "stage_status.json", {"stage": "collection", "at": _now()})
    frozen = verify_frozen_inputs(root)
    pd.DataFrame(frozen).to_csv(run_dir / "frozen_input_verification.csv", index=False)
    set_global_seed(0, torch_threads=int(config["torch_threads"]))
    device = torch.device(config["device"])
    scenarios = [str(value) for value in config["scenarios"]]
    mdps = {scenario: _mdp(config, scenario) for scenario in scenarios}
    optima = {scenario: solve_action_dp(mdp) for scenario, mdp in mdps.items()}
    aliases = pd.read_csv(
        root
        / "results/direct_action_planning_context_value"
        / oracle_run_id
        / "exact_alias_states.csv.gz"
    )
    old_selection = _old_selection(root)
    train_frames: dict[tuple[str, int], pd.DataFrame] = {}
    validation_frames: dict[tuple[str, int], pd.DataFrame] = {}
    base_tables: dict[str, BudgetValueTable] = {}
    transitions: dict[str, object] = {}
    fresh_values: dict[str, BudgetValueTable] = {}
    fresh_transitions: dict[str, object] = {}
    leakage_reports: list[dict[str, object]] = []
    for slot, (model_seed, train_seed, validation_seed) in enumerate(
        zip(
            config["predecessor_model_seeds"],
            config["seed_split"]["train"],
            config["seed_split"]["validation"],
        )
    ):
        for scenario in scenarios:
            mdp, optimum = mdps[scenario], optima[scenario]
            policies, fixed, transition, fresh_value, fresh_transition = _context_source_policies(
                root, scenario, int(model_seed), 25 + slot, mdp, old_selection
            )
            key = f"{scenario}::{int(model_seed)}"
            base_tables[key] = fixed
            transitions[key] = transition
            fresh_values[key] = fresh_value
            fresh_transitions[key] = fresh_transition
            train = collect_context_trajectories(
                mdp,
                optimum,
                fixed,
                policies,
                [int(value) for value in config["budgets"]],
                int(train_seed),
                int(config["collection"]["episodes_per_source"]),
                device,
            )
            validation = collect_context_trajectories(
                mdp,
                optimum,
                fixed,
                policies,
                [int(value) for value in config["budgets"]],
                int(validation_seed),
                int(config["collection"]["episodes_per_source"]),
                device,
            )
            for frame in (train, validation):
                frame["model_seed"] = int(model_seed)
            train_frames[(scenario, int(model_seed))] = train
            validation_frames[(scenario, int(model_seed))] = validation
            train.to_csv(
                run_dir / "data" / f"{scenario}__s{model_seed}__train.csv.gz",
                index=False,
                compression="gzip",
            )
            validation.to_csv(
                run_dir / "data" / f"{scenario}__s{model_seed}__validation.csv.gz",
                index=False,
                compression="gzip",
            )
            for split, frame in (("train", train), ("validation", validation)):
                report = leakage_audit(frame)
                leakage_reports.append(
                    {
                        "scenario": scenario,
                        "model_seed": int(model_seed),
                        "split": split,
                        **report,
                    }
                )
    write_json(
        run_dir / "leakage_audit.json",
        {
            "schema": "direct_action_planning_context_value.leakage_collection.v1",
            "passed": all(bool(row["passed"]) for row in leakage_reports),
            "reports": leakage_reports,
        },
    )
    if not all(bool(row["passed"]) for row in leakage_reports):
        raise RuntimeError("future-information leakage gate failed")

    write_json(run_dir / "stage_status.json", {"stage": "validation_training", "at": _now()})
    trained_models: dict[tuple[str, str, int], object] = {}
    validation_rows: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    windows = [int(value) for value in config["history_windows"]]
    variants = [("current", windows[0])] + [
        (mode, window) for mode in ("feature", "gru") for window in windows
    ]
    source_weights = {
        str(key): float(value)
        for key, value in config["collection"]["source_weights"].items()
    }
    for heldout_index, heldout in enumerate(scenarios):
        source_scenarios = [scenario for scenario in scenarios if scenario != heldout]
        training = pd.concat(
            [
                frame
                for (scenario, _), frame in train_frames.items()
                if scenario in source_scenarios
            ],
            ignore_index=True,
        )
        model_validation = pd.concat(
            [
                frame
                for (scenario, _), frame in validation_frames.items()
                if scenario in source_scenarios
            ],
            ignore_index=True,
        )
        for variant_index, (mode, window) in enumerate(variants):
            trained = train_context_value(
                mode,
                window,
                training,
                model_validation,
                mdps,
                optima,
                base_tables,  # type: ignore[arg-type]
                transitions,  # type: ignore[arg-type]
                _loss_weights(config),
                _training_config(config, 10_000 + heldout_index * 100 + variant_index),
                source_weights,
                device,
            )
            trained_models[(heldout, mode, window)] = trained
            stem = f"loso_{heldout}__{mode}__L{window}"
            torch.save(context_checkpoint_payload(trained), run_dir / "models" / f"{stem}.pt")
            trained.history.to_csv(run_dir / "training" / f"{stem}.csv", index=False)
            training_rows.append(
                {
                    "heldout_scenario": heldout,
                    "mode": mode,
                    "window": window,
                    "best_epoch": trained.best_epoch,
                    **trained.selection_metrics,
                }
            )
            for slot, (model_seed, validation_seed) in enumerate(
                zip(config["predecessor_model_seeds"], config["seed_split"]["validation"])
            ):
                key = f"{heldout}::{int(model_seed)}"
                predictor = ContextValuePredictor(trained.model, base_tables[key], window)
                agent = ContextPlanningAgent(
                    mdps[heldout], predictor, transitions[key]  # type: ignore[arg-type]
                )
                episodes, _, runtime = evaluate_policy_rollouts(
                    agent,
                    f"{mode}_L{window}",
                    mdps[heldout],
                    optima[heldout],
                    [int(value) for value in config["budgets"]],
                    heldout,
                    int(validation_seed),
                    int(config["evaluation_episodes"]),
                    device,
                )
                episodes["heldout_scenario"] = heldout
                episodes["model_seed"] = int(model_seed)
                episodes["validation_seed"] = int(validation_seed)
                episodes["mode"] = mode
                episodes["window"] = window
                episodes["decision_latency_ms_mean"] = runtime[
                    "decision_latency_ms_mean"
                ]
                validation_rows.append(episodes)
    validation = pd.concat(validation_rows, ignore_index=True)
    validation.to_csv(run_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(run_dir / "training_summary.csv", index=False)
    selection = _select_validation_configuration(validation)
    write_json(run_dir / "validation_choice.json", selection)
    if smoke:
        write_json(
            run_dir / "manifest.json",
            {
                "schema": "direct_action_planning_context_value.smoke.v1",
                "status": "completed",
                "decision": "SMOKE_ONLY",
                "started_at": started_at,
                "completed_at": _now(),
                "elapsed_seconds": time.perf_counter() - started,
                "final_test_evaluations": 0,
            },
        )
        write_json(
            run_dir / "stage_status.json",
            {"stage": "completed", "at": _now(), "decision": "SMOKE_ONLY"},
        )
        return run_dir

    selected_mode = str(selection["selected_mode"])
    selected_window = int(selection["selected_window"])
    family_windows = {
        mode: int(selection["family_choices"][mode]["window"])  # type: ignore[index]
        for mode in ("feature", "gru")
    }
    all_training = pd.concat(list(train_frames.values()), ignore_index=True)
    all_validation = pd.concat(list(validation_frames.values()), ignore_index=True)
    in_domain = train_context_value(
        selected_mode,
        selected_window,
        all_training,
        all_validation,
        mdps,
        optima,
        base_tables,  # type: ignore[arg-type]
        transitions,  # type: ignore[arg-type]
        _loss_weights(config),
        _training_config(config, 20_000),
        source_weights,
        device,
    )
    torch.save(context_checkpoint_payload(in_domain), run_dir / "models/in_domain_selected.pt")
    in_domain.history.to_csv(run_dir / "training/in_domain_selected.csv", index=False)
    validation_sha = sha256_file(run_dir / "validation_metrics.csv").removeprefix("sha256:")
    ledger = FinalTestLedger.create(
        run_dir / "validation_selection.json", selection, validation_sha
    )
    ledger.mark_test_started()
    write_json(run_dir / "stage_status.json", {"stage": "single_final_test", "at": _now()})

    episode_frames: list[pd.DataFrame] = []
    step_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    state_frames: list[pd.DataFrame] = []
    state_summaries: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    for slot, (model_seed, test_seed) in enumerate(
        zip(config["predecessor_model_seeds"], config["seed_split"]["test"])
    ):
        for heldout in scenarios:
            mdp, optimum = mdps[heldout], optima[heldout]
            key = f"{heldout}::{int(model_seed)}"
            oracle = BudgetValueTable.from_exact_dp(optimum.values)
            regular_agents = {
                "exact_dp": DirectPlanningAgent(mdp, oracle),
                "frozen_D1": StructuredPlanningAgent(
                    mdp, base_tables[key], transitions[key]  # type: ignore[arg-type]
                ),
                "domain_value_refresh": StructuredPlanningAgent(
                    mdp, fresh_values[key], fresh_transitions[key]  # type: ignore[arg-type]
                ),
                "oracle_scenario_value": StructuredPlanningAgent(
                    mdp, oracle, transitions[key]  # type: ignore[arg-type]
                ),
            }
            context_specs = {
                "current_state_value": (trained_models[(heldout, "current", windows[0])], windows[0]),
                "history_feature_value": (
                    trained_models[(heldout, "feature", family_windows["feature"])],
                    family_windows["feature"],
                ),
                "gru_history_value": (
                    trained_models[(heldout, "gru", family_windows["gru"])],
                    family_windows["gru"],
                ),
                "in_domain_context_value": (in_domain, selected_window),
            }
            agents: dict[str, object] = dict(regular_agents)
            for method, (trained, window) in context_specs.items():
                agents[method] = ContextPlanningAgent(
                    mdp,
                    ContextValuePredictor(trained.model, base_tables[key], int(window)),
                    transitions[key],  # type: ignore[arg-type]
                )
            for method, agent in agents.items():
                episodes, steps, runtime = evaluate_policy_rollouts(
                    agent,
                    method,
                    mdp,
                    optimum,
                    [int(value) for value in config["budgets"]],
                    heldout,
                    int(test_seed),
                    int(config["evaluation_episodes"]),
                    device,
                )
                for frame in (episodes, steps):
                    frame["model_seed"] = int(model_seed)
                    frame["test_seed"] = int(test_seed)
                episode_frames.append(episodes)
                step_frames.append(steps)
                runtime_rows.append(
                    {
                        "method": method,
                        "scenario": heldout,
                        "model_seed": int(model_seed),
                        "test_seed": int(test_seed),
                        **runtime,
                    }
                )
            # Diagnostic labels share the exact same decisions; duplicate instead of rerunning.
            for source_method, target_method in (
                ("oracle_scenario_value", "oracle_phase_value"),
                (
                    "history_feature_value"
                    if selected_mode == "feature"
                    else "gru_history_value",
                    "final_context_value",
                ),
            ):
                for collection in (episode_frames, step_frames):
                    source_frame = collection[-len(agents) :]
                    match = [frame for frame in source_frame if str(frame.method.iloc[0]) == source_method][0]
                    clone = match.copy()
                    clone["method"] = target_method
                    collection.append(clone)
                source_runtime = next(
                    row
                    for row in runtime_rows[-len(agents) :]
                    if row["method"] == source_method
                )
                runtime_rows.append({**source_runtime, "method": target_method})

            context_bank = validation_frames[(heldout, int(model_seed))]
            for method, (trained, window) in context_specs.items():
                state, summary, pairs = evaluate_context_state_grid(
                    ContextValuePredictor(trained.model, base_tables[key], int(window)),
                    transitions[key],  # type: ignore[arg-type]
                    mdp,
                    optimum,
                    context_bank,
                    aliases,
                    method,
                    int(model_seed),
                )
                state["test_seed"] = int(test_seed)
                summary["test_seed"] = int(test_seed)
                pairs["test_seed"] = int(test_seed)
                state_frames.append(state)
                state_summaries.append(summary)
                pair_frames.append(pairs)
            selected_source = (
                "history_feature_value" if selected_mode == "feature" else "gru_history_value"
            )
            selected_state = next(
                frame
                for frame in state_frames[-len(context_specs) :]
                if str(frame.method.iloc[0]) == selected_source
            ).copy()
            selected_state["method"] = "final_context_value"
            state_frames.append(selected_state)
            selected_summary = next(
                frame
                for frame in state_summaries[-len(context_specs) :]
                if str(frame.method.iloc[0]) == selected_source
            ).copy()
            selected_summary["method"] = "final_context_value"
            state_summaries.append(selected_summary)
            selected_pairs = next(
                frame
                for frame in pair_frames[-len(context_specs) :]
                if str(frame.method.iloc[0]) == selected_source
            ).copy()
            selected_pairs["method"] = "final_context_value"
            pair_frames.append(selected_pairs)
            for method, agent in regular_agents.items():
                states, _ = evaluate_full_state_policy(
                    agent, method, mdp, optimum, heldout, int(test_seed), device
                )
                states["model_seed"] = int(model_seed)
                states["test_seed"] = int(test_seed)
                state_frames.append(states)
                state_summaries.append(
                    _noncontext_alias_summary(states, aliases, int(model_seed)).assign(
                        test_seed=int(test_seed)
                    )
                )
                q_rows = _planning_q_rows(
                    agent, method, mdp, optimum, heldout, int(test_seed), device
                )
                q_rows["model_seed"] = int(model_seed)
                pairs = _pair_ranking_summary(q_rows)
                if not pairs.empty:
                    pairs["model_seed"] = int(model_seed)
                    pairs["test_seed"] = int(test_seed)
                    pairs["alias_region"] = "all"
                    pair_frames.append(pairs)

    episodes = pd.concat(episode_frames, ignore_index=True)
    steps = pd.concat(step_frames, ignore_index=True)
    states = pd.concat(state_frames, ignore_index=True, sort=False)
    state_summary = pd.concat(state_summaries, ignore_index=True, sort=False)
    pairs = pd.concat(pair_frames, ignore_index=True, sort=False)
    runtime = pd.DataFrame(runtime_rows)
    episodes.to_csv(run_dir / "test_metrics.csv", index=False)
    steps.to_csv(run_dir / "test_steps.csv.gz", index=False, compression="gzip")
    states.to_csv(run_dir / "test_state_actions.csv.gz", index=False, compression="gzip")
    state_summary.to_csv(run_dir / "test_state_summary.csv", index=False)
    pairs.to_csv(run_dir / "test_pair_ranking.csv", index=False)
    runtime.to_csv(run_dir / "runtime.csv", index=False)
    ledger.mark_test_completed()
    gate = assess_context_gate(
        episodes,
        state_summary,
        oracle_gate,
        bool(json.loads((run_dir / "leakage_audit.json").read_text())["passed"]),
        config["gate"],
    )
    write_json(run_dir / "continuation_gate.json", gate)
    write_json(
        run_dir / "manifest.json",
        {
            "schema": "direct_action_planning_context_value.manifest.v1",
            "status": "completed",
            "decision": gate["decision"],
            "started_at": started_at,
            "completed_at": _now(),
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_inputs_verified": len(frozen),
            "final_test_evaluations": 1,
            "selection_source": "validation_only",
        },
    )
    write_json(
        run_dir / "stage_status.json",
        {"stage": "completed", "at": _now(), "decision": gate["decision"]},
    )
    return run_dir
