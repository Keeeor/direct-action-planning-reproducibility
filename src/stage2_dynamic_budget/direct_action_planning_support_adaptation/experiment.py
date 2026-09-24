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
    ActionDPResult,
    solve_action_dp,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.evaluation import ExactDPAgent
from stage2_dynamic_budget.direct_action_planning.planning import (
    BudgetValueTable,
    DirectPlanningAgent,
)
from stage2_dynamic_budget.direct_action_planning_repair.planning import (
    StructuredPlanningAgent,
)
from stage2_dynamic_budget.direct_action_planning_controlled_aggregation.diagnosis import (
    load_structured_checkpoint,
)
from stage2_dynamic_budget.experiment import load_config
from stage2_dynamic_budget.utils.artifacts import environment_record, sha256_file, write_json
from stage2_dynamic_budget.utils.seed import set_global_seed

from .data import collect_value_trajectories, full_state_value_frame, sample_value_states
from .evaluation import evaluate_full_state, evaluate_rollouts
from .gate import assess_support_adaptation_gate, oracle_recovery
from .models import PooledValueNetwork
from .protocol import (
    FinalTestLedger,
    assert_parameter_splits_disjoint,
    split_target_prefix_suffix,
)
from .scenario import ContinuousScenario, build_continuous_mdp
from .support import build_support_features, compute_support_distances, support_relationship
from .training import (
    AdapterTrainingConfig,
    PooledTrainingConfig,
    ValueLossWeights,
    dense_value_table,
    train_pooled_value,
    train_value_adapter,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _original_mdp(config: dict, scenario: str) -> ActionConditionedBudgetMDP:
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


def _continuous_mdp(config: dict, row: dict[str, object]):
    env = config["environment"]
    return build_continuous_mdp(
        ContinuousScenario.from_mapping(row),
        horizon=int(env["horizon"]),
        max_budget=int(env["max_budget"]),
        max_queue=int(env["max_queue"]),
        gamma=float(env["gamma"]),
        action_costs=tuple(map(int, env["action_costs"])),
        action_capacity=tuple(map(int, env["action_capacity"])),
    )


def verify_frozen_inputs(project_root: str | Path) -> list[dict[str, object]]:
    root = Path(project_root).resolve()
    manifest = root / "research/direct_action_planning_support_adaptation/FROZEN_INPUTS.sha256"
    rows: list[dict[str, object]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
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
        bad = [str(row["path"]) for row in rows if not row["matches"]]
        raise RuntimeError(f"frozen predecessor input changed: {bad}")
    return rows


def _repair_cell(root: Path, scenario: str, model_seed: int) -> Path:
    return root / "results/direct_action_planning_repair/minimal_v1/cells" / (
        f"{scenario}__s{model_seed}"
    )


def _load_d1_table(root: Path, scenario: str, model_seed: int) -> BudgetValueTable:
    path = _repair_cell(root, scenario, model_seed) / "learned_value.npz"
    payload = np.load(path, allow_pickle=False)
    return BudgetValueTable(payload["values"], source=f"frozen_D1:{scenario}:s{model_seed}")


def _pooled_d1_table(root: Path, model_seed: int, scenarios: list[str]) -> BudgetValueTable:
    values = np.mean(
        np.stack([_load_d1_table(root, scenario, model_seed).values for scenario in scenarios]),
        axis=0,
    )
    return BudgetValueTable(values, source=f"frozen_D1_pooled:s{model_seed}")


def _load_repaired_transition(root: Path, scenario: str, model_seed: int):
    path = _repair_cell(root, scenario, model_seed) / "models/aggregation_round_1.pt"
    return load_structured_checkpoint(path)


def _resolved(config: dict, run_id: str, smoke: bool) -> dict:
    value = json.loads(json.dumps(config))
    value["run_id"] = run_id
    value["smoke"] = bool(smoke)
    value["resolved_at"] = _now()
    if smoke:
        value["seed_split"] = {"train": [60], "validation": [65], "test": [70]}
        value["predecessor_model_seeds"] = [5]
        for key in ("train_interpolation",):
            value["scenario_split"][key] = value["scenario_split"][key][:2]
        for key in ("validation_interpolation", "test_interpolation", "boundary", "extrapolation"):
            value["scenario_split"][key] = value["scenario_split"][key][:1]
        value["collection"].update(
            {
                "train_episodes_per_scenario": 4,
                "validation_episodes_per_scenario": 2,
                "calibration_episodes": 8,
                "evaluation_episodes": 3,
            }
        )
        value["training"].update({"max_epochs": 20, "patience": 5, "rank_states_per_scenario": 12})
        value["adaptation"].update({"max_epochs": 20, "patience": 5})
    return value


def _model_payload(model: PooledValueNetwork, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "horizon": model.horizon,
        "n_loads": model.n_loads,
        "max_queue": model.max_queue,
        "max_budget": model.max_budget,
        "hidden_dim": model.encoder[0].out_features,
        "validation_metrics": metrics,
        "scenario_input": False,
        "history_input": False,
    }


def _train_models(
    root: Path,
    run_dir: Path,
    config: dict,
    continuous_mdps: dict[str, ActionConditionedBudgetMDP],
    optima: dict[str, ActionDPResult],
) -> tuple[
    dict[int, dict[str, PooledValueNetwork]],
    dict[int, pd.DataFrame],
    pd.DataFrame,
]:
    train_rows = config["scenario_split"]["train_interpolation"]
    validation_rows = config["scenario_split"]["validation_interpolation"]
    model_seeds = list(map(int, config["predecessor_model_seeds"]))
    train_seeds = list(map(int, config["seed_split"]["train"]))
    validation_seeds = list(map(int, config["seed_split"]["validation"]))
    collection = config["collection"]
    training_cfg = config["training"]
    trained_models: dict[int, dict[str, PooledValueNetwork]] = {}
    support_frames: dict[int, pd.DataFrame] = {}
    history_rows: list[pd.DataFrame] = []
    for slot, model_seed in enumerate(model_seeds):
        train_seed = train_seeds[slot]
        validation_seed = validation_seeds[slot]
        train_parts, validation_parts = [], []
        train_rank, validation_rank = {}, {}
        for row in train_rows:
            scenario_id = str(row["id"])
            mdp, optimum = continuous_mdps[scenario_id], optima[scenario_id]
            collected = collect_value_trajectories(
                mdp,
                optimum,
                scenario_id=scenario_id,
                region="train_interpolation",
                budgets=list(map(int, config["budgets"])),
                seed=train_seed,
                episodes=int(collection["train_episodes_per_scenario"]),
            )
            train_parts.append(collected)
            state_grid = full_state_value_frame(mdp, optimum, scenario_id, region="train_interpolation")
            train_rank[scenario_id] = (
                mdp,
                sample_value_states(
                    state_grid,
                    count=int(training_cfg["rank_states_per_scenario"]),
                    seed=train_seed + len(train_rank) * 101,
                ),
            )
        for row in validation_rows:
            scenario_id = str(row["id"])
            mdp, optimum = continuous_mdps[scenario_id], optima[scenario_id]
            collected = collect_value_trajectories(
                mdp,
                optimum,
                scenario_id=scenario_id,
                region="validation_interpolation",
                budgets=list(map(int, config["budgets"])),
                seed=validation_seed,
                episodes=int(collection["validation_episodes_per_scenario"]),
            )
            validation_parts.append(collected)
            state_grid = full_state_value_frame(
                mdp, optimum, scenario_id, region="validation_interpolation"
            )
            validation_rank[scenario_id] = (
                mdp,
                sample_value_states(
                    state_grid,
                    count=int(training_cfg["rank_states_per_scenario"]),
                    seed=validation_seed + len(validation_rank) * 103,
                ),
            )
        train = pd.concat(train_parts, ignore_index=True)
        validation = pd.concat(validation_parts, ignore_index=True)
        anchors = sample_value_states(train, count=min(768, len(train)), seed=train_seed + 911)
        base = PooledValueNetwork(
            horizon=int(config["environment"]["horizon"]),
            n_loads=3,
            max_queue=int(config["environment"]["max_queue"]),
            max_budget=int(config["environment"]["max_budget"]),
            hidden_dim=int(training_cfg["hidden_dim"]),
        )
        train_config = PooledTrainingConfig(
            learning_rate=float(training_cfg["learning_rate"]),
            max_epochs=int(training_cfg["max_epochs"]),
            patience=int(training_cfg["patience"]),
            seed=train_seed,
        )
        pooled_weights = ValueLossWeights(**training_cfg["losses"]["pooled"])
        joint_weights = ValueLossWeights(**training_cfg["losses"]["joint"])
        pooled = train_pooled_value(
            train,
            validation,
            rank_frames={},
            validation_rank_frames=validation_rank,
            anchor_states=anchors,
            model=base,
            weights=pooled_weights,
            config=train_config,
        )
        joint = train_pooled_value(
            train,
            validation,
            rank_frames=train_rank,
            validation_rank_frames=validation_rank,
            anchor_states=anchors,
            model=base,
            weights=joint_weights,
            config=train_config,
        )
        trained_models[model_seed] = {"pooled": pooled.model, "joint": joint.model}
        support_frames[model_seed] = train
        torch.save(
            _model_payload(pooled.model, pooled.validation_metrics),
            run_dir / "models" / f"pooled_current_state__s{model_seed}.pt",
        )
        torch.save(
            _model_payload(joint.model, joint.validation_metrics),
            run_dir / "models" / f"continuous_joint_refresh__s{model_seed}.pt",
        )
        for family, trained in (("pooled", pooled), ("joint", joint)):
            current = trained.history.copy()
            current["family"] = family
            current["model_seed"] = model_seed
            current["train_seed"] = train_seed
            current["validation_seed"] = validation_seed
            history_rows.append(current)
    history = pd.concat(history_rows, ignore_index=True)
    history.to_csv(run_dir / "training/value_training_history.csv.gz", index=False, compression="gzip")
    return trained_models, support_frames, history


def _agent_for_value(
    mdp: ActionConditionedBudgetMDP,
    table: BudgetValueTable,
    transition,
):
    return (
        DirectPlanningAgent(mdp, table)
        if transition is None
        else StructuredPlanningAgent(mdp, table, transition)
    )


def _evaluate_zero_shot(
    root: Path,
    config: dict,
    trained_models: dict[int, dict[str, PooledValueNetwork]],
    continuous_mdps: dict[str, ActionConditionedBudgetMDP],
    optima: dict[str, ActionDPResult],
    scenario_regions: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = list(map(str, config["original_loso"]))
    model_seeds = list(map(int, config["predecessor_model_seeds"]))
    test_seeds = list(map(int, config["seed_split"]["test"]))
    episodes = int(config["collection"]["evaluation_episodes"])
    episode_frames, step_frames, state_frames, state_summaries = [], [], [], []
    test_ids = [
        str(row["id"])
        for region in ("test_interpolation", "boundary", "extrapolation")
        for row in config["scenario_split"][region]
    ] + original
    for slot, model_seed in enumerate(model_seeds):
        test_seed = test_seeds[slot]
        pooled_d1 = _pooled_d1_table(root, model_seed, original)
        for scenario_id in test_ids:
            is_original = scenario_id in original
            if is_original:
                mdp = _original_mdp(config, scenario_id)
                optimum = solve_action_dp(mdp)
                transition = _load_repaired_transition(root, scenario_id, model_seed)
                d1 = _load_d1_table(root, scenario_id, model_seed)
                region = "original_loso"
            else:
                mdp, optimum = continuous_mdps[scenario_id], optima[scenario_id]
                transition = None
                d1 = pooled_d1
                region = scenario_regions[scenario_id]
            pooled_table = dense_value_table(
                trained_models[model_seed]["pooled"], mdp, source="pooled_current_state"
            )
            joint_table = dense_value_table(
                trained_models[model_seed]["joint"], mdp, source="continuous_joint_refresh"
            )
            oracle = BudgetValueTable.from_exact_dp(optimum.values)
            agents = {
                "exact_dp": ExactDPAgent(mdp, optimum),
                "frozen_D1": _agent_for_value(mdp, d1, transition),
                "domain_value_refresh": _agent_for_value(mdp, oracle, transition),
                "pooled_current_state": _agent_for_value(mdp, pooled_table, transition),
                "continuous_joint_refresh": _agent_for_value(mdp, joint_table, transition),
            }
            for method, agent in agents.items():
                states, summary = evaluate_full_state(
                    agent,
                    method,
                    mdp,
                    optimum,
                    scenario_id=scenario_id,
                    region=region,
                    seed=test_seed,
                )
                episodes_frame, steps, runtime = evaluate_rollouts(
                    agent,
                    method,
                    mdp,
                    optimum,
                    budgets=list(map(int, config["budgets"])),
                    scenario_id=scenario_id,
                    seed=test_seed,
                    episodes=episodes,
                )
                for frame in (states, episodes_frame, steps):
                    frame["model_seed"] = model_seed
                    frame["test_seed"] = test_seed
                    frame["region"] = region
                summary.update(
                    {
                        "model_seed": model_seed,
                        "test_seed": test_seed,
                        "rollout_Q_star_regret": float(episodes_frame.mean_Q_star_regret.mean()),
                        "rollout_action_consistency": float(
                            episodes_frame.action_consistency_rate.mean()
                        ),
                        "return_gap": float(episodes_frame.return_gap_to_paired_optimal.mean()),
                        "budget_trajectory_mae": float(
                            episodes_frame.budget_trajectory_mae.mean()
                        ),
                        "completion_rate": float(episodes_frame.completion_rate.mean()),
                        "slo_violation_rate": float(episodes_frame.slo_violation_rate.mean()),
                        "total_cost": float(episodes_frame.total_cost.mean()),
                        **runtime,
                    }
                )
                state_frames.append(states)
                episode_frames.append(episodes_frame)
                step_frames.append(steps)
                state_summaries.append(summary)
    return (
        pd.concat(episode_frames, ignore_index=True),
        pd.concat(step_frames, ignore_index=True),
        pd.concat(state_frames, ignore_index=True),
        pd.DataFrame(state_summaries),
    )


def _support_diagnosis(
    config: dict,
    trained_models: dict[int, dict[str, PooledValueNetwork]],
    training_support: dict[int, pd.DataFrame],
    zero_states: pd.DataFrame,
    continuous_mdps: dict[str, ActionConditionedBudgetMDP],
    optima: dict[str, ActionDPResult],
) -> tuple[pd.DataFrame, dict[str, object]]:
    original = list(map(str, config["original_loso"]))
    rows = []
    for model_seed, model_group in trained_models.items():
        train_parts = []
        for scenario_id, group in training_support[model_seed].groupby("scenario_id"):
            mdp, optimum = continuous_mdps[str(scenario_id)], optima[str(scenario_id)]
            sampled = group.sample(
                n=min(180, len(group)), random_state=model_seed + len(train_parts) * 19
            )
            train_parts.append(build_support_features(sampled, mdp, optimum, model_group["joint"]))
        train_features = pd.concat(train_parts, ignore_index=True)
        selected = zero_states[
            (zero_states.model_seed == model_seed)
            & (zero_states.method == "continuous_joint_refresh")
        ]
        for scenario_id, group in selected.groupby("scenario_id"):
            if scenario_id in original:
                mdp = _original_mdp(config, str(scenario_id))
                optimum = solve_action_dp(mdp)
            else:
                mdp, optimum = continuous_mdps[str(scenario_id)], optima[str(scenario_id)]
            sampled = group.sample(
                n=min(720, len(group)), random_state=model_seed + len(rows) * 23
            )
            test_features = build_support_features(sampled, mdp, optimum, model_group["joint"])
            distances = compute_support_distances(train_features, test_features)
            distances["model_seed"] = model_seed
            rows.append(distances)
    frame = pd.concat(rows, ignore_index=True)
    relation = support_relationship(
        frame,
        spearman_floor=float(config["gate"]["support_spearman_floor"]),
        regret_share_floor=float(config["gate"]["low_support_regret_share_floor"]),
    )
    region_means = (
        frame.groupby("region")[["raw", "structured", "hidden", "q_vector"]]
        .mean()
        .to_dict(orient="index")
    )
    deployable_clear_spaces = {
        str(row["space"])
        for row in relation["spaces"]
        if row["space"] != "q_vector" and bool(row["clear"])
    }
    distance_order = any(
        float(region_means.get("test_interpolation", {}).get(space, np.inf))
        < float(region_means.get("extrapolation", {}).get(space, -np.inf))
        for space in deployable_clear_spaces
    )
    relation["region_mean_distances"] = region_means
    relation["interpolation_closer_than_extrapolation"] = distance_order
    relation["deployable_support_clear"] = bool(
        relation["deployable_support_clear"] and distance_order
    )
    return frame, relation


def _evaluate_adaptation(
    root: Path,
    run_dir: Path,
    config: dict,
    trained_models: dict[int, dict[str, PooledValueNetwork]],
    training_support: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    original = list(map(str, config["original_loso"]))
    model_seeds = list(map(int, config["predecessor_model_seeds"]))
    test_seeds = list(map(int, config["seed_split"]["test"]))
    ratios = list(map(float, config["adaptation"]["ratios"]))
    variants = ["regression", "regression_anchor", "regression_rank", "full"]
    episodes = int(config["collection"]["evaluation_episodes"])
    calibration_episodes = int(config["collection"]["calibration_episodes"])
    prefix_fraction = float(config["adaptation"]["prefix_fraction"])
    metric_start = int(np.ceil(int(config["environment"]["horizon"]) * prefix_fraction))
    metric_frames, step_frames, history_frames, calibration_rows = [], [], [], []
    audits: list[dict[str, object]] = []
    for slot, model_seed in enumerate(model_seeds):
        test_seed = test_seeds[slot]
        base_model = trained_models[model_seed]["joint"]
        anchors = sample_value_states(
            training_support[model_seed],
            count=min(512, len(training_support[model_seed])),
            seed=test_seed + 404,
        )
        for scenario in original:
            mdp = _original_mdp(config, scenario)
            optimum = solve_action_dp(mdp)
            transition = _load_repaired_transition(root, scenario, model_seed)
            calibration_pool = collect_value_trajectories(
                mdp,
                optimum,
                scenario_id=scenario,
                region="original_loso_calibration",
                budgets=list(map(int, config["budgets"])),
                seed=test_seed + 2_000,
                episodes=calibration_episodes,
            )
            base_table = dense_value_table(base_model, mdp, source="no_calibration")
            oracle_table = BudgetValueTable.from_exact_dp(optimum.values)
            baseline_agents = {
                "no_calibration": _agent_for_value(mdp, base_table, transition),
                "oracle_target_value": _agent_for_value(mdp, oracle_table, transition),
            }
            for method, agent in baseline_agents.items():
                episode_frame, steps, _ = evaluate_rollouts(
                    agent,
                    method,
                    mdp,
                    optimum,
                    budgets=list(map(int, config["budgets"])),
                    scenario_id=scenario,
                    seed=test_seed,
                    episodes=episodes,
                    metric_start_t=metric_start,
                )
                for frame in (episode_frame, steps):
                    frame["model_seed"] = model_seed
                    frame["test_seed"] = test_seed
                    frame["ratio"] = 0.0 if method == "no_calibration" else 1.0
                    frame["variant"] = method
                metric_frames.append(episode_frame)
                step_frames.append(steps)
            for ratio in ratios:
                calibration, _, audit = split_target_prefix_suffix(
                    calibration_pool,
                    ratio=ratio,
                    horizon=mdp.config.horizon,
                    prefix_fraction=prefix_fraction,
                    seed=test_seed + 3_001,
                )
                audit.update(
                    {"scenario_id": scenario, "model_seed": model_seed, "test_seed": test_seed}
                )
                audits.append(audit)
                calibration_rows.append(
                    {
                        "scenario_id": scenario,
                        "model_seed": model_seed,
                        "test_seed": test_seed,
                        "ratio": ratio,
                        "samples": len(calibration),
                        "realized_ratio": audit["realized_ratio"],
                    }
                )
                if ratio == 0.0:
                    continue
                rank_states = calibration.drop_duplicates(
                    ["t", "load", "queue", "remaining_budget"]
                ).reset_index(drop=True)
                for variant in variants:
                    started = time.perf_counter()
                    trained = train_value_adapter(
                        base_model,
                        calibration=calibration,
                        anchors=anchors,
                        rank_frame=(mdp, rank_states),
                        variant=variant,
                        config=AdapterTrainingConfig(
                            learning_rate=float(config["adaptation"]["learning_rate"]),
                            max_epochs=int(config["adaptation"]["max_epochs"]),
                            patience=int(config["adaptation"]["patience"]),
                            anchor_weight=float(config["adaptation"]["anchor_weight"]),
                            rank_weight=float(config["adaptation"]["rank_weight"]),
                            seed=test_seed,
                        ),
                    )
                    elapsed = time.perf_counter() - started
                    method = f"adapt_{variant}_{int(round(ratio * 100)):02d}pct"
                    table = dense_value_table(trained.model, mdp, source=method)
                    agent = _agent_for_value(mdp, table, transition)
                    episode_frame, steps, _ = evaluate_rollouts(
                        agent,
                        method,
                        mdp,
                        optimum,
                        budgets=list(map(int, config["budgets"])),
                        scenario_id=scenario,
                        seed=test_seed,
                        episodes=episodes,
                        metric_start_t=metric_start,
                    )
                    anchor_x = torch.as_tensor(
                        np.stack(
                            [
                                anchors.remaining_horizon.to_numpy(float) / mdp.config.horizon,
                                anchors.load.to_numpy(float) / max(mdp.n_loads - 1, 1),
                                anchors.queue.to_numpy(float) / mdp.config.max_queue,
                                anchors.remaining_budget.to_numpy(float) / mdp.config.max_budget,
                            ],
                            axis=1,
                        ),
                        dtype=torch.float64,
                    )
                    with torch.no_grad():
                        anchor_delta = float(
                            torch.mean(torch.abs(trained.model(anchor_x) - trained.model.base(anchor_x)))
                        )
                    for frame in (episode_frame, steps):
                        frame["model_seed"] = model_seed
                        frame["test_seed"] = test_seed
                        frame["ratio"] = ratio
                        frame["variant"] = variant
                        frame["calibration_samples"] = len(calibration)
                        frame["calibration_seconds"] = elapsed
                        frame["anchor_mae_increase"] = anchor_delta
                    metric_frames.append(episode_frame)
                    step_frames.append(steps)
                    history = trained.history.copy()
                    history["method"] = method
                    history["scenario_id"] = scenario
                    history["model_seed"] = model_seed
                    history["test_seed"] = test_seed
                    history["ratio"] = ratio
                    history["calibration_samples"] = len(calibration)
                    history["calibration_seconds"] = elapsed
                    history["anchor_mae_increase"] = anchor_delta
                    history_frames.append(history)
                    torch.save(
                        {
                            "state_dict": trained.model.adapter.state_dict(),
                            "variant": variant,
                            "ratio": ratio,
                            "base_model_seed": model_seed,
                            "base_frozen": True,
                            "trainable_parameters": sum(
                                parameter.numel() for parameter in trained.model.adapter.parameters()
                            ),
                        },
                        run_dir / "models" / f"adapter__{scenario}__s{model_seed}__{variant}__{int(ratio*100):02d}.pt",
                    )
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(step_frames, ignore_index=True),
        pd.concat(history_frames, ignore_index=True),
        pd.DataFrame(calibration_rows),
        audits,
    )


def _gate_summary(
    config: dict,
    zero_metrics: pd.DataFrame,
    adaptation_metrics: pd.DataFrame,
    support_relation: dict[str, object],
    audits: list[dict[str, object]],
) -> dict[str, object]:
    zero_cells = (
        zero_metrics.groupby(["region", "scenario_id", "model_seed", "method"], as_index=False)
        .mean(numeric_only=True)
    )
    interpolation = zero_cells[zero_cells.region == "test_interpolation"]
    pivot = interpolation.pivot_table(
        index=["scenario_id", "model_seed"], columns="method", values="mean_Q_star_regret"
    )
    interpolation_win_fraction = float(
        (pivot["continuous_joint_refresh"] < pivot["frozen_D1"] - 1.0e-12).mean()
    )
    joint_region = zero_cells[zero_cells.method == "continuous_joint_refresh"]
    interpolation_regret = float(
        joint_region[joint_region.region == "test_interpolation"].mean_Q_star_regret.mean()
    )
    extrapolation_regret = float(
        joint_region[joint_region.region == "extrapolation"].mean_Q_star_regret.mean()
    )
    domain_regret = float(
        zero_cells[
            (zero_cells.region == "test_interpolation")
            & (zero_cells.method == "domain_value_refresh")
        ].mean_Q_star_regret.mean()
    )
    in_domain_increase = interpolation_regret - domain_regret

    adapted_cells = (
        adaptation_metrics.groupby(["scenario_id", "model_seed", "method"], as_index=False)
        .mean(numeric_only=True)
    )
    wide = adapted_cells.pivot_table(
        index=["scenario_id", "model_seed"], columns="method", values="mean_Q_star_regret"
    )
    recoveries: dict[str, float] = {}
    for ratio in (5, 10):
        method = f"adapt_full_{ratio:02d}pct"
        recoveries[str(ratio)] = float(
            np.mean(
                [
                    oracle_recovery(base, adapted, oracle)
                    for base, adapted, oracle in zip(
                        wide["no_calibration"], wide[method], wide["oracle_target_value"]
                    )
                ]
            )
        )
    final_method = "adapt_full_10pct"
    scenario_improvement = {
        scenario: bool(
            wide.loc[scenario][final_method].mean()
            < wide.loc[scenario]["no_calibration"].mean() - 1.0e-12
        )
        for scenario in ("early_burst", "late_burst", "periodic")
    }
    cell_wins = int((wide[final_method] < wide["no_calibration"] - 1.0e-12).sum())
    final_rows = adaptation_metrics[adaptation_metrics.method == final_method]
    base_rows = adaptation_metrics[adaptation_metrics.method == "no_calibration"]
    final_mean = final_rows.mean(numeric_only=True)
    base_mean = base_rows.mean(numeric_only=True)
    completion_delta = float(final_mean.completion_rate - base_mean.completion_rate)
    slo_delta = float(final_mean.slo_violation_rate - base_mean.slo_violation_rate)
    cost_delta_fraction = float(
        (final_mean.total_cost - base_mean.total_cost) / max(abs(base_mean.total_cost), 1.0)
    )
    gate_cfg = config["gate"]
    service_guard = bool(
        completion_delta >= -float(gate_cfg["material_completion_loss"])
        and slo_delta <= float(gate_cfg["material_slo_increase"])
        and cost_delta_fraction <= float(gate_cfg["material_cost_increase_fraction"])
    )
    return {
        "interpolation_win_fraction": interpolation_win_fraction,
        "interpolation_regret": interpolation_regret,
        "extrapolation_regret": extrapolation_regret,
        "deployable_support_clear": bool(support_relation["deployable_support_clear"]),
        "recovery_5pct": recoveries["5"],
        "recovery_10pct": recoveries["10"],
        "original_scenario_improvement": scenario_improvement,
        "original_cell_wins": cell_wins,
        "anchor_mae_increase": float(final_rows.anchor_mae_increase.mean()),
        "in_domain_regret_increase": in_domain_increase,
        "closed_loop_return_improved": bool(
            final_mean.return_gap_to_paired_optimal
            < base_mean.return_gap_to_paired_optimal - 1.0e-12
        ),
        "service_cost_guardrail": service_guard,
        "leakage_passed": bool(audits and all(bool(row["passed"]) for row in audits)),
        "independent_test_better": bool(
            final_mean.mean_Q_star_regret < base_mean.mean_Q_star_regret - 1.0e-12
        ),
        "service_cost": {
            "completion_delta": completion_delta,
            "slo_delta": slo_delta,
            "cost_delta_fraction": cost_delta_fraction,
            "return_gap_delta": float(
                final_mean.return_gap_to_paired_optimal
                - base_mean.return_gap_to_paired_optimal
            ),
        },
    }


def run_support_adaptation(
    project_root: str | Path,
    config_path: str | Path,
    *,
    run_id: str,
    smoke: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    config = _resolved(load_config(Path(config_path).resolve()), run_id, smoke)
    run_dir = root / "results/direct_action_planning_support_adaptation" / run_id
    if run_dir.exists():
        manifest = run_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return run_dir
        raise RuntimeError(f"append-only support-adaptation run exists: {run_dir}")
    for subdir in ("models", "training", "analysis"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    started_at = _now()
    started = time.perf_counter()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment_record())
    write_json(run_dir / "stage_status.json", {"stage": "protocol_audit", "at": _now()})
    frozen = verify_frozen_inputs(root)
    pd.DataFrame(frozen).to_csv(run_dir / "frozen_input_verification.csv", index=False)
    parameter_audit = assert_parameter_splits_disjoint(config["scenario_split"])
    write_json(run_dir / "parameter_split_audit.json", parameter_audit)
    set_global_seed(0, torch_threads=int(config["torch_threads"]))

    scenario_regions: dict[str, str] = {}
    continuous_mdps: dict[str, ActionConditionedBudgetMDP] = {}
    for region, rows in config["scenario_split"].items():
        for row in rows:
            scenario_id = str(row["id"])
            scenario_regions[scenario_id] = region
            continuous_mdps[scenario_id] = _continuous_mdp(config, row)
    optima = {scenario_id: solve_action_dp(mdp) for scenario_id, mdp in continuous_mdps.items()}
    if any(result.max_bellman_residual > 1.0e-10 for result in optima.values()):
        raise RuntimeError("continuous exact DP Bellman residual is non-zero")

    write_json(run_dir / "stage_status.json", {"stage": "value_training", "at": _now()})
    models, training_support, _ = _train_models(root, run_dir, config, continuous_mdps, optima)
    selection = {
        "base_family": "continuous_joint_refresh",
        "adapter_family": "linear_full",
        "selection_source": "preregistered_plus_validation_only",
        "test_seeds": config["seed_split"]["test"],
    }
    ledger = FinalTestLedger.create(run_dir / "final_test_ledger.json", selection)
    ledger.mark_test_started()

    write_json(run_dir / "stage_status.json", {"stage": "one_shot_zero_shot_test", "at": _now()})
    zero_metrics, zero_steps, zero_states, zero_summary = _evaluate_zero_shot(
        root, config, models, continuous_mdps, optima, scenario_regions
    )
    zero_metrics.to_csv(run_dir / "zero_shot_metrics.csv.gz", index=False, compression="gzip")
    zero_steps.to_csv(run_dir / "zero_shot_steps.csv.gz", index=False, compression="gzip")
    zero_states.to_csv(run_dir / "zero_shot_state_actions.csv.gz", index=False, compression="gzip")
    zero_summary.to_csv(run_dir / "zero_shot_summary.csv", index=False)

    write_json(run_dir / "stage_status.json", {"stage": "support_diagnosis", "at": _now()})
    support_rows, support_relation = _support_diagnosis(
        config, models, training_support, zero_states, continuous_mdps, optima
    )
    support_rows.to_csv(run_dir / "support_distances.csv.gz", index=False, compression="gzip")
    write_json(run_dir / "support_relationship.json", support_relation)

    write_json(run_dir / "stage_status.json", {"stage": "target_prefix_adaptation", "at": _now()})
    adaptation_metrics, adaptation_steps, adaptation_history, calibration_counts, audits = (
        _evaluate_adaptation(root, run_dir, config, models, training_support)
    )
    adaptation_metrics.to_csv(run_dir / "adaptation_metrics.csv.gz", index=False, compression="gzip")
    adaptation_steps.to_csv(run_dir / "adaptation_steps.csv.gz", index=False, compression="gzip")
    adaptation_history.to_csv(
        run_dir / "training/adaptation_training_history.csv.gz", index=False, compression="gzip"
    )
    calibration_counts.to_csv(run_dir / "calibration_counts.csv", index=False)
    write_json(
        run_dir / "prefix_suffix_leakage_audit.json",
        {
            "schema": "direct_action_planning_support_adaptation.leakage_collection.v1",
            "passed": all(bool(row["passed"]) for row in audits),
            "audits": audits,
        },
    )
    ledger.mark_test_completed()

    summary = _gate_summary(config, zero_metrics, adaptation_metrics, support_relation, audits)
    gate = assess_support_adaptation_gate(summary)
    write_json(run_dir / "continuation_gate.json", gate)
    elapsed = time.perf_counter() - started
    write_json(
        run_dir / "manifest.json",
        {
            "schema": "direct_action_planning_support_adaptation.minimal_run.v1",
            "status": "completed",
            "scientific_status": gate["decision"],
            "started_at": started_at,
            "completed_at": _now(),
            "elapsed_seconds": elapsed,
            "frozen_inputs_verified": len(frozen),
            "final_test_evaluations": 1,
            "test_used_for_selection": False,
            "full_expansion_authorized": gate["decision"] == "CONTINUE",
        },
    )
    write_json(
        run_dir / "stage_status.json",
        {"stage": "completed", "at": _now(), "decision": gate["decision"]},
    )
    return run_dir
