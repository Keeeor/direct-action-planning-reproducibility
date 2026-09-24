from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable
from stage2_dynamic_budget.direct_action_planning_repair.model import (
    StructuredActionEffectModel,
    planning_predictions,
    selection_metrics,
)


STATE_KEYS = ["t", "load", "queue"]
ACTION_KEYS = [*STATE_KEYS, "action"]
PLANNING_KEYS = [*STATE_KEYS, "remaining_budget", "action"]


@dataclass(frozen=True)
class DiagnosisOutputs:
    rounds: pd.DataFrame
    distributions: pd.DataFrame
    anchor_errors: pd.DataFrame
    mechanisms: pd.DataFrame


def load_structured_checkpoint(path: str | Path) -> StructuredActionEffectModel:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = StructuredActionEffectModel(
        int(payload["horizon"]),
        int(payload["n_loads"]),
        int(payload["n_actions"]),
        int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def empirical_anchor_labels(
    base_labels: pd.DataFrame, samples: pd.DataFrame
) -> pd.DataFrame:
    """Create a fixed evaluation anchor from rollout branches without changing Q labels."""

    counts = samples.groupby(ACTION_KEYS + ["true_next_load"], as_index=False).size()
    totals = counts.groupby(ACTION_KEYS, as_index=False)["size"].sum().rename(
        columns={"size": "anchor_sample_count"}
    )
    probabilities = counts.pivot_table(
        index=ACTION_KEYS,
        columns="true_next_load",
        values="size",
        fill_value=0,
    ).reset_index()
    for next_load in (0, 1, 2):
        if next_load not in probabilities:
            probabilities[next_load] = 0
    probabilities = probabilities.merge(totals, on=ACTION_KEYS, validate="one_to_one")
    for next_load in (0, 1, 2):
        probabilities[f"anchor_prob_{next_load}"] = (
            probabilities[next_load] / probabilities.anchor_sample_count
        )
    anchor = base_labels[base_labels.split == "validation"].drop(
        columns=["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]
    )
    anchor = anchor.merge(
        probabilities[
            ACTION_KEYS
            + ["anchor_prob_0", "anchor_prob_1", "anchor_prob_2", "anchor_sample_count"]
        ],
        on=ACTION_KEYS,
        how="inner",
        validate="many_to_one",
    ).rename(
        columns={
            "anchor_prob_0": "next_load_prob_0",
            "anchor_prob_1": "next_load_prob_1",
            "anchor_prob_2": "next_load_prob_2",
        }
    )
    anchor["split"] = "validation"
    return anchor


def duplicate_and_coverage(samples: pd.DataFrame, d0_states: pd.DataFrame) -> dict[str, float]:
    state_columns = [*STATE_KEYS]
    if "remaining_budget" in samples:
        state_columns.append("remaining_budget")
    transition_columns = [*state_columns, "action", "true_next_load"]
    unique_states = samples[state_columns].drop_duplicates()
    grid_states = samples[STATE_KEYS].drop_duplicates()
    d0_grid = d0_states[STATE_KEYS].drop_duplicates()
    coverage = len(grid_states.merge(d0_grid, on=STATE_KEYS, how="inner")) / max(len(d0_grid), 1)
    state_counts = samples.groupby(state_columns, dropna=False).size().to_numpy(dtype=float)
    probability = state_counts / max(state_counts.sum(), 1.0)
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1.0e-15))))
    normalized_entropy = entropy / max(np.log(len(d0_grid)), 1.0)
    return {
        "rows": float(len(samples)),
        "unique_states": float(len(unique_states)),
        "grid_coverage": float(coverage),
        "transition_duplicate_rate": float(
            1.0 - len(samples[transition_columns].drop_duplicates()) / max(len(samples), 1)
        ),
        "state_revisit_rate": float(1.0 - len(unique_states) / max(len(samples), 1)),
        "normalized_state_entropy": float(normalized_entropy),
    }


def _distribution_rows(
    samples: pd.DataFrame, scenario: str, seed: int, round_index: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    columns = ["load", "queue", "action"]
    if "remaining_budget" in samples:
        columns.append("remaining_budget")
    if "remaining_horizon" in samples:
        columns.append("remaining_horizon")
    else:
        samples = samples.assign(remaining_horizon=16 - samples.t)
        columns.append("remaining_horizon")
    for column in columns:
        counts = samples[column].value_counts(dropna=False, normalize=True).sort_index()
        for value, fraction in counts.items():
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "round": round_index,
                    "dimension": column,
                    "value": value,
                    "fraction": float(fraction),
                }
            )
    return rows


@torch.no_grad()
def _anchor_metrics(
    model: StructuredActionEffectModel,
    labels: pd.DataFrame,
    mdp: ActionConditionedBudgetMDP,
    anchor: str,
    model_round: int,
) -> dict[str, object]:
    metrics = selection_metrics(model, labels, mdp.config.gamma, torch.device("cpu"))
    predicted_q = planning_predictions(
        model, labels, mdp.config.gamma, torch.device("cpu")
    ).numpy()
    target_probability = labels[
        ["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]
    ].to_numpy()
    predicted_probability = model(
        torch.as_tensor(labels.t.to_numpy(), dtype=torch.long),
        torch.as_tensor(labels.load.to_numpy(), dtype=torch.long),
        torch.as_tensor(labels.action.to_numpy(), dtype=torch.long),
    ).numpy()
    return {
        "anchor": anchor,
        "model_round": model_round,
        **metrics,
        "state_probability_mae": float(np.mean(np.abs(predicted_probability - target_probability))),
        "planning_q_bias": float(np.mean(predicted_q - labels.q_lv.to_numpy())),
    }


def _true_transition_error(
    model: StructuredActionEffectModel, mdp: ActionConditionedBudgetMDP
) -> float:
    errors = []
    for t, load, action in np.ndindex(mdp.config.horizon, mdp.n_loads, mdp.n_actions):
        predicted = model.predict_probabilities(t, load, action)
        errors.append(0.5 * np.abs(predicted - mdp.load_probabilities(t, load)).sum())
    return float(np.mean(errors))


def _value_extrapolation_error(
    visits: pd.DataFrame,
    value: BudgetValueTable,
    optimum,
    horizon: int,
) -> float:
    errors = []
    for row in visits.itertuples(index=False):
        errors.append(
            abs(
                value.predict(
                    int(row.load),
                    int(row.queue),
                    int(row.remaining_budget),
                    horizon - int(row.t),
                )
                - float(
                    optimum.values[
                        int(row.t),
                        int(row.load),
                        int(row.queue),
                        int(row.remaining_budget),
                    ]
                )
            )
        )
    return float(np.mean(errors)) if errors else float("nan")


def diagnose_frozen_aggregation(
    old_run_dir: str | Path,
    config: ACBADPConfig,
) -> DiagnosisOutputs:
    old_run = Path(old_run_dir)
    round_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    for cell in sorted((old_run / "cells").iterdir()):
        if not cell.is_dir() or "__s" not in cell.name:
            continue
        scenario, seed_text = cell.name.split("__s", maxsplit=1)
        seed = int(seed_text)
        mdp = ActionConditionedBudgetMDP(
            ACBADPConfig(
                horizon=config.horizon,
                max_budget=config.max_budget,
                max_queue=config.max_queue,
                scenario=scenario,
                gamma=config.gamma,
                action_costs=config.action_costs,
                action_capacity=config.action_capacity,
            )
        )
        optimum = solve_action_dp(mdp)
        value_payload = np.load(cell / "learned_value.npz", allow_pickle=False)
        value = BudgetValueTable(value_payload["values"], str(value_payload["source"].item()))
        base_labels = pd.read_csv(cell / "D0_branch_labels.csv.gz")
        d0_samples = pd.read_csv(cell / "D0_branch_samples.csv.gz")
        anchors = {"D0_fixed": base_labels[base_labels.split == "validation"].copy()}
        d1_path = cell / "D1_branch_samples.csv.gz"
        if d1_path.exists():
            d1_samples = pd.read_csv(d1_path)
            anchors["D1_fixed"] = empirical_anchor_labels(base_labels, d1_samples)
        else:
            d1_samples = pd.DataFrame()
        model_paths = {0: cell / "models" / "structured_effect_q_rank.pt"}
        for round_index in (1, 2, 3):
            model_paths[round_index] = cell / "models" / f"aggregation_round_{round_index}.pt"
        for round_index in (0, 1, 2, 3):
            sample_path = (
                cell / "D0_branch_samples.csv.gz"
                if round_index == 0
                else cell / f"D{round_index}_branch_samples.csv.gz"
            )
            if not sample_path.exists():
                continue
            samples = d0_samples if round_index == 0 else pd.read_csv(sample_path)
            stats = duplicate_and_coverage(samples, d0_samples)
            distribution_rows.extend(_distribution_rows(samples, scenario, seed, round_index))
            visits_path = cell / f"D{round_index}_visits.csv"
            visits = pd.read_csv(visits_path) if round_index > 0 and visits_path.exists() else pd.DataFrame()
            threshold = float(base_labels.high_regret_threshold.dropna().iloc[0])
            model_path = model_paths[round_index]
            model = load_structured_checkpoint(model_path) if model_path.exists() else None
            row = {
                "scenario": scenario,
                "seed": seed,
                "round": round_index,
                **stats,
                "model_available": model is not None,
                "effective_training_label_share": 1.0 if round_index == 0 else 0.0,
                "high_regret_visit_fraction": (
                    float(np.mean(visits.q_star_regret > threshold)) if not visits.empty else np.nan
                ),
                "value_extrapolation_mae": (
                    _value_extrapolation_error(visits, value, optimum, config.horizon)
                    if not visits.empty
                    else np.nan
                ),
            }
            if round_index > 0:
                d0_per_cell = 48.0
                source_count = samples.groupby(ACTION_KEYS).size()
                row["mean_new_samples_per_visited_state_action"] = float(source_count.mean())
                row["mean_new_target_mass_fraction"] = float(
                    np.mean(source_count.to_numpy() / (source_count.to_numpy() + d0_per_cell))
                )
            if model is not None:
                row["true_transition_tv"] = _true_transition_error(model, mdp)
                for anchor_name, labels in anchors.items():
                    if not labels.empty:
                        anchor_rows.append(
                            {
                                "scenario": scenario,
                                "seed": seed,
                                **_anchor_metrics(model, labels, mdp, anchor_name, round_index),
                            }
                        )
            round_rows.append(row)

    rounds = pd.DataFrame(round_rows)
    anchors = pd.DataFrame(anchor_rows)
    mechanisms = infer_mechanisms(rounds, anchors)
    return DiagnosisOutputs(rounds, pd.DataFrame(distribution_rows), anchors, mechanisms)


def infer_mechanisms(rounds: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    later = rounds[rounds["round"] > 0]
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "mechanism": "later_rows_dominate_batches",
            "status": "unsupported",
            "evidence": "training retains the D0 label grid; rollout rows alter transition targets but are not appended as loss rows",
        }
    )
    coverage_drop = 1.0 - float(later.grid_coverage.mean()) if not later.empty else 0.0
    entropy = float(later.normalized_state_entropy.mean()) if not later.empty else np.nan
    rows.append(
        {
            "mechanism": "state_coverage_contraction",
            "status": "supported" if coverage_drop > 0.10 or entropy < 0.80 else "mixed",
            "evidence": f"mean grid coverage drop={coverage_drop:.4f}; normalized visitation entropy={entropy:.4f}",
        }
    )
    duplicate = float(later.transition_duplicate_rate.mean()) if not later.empty else np.nan
    rows.append(
        {
            "mechanism": "repeated_easy_samples",
            "status": "supported" if duplicate > 0.50 else "mixed",
            "evidence": f"mean transition-signature duplicate rate={duplicate:.4f}",
        }
    )
    forgetting = []
    for anchor in ("D0_fixed", "D1_fixed"):
        subset = anchors[anchors.anchor == anchor]
        if subset.empty:
            continue
        baseline_round = 0 if anchor == "D0_fixed" else 1
        baseline = subset[subset.model_round == baseline_round][
            ["scenario", "seed", "mean_exact_q_damage_vs_lv"]
        ].rename(columns={"mean_exact_q_damage_vs_lv": "baseline"})
        joined = subset.merge(baseline, on=["scenario", "seed"], how="inner")
        forgetting.extend(
            (joined.mean_exact_q_damage_vs_lv - joined.baseline).to_numpy().tolist()
        )
    mean_forgetting = float(np.mean(forgetting)) if forgetting else np.nan
    rows.append(
        {
            "mechanism": "catastrophic_forgetting",
            "status": "supported" if mean_forgetting > 0.005 else "mixed",
            "evidence": f"mean anchor exact-Q damage increase={mean_forgetting:.6f}",
        }
    )
    correlation = float(
        later[["value_extrapolation_mae", "high_regret_visit_fraction"]]
        .dropna()
        .corr()
        .iloc[0, 1]
    ) if len(later[["value_extrapolation_mae", "high_regret_visit_fraction"]].dropna()) > 2 else np.nan
    rows.append(
        {
            "mechanism": "value_transition_distribution_mismatch",
            "status": "supported" if np.isfinite(correlation) and correlation > 0.20 else "mixed",
            "evidence": f"value extrapolation/high-regret-visit correlation={correlation:.4f}",
        }
    )
    return pd.DataFrame(rows)
