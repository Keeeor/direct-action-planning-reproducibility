from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dap.utils.artifacts import sha256_file, write_json

from .gate import assess_continuation


METRICS = {
    "action_consistency_rate": "higher",
    "mean_Q_star_regret": "lower",
    "return_gap_to_paired_optimal": "lower",
    "budget_trajectory_mae": "lower",
    "completion_rate": "higher",
    "slo_violation_rate": "lower",
    "total_cost": "context",
}


def _bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def paired_comparisons(episodes: pd.DataFrame) -> pd.DataFrame:
    cells = episodes.groupby(["method", "scenario", "budget", "seed"], as_index=False)[
        list(METRICS)
    ].mean()
    baseline = cells[cells.method == "b4_budget_state"].set_index(
        ["scenario", "budget", "seed"]
    )
    rows: list[dict[str, object]] = []
    candidates = [method for method in cells.method.unique() if method not in {"b4_budget_state", "optimal"}]
    for candidate_index, candidate in enumerate(candidates):
        candidate_frame = cells[cells.method == candidate].set_index(
            ["scenario", "budget", "seed"]
        )
        paired = candidate_frame.join(baseline, lsuffix="_candidate", rsuffix="_b4", how="inner")
        for metric_index, (metric, direction) in enumerate(METRICS.items()):
            difference = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_b4"]
            ).to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_ci(
                difference, seed=20260802 + candidate_index * 100 + metric_index
            )
            nonzero = difference[np.abs(difference) > 1e-12]
            p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "b4_budget_state",
                    "metric": metric,
                    "preferred_direction": direction,
                    "paired_cells": len(difference),
                    "candidate_mean": float(paired[f"{metric}_candidate"].mean()),
                    "baseline_mean": float(paired[f"{metric}_b4"].mean()),
                    "mean_difference_candidate_minus_b4": float(difference.mean()),
                    "bootstrap_95ci_lower": lower,
                    "bootstrap_95ci_upper": upper,
                    "wilcoxon_p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame["bh_q"] = _bh_adjust(frame.wilcoxon_p.tolist())
    return frame


def action_conditioning_diagnostics(truth: pd.DataFrame) -> dict[str, object]:
    feasible = truth[truth.feasible].copy()
    all_actions = feasible.action.nunique()
    complete = feasible.groupby(
        ["scenario", "t", "load", "queue", "remaining_budget"]
    ).filter(lambda frame: len(frame) == all_actions)
    pivot = complete.pivot(
        index=["scenario", "t", "load", "queue", "remaining_budget"],
        columns="action",
        values="A_star",
    )
    matrix = pivot.to_numpy(dtype=float)
    grand = matrix.mean()
    additive = matrix.mean(axis=1, keepdims=True) + matrix.mean(axis=0, keepdims=True) - grand
    total_energy = float(np.sum((matrix - grand) ** 2))
    interaction_energy = float(np.sum((matrix - additive) ** 2))
    rank_reversals = []
    for left in range(all_actions):
        for right in range(left + 1, all_actions):
            delta = matrix[:, right] - matrix[:, left]
            strict = delta[np.abs(delta) > 1e-10]
            rank_reversals.append(
                {
                    "left_action": left,
                    "right_action": right,
                    "strict_states": len(strict),
                    "right_preferred_proportion": float((strict > 0).mean()),
                    "left_preferred_proportion": float((strict < 0).mean()),
                    "both_orderings_observed": bool((strict > 0).any() and (strict < 0).any()),
                }
            )
    action_variability = (
        feasible.groupby("action")["A_star"].agg(["mean", "std", "min", "max"]).reset_index()
    )
    return {
        "schema": "acba.action_conditioning_diagnostics.v1",
        "complete_feasible_states": len(pivot),
        "state_action_interaction_energy_ratio": interaction_energy / max(total_energy, 1e-12),
        "action_advantage_variability": action_variability.to_dict(orient="records"),
        "pairwise_rank_reversals": rank_reversals,
        "interpretation_guardrail": "Interaction and rank reversal establish non-separability in this DP; they do not establish cross-domain superiority of ACBA.",
    }


def run_analysis(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir / "analysis_v1"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    episodes = pd.read_csv(run_dir / "metrics.csv")
    state_summaries = pd.read_csv(run_dir / "state_policy_summary.csv")
    truth = pd.read_csv(run_dir / "action_truth.csv.gz")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    method_summary = episodes.groupby("method")[list(METRICS)].agg(["mean", "std", "sem"])
    method_summary.columns = ["_".join(column) for column in method_summary.columns]
    method_summary.reset_index().to_csv(output_dir / "method_summary.csv", index=False)
    comparisons = paired_comparisons(episodes)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    write_json(
        output_dir / "action_conditioning_diagnostics.json",
        action_conditioning_diagnostics(truth),
    )

    gate_config = config["continuation_gate"]
    corrected_gate = assess_continuation(
        episodes,
        state_summaries,
        int(gate_config["paired_budget_seed_wins_required"]),
        int(gate_config["burst_seed_wins_required"]),
        float(gate_config["high_risk_balanced_accuracy_floor"]),
    )
    corrected_gate["correction"] = {
        "source_gate": "../continuation_gate.json",
        "change": "Strict service deterioration now uses Pareto domination across completion, SLO, and cost instead of an OR over service metrics.",
        "decision_changed": False,
    }
    write_json(output_dir / "continuation_gate_corrected.json", corrected_gate)
    artifacts = [
        output_dir / "method_summary.csv",
        output_dir / "paired_comparisons.csv",
        output_dir / "action_conditioning_diagnostics.json",
        output_dir / "continuation_gate_corrected.json",
    ]
    write_json(
        output_dir / "manifest.json",
        {
            "schema": "light.analysis_manifest.v1",
            "status": "completed",
            "source_run": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {path.name: sha256_file(path) for path in artifacts},
            "multiple_comparison_control": "Benjamini-Hochberg across all generated paired tests",
            "bootstrap_unit": "scenario-budget-seed cell",
            "bootstrap_samples": 10000,
        },
    )
    return output_dir

