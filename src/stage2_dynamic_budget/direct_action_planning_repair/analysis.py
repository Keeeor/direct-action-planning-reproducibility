from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = {
    "action_consistency_rate": "higher",
    "mean_Q_star_regret": "lower",
    "return_gap_to_paired_optimal": "lower",
    "budget_trajectory_mae": "lower",
    "completion_rate": "higher",
    "slo_violation_rate": "lower",
    "total_cost": "neutral",
}
BASELINES = ("original_learned_model", "learned_value_branch", "dsp_b")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000):
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_comparisons(episodes: pd.DataFrame) -> pd.DataFrame:
    units = (
        episodes.groupby(["method", "budget", "seed"], as_index=False)
        .mean(numeric_only=True)
    )
    candidate = units[units.method == "full_repair"].set_index(["budget", "seed"])
    rows = []
    comparison_index = 0
    for baseline in BASELINES:
        reference = units[units.method == baseline].set_index(["budget", "seed"])
        paired = candidate.join(reference, lsuffix="_repair", rsuffix="_baseline", how="inner")
        for metric, direction in METRICS.items():
            raw = (
                paired[f"{metric}_repair"] - paired[f"{metric}_baseline"]
            ).to_numpy(dtype=float)
            improvement = -raw if direction == "lower" else raw
            if direction == "neutral":
                improvement = raw
            lower, upper = _bootstrap_mean_ci(improvement, 8_300 + comparison_index)
            nonzero = improvement[np.abs(improvement) > 1.0e-12]
            if len(nonzero):
                wilcoxon = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
                statistic, p_value = float(wilcoxon.statistic), float(wilcoxon.pvalue)
            else:
                statistic, p_value = 0.0, 1.0
            std = float(np.std(improvement, ddof=1)) if len(improvement) > 1 else 0.0
            rows.append(
                {
                    "candidate": "full_repair",
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "paired_units": len(improvement),
                    "mean_improvement": float(np.mean(improvement)),
                    "bootstrap_95ci_lower": lower,
                    "bootstrap_95ci_upper": upper,
                    "paired_cohens_dz": float(np.mean(improvement) / std) if std else 0.0,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
            comparison_index += 1
    result = pd.DataFrame(rows)
    result["bh_q"] = _bh_adjust(result.wilcoxon_p.to_numpy())
    result["ci_excludes_zero"] = (
        (result.bootstrap_95ci_lower > 0) | (result.bootstrap_95ci_upper < 0)
    )
    return result


def method_summary(
    episodes: pd.DataFrame, state_summary: pd.DataFrame, runtime: pd.DataFrame
) -> pd.DataFrame:
    rollout = episodes.groupby("method", as_index=False).agg(
        rollout_action_agreement=("action_consistency_rate", "mean"),
        rollout_q_star_regret=("mean_Q_star_regret", "mean"),
        paired_return_gap=("return_gap_to_paired_optimal", "mean"),
        budget_trajectory_mae=("budget_trajectory_mae", "mean"),
        completion_rate=("completion_rate", "mean"),
        slo_violation_rate=("slo_violation_rate", "mean"),
        total_cost=("total_cost", "mean"),
    )
    state = state_summary.groupby("method", as_index=False).agg(
        full_state_action_agreement=("action_consistency_rate", "mean"),
        full_state_q_star_regret=("mean_Q_star_regret", "mean"),
        high_risk_low_cost_balanced_accuracy=(
            "high_risk_low_cost_balanced_accuracy",
            "mean",
        ),
    )
    timing = runtime.groupby("method", as_index=False).agg(
        planning_ms_mean=("decision_latency_ms_mean", "mean"),
        planning_ms_p95=("decision_latency_ms_p95", "mean"),
        fallback_rate=("fallback_rate", "mean"),
    )
    return rollout.merge(state, on="method").merge(timing, on="method")


def pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    dominated_by = []
    for row in result.itertuples():
        dominators = []
        for other in result.itertuples():
            if row.method == other.method:
                continue
            weak = (
                other.completion_rate >= row.completion_rate
                and other.slo_violation_rate <= row.slo_violation_rate
                and other.total_cost <= row.total_cost
            )
            strict = (
                other.completion_rate > row.completion_rate
                or other.slo_violation_rate < row.slo_violation_rate
                or other.total_cost < row.total_cost
            )
            if weak and strict:
                dominators.append(other.method)
        dominated_by.append("+".join(dominators))
    result["pareto_dominated"] = [bool(value) for value in dominated_by]
    result["dominated_by"] = dominated_by
    return result


def run_analysis(project_root: str | Path, run_id: str = "minimal_v1") -> Path:
    root = Path(project_root).resolve()
    source = root / "results/direct_action_planning_repair" / run_id
    output = source / "analysis_v1"
    if output.exists():
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return output
        raise RuntimeError(f"analysis output exists and is incomplete: {output}")
    output.mkdir()
    episodes = pd.read_csv(source / "metrics.csv")
    state = pd.read_csv(source / "state_policy_summary.csv")
    runtime = pd.read_csv(source / "runtime.csv")
    summary = method_summary(episodes, state, runtime)
    comparisons = paired_comparisons(episodes)
    pareto = pareto_table(summary)
    summary.to_csv(output / "method_summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    pareto.to_csv(output / "pareto_summary.csv", index=False)
    gate = json.loads((source / "continuation_gate.json").read_text())
    claims = [
        "# Claim-Evidence Table",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        (
            "| Structured repair closes most old model regret | Full-state exact-Q regret "
            f"falls {100 * gate['q_star_regret']['relative_reduction']:.2f}% | Supported in minimal DP |"
        ),
        (
            "| Repair stays close to Learned-Value ordering | Agreement loss is "
            f"{gate['agreement']['loss']:.5f} against a 0.03 ceiling | Supported in minimal DP |"
        ),
        (
            "| Closed-loop aggregation improves continuously | Round means are "
            + ", ".join(
                f"D{row['model_round']}={row['rollout_q_star_regret']:.5f}"
                for row in gate["aggregation_round_means"]
            )
            + " | Refuted; STOP trigger |"
        ),
        (
            "| Expansion/public-trace superiority | Not run after minimal STOP | Not tested |"
        ),
    ]
    (output / "claim_evidence_table.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    inputs = [source / "metrics.csv", source / "state_policy_summary.csv", source / "runtime.csv", source / "continuation_gate.json"]
    outputs = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema": "direct_action_planning_repair.analysis.v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source.relative_to(root)),
        "source_sha256": {str(path.relative_to(root)): _sha256(path) for path in inputs},
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "paired_unit": "budget_seed_after_scenario_and_episode_averaging",
        "paired_units": 15,
        "bootstrap_draws": 10_000,
        "multiple_testing": "single_Benjamini_Hochberg_family_over_21_comparisons",
        "small_sample_evidence_cap": "weak",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
