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
BASELINES = (
    "frozen_best_D1",
    "learned_value",
    "original_learned_model",
    "controlled_full",
    "aggregate_only",
    "aggregate_anchor",
    "aggregate_anchor_mono",
    "aggregate_anchor_mono_rank",
    "full_value_refresh",
    "repaired_transition_oracle_value",
)


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
    units = episodes.groupby(
        ["method", "scenario", "train_seed"], as_index=False
    ).mean(numeric_only=True)
    candidate = units[units.method == "final_value_refresh"].set_index(
        ["scenario", "train_seed"]
    )
    rows = []
    comparison_index = 0
    for baseline in BASELINES:
        reference = units[units.method == baseline].set_index(["scenario", "train_seed"])
        paired = candidate.join(reference, lsuffix="_candidate", rsuffix="_baseline")
        for metric, direction in METRICS.items():
            raw = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            ).to_numpy(dtype=float)
            improvement = -raw if direction == "lower" else raw
            if direction == "neutral":
                improvement = raw
            lower, upper = _bootstrap_mean_ci(improvement, 9_700 + comparison_index)
            nonzero = improvement[np.abs(improvement) > 1.0e-12]
            if len(nonzero):
                test = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            else:
                statistic, p_value = 0.0, 1.0
            deviation = float(np.std(improvement, ddof=1)) if len(improvement) > 1 else 0.0
            rows.append(
                {
                    "candidate": "final_value_refresh",
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "paired_units": len(improvement),
                    "mean_improvement": float(np.mean(improvement)),
                    "bootstrap_95ci_lower": lower,
                    "bootstrap_95ci_upper": upper,
                    "paired_cohens_dz": float(np.mean(improvement) / deviation)
                    if deviation
                    else 0.0,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p": p_value,
                }
            )
            comparison_index += 1
    result = pd.DataFrame(rows)
    result["bh_q"] = _bh_adjust(result.wilcoxon_p.to_numpy())
    result["ci_excludes_zero"] = (
        (result.bootstrap_95ci_lower > 0.0) | (result.bootstrap_95ci_upper < 0.0)
    )
    return result


def method_summary(
    episodes: pd.DataFrame, states: pd.DataFrame, runtime: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    rollout = episodes.groupby("method", as_index=False).agg(
        rollout_action_agreement=("action_consistency_rate", "mean"),
        rollout_q_star_regret=("mean_Q_star_regret", "mean"),
        paired_return_gap=("return_gap_to_paired_optimal", "mean"),
        budget_trajectory_mae=("budget_trajectory_mae", "mean"),
        completion_rate=("completion_rate", "mean"),
        slo_violation_rate=("slo_violation_rate", "mean"),
        total_cost=("total_cost", "mean"),
        high_risk_low_cost_balanced_accuracy=(
            "high_risk_low_cost_balanced_accuracy",
            "mean",
        ),
    )
    full = states.groupby("method", as_index=False).agg(
        full_state_action_agreement=("action_consistency_rate", "mean"),
        full_state_q_star_regret=("mean_Q_star_regret", "mean"),
        full_state_high_risk_low_cost_balanced_accuracy=(
            "high_risk_low_cost_balanced_accuracy",
            "mean",
        ),
    )
    timing = runtime.groupby("method", as_index=False).agg(
        decision_ms_mean=("decision_latency_ms_mean", "mean"),
        decision_ms_p95=("decision_latency_ms_p95", "mean"),
    )
    ranking = pairs.groupby("method", as_index=False).agg(
        pair_ranking_accuracy=("pair_ranking_accuracy", "mean"),
        compared_pairs=("count", "sum"),
    )
    return rollout.merge(full, on="method").merge(timing, on="method").merge(
        ranking, on="method"
    )


def paired_service_cost_pareto(episodes: pd.DataFrame) -> pd.DataFrame:
    units = episodes.groupby(
        ["method", "scenario", "train_seed", "budget"], as_index=False
    ).mean(numeric_only=True)
    candidate = units[units.method == "final_value_refresh"]
    rows = []
    for baseline in BASELINES:
        paired = candidate.merge(
            units[units.method == baseline],
            on=["scenario", "train_seed", "budget"],
            suffixes=("_candidate", "_baseline"),
        )
        service = paired.discounted_return_candidate - paired.discounted_return_baseline
        slo = paired.slo_violation_rate_candidate - paired.slo_violation_rate_baseline
        cost = paired.total_cost_candidate - paired.total_cost_baseline
        candidate_dominates = (service >= 0) & (slo <= 0) & (cost <= 0) & (
            (service > 0) | (slo < 0) | (cost < 0)
        )
        baseline_dominates = (service <= 0) & (slo >= 0) & (cost >= 0) & (
            (service < 0) | (slo > 0) | (cost > 0)
        )
        equal = (service == 0) & (slo == 0) & (cost == 0)
        rows.append(
            {
                "candidate": "final_value_refresh",
                "baseline": baseline,
                "descriptive_budget_units": len(paired),
                "candidate_dominates": int(candidate_dominates.sum()),
                "baseline_dominates": int(baseline_dominates.sum()),
                "equal": int(equal.sum()),
                "tradeoff": int((~candidate_dominates & ~baseline_dominates & ~equal).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(project_root: str | Path, run_id: str = "minimal_v1") -> Path:
    root = Path(project_root).resolve()
    source = root / "results/direct_action_planning_value_refresh" / run_id
    output = source / "analysis_v1"
    if output.exists():
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return output
        raise RuntimeError(f"append-only analysis output already exists: {output}")
    output.mkdir()
    episodes = pd.read_csv(source / "test_metrics.csv")
    states = pd.read_csv(source / "test_state_policy_summary.csv")
    runtime = pd.read_csv(source / "runtime.csv")
    pairs = pd.read_csv(source / "test_pair_ranking.csv")
    loso = pd.read_csv(source / "loso_validation.csv")
    gate = json.loads((source / "continuation_gate.json").read_text())
    summary = method_summary(episodes, states, runtime, pairs)
    comparisons = paired_comparisons(episodes)
    pareto = paired_service_cost_pareto(episodes)
    loso_summary = loso.groupby("heldout_scenario", as_index=False).agg(
        frozen_D1_regret=("frozen_D1_regret", "mean"),
        loso_regret=("loso_regret", "mean"),
        regret_delta=("regret_delta", "mean"),
        material_reversal_rate=(
            "regret_delta",
            lambda values: float(np.mean(values > 0.0)),
        ),
    )
    summary.to_csv(output / "method_summary.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    pareto.to_csv(output / "paired_service_cost_pareto.csv", index=False)
    loso_summary.to_csv(output / "loso_summary.csv", index=False)
    claims = [
        "# Claim-Evidence Table",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        (
            "| Fixed value mismatch is causally repairable | Oracle replacement recovered "
            "99.48% static and 100% closed-loop regret | Supported in small DP |"
        ),
        (
            "| Value Refresh beats frozen D1 in-domain | Test regret "
            f"{gate['test_regret']['final_value_refresh']:.6f} versus "
            f"{gate['test_regret']['frozen_best_D1']:.6f}; 14/15 cells | Supported |"
        ),
        (
            "| Value Refresh generalizes across scenarios | 12/15 LOSO cells materially "
            "reversed versus frozen D1 | Refuted; STOP trigger |"
        ),
        "| Full synthetic/public-trace advantage | Not run after LOSO STOP | Not tested |",
    ]
    (output / "claim_evidence_table.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    inputs = [
        source / "test_metrics.csv",
        source / "test_state_policy_summary.csv",
        source / "runtime.csv",
        source / "test_pair_ranking.csv",
        source / "loso_validation.csv",
        source / "continuation_gate.json",
    ]
    outputs = sorted(path for path in output.iterdir() if path.is_file())
    payload = {
        "schema": "direct_action_planning_value_refresh.analysis.v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source.relative_to(root)),
        "source_sha256": {str(path.relative_to(root)): _sha256(path) for path in inputs},
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "paired_unit": "scenario_train_seed_after_budget_and_episode_averaging",
        "paired_units": 15,
        "bootstrap_draws": 10_000,
        "multiple_testing": "single_Benjamini_Hochberg_family_over_70_planned_comparisons",
        "pareto_units": "45 descriptive scenario_seed_budget cells; not inferential replicates",
        "evidence_scope": "small exact-DP gate only",
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
