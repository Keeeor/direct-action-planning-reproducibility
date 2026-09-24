from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json


METRICS = {
    "action_consistency_rate": "higher",
    "mean_Q_star_regret": "lower",
    "return_gap_to_paired_optimal": "lower",
    "high_risk_low_cost_balanced_accuracy": "higher",
    "budget_trajectory_mae": "lower",
    "completion_rate": "higher",
    "slo_violation_rate": "lower",
    "total_cost": "context",
}

PAIRINGS = (
    ("oracle_branch", "optimal"),
    ("b4_budget_state", "optimal"),
    ("dsp_b", "optimal"),
    ("acba_a", "optimal"),
    ("learned_value_branch", "dsp_b"),
    ("learned_model_branch", "learned_value_branch"),
    ("learned_model_branch", "dsp_b"),
)


def _bootstrap_mean_ci(
    values: np.ndarray, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def paired_comparisons(episodes: pd.DataFrame) -> pd.DataFrame:
    # The frozen pooled unit is (budget, seed); scenarios and episodes are averaged
    # within that unit before inference. Burst robustness is handled by the gate.
    cells = episodes.groupby(["method", "budget", "seed"], as_index=False)[
        list(METRICS)
    ].mean()
    rows: list[dict[str, object]] = []
    for pair_index, (candidate, baseline) in enumerate(PAIRINGS):
        candidate_frame = cells[cells.method == candidate].set_index(["budget", "seed"])
        baseline_frame = cells[cells.method == baseline].set_index(["budget", "seed"])
        paired = candidate_frame.join(
            baseline_frame,
            lsuffix="_candidate",
            rsuffix="_baseline",
            how="inner",
        )
        for metric_index, (metric, direction) in enumerate(METRICS.items()):
            difference = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            ).to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_ci(
                difference, seed=20260802 + pair_index * 100 + metric_index
            )
            nonzero = difference[np.abs(difference) > 1e-12]
            p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            standard_deviation = float(np.std(difference, ddof=1))
            paired_effect = (
                float(np.mean(difference) / standard_deviation)
                if standard_deviation > 1e-12
                else 0.0
            )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "preferred_direction": direction,
                    "paired_cells": len(difference),
                    "candidate_mean": float(paired[f"{metric}_candidate"].mean()),
                    "baseline_mean": float(paired[f"{metric}_baseline"].mean()),
                    "mean_difference": float(np.mean(difference)),
                    "bootstrap_95ci_lower": lower,
                    "bootstrap_95ci_upper": upper,
                    "paired_cohens_dz": paired_effect,
                    "wilcoxon_p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame["bh_q"] = _bh_adjust(frame.wilcoxon_p.tolist())
    return frame


def _evidence_grade(row: pd.Series) -> str:
    excludes_zero = row.bootstrap_95ci_lower > 0 or row.bootstrap_95ci_upper < 0
    if not excludes_zero or row.bh_q >= 0.05:
        return "none"
    if row.paired_cells < 30:
        return "weak"
    if row.bh_q < 0.01 and abs(row.paired_cohens_dz) >= 0.5 and row.paired_cells >= 30:
        return "strong"
    if abs(row.paired_cohens_dz) >= 0.5:
        return "moderate"
    return "weak"


def _claim_table(comparisons: pd.DataFrame) -> tuple[str, dict[str, object]]:
    selected = comparisons[
        comparisons.apply(
            lambda row: (row.candidate, row.baseline)
            in {
                ("oracle_branch", "optimal"),
                ("learned_value_branch", "dsp_b"),
                ("learned_model_branch", "learned_value_branch"),
                ("learned_model_branch", "dsp_b"),
            },
            axis=1,
        )
    ].copy()
    selected["grade"] = selected.apply(_evidence_grade, axis=1)
    paired_cells = int(selected.paired_cells.iloc[0])
    lines = [
        "# Claim-Evidence Table",
        "",
        f"All pooled tests use {paired_cells} paired budget-seed cells after averaging scenarios and episodes within each cell. `q` is Benjamini-Hochberg adjusted across all 56 planned comparisons.",
        "",
        "| Candidate vs baseline | Metric | Difference | 95% CI | p | q | dz | Grade |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    claims: list[dict[str, object]] = []
    for index, row in selected.reset_index(drop=True).iterrows():
        claim_id = f"DAP-C{index + 1:02d}"
        lines.append(
            f"| {row.candidate} vs {row.baseline} | {row.metric} | "
            f"{row.mean_difference:.6f} | [{row.bootstrap_95ci_lower:.6f}, "
            f"{row.bootstrap_95ci_upper:.6f}] | {row.wilcoxon_p:.6g} | "
            f"{row.bh_q:.6g} | {row.paired_cohens_dz:.3f} | {row.grade} |"
        )
        claims.append(
            {
                "claim_id": claim_id,
                "candidate": row.candidate,
                "baseline": row.baseline,
                "metric": row.metric,
                "mean_difference": float(row.mean_difference),
                "ci95": [
                    float(row.bootstrap_95ci_lower),
                    float(row.bootstrap_95ci_upper),
                ],
                "p": float(row.wilcoxon_p),
                "q": float(row.bh_q),
                "effect_size": float(row.paired_cohens_dz),
                "n": int(row.paired_cells),
                "grade": row.grade,
                "allowed_language": (
                    "demonstrates in this frozen DP matrix"
                    if row.grade == "strong"
                    else "supports in this frozen DP matrix"
                    if row.grade == "moderate"
                    else "suggests in this frozen DP matrix"
                    if row.grade == "weak"
                    else "no statistically resolved difference in this matrix"
                ),
            }
        )
    evidence = {
        "schema": "light.evidence_strength.v1",
        "project": "stage2_dynamic_budget/direct_action_planning",
        "comparison_family": "DAP-MINIMAL-PRIMARY",
        "correction": "Benjamini-Hochberg across 56 planned paired comparisons",
        "claims": claims,
    }
    return "\n".join(lines) + "\n", evidence


def _plot_corrected_policy_metrics(
    episodes: pd.DataFrame, output_dir: Path
) -> list[Path]:
    cells = episodes.groupby(["method", "budget", "seed"], as_index=False)[
        ["action_consistency_rate", "mean_Q_star_regret"]
    ].mean()
    labels = {
        "optimal": "Exact DP",
        "b4_budget_state": "B4",
        "dsp_b": "DSP-B",
        "acba_a": "ACBA-A",
        "oracle_branch": "Oracle-Branch",
        "learned_value_branch": "Learned-Value",
        "learned_model_branch": "Learned-Model",
    }
    colors = ("#000000", "#E69F00", "#0072B2", "#CC79A7", "#666666", "#009E73", "#56B4E9")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for axis, metric, ylabel in (
        (axes[0], "action_consistency_rate", "Exact optimal-action agreement"),
        (axes[1], "mean_Q_star_regret", "Mean Q* regret"),
    ):
        for index, (method, label) in enumerate(labels.items()):
            values = cells.loc[cells.method == method, metric].to_numpy(float)
            mean = float(values.mean())
            lower, upper = _bootstrap_mean_ci(values, 20260802 + index)
            axis.errorbar(
                index, mean, yerr=[[mean - lower], [upper - mean]],
                color=colors[index], marker="o", capsize=3, linestyle="none"
            )
        axis.set_xticks(range(len(labels)), list(labels.values()), rotation=32, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(bottom=0.0)
    fig.suptitle("Frozen pooled analysis (mean and bootstrap 95% CI, n=15 budget-seed cells)")
    fig.tight_layout()
    paths = [
        output_dir / "policy_agreement_and_regret_corrected.png",
        output_dir / "policy_agreement_and_regret_corrected.pdf",
    ]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def run_analysis(run_dir: str | Path, output_name: str = "analysis_v3") -> Path:
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir / output_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    episodes = pd.read_csv(run_dir / "metrics.csv")
    runtime = pd.read_csv(run_dir / "runtime.csv")
    value = pd.read_csv(run_dir / "value_diagnostics.csv")
    model = pd.read_csv(run_dir / "model_diagnostics.csv")

    method_summary = episodes.groupby("method")[list(METRICS)].agg(["mean", "std", "sem"])
    method_summary.columns = ["_".join(column) for column in method_summary.columns]
    method_summary.reset_index().to_csv(output_dir / "method_summary.csv", index=False)
    runtime.groupby("method")[["decision_latency_ms_mean", "decision_latency_ms_p95"]].agg(
        ["mean", "std"]
    ).to_csv(output_dir / "runtime_summary.csv")
    value.groupby("scenario").mean(numeric_only=True).to_csv(
        output_dir / "value_diagnostic_summary.csv"
    )
    model.groupby("scenario").mean(numeric_only=True).to_csv(
        output_dir / "model_diagnostic_summary.csv"
    )
    comparisons = paired_comparisons(episodes)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    claim_table, evidence = _claim_table(comparisons)
    (output_dir / "claim_evidence_table.md").write_text(claim_table, encoding="utf-8")
    write_json(output_dir / "evidence_strength.json", evidence)
    _plot_corrected_policy_metrics(episodes, output_dir)

    artifacts = sorted(path for path in output_dir.iterdir() if path.name != "manifest.json")
    write_json(
        output_dir / "manifest.json",
        {
            "schema": "light.analysis_manifest.v1",
            "status": "completed",
            "source_run": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {path.name: sha256_file(path) for path in artifacts},
            "comparison_family": "DAP-MINIMAL-PRIMARY",
            "multiple_comparison_control": "Benjamini-Hochberg across 56 planned paired comparisons",
            "unit_of_analysis": "budget-seed cell after averaging scenarios and episodes",
            "paired_cells": 15,
            "bootstrap_samples": 10000,
        },
    )
    return output_dir
