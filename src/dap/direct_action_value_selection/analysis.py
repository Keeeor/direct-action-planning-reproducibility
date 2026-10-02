from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dap.utils.artifacts import sha256_file, write_json


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
    ("learned_value_branch", "dsp_b"),
    ("learned_model_branch", "dsp_b"),
    ("davs_r", "dsp_b"),
    ("davs_rank", "dsp_b"),
    ("davs_ensemble", "dsp_b"),
    ("davs_r", "learned_value_branch"),
    ("davs_rank", "learned_value_branch"),
    ("davs_ensemble", "learned_value_branch"),
    ("davs_rank", "davs_r"),
    ("davs_ensemble", "davs_r"),
)


def _bootstrap_mean_ci(
    values: np.ndarray, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("cannot bootstrap an empty paired sample")
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


def _evidence_grade(row: pd.Series) -> str:
    excludes_zero = row.bootstrap_95ci_lower > 0 or row.bootstrap_95ci_upper < 0
    if not excludes_zero or row.bh_q >= 0.05:
        return "none"
    return "weak"


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = set(METRICS) - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics table is missing: {sorted(missing)}")
    cells = metrics.groupby(["method", "budget", "seed"], as_index=False)[
        list(METRICS)
    ].mean()
    rows: list[dict[str, object]] = []
    for pair_index, (candidate, baseline) in enumerate(PAIRINGS):
        left = cells[cells.method == candidate].set_index(["budget", "seed"])
        right = cells[cells.method == baseline].set_index(["budget", "seed"])
        paired = left.join(right, lsuffix="_candidate", rsuffix="_baseline", how="inner")
        if len(paired) != 15:
            raise ValueError(
                f"{candidate}/{baseline} has {len(paired)} paired cells; expected 15"
            )
        for metric_index, (metric, direction) in enumerate(METRICS.items()):
            difference = (
                paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
            ).to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_ci(
                difference, 20260802 + pair_index * 100 + metric_index
            )
            nonzero = difference[np.abs(difference) > 1e-12]
            p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            standard_deviation = float(np.std(difference, ddof=1))
            effect = (
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
                    "paired_cohens_dz": effect,
                    "wilcoxon_p": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame["bh_q"] = _bh_adjust(frame.wilcoxon_p.tolist())
    frame["evidence_grade"] = frame.apply(_evidence_grade, axis=1)
    return frame


def _claim_evidence_table(comparisons: pd.DataFrame) -> tuple[str, dict[str, object]]:
    selected_pairs = {
        ("learned_value_branch", "dsp_b"),
        ("davs_r", "dsp_b"),
        ("davs_rank", "dsp_b"),
        ("davs_ensemble", "dsp_b"),
        ("davs_ensemble", "learned_value_branch"),
    }
    selected = comparisons[
        comparisons.apply(
            lambda row: (row.candidate, row.baseline) in selected_pairs, axis=1
        )
    ]
    lines = [
        "# Claim-Evidence Table",
        "",
        "The inference unit is the paired `(budget, seed)` cell after averaging scenarios and episodes. All 80 primary tests share one Benjamini-Hochberg family; with n=15, positive evidence is capped at weak.",
        "",
        "| Candidate vs baseline | Metric | Difference | 95% CI | p | q | dz | Grade |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    claims: list[dict[str, object]] = []
    for index, row in selected.reset_index(drop=True).iterrows():
        claim_id = f"DAVS-C{index + 1:02d}"
        lines.append(
            f"| {row.candidate} vs {row.baseline} | {row.metric} | "
            f"{row.mean_difference:.6f} | [{row.bootstrap_95ci_lower:.6f}, "
            f"{row.bootstrap_95ci_upper:.6f}] | {row.wilcoxon_p:.6g} | "
            f"{row.bh_q:.6g} | {row.paired_cohens_dz:.3f} | {row.evidence_grade} |"
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
                "grade": row.evidence_grade,
                "allowed_language": (
                    "suggests in this frozen DP matrix"
                    if row.evidence_grade == "weak"
                    else "no statistically resolved difference in this matrix"
                ),
            }
        )
    evidence = {
        "schema": "light.evidence_strength.v1",
        "project": "dap/direct_action_value_selection",
        "comparison_family": "DAVS-MINIMAL-PRIMARY",
        "correction": "Benjamini-Hochberg across 80 planned paired comparisons",
        "unit_of_analysis": "budget-seed cell after averaging scenarios and episodes",
        "claims": claims,
    }
    return "\n".join(lines) + "\n", evidence


def run_analysis(run_dir: str | Path, output_name: str = "analysis_v1") -> Path:
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir / output_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"analysis output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.read_csv(run_dir / "gate_metrics.csv")
    runtime = pd.read_csv(run_dir / "runtime.csv")
    diagnostics = pd.read_csv(run_dir / "value_ranking_diagnostics.csv")

    summary = metrics.groupby("method")[list(METRICS)].agg(["mean", "std", "sem"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output_dir / "method_summary.csv", index=False)
    runtime.groupby("method").mean(numeric_only=True).to_csv(
        output_dir / "runtime_summary.csv"
    )
    diagnostics.groupby("method").mean(numeric_only=True).to_csv(
        output_dir / "value_ranking_summary.csv"
    )
    comparisons = paired_comparisons(metrics)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    claim_table, evidence = _claim_evidence_table(comparisons)
    (output_dir / "claim_evidence_table.md").write_text(claim_table, encoding="utf-8")
    write_json(output_dir / "evidence_strength.json", evidence)

    artifacts = sorted(path for path in output_dir.iterdir() if path.name != "manifest.json")
    write_json(
        output_dir / "manifest.json",
        {
            "schema": "light.analysis_manifest.v1",
            "status": "completed",
            "source_run": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {path.name: sha256_file(path) for path in artifacts},
            "comparison_family": "DAVS-MINIMAL-PRIMARY",
            "multiple_comparison_control": (
                "Benjamini-Hochberg across 80 planned paired comparisons"
            ),
            "unit_of_analysis": (
                "budget-seed cell after averaging scenarios and episodes"
            ),
            "paired_cells": 15,
            "bootstrap_samples": 10_000,
            "evidence_strength_cap": "weak because paired n=15",
        },
    )
    return output_dir
