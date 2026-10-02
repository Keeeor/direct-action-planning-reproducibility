from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dap.utils.artifacts import sha256_file, write_json


METRICS = {
    "episode_reward": "higher",
    "completion_rate": "higher",
    "slo_violation_rate": "lower",
    "total_cost": "context",
}
PAIRINGS = tuple(
    (candidate, baseline)
    for candidate in ("davs_r", "davs_rank", "davs_ensemble")
    for baseline in ("dsp_b", "b4_joint_hard")
)
COLORS = {
    "b4_joint_hard": "#000000",
    "dsp_b": "#0072B2",
    "davs_r": "#E69F00",
    "davs_rank": "#D55E00",
    "davs_ensemble": "#666666",
}
LABELS = {
    "b4_joint_hard": "B4",
    "dsp_b": "DSP-B",
    "davs_r": "DAVS-R",
    "davs_rank": "DAVS-Rank",
    "davs_ensemble": "DAVS-Ensemble",
}


def _bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def _grade(row: pd.Series) -> str:
    excludes_zero = row.bootstrap_95ci_lower > 0 or row.bootstrap_95ci_upper < 0
    if not excludes_zero or row.bh_q >= 0.05:
        return "none"
    if row.bh_q < 0.01 and abs(row.paired_cohens_dz) >= 0.5:
        return "strong"
    if abs(row.paired_cohens_dz) >= 0.5:
        return "moderate"
    return "weak"


def paired_comparisons(
    synthetic: pd.DataFrame, public: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain_index, (domain, frame) in enumerate(
        (("synthetic", synthetic), ("public_trace", public))
    ):
        cells = frame.groupby(["method", "budget", "seed"], as_index=False)[
            list(METRICS)
        ].mean()
        for pair_index, (candidate, baseline) in enumerate(PAIRINGS):
            left = cells[cells.method == candidate].set_index(["budget", "seed"])
            right = cells[cells.method == baseline].set_index(["budget", "seed"])
            paired = left.join(
                right, lsuffix="_candidate", rsuffix="_baseline", how="inner"
            )
            if len(paired) != 50:
                raise ValueError(f"{domain}/{candidate}/{baseline} has n={len(paired)}")
            for metric_index, (metric, direction) in enumerate(METRICS.items()):
                difference = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
                ).to_numpy(dtype=float)
                lower, upper = _bootstrap_ci(
                    difference,
                    20260802 + domain_index * 10_000 + pair_index * 100 + metric_index,
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
                        "domain": domain,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "preferred_direction": direction,
                        "paired_cells": len(difference),
                        "candidate_mean": float(
                            paired[f"{metric}_candidate"].mean()
                        ),
                        "baseline_mean": float(paired[f"{metric}_baseline"].mean()),
                        "mean_difference": float(np.mean(difference)),
                        "bootstrap_95ci_lower": lower,
                        "bootstrap_95ci_upper": upper,
                        "paired_cohens_dz": effect,
                        "wilcoxon_p": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["bh_q"] = _bh_adjust(result.wilcoxon_p.tolist())
    result["evidence_grade"] = result.apply(_grade, axis=1)
    return result


def _save(fig: plt.Figure, output: Path, name: str) -> list[Path]:
    png = output / f"{name}.png"
    pdf = output / f"{name}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _pareto_figure(
    synthetic: pd.DataFrame, public: pd.DataFrame, output: Path
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for axis, (title, frame) in zip(
        axes, (("Eight synthetic scenarios", synthetic), ("Public trace test", public))
    ):
        summary = frame.groupby("method")[["total_cost", "completion_rate"]].mean()
        for method, row in summary.iterrows():
            axis.scatter(
                row.total_cost,
                row.completion_rate,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.5,
                s=55,
                label=LABELS[method],
            )
        axis.set_xlabel("Mean total resource cost")
        axis.set_ylabel("Mean completion rate")
        axis.set_title(title)
        axis.grid(color="#DDDDDD", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Service-cost relation; SLO and reward remain separate gate metrics")
    fig.tight_layout()
    return _save(fig, output, "full_service_cost_pareto")


def _difference_figure(comparisons: pd.DataFrame, output: Path) -> list[Path]:
    selected = comparisons[
        (comparisons.candidate == "davs_ensemble")
        & (comparisons.baseline == "dsp_b")
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6))
    for row_index, domain in enumerate(("synthetic", "public_trace")):
        group = selected[selected.domain == domain].set_index("metric")
        for column_index, (metric, label) in enumerate(
            zip(METRICS, ("Reward", "Completion", "SLO", "Cost"))
        ):
            axis = axes[row_index, column_index]
            row = group.loc[metric]
            value = float(row.mean_difference)
            lower = float(row.bootstrap_95ci_lower)
            upper = float(row.bootstrap_95ci_upper)
            axis.errorbar(
                [0],
                [value],
                yerr=[[value - lower], [upper - value]],
                marker="o",
                linestyle="none",
                color=COLORS["davs_ensemble"],
                capsize=3,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks([])
            axis.set_title(label)
            axis.set_ylabel(
                ("Synthetic" if row_index == 0 else "Public trace") + " difference"
            )
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Paired mean differences and bootstrap 95% CI (n=50 budget-seed cells)")
    fig.tight_layout()
    return _save(fig, output, "full_ensemble_paired_differences")


def _generalization_figure(diagnostics: pd.DataFrame, output: Path) -> list[Path]:
    synth = diagnostics[
        (diagnostics.domain == "synthetic")
        & (diagnostics.method == "davs_ensemble")
    ]
    scenario = synth.groupby("scenario")[[
        "action_consistency_rate",
        "pairwise_action_ranking_accuracy",
        "error_auroc",
    ]].mean()
    order = [
        "stable",
        "periodic",
        "early_burst",
        "late_burst",
        "multi_burst",
        "gradual",
        "abrupt",
        "ood_burst",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    positions = np.arange(len(order))
    axes[0].plot(
        positions,
        scenario.loc[order, "action_consistency_rate"],
        marker="o",
        label="Top-1 agreement",
        color="#D55E00",
    )
    axes[0].plot(
        positions,
        scenario.loc[order, "pairwise_action_ranking_accuracy"],
        marker="s",
        label="Pairwise ranking",
        color="#0072B2",
    )
    axes[1].bar(
        positions,
        scenario.loc[order, "error_auroc"],
        color="#777777",
        edgecolor="black",
        linewidth=0.4,
    )
    for axis in axes:
        axis.set_xticks(positions, [value.replace("_", " ") for value in order], rotation=35, ha="right")
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Held-out branch-label accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("Uncertainty AUROC for wrong action")
    fig.suptitle("Synthetic branch ranking and uncertainty by scenario")
    fig.tight_layout()
    return _save(fig, output, "full_ranking_generalization")


def _sensitivity_figure(sensitivity: pd.DataFrame, output: Path) -> list[Path]:
    families = ("rank_beta", "data_scale", "label_noise", "ensemble_members")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    for axis, family in zip(axes.flat, families):
        group = sensitivity[sensitivity.variant == family].groupby("setting")[
            ["action_consistency_rate", "mean_Q_branch_regret"]
        ].mean()
        positions = np.arange(len(group))
        axis.bar(
            positions,
            group.action_consistency_rate,
            color="#56B4E9",
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_xticks(positions, group.index, rotation=25, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Action agreement")
        axis.set_title(family.replace("_", " ").title())
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Held-out synthetic branch sensitivity (same frozen labels)")
    fig.tight_layout()
    return _save(fig, output, "full_sensitivity")


def _claim_table(comparisons: pd.DataFrame) -> tuple[str, dict[str, object]]:
    selected = comparisons[
        (comparisons.candidate == "davs_ensemble")
        & comparisons.baseline.isin(["dsp_b", "b4_joint_hard"])
    ]
    lines = [
        "# Full Claim-Evidence Table",
        "",
        "Each domain uses 50 paired budget-seed cells after averaging scenarios/domains and episodes. All 48 planned tests share one Benjamini-Hochberg family.",
        "",
        "| Domain | Candidate vs baseline | Metric | Difference | 95% CI | q | dz | Grade |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    claims: list[dict[str, object]] = []
    for index, row in selected.reset_index(drop=True).iterrows():
        lines.append(
            f"| {row.domain} | {row.candidate} vs {row.baseline} | {row.metric} | "
            f"{row.mean_difference:.6f} | [{row.bootstrap_95ci_lower:.6f}, "
            f"{row.bootstrap_95ci_upper:.6f}] | {row.bh_q:.6g} | "
            f"{row.paired_cohens_dz:.3f} | {row.evidence_grade} |"
        )
        claims.append(
            {
                "claim_id": f"DAVS-FULL-C{index + 1:02d}",
                "domain": row.domain,
                "candidate": row.candidate,
                "baseline": row.baseline,
                "metric": row.metric,
                "mean_difference": float(row.mean_difference),
                "ci95": [float(row.bootstrap_95ci_lower), float(row.bootstrap_95ci_upper)],
                "q": float(row.bh_q),
                "effect_size": float(row.paired_cohens_dz),
                "n": int(row.paired_cells),
                "grade": row.evidence_grade,
                "allowed_language": (
                    "supports in the frozen expansion"
                    if row.evidence_grade in {"strong", "moderate"}
                    else "suggests in the frozen expansion"
                    if row.evidence_grade == "weak"
                    else "no statistically resolved difference in the frozen expansion"
                ),
            }
        )
    return "\n".join(lines) + "\n", {
        "schema": "light.evidence_strength.v1",
        "project": "dap/direct_action_value_selection",
        "comparison_family": "DAVS-FULL-PRIMARY",
        "correction": "Benjamini-Hochberg across 48 planned paired comparisons",
        "claims": claims,
    }


def run_full_analysis(run_dir: str | Path, output_name: str = "analysis_v1") -> Path:
    run_dir = Path(run_dir).resolve()
    output = run_dir / output_name
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"analysis output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    synthetic = pd.read_csv(run_dir / "synthetic_metrics.csv")
    public = pd.read_csv(run_dir / "public_trace_metrics.csv")
    diagnostics = pd.read_csv(run_dir / "branch_diagnostics.csv")
    sensitivity = pd.read_csv(run_dir / "sensitivity.csv")
    runtime = pd.read_csv(run_dir / "runtime.csv")

    comparisons = paired_comparisons(synthetic, public)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    for name, frame in (("synthetic", synthetic), ("public_trace", public)):
        frame.groupby("method")[list(METRICS)].agg(["mean", "std", "sem"]).to_csv(
            output / f"{name}_method_summary.csv"
        )
    diagnostics.groupby(["domain", "method"]).mean(numeric_only=True).to_csv(
        output / "branch_diagnostic_summary.csv"
    )
    runtime.groupby(["domain", "method"])[
        ["decision_latency_ms_mean", "decision_latency_ms_p95"]
    ].agg(["mean", "std"]).to_csv(output / "runtime_summary.csv")
    claim_table, evidence = _claim_table(comparisons)
    (output / "claim_evidence_table.md").write_text(claim_table, encoding="utf-8")
    write_json(output / "evidence_strength.json", evidence)

    _pareto_figure(synthetic, public, output)
    _difference_figure(comparisons, output)
    _generalization_figure(diagnostics, output)
    _sensitivity_figure(sensitivity, output)
    artifacts = sorted(path for path in output.iterdir() if path.name != "manifest.json")
    write_json(
        output / "manifest.json",
        {
            "schema": "light.analysis_manifest.v1",
            "status": "completed",
            "source_run": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {path.name: sha256_file(path) for path in artifacts},
            "comparison_family": "DAVS-FULL-PRIMARY",
            "multiple_comparison_control": (
                "Benjamini-Hochberg across 48 planned paired comparisons"
            ),
            "unit_of_analysis": (
                "budget-seed cell after within-domain scenario/episode averaging"
            ),
            "paired_cells_per_domain": 50,
            "bootstrap_samples": 10_000,
        },
    )
    return output
