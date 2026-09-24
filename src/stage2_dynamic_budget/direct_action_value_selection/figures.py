from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "optimal",
    "dsp_b",
    "acba_a",
    "learned_value_branch",
    "learned_model_branch",
    "davs_r",
    "davs_rank",
    "davs_ensemble",
)

LABELS = {
    "optimal": "Exact DP",
    "dsp_b": "DSP-B",
    "acba_a": "ACBA-A",
    "learned_value_branch": "Learned-Value",
    "learned_model_branch": "Learned-Model",
    "davs_r": "DAVS-R",
    "davs_rank": "DAVS-Rank",
    "davs_ensemble": "DAVS-Ensemble",
}

COLORS = {
    "optimal": "#000000",
    "dsp_b": "#0072B2",
    "acba_a": "#CC79A7",
    "learned_value_branch": "#009E73",
    "learned_model_branch": "#56B4E9",
    "davs_r": "#E69F00",
    "davs_rank": "#D55E00",
    "davs_ensemble": "#666666",
}


def _bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(5_000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _save(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _policy_figure(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    cells = metrics.groupby(["method", "budget", "seed"], as_index=False)[
        ["action_consistency_rate", "mean_Q_star_regret"]
    ].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for axis, metric, ylabel in (
        (axes[0], "action_consistency_rate", "Exact optimal-action agreement"),
        (axes[1], "mean_Q_star_regret", "Mean Q* regret"),
    ):
        for index, method in enumerate(METHOD_ORDER):
            values = cells.loc[cells.method == method, metric].to_numpy(float)
            mean = float(values.mean())
            lower, upper = _bootstrap_ci(values, 20260802 + index)
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                color=COLORS[method],
                marker="o",
                capsize=3,
                linestyle="none",
            )
        axis.set_xticks(range(len(METHOD_ORDER)), [LABELS[item] for item in METHOD_ORDER])
        axis.tick_params(axis="x", rotation=32)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylim(0.0, 1.02)
    axes[1].set_ylim(bottom=0.0)
    fig.suptitle("Held-out time-block policy metrics (mean and bootstrap 95% CI, n=15)")
    fig.tight_layout()
    return _save(fig, output_dir, "policy_agreement_and_regret")


def _value_diagnostic_figure(diagnostics: pd.DataFrame, output_dir: Path) -> list[Path]:
    methods = ("davs_r", "davs_rank", "davs_ensemble")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for index, method in enumerate(methods):
        group = diagnostics[diagnostics.method == method]
        axes[0].bar(
            index,
            float(group.action_value_mae_q_star.mean()),
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].bar(
            index,
            float(group.pairwise_action_ranking_accuracy.mean()),
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
        )
    labels = [LABELS[item] for item in methods]
    axes[0].set_xticks(range(len(methods)), labels, rotation=22)
    axes[1].set_xticks(range(len(methods)), labels, rotation=22)
    axes[0].set_ylabel("Action-value MAE vs Q*")
    axes[1].set_ylabel("Feasible-pair ranking accuracy")
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(0.0, 1.02)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Direct action-value and ranking diagnostics (15 scenario-seed fits)")
    fig.tight_layout()
    return _save(fig, output_dir, "value_and_ranking_diagnostics")


def _service_cost_figure(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    summary = metrics.groupby("method")[["total_cost", "completion_rate", "slo_violation_rate"]].mean()
    fig, axis = plt.subplots(figsize=(6.2, 4.5))
    for method in METHOD_ORDER:
        row = summary.loc[method]
        axis.scatter(
            row.total_cost,
            row.completion_rate,
            s=58,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            label=LABELS[method],
        )
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Mean resource cost in held-out two-step window")
    axis.set_ylabel("Completion rate in held-out two-step window")
    axis.grid(color="#DDDDDD", linewidth=0.6)
    axis.legend(fontsize=8, ncol=2, frameon=False)
    axis.set_title("Service-cost relation; SLO is reported separately in the result table")
    fig.tight_layout()
    return _save(fig, output_dir, "service_cost_pareto")


def _generalization_uncertainty_figure(
    loso: pd.DataFrame, uncertainty_states: pd.DataFrame, output_dir: Path
) -> list[Path]:
    methods = ("dsp_b", "davs_r", "davs_rank", "davs_ensemble")
    scenarios = ("early_burst", "late_burst", "periodic")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    width = 0.19
    for method_index, method in enumerate(methods):
        values = [
            float(
                loso[
                    (loso.method == method) & (loso.heldout_scenario == scenario)
                ].action_consistency_rate.mean()
            )
            for scenario in scenarios
        ]
        positions = np.arange(len(scenarios)) + (method_index - 1.5) * width
        axes[0].bar(
            positions,
            values,
            width=width,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.4,
            label=LABELS[method],
        )
    axes[0].set_xticks(range(len(scenarios)), [item.replace("_", " ") for item in scenarios])
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Leave-one-scenario action agreement")
    axes[0].legend(fontsize=7, frameon=False)
    ensemble = uncertainty_states[uncertainty_states.method == "davs_ensemble"]
    correct = ensemble.loc[ensemble.error == 0, "ranking_uncertainty"].to_numpy(float)
    wrong = ensemble.loc[ensemble.error == 1, "ranking_uncertainty"].to_numpy(float)
    axes[1].boxplot(
        [correct, wrong],
        tick_labels=[f"Correct\n(n={len(correct)})", f"Wrong\n(n={len(wrong)})"],
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": "#BDBDBD"},
        medianprops={"color": "black"},
    )
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("DAVS-Ensemble ranking uncertainty")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("Cross-scenario transfer and uncertainty/error separation")
    fig.tight_layout()
    return _save(fig, output_dir, "generalization_and_uncertainty")


def generate_all_figures(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    loso: pd.DataFrame,
    uncertainty_states: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(_policy_figure(metrics, output_dir))
    paths.extend(_value_diagnostic_figure(diagnostics, output_dir))
    paths.extend(_service_cost_figure(metrics, output_dir))
    if not loso.empty:
        paths.extend(
            _generalization_uncertainty_figure(loso, uncertainty_states, output_dir)
        )
    return paths
