#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
SUM = ROOT / "results" / "summaries"
sys.path.insert(0, str(ROOT / "src"))


COLORS = {
    "b3_lagrangian": "#0072B2",
    "b4_budget_state": "#E69F00",
    "b5_fixed_local": "#009E73",
    "cdba": "#D55E00",
    "cdba_discrete": "#CC79A7",
    "b2_ppo": "#56B4E9",
    "b4_budget_state_matched": "#000000",
}
LABELS = {
    "b3_lagrangian": "Fixed-Lagrangian PPO",
    "b4_budget_state": "Budget-state PPO",
    "b5_fixed_local": "Fixed local quota",
    "cdba": "CDBA (continuous)",
    "cdba_discrete": "CDBA (discrete)",
    "b2_ppo": "PPO",
    "b4_budget_state_matched": "Budget-state PPO (param.-matched)",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _across_scenario_ci(seeds: pd.DataFrame, metric: str) -> pd.DataFrame:
    per_seed = seeds.groupby(["method", "budget", "seed"], as_index=False)[metric].mean()
    rows = []
    for keys, group in per_seed.groupby(["method", "budget"]):
        values = group[metric].to_numpy(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
        rows.append({"method": keys[0], "budget": keys[1], "mean": mean, "low": mean - half, "high": mean + half})
    return pd.DataFrame(rows)


def plot_main_synthetic() -> None:
    seeds = pd.read_csv(SUM / "final_synthetic_seed_metrics.csv")
    methods = ["b3_lagrangian", "b4_budget_state", "b4_budget_state_matched", "b5_fixed_local", "cdba", "cdba_discrete"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    curve = _across_scenario_ci(seeds[seeds.method.isin(methods)], "slo_violation_rate")
    for method in methods:
        part = curve[curve.method == method].sort_values("budget")
        axes[0].plot(part.budget, part["mean"], marker="o", lw=1.4, color=COLORS[method], label=LABELS[method])
        axes[0].fill_between(part.budget, part.low, part.high, color=COLORS[method], alpha=0.13, linewidth=0)
    axes[0].set(title="a  Service violations across budgets", xlabel="Episode budget", ylabel="SLO violation rate")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, ncol=2)

    points = seeds[seeds.method.isin(methods)].groupby(["method", "budget"], as_index=False)[["total_cost", "slo_violation_rate"]].mean()
    for method in methods:
        part = points[points.method == method].sort_values("total_cost")
        axes[1].plot(part.total_cost, part.slo_violation_rate, marker="o", lw=1.4, color=COLORS[method], label=LABELS[method])
    axes[1].set(title="b  Empirical service–cost frontier", xlabel="Mean realized episode cost", ylabel="SLO violation rate")
    axes[1].grid(alpha=0.22)
    _save(fig, "fig01_synthetic_main")


def plot_mechanism() -> None:
    seeds = pd.read_csv(SUM / "final_synthetic_seed_metrics.csv")
    cdba = seeds[seeds.method == "cdba"]
    aggregate = cdba.groupby("budget", as_index=False)[
        ["high_risk_budget_mean", "low_risk_budget_mean", "risk_budget_correlation", "budget_reallocation_ratio"]
    ].mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), constrained_layout=True)
    x = np.arange(len(aggregate))
    width = 0.36
    axes[0].bar(x - width / 2, aggregate.low_risk_budget_mean, width, color="#56B4E9", label="Low-risk quartile")
    axes[0].bar(x + width / 2, aggregate.high_risk_budget_mean, width, color="#D55E00", label="High-risk quartile")
    axes[0].set_xticks(x, aggregate.budget.astype(int))
    axes[0].set(title="a  Context-dependent quota", xlabel="Episode budget", ylabel="Mean local quota")
    axes[0].legend(frameon=False)
    axes[1].plot(aggregate.budget, aggregate.risk_budget_correlation, color="#D55E00", marker="o")
    axes[1].axhline(0, color="0.45", lw=0.8)
    axes[1].set(title="b  Risk–quota association", xlabel="Episode budget", ylabel="Spearman correlation")
    exhaustion = seeds[seeds.method.isin(["b3_lagrangian", "b4_budget_state", "b5_fixed_local", "cdba", "cdba_discrete"])].groupby(["method", "budget"], as_index=False).budget_exhaustion_ratio.mean()
    for method in exhaustion.method.unique():
        part = exhaustion[exhaustion.method == method]
        axes[2].plot(part.budget, part.budget_exhaustion_ratio, marker="o", color=COLORS[method], label=LABELS[method])
    axes[2].set(title="c  Budget exhaustion timing", xlabel="Episode budget", ylabel="Exhaustion step / horizon", ylim=(0, 1.05))
    axes[2].legend(frameon=False, fontsize=6)
    for ax in axes:
        ax.grid(alpha=0.2)
    _save(fig, "fig02_mechanism_summary")


def plot_mechanism_timeline() -> None:
    paths = {
        "cdba": ROOT / "results/raw_logs/formal/formal__valid_v2__cdba__b220__s0/steps.csv.gz",
        "b4": ROOT / "results/raw_logs/formal/formal__valid_v2__b4_budget_state__b220__s0/steps.csv.gz",
    }
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    frames = {key: value[(value.scenario == "late_burst") & (value.episode == 0)].sort_values("step") for key, value in frames.items()}
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 4.6), sharex=True, constrained_layout=True)
    axes[0].plot(frames["cdba"].step, frames["cdba"].load_level, color="#0072B2", lw=1.2, label="Arrival load")
    axes[0].plot(frames["cdba"].step, frames["cdba"].risk_level * 10, color="#D55E00", lw=1.1, label="Risk × 10")
    axes[0].set(ylabel="Load / scaled risk", title="Context-aware reallocation during a late burst")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(frames["cdba"].step, frames["cdba"].local_budget, color="#D55E00", lw=1.2, label="CDBA local quota")
    axes[1].axhline(220 / 128, color="0.35", ls="--", lw=0.9, label="Fixed B/T quota")
    axes[1].set(ylabel="Local quota")
    axes[1].legend(frameon=False, ncol=2)
    for label, frame, color in (("CDBA", frames["cdba"], "#D55E00"), ("Budget-state PPO", frames["b4"], "#E69F00")):
        axes[2].plot(frame.step, frame.resource_cost.cumsum(), color=color, lw=1.2, label=label)
    axes[2].axhline(220, color="0.35", ls="--", lw=0.9, label="Budget")
    axes[2].set(xlabel="Decision step", ylabel="Cumulative cost")
    axes[2].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.grid(alpha=0.2)
    _save(fig, "fig03_mechanism_timeline")


def plot_ablation() -> None:
    paired = pd.read_csv(SUM / "final_ablation_paired_tests.csv")
    overall = paired[paired.budget.astype(str) == "ALL"].copy()
    order = overall[overall.metric == "slo_violation_rate"].sort_values("mean_difference").candidate.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7), sharey=True, constrained_layout=True)
    for ax, metric, title, xlabel in (
        (axes[0], "slo_violation_rate", "a  Service effect", "SLO violation difference\n(lower is better)"),
        (axes[1], "total_cost", "b  Cost effect", "Episode-cost difference\n(lower is better)"),
    ):
        data = overall[overall.metric == metric].set_index("candidate").loc[order].reset_index()
        y = np.arange(len(data))
        ax.errorbar(data.mean_difference, y, xerr=[data.mean_difference - data.ci_low, data.ci_high - data.mean_difference], fmt="o", color="#0072B2", ecolor="#56B4E9", capsize=2)
        ax.axvline(0, color="0.35", lw=0.9)
        ax.set_yticks(y, data.candidate.str.replace("_", " "))
        ax.set(title=title, xlabel=xlabel)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Ablation effects vs full CDBA, paired across seeds", fontsize=10)
    _save(fig, "fig04_ablation")


def plot_sensitivity() -> None:
    seeds = pd.read_csv(SUM / "final_sensitivity_seed_metrics.csv")
    means = seeds.groupby("method", as_index=False)[["total_cost", "slo_violation_rate", "risk_budget_correlation"]].mean()
    paired = pd.read_csv(SUM / "final_sensitivity_paired_tests.csv")
    cost_variants = [name for name in means.method if name.startswith("benefit") or name.startswith("cost_")]
    hyper = paired[(paired.metric == "slo_violation_rate") & (paired.budget.astype(str) == "ALL") & ~paired.candidate.isin(cost_variants)].sort_values("mean_difference")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5), constrained_layout=True)
    y = np.arange(len(hyper))
    axes[0].errorbar(hyper.mean_difference, y, xerr=[hyper.mean_difference - hyper.ci_low, hyper.ci_high - hyper.mean_difference], fmt="o", color="#0072B2", ecolor="#56B4E9", capsize=2)
    axes[0].axvline(0, color="0.35", lw=0.9)
    axes[0].set_yticks(y, hyper.candidate.str.replace("_", " "))
    axes[0].set(title="a  Hyperparameter sensitivity", xlabel="SLO difference vs default\n(lower is better)")
    axes[0].grid(axis="x", alpha=0.2)
    curve = means[means.method.isin(cost_variants + ["default_reference"])]
    for _, row in curve.iterrows():
        marker = "o" if row.method == "default_reference" else "s"
        color = "#0072B2" if row.method == "default_reference" else "#D55E00"
        axes[1].scatter(row.total_cost, row.slo_violation_rate, color=color, marker=marker)
        axes[1].annotate(row.method.replace("_", " "), (row.total_cost, row.slo_violation_rate), xytext=(3, 2), textcoords="offset points", fontsize=6)
    axes[1].set(title="b  Action cost–benefit curves", xlabel="Mean realized episode cost", ylabel="SLO violation rate")
    axes[1].grid(alpha=0.2)
    _save(fig, "fig05_sensitivity")


def plot_trace() -> None:
    seeds = pd.read_csv(SUM / "final_trace_seed_metrics.csv")
    methods = ["b3_lagrangian", "b4_budget_state", "b5_fixed_local", "cdba", "cdba_discrete"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    curve = _across_scenario_ci(seeds[seeds.method.isin(methods)], "slo_violation_rate")
    for method in methods:
        part = curve[curve.method == method].sort_values("budget")
        axes[0].plot(part.budget, part["mean"], marker="o", lw=1.3, color=COLORS[method], label=LABELS[method])
        axes[0].fill_between(part.budget, part.low, part.high, color=COLORS[method], alpha=0.13)
    axes[0].set(title="a  Chronological Azure evaluation", xlabel="Episode budget", ylabel="SLO violation rate")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=6)
    profile_root = ROOT / "data/processed/azure_functions_2019"
    for domain, color in (("http", "#0072B2"), ("async", "#D55E00")):
        with np.load(profile_root / f"{domain}.npz") as bundle:
            trace = bundle["test"][:512]
        axes[1].plot(np.arange(len(trace)), trace, color=color, lw=0.9, label=f"{domain} test")
    axes[1].set(title="b  Public test-trace excerpt", xlabel="Chronological minute", ylabel="Scaled invocations")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    _save(fig, "fig06_public_trace")


def plot_generalization() -> None:
    seeds = pd.read_csv(SUM / "final_generalization_seed_metrics.csv")
    data = seeds.groupby(["variant", "method", "scenario"], as_index=False).slo_violation_rate.mean()
    scenarios = sorted(data.scenario.unique())
    labels = [scenario.replace("azure_", "").replace("_", " ") for scenario in scenarios]
    fig, axes = plt.subplots(1, len(data.variant.unique()), figsize=(7.1, 2.8), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, variant in zip(axes, sorted(data.variant.unique())):
        subset = data[data.variant == variant]
        x = np.arange(len(scenarios))
        for offset, method in zip((-0.22, 0, 0.22), ["b4_budget_state", "cdba", "cdba_discrete"]):
            values = subset.set_index(["method", "scenario"]).reindex(pd.MultiIndex.from_product([[method], scenarios])).slo_violation_rate.to_numpy()
            ax.bar(x + offset, values, 0.21, color=COLORS[method], label=LABELS[method])
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set(title=variant.replace("train_", "Train: ").replace("_", " "), ylabel="SLO violation rate")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=6)
    _save(fig, "fig07_cross_domain_generalization")


def main() -> int:
    _style()
    plot_main_synthetic()
    plot_mechanism()
    plot_mechanism_timeline()
    plot_ablation()
    plot_sensitivity()
    plot_trace()
    plot_generalization()
    print("figures written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
