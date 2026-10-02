from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dap.analysis.aggregation import paired_comparisons, pareto_table
from dap.analysis.statistics import benjamini_hochberg
from dap.utils.artifacts import sha256_file, write_json


def collect(root: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(pattern)):
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "completed" or manifest["completion"]["oracle"] != "PASS":
            raise RuntimeError(f"ineligible run {path.parent}")
        frame = pd.read_csv(path)
        frame["run_id"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(pattern)
    return pd.concat(frames, ignore_index=True)


def seed_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["variant", "method", "budget", "scenario", "seed"]
    numeric = [
        col
        for col in frame.select_dtypes(include=np.number).columns
        if col not in {"budget", "seed", "eval_episode", "eval_seed", "episode"}
    ]
    return frame.groupby(identifiers, as_index=False)[numeric].mean()


def compare_all(seeds, candidates, baseline, metrics):
    tables = [paired_comparisons(seeds, candidate, baseline, metrics) for candidate in candidates]
    result = pd.concat(tables, ignore_index=True)
    result["q_value_bh_global"] = benjamini_hochberg(result.p_value)
    return result


def runtime_table(run_roots):
    rows = []
    for root, pattern in run_roots:
        for path in sorted(root.glob(pattern)):
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            runtime = json.loads(path.read_text(encoding="utf-8"))
            latency = runtime["decision_latency"]
            means = [row["decision_latency_ms_mean"] for row in latency]
            rows.append(
                {
                    "run_id": path.parent.name,
                    "method": config["method"],
                    "variant": config["variant"],
                    "parameter_count": runtime["parameter_count"],
                    "training_seconds": runtime["training_seconds"],
                    "decision_latency_ms_mean": float(np.mean(means)),
                    "device": runtime["device"],
                }
            )
    return pd.DataFrame(rows)


def plot_main(dp, synthetic, trace, out):
    colors = {"b4_budget_state": "#444444", "b4_joint_hard": "#444444", "dsp_a": "#0072B2", "dsp_b": "#D55E00", "cdba": "#999999"}
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0), constrained_layout=True)
    for method in ["b4_budget_state", "dsp_a", "dsp_b", "cdba"]:
        data = dp[dp.method == method].groupby("budget").return_gap_to_paired_optimal.agg(["mean", "sem"])
        axes[0].errorbar(data.index, data["mean"], yerr=1.96 * data["sem"], marker="o", label=method, color=colors[method])
    axes[0].set(xlabel="DP budget", ylabel="Return gap to exact optimum", title="a  Exact-reference validation")
    for method in ["b4_joint_hard", "dsp_a", "dsp_b"]:
        data = synthetic[synthetic.method == method].groupby("budget").slo_violation_rate.agg(["mean", "sem"])
        axes[1].errorbar(data.index, data["mean"], yerr=1.96 * data["sem"], marker="o", label=method, color=colors[method])
        data = trace[trace.method == method].groupby("budget").slo_violation_rate.agg(["mean", "sem"])
        axes[2].errorbar(data.index, data["mean"], yerr=1.96 * data["sem"], marker="o", label=method, color=colors[method])
    axes[1].set(xlabel="Episode budget", ylabel="SLO violation rate", title="b  Synthetic nonstationarity")
    axes[2].set(xlabel="Episode budget", ylabel="SLO violation rate", title="c  Chronological public trace")
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_mechanism(synthetic, trace, ablation, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    for dataset, data, marker in [("synthetic", synthetic, "o"), ("trace", trace, "s")]:
        selected = data[data.method.isin(["dsp_a", "dsp_b"])]
        grouped = selected.groupby(["method", "budget"]).shadow_price_mean.mean().reset_index()
        for method in ["dsp_a", "dsp_b"]:
            part = grouped[grouped.method == method]
            axes[0].plot(part.budget, part.shadow_price_mean, marker=marker, label=f"{dataset}:{method}")
    axes[0].set(xlabel="Budget", ylabel="Mean finite-difference price", title="a  Learned price magnitude")
    axes[0].legend(frameon=False, fontsize=7)
    grouped = pd.concat([synthetic.assign(dataset="synthetic"), trace.assign(dataset="trace")])
    grouped = grouped.groupby(["dataset", "method"]).action_reallocation_difference.mean().reset_index()
    labels = [f"{row.dataset}\n{row.method}" for row in grouped.itertuples()]
    axes[1].bar(range(len(grouped)), grouped.action_reallocation_difference, color="#56B4E9")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(range(len(grouped)), labels, rotation=45, ha="right", fontsize=7)
    axes[1].set(ylabel="High-risk minus low-risk action cost", title="b  Actual reallocation")
    abl = ablation.groupby("variant")[["slo_violation_rate", "total_cost"]].mean().sort_index()
    axes[2].scatter(abl.total_cost, abl.slo_violation_rate, s=55)
    annotation_offsets = {
        "fixed_price": (6, 5),
        "no_actor_budget": (-76, 4),
        "no_actor_horizon": (-78, -18),
        "no_mono": (-38, 20),
    }
    for name, row in abl.iterrows():
        label = name.replace("ablation_v2_", "").replace("ablation_", "")
        axes[2].annotate(
            label,
            (row.total_cost, row.slo_violation_rate),
            xytext=annotation_offsets.get(label, (6, 5)),
            textcoords="offset points",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5},
        )
    axes[2].set(xlabel="Mean episode cost", ylabel="SLO violation rate", title="c  Ablation trade-offs")
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results/dynamic_shadow_price/summaries"
    figures = root / "results/dynamic_shadow_price/figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    dp_episodes = collect(root / "results/dynamic_shadow_price/dp_learning", "dp_joint__formal__*/metrics.csv")
    synth_episodes = collect(root / "results/dynamic_shadow_price/synthetic", "synthetic_joint__formal__*/metrics.csv")
    trace_episodes = collect(root / "results/dynamic_shadow_price/trace", "trace_joint__formal__*/metrics.csv")
    ablation = pd.concat(
        [
            collect(root / "results/dynamic_shadow_price/synthetic", "synthetic_joint__ablation_no_mono__*/metrics.csv"),
            collect(root / "results/dynamic_shadow_price/synthetic", "synthetic_joint__ablation_v2_*/metrics.csv"),
        ],
        ignore_index=True,
    )
    dp = seed_aggregate(dp_episodes)
    synthetic = seed_aggregate(synth_episodes)
    trace = seed_aggregate(trace_episodes)
    for name, table in [("dp", dp), ("synthetic", synthetic), ("trace", trace)]:
        table.to_csv(out / f"{name}_formal_seed_metrics.csv", index=False)
    dp_tests = compare_all(
        dp,
        ["cdba", "dsp_a", "dsp_b"],
        "b4_budget_state",
        ["action_consistency_rate", "return_gap_to_paired_optimal", "budget_trajectory_mae", "slo_violation_rate"],
    )
    synth_tests = compare_all(
        synthetic,
        ["dsp_a", "dsp_b"],
        "b4_joint_hard",
        ["slo_violation_rate", "total_cost", "completion_rate", "action_reallocation_difference"],
    )
    trace_tests = compare_all(
        trace,
        ["dsp_a", "dsp_b"],
        "b4_joint_hard",
        ["slo_violation_rate", "total_cost", "completion_rate", "action_reallocation_difference"],
    )
    dp_tests.to_csv(out / "dp_formal_paired_tests.csv", index=False)
    synth_tests.to_csv(out / "synthetic_formal_paired_tests.csv", index=False)
    trace_tests.to_csv(out / "trace_formal_paired_tests.csv", index=False)
    pareto_table(synthetic).to_csv(out / "synthetic_formal_pareto.csv", index=False)
    pareto_table(trace).to_csv(out / "trace_formal_pareto.csv", index=False)
    full = synth_episodes[synth_episodes.method == "dsp_a"].copy()
    full["method"] = "formal_dsp_a"
    full["variant"] = "formal_dsp_a"
    ablation_named = ablation.copy()
    ablation_named["method"] = ablation_named.variant
    abl_seed = seed_aggregate(pd.concat([full, ablation_named], ignore_index=True))
    abl_tests = compare_all(
        abl_seed,
        sorted(ablation_named.method.unique()),
        "formal_dsp_a",
        ["slo_violation_rate", "total_cost", "completion_rate", "action_reallocation_difference"],
    )
    abl_seed.to_csv(out / "synthetic_ablation_seed_metrics.csv", index=False)
    abl_tests.to_csv(out / "synthetic_ablation_paired_tests.csv", index=False)
    runtime = runtime_table(
        [
            (root / "results/dynamic_shadow_price/dp_learning", "dp_joint__formal__*/runtime.json"),
            (root / "results/dynamic_shadow_price/synthetic", "synthetic_joint__formal__*/runtime.json"),
            (root / "results/dynamic_shadow_price/trace", "trace_joint__formal__*/runtime.json"),
        ]
    )
    runtime.to_csv(out / "formal_runtime.csv", index=False)
    plot_main(dp, synthetic, trace, figures / "dsp_main_results")
    plot_mechanism(synthetic, trace, ablation, figures / "dsp_mechanism_ablation")
    overall = pd.concat(
        [
            dp_tests.assign(dataset="dp"),
            synth_tests.assign(dataset="synthetic"),
            trace_tests.assign(dataset="trace"),
        ],
        ignore_index=True,
    )
    overall.to_csv(out / "all_primary_paired_tests.csv", index=False)
    formal_manifests = list((root / "results/dynamic_shadow_price").glob("**/manifest.json"))
    failures = list((root / "results/dynamic_shadow_price").glob("**/failure.json"))
    write_json(
        out / "negative_evidence_audit.json",
        {
            "schema": "dynamic_shadow_price.negative_evidence.v1",
            "triggered": ["B1", "B3"],
            "reason": "No single DSP variant has a stable Pareto/service advantage over the fair B4 comparator across synthetic and public-trace domains.",
            "formal_dp_runs": 40,
            "formal_synthetic_runs": 30,
            "formal_trace_runs": 30,
            "formal_ablation_runs": 40,
            "all_branch_manifests": len(formal_manifests),
            "failure_files": len(failures),
            "budgets_dp": [4, 8, 12],
            "budgets_full": [110, 165, 220, 275, 330],
            "formal_seeds": list(range(10)),
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(out.glob("*.csv"))
            },
        },
    )


if __name__ == "__main__":
    main()
