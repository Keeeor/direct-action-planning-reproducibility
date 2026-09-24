#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2_dynamic_budget.analysis.aggregation import (  # noqa: E402
    collect_runs,
    paired_comparisons,
    pareto_table,
    seed_aggregate,
    summary_table,
)
from stage2_dynamic_budget.analysis.statistics import benjamini_hochberg  # noqa: E402


OUT = ROOT / "results" / "summaries"


def _write_core(name: str, seeds: pd.DataFrame) -> None:
    seeds.to_csv(OUT / f"{name}_seed_metrics.csv", index=False)
    summary_table(seeds).to_csv(OUT / f"{name}_summary.csv", index=False)
    pareto_table(seeds).to_csv(OUT / f"{name}_pareto.csv", index=False)


def _all_pairwise(seeds: pd.DataFrame, baseline: str, candidates: list[str], name: str) -> None:
    tables = [paired_comparisons(seeds, candidate, baseline) for candidate in candidates]
    result = pd.concat(tables, ignore_index=True)
    result["q_value_bh_global"] = benjamini_hochberg(result.p_value)
    result.to_csv(OUT / f"{name}_paired_tests.csv", index=False)


def _matched_tradeoffs(seeds: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    means = seeds.groupby(["method", "budget"], as_index=False)[
        ["total_cost", "slo_violation_rate", "completion_rate"]
    ].mean()
    base = means[means.method == baseline].copy()
    cand = means[means.method == candidate].copy()
    cost_order = base.sort_values("total_cost")
    service_order = base.sort_values("slo_violation_rate")
    rows = []
    for _, row in cand.iterrows():
        cost = float(row.total_cost)
        service = float(row.slo_violation_rate)
        same_cost_service = (
            float(np.interp(cost, cost_order.total_cost, cost_order.slo_violation_rate))
            if cost_order.total_cost.min() <= cost <= cost_order.total_cost.max()
            else np.nan
        )
        same_service_cost = (
            float(np.interp(service, service_order.slo_violation_rate, service_order.total_cost))
            if service_order.slo_violation_rate.min() <= service <= service_order.slo_violation_rate.max()
            else np.nan
        )
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "budget": row.budget,
                "candidate_cost": cost,
                "candidate_slo_violation": service,
                "baseline_slo_at_same_cost": same_cost_service,
                "slo_difference_at_same_cost": service - same_cost_service,
                "baseline_cost_at_same_service": same_service_cost,
                "cost_difference_at_same_service": cost - same_service_cost,
            }
        )
    return pd.DataFrame(rows)


def _anomalies(seeds: pd.DataFrame, name: str) -> None:
    rows = []
    core = ["total_cost", "slo_violation_rate", "completion_rate"]
    for keys, group in seeds.groupby(["method", "budget", "scenario"]):
        for metric in core:
            values = group[metric].to_numpy(float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad <= 1e-12:
                continue
            robust_z = 0.6745 * (values - median) / mad
            for (_, row), score in zip(group.iterrows(), robust_z):
                if abs(score) > 3.5:
                    rows.append(
                        {
                            "method": keys[0],
                            "budget": keys[1],
                            "scenario": keys[2],
                            "seed": row.seed,
                            "metric": metric,
                            "value": row[metric],
                            "median": median,
                            "robust_z": score,
                            "classification": "retained_statistical_outlier",
                        }
                    )
    pd.DataFrame(rows).to_csv(OUT / f"{name}_anomalies.csv", index=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    formal = pd.read_csv(OUT / "formal_valid_v2_seed_metrics.csv")
    corrected_b5 = pd.read_csv(OUT / "formal_valid_v2_b5_corrected_seed_metrics.csv")
    matched_episodes, matched_runtime = collect_runs(
        ROOT / "results" / "raw_logs" / "formal_matched", "formal_matched__hidden85__"
    )
    matched = seed_aggregate(matched_episodes)
    matched["method"] = "b4_budget_state_matched"
    matched["variant"] = "hidden85"
    matched_runtime["method"] = "b4_budget_state_matched"
    synthetic = pd.concat(
        [formal[formal.method != "b5_fixed_local"], corrected_b5, matched], ignore_index=True
    )
    _write_core("final_synthetic", synthetic)
    _all_pairwise(
        synthetic,
        "b4_budget_state",
        ["b3_lagrangian", "b5_fixed_local", "cdba", "cdba_discrete"],
        "final_synthetic",
    )
    _all_pairwise(
        synthetic,
        "b4_budget_state_matched",
        ["cdba", "cdba_discrete"],
        "final_synthetic_matched",
    )
    pd.concat(
        [
            _matched_tradeoffs(synthetic, "cdba", "b4_budget_state"),
            _matched_tradeoffs(synthetic, "cdba_discrete", "b4_budget_state"),
            _matched_tradeoffs(synthetic, "cdba", "b4_budget_state_matched"),
        ],
        ignore_index=True,
    ).to_csv(OUT / "final_synthetic_matched_tradeoffs.csv", index=False)
    _anomalies(synthetic, "final_synthetic")
    formal_runtime = pd.read_csv(OUT / "formal_valid_v2_runtime.csv")
    corrected_runtime = pd.read_csv(OUT / "formal_valid_v2_b5_corrected_runtime.csv")
    final_runtime = pd.concat(
        [formal_runtime[formal_runtime.method != "b5_fixed_local"], corrected_runtime, matched_runtime],
        ignore_index=True,
    )
    final_runtime.to_csv(OUT / "final_synthetic_runtime.csv", index=False)

    ablation_episodes, _ = collect_runs(ROOT / "results" / "raw_logs" / "ablation", "ablation__")
    ablation = seed_aggregate(ablation_episodes)
    ablation = ablation[
        ~ablation.variant.isin(["no_remaining_budget", "no_remaining_horizon"])
    ].copy()
    ablation["method"] = ablation["variant"]
    reference = synthetic[
        (synthetic.method == "cdba") & synthetic.budget.isin([110.0, 220.0, 330.0])
    ].copy()
    reference["method"] = "full_cdba"
    reference["variant"] = "full_cdba"
    ablation_all = pd.concat([reference, ablation], ignore_index=True)
    _write_core("final_ablation", ablation_all)
    _all_pairwise(
        ablation_all,
        "full_cdba",
        sorted(method for method in ablation.method.unique()),
        "final_ablation",
    )

    sensitivity_episodes, _ = collect_runs(
        ROOT / "results" / "raw_logs" / "sensitivity", "sensitivity__"
    )
    sensitivity = seed_aggregate(sensitivity_episodes)
    sensitivity["method"] = sensitivity["variant"]
    _write_core("final_sensitivity", sensitivity)
    if "default_reference" in set(sensitivity.method):
        _all_pairwise(
            sensitivity,
            "default_reference",
            sorted(method for method in sensitivity.method.unique() if method != "default_reference"),
            "final_sensitivity",
        )

    trace_episodes, trace_runtime = collect_runs(
        ROOT / "results" / "raw_logs" / "trace_formal", "trace_formal__train_combined__"
    )
    trace = seed_aggregate(trace_episodes)
    _write_core("final_trace", trace)
    _all_pairwise(
        trace,
        "b4_budget_state",
        ["b3_lagrangian", "b5_fixed_local", "cdba", "cdba_discrete"],
        "final_trace",
    )
    _anomalies(trace, "final_trace")
    trace_runtime.to_csv(OUT / "final_trace_runtime.csv", index=False)

    generalization_episodes, generalization_runtime = collect_runs(
        ROOT / "results" / "raw_logs" / "trace_generalization", "trace_generalization__"
    )
    generalization = seed_aggregate(generalization_episodes)
    _write_core("final_generalization", generalization)
    generalization_tests = []
    for variant in sorted(generalization.variant.unique()):
        subset = generalization[generalization.variant == variant]
        for candidate in ("cdba", "cdba_discrete"):
            comparison = paired_comparisons(subset, candidate, "b4_budget_state")
            comparison["training_variant"] = variant
            generalization_tests.append(comparison)
    generalization_tests = pd.concat(generalization_tests, ignore_index=True)
    generalization_tests["q_value_bh_global"] = benjamini_hochberg(
        generalization_tests.p_value
    )
    generalization_tests.to_csv(
        OUT / "final_generalization_paired_tests.csv", index=False
    )
    generalization_runtime.to_csv(OUT / "final_generalization_runtime.csv", index=False)
    print("final tables written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
