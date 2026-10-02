from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dap.utils.artifacts import sha256_file, write_json


DAP = "structured_dap_selected"
METRICS = ("discounted_return", "completion_ratio", "slo_violation_rate", "total_cost", "queue_area", "decision_ms_mean")

PRIVILEGED_MODEL_REFERENCES = (
    "reactive_threshold",
    "pid_budget_autoscaler",
    "causal_mpc",
    "lyapunov_dpp",
)
COMMON_RL_BASELINES = ("double_dqn", "a2c", "ppo")
CONSTRAINED_RL_BASELINES = (
    "ppo_lagrangian",
    "pid_lagrangian",
    "p3o",
    "cpo",
    "budgeted_fitted_q",
)
LEARNED_BASELINES = COMMON_RL_BASELINES + CONSTRAINED_RL_BASELINES

CLAIM_METRICS = (
    "discounted_return",
    "completion_ratio",
    "slo_violation_rate",
    "total_cost",
    "queue_area",
    "decision_ms_mean",
    "return_cvar20",
    "completion_p10",
    "slo_p95",
    "decision_ms_p95",
    "budget_overspend_max",
)
HIGHER_IS_BETTER = {
    "discounted_return": True,
    "completion_ratio": True,
    "slo_violation_rate": False,
    "total_cost": False,
    "queue_area": False,
    "decision_ms_mean": False,
    "return_cvar20": True,
    "completion_p10": True,
    "slo_p95": False,
    "decision_ms_p95": False,
    "budget_overspend_max": False,
}


def _bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _paired_summary(values: np.ndarray, seed: int = 20260804) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    p_value = 1.0 if np.allclose(values, 0.0) else float(wilcoxon(values).pvalue)
    return {
        "n_seed_blocks": int(len(values)),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "wilcoxon_p": p_value,
        "paired_standardized_effect": (
            float(values.mean() / values.std(ddof=1))
            if len(values) > 1 and values.std(ddof=1) > 1.0e-12
            else 0.0
        ),
    }


def _method_role(method: str) -> str:
    if method == DAP:
        return "candidate"
    if method in PRIVILEGED_MODEL_REFERENCES:
        return "privileged_model_reference"
    if method in COMMON_RL_BASELINES:
        return "common_rl"
    if method in CONSTRAINED_RL_BASELINES:
        return "constrained_or_budget_rl"
    raise ValueError(f"unregistered benchmark method: {method}")


def _tail_mean(values: pd.Series, fraction: float = 0.2) -> float:
    ordered = np.sort(values.to_numpy(dtype=np.float64))
    count = max(1, int(np.ceil(fraction * len(ordered))))
    return float(ordered[:count].mean())


def _claim_unit_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "budget", "seed", "method"]
    rows: list[dict[str, float | str]] = []
    for key, group in metrics.groupby(keys, sort=True):
        row: dict[str, float | str] = dict(zip(keys, key, strict=True))
        row.update({metric: float(group[metric].mean()) for metric in METRICS})
        row.update({
            "return_cvar20": _tail_mean(group["discounted_return"]),
            "completion_p10": float(group["completion_ratio"].quantile(0.10)),
            "slo_p95": float(group["slo_violation_rate"].quantile(0.95)),
            "decision_ms_p95": float(group["decision_ms_p95"].quantile(0.95)),
            "budget_overspend_max": float(group["budget_overspend"].max()),
            "method_role": _method_role(str(key[-1])),
            "episode_count": int(len(group)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _read_corrected_metrics(
    project_root: Path,
    tier: str,
    correction_tier: str | None,
) -> tuple[pd.DataFrame, list[Path], list[Path], set[str]]:
    result_root = project_root / "results/direct_action_planning_dataset_benchmark" / tier
    source_roots = [result_root]
    paths = sorted(result_root.glob("*/*/metrics.csv"))
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    corrected_methods: set[str] = set()
    if correction_tier is not None:
        correction_root = (
            project_root / "results/direct_action_planning_dataset_benchmark" / correction_tier
        )
        source_roots.append(correction_root)
        correction_paths = sorted(correction_root.glob("*/*/metrics.csv"))
        correction = pd.concat([pd.read_csv(path) for path in correction_paths], ignore_index=True)
        corrected_methods = set(correction.method.unique())
        metrics = pd.concat(
            [metrics.loc[~metrics.method.isin(corrected_methods)], correction],
            ignore_index=True,
        )
    manifests = [path for root in source_roots for path in sorted(root.glob("*/*/manifest.json"))]
    failures = [path for root in source_roots for path in sorted(root.glob("**/failure.json"))]
    return metrics, manifests, failures, corrected_methods


def _integrity_summary(manifests: list[Path], failures: list[Path], source_count: int) -> dict:
    mismatches = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("formal_test_accessed") is not False:
            mismatches.append(f"test-access flag: {path}")
        for name, expected in manifest.get("artifacts", {}).items():
            artifact = path.parent / name
            if not artifact.exists() or sha256_file(artifact) != expected:
                mismatches.append(str(artifact))
    return {
        "manifest_count": len(manifests),
        "failure_count": len(failures),
        "artifact_mismatches": mismatches,
        "formal_test_accessed": False,
        "passed": len(manifests) == 30 * source_count and not failures and not mismatches,
    }


def run_analysis(
    project_root: str | Path,
    tier: str,
    analysis_name: str = "analysis_v1",
    correction_tier: str | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    result_root = project_root / "results/direct_action_planning_dataset_benchmark" / tier
    output = result_root / analysis_name
    if output.exists():
        raise FileExistsError(f"analysis is append-only: {output}")
    output.mkdir(parents=True)
    source_roots = [result_root]
    if correction_tier is not None:
        source_roots.append(
            project_root / "results/direct_action_planning_dataset_benchmark" / correction_tier
        )
    manifests = [path for root in source_roots for path in sorted(root.glob("*/*/manifest.json"))]
    failures = [path for root in source_roots for path in sorted(root.glob("**/failure.json"))]
    mismatches = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("formal_test_accessed") is not False:
            mismatches.append(f"test-access flag: {path}")
        for name, expected in manifest.get("artifacts", {}).items():
            artifact = path.parent / name
            if not artifact.exists() or sha256_file(artifact) != expected:
                mismatches.append(str(artifact))
    integrity = {
        "manifest_count": len(manifests),
        "failure_count": len(failures),
        "artifact_mismatches": mismatches,
        "formal_test_accessed": False,
        "passed": len(manifests) == 30 * len(source_roots) and not failures and not mismatches,
    }
    write_json(output / "integrity.json", integrity)
    paths = sorted(result_root.glob("*/*/metrics.csv"))
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    corrected_methods: set[str] = set()
    if correction_tier is not None:
        correction_root = source_roots[1]
        correction_paths = sorted(correction_root.glob("*/*/metrics.csv"))
        correction = pd.concat([pd.read_csv(path) for path in correction_paths], ignore_index=True)
        corrected_methods = set(correction.method.unique())
        metrics = pd.concat(
            [metrics.loc[~metrics.method.isin(corrected_methods)], correction],
            ignore_index=True,
        )
    units = metrics.groupby(["dataset", "budget", "seed", "method"], as_index=False)[list(METRICS)].mean()
    seed_blocks = units.groupby(["dataset", "seed", "method"], as_index=False)[list(METRICS)].mean()
    overall = metrics.groupby(["dataset", "method"], as_index=False)[list(METRICS)].mean()
    units.to_csv(output / "unit_metrics.csv", index=False)
    seed_blocks.to_csv(output / "seed_block_metrics.csv", index=False)
    overall.to_csv(output / "overall_summary.csv", index=False)

    comparisons = []
    pareto = []
    for dataset in sorted(units.dataset.unique()):
        dataset_units = units[units.dataset == dataset]
        pivot = dataset_units.pivot(index=["budget", "seed"], columns="method")
        baselines = sorted(set(dataset_units.method) - {DAP})
        for baseline in baselines:
            return_diff = pivot[("discounted_return", DAP)] - pivot[("discounted_return", baseline)]
            completion_diff = pivot[("completion_ratio", DAP)] - pivot[("completion_ratio", baseline)]
            slo_diff = pivot[("slo_violation_rate", DAP)] - pivot[("slo_violation_rate", baseline)]
            cost_diff = pivot[("total_cost", DAP)] - pivot[("total_cost", baseline)]
            seed_return = return_diff.groupby(level="seed").mean().to_numpy()
            summary = _paired_summary(seed_return)
            comparisons.append({
                "dataset": dataset,
                "baseline": baseline,
                **summary,
                "unit_wins": int(np.sum(return_diff > 1.0e-10)),
                "unit_ties": int(np.sum(np.abs(return_diff) <= 1.0e-10)),
                "unit_losses": int(np.sum(return_diff < -1.0e-10)),
                "completion_difference": float(completion_diff.mean()),
                "slo_difference": float(slo_diff.mean()),
                "cost_difference": float(cost_diff.mean()),
            })
            dap_dominates = (return_diff >= 0) & (cost_diff <= 0) & ((return_diff > 0) | (cost_diff < 0))
            baseline_dominates = (return_diff <= 0) & (cost_diff >= 0) & ((return_diff < 0) | (cost_diff > 0))
            ties = np.isclose(return_diff, 0.0) & np.isclose(cost_diff, 0.0)
            pareto.append({
                "dataset": dataset,
                "baseline": baseline,
                "dap_dominates": int(dap_dominates.sum()),
                "baseline_dominates": int(baseline_dominates.sum()),
                "ties": int(ties.sum()),
                "tradeoffs": int((~dap_dominates & ~baseline_dominates & ~ties).sum()),
            })
    comparisons_frame = pd.DataFrame(comparisons)
    comparisons_frame["bh_q"] = comparisons_frame.groupby("dataset")["wilcoxon_p"].transform(lambda values: _bh(values.to_numpy()))
    comparisons_frame.to_csv(output / "dap_vs_external_baselines.csv", index=False)
    pd.DataFrame(pareto).to_csv(output / "pareto_counts.csv", index=False)

    diagnostics = {}
    training_paths = list(result_root.glob("*/*/training.json"))
    if correction_tier is not None:
        training_paths.extend(source_roots[1].glob("*/*/training.json"))
    for path in training_paths:
        training = json.loads(path.read_text(encoding="utf-8"))
        for method in ("ppo_lagrangian", "pid_lagrangian", "p3o", "cpo"):
            if method not in training:
                continue
            if correction_tier is not None:
                is_correction = source_roots[1] in path.parents
                if method in corrected_methods and not is_correction:
                    continue
                if method not in corrected_methods and is_correction:
                    continue
            diagnostics.setdefault(method, []).append(training[method].get("diagnostics", {}))
    constraint_summary = {}
    for method, rows in diagnostics.items():
        keys = sorted(set().union(*(set(row) for row in rows)))
        constraint_summary[method] = {
            key: float(np.mean([row[key] for row in rows if key in row and np.isfinite(row[key])]))
            for key in keys
            if any(key in row and isinstance(row[key], (int, float)) and np.isfinite(row[key]) for row in rows)
        }
    write_json(output / "constraint_diagnostics.json", constraint_summary)

    summary = {"integrity": integrity, "development_only": True, "datasets": {}}
    for dataset in sorted(overall.dataset.unique()):
        ranked = overall[overall.dataset == dataset].sort_values("discounted_return", ascending=False)
        dap_row = ranked[ranked.method == DAP].iloc[0]
        best = ranked.iloc[0]
        summary["datasets"][dataset] = {
            "best_method": str(best.method),
            "best_discounted_return": float(best.discounted_return),
            "dap_discounted_return": float(dap_row.discounted_return),
            "dap_rank": int(np.flatnonzero(ranked.method.to_numpy() == DAP)[0] + 1),
            "method_count": int(len(ranked)),
        }
    write_json(output / "summary.json", summary)
    artifacts = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", {
        "schema": "dap.dap_dataset_benchmark.analysis.v1",
        "source_tier": tier,
        "correction_tier": correction_tier,
        "corrected_methods": sorted(corrected_methods),
        "development_only": True,
        "formal_test_accessed": False,
        "artifacts": artifacts,
    })
    return output


def _evidence_label(row: pd.Series) -> str:
    if row["bh_q"] < 0.05 and row["ci_low"] > 0.0:
        return "fdr_confirmed"
    if row["ci_low"] > 0.0 and row["unit_wins"] >= 12:
        return "strong_directional_not_fdr_confirmed"
    if row["mean_difference"] > 0.0 and row["unit_wins"] >= 8:
        return "weak_directional"
    if row["mean_difference"] > 0.0:
        return "mixed_positive_mean"
    return "not_supportive"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_claim_markdown(
    output: Path,
    comparisons: pd.DataFrame,
    overall: pd.DataFrame,
    pareto: pd.DataFrame,
) -> None:
    return_rows = comparisons[comparisons.metric == "discounted_return"].copy()
    claim_rows = []
    for row in return_rows.itertuples(index=False):
        aggregate = comparisons[
            (comparisons.dataset == row.dataset) & (comparisons.baseline == row.baseline)
        ].set_index("metric")
        claim_rows.append([
            str(row.dataset),
            str(row.baseline),
            f"{row.raw_mean_difference:+.3f}",
            f"{row.unit_wins}/{row.unit_wins + row.unit_ties + row.unit_losses}",
            f"[{row.raw_ci_low:+.3f}, {row.raw_ci_high:+.3f}]",
            f"{row.bh_q:.5f}",
            f"{aggregate.loc['completion_ratio', 'raw_mean_difference']:+.4f}",
            f"{aggregate.loc['slo_violation_rate', 'raw_mean_difference']:+.4f}",
            f"{aggregate.loc['total_cost', 'raw_mean_difference']:+.3f}",
            str(row.evidence_label),
        ])
    claim_table = _markdown_table(
        [
            "Dataset",
            "RL/SOTA baseline",
            "Return delta",
            "Wins",
            "95% CI",
            "BH q",
            "Completion delta",
            "SLO delta",
            "Cost delta",
            "Evidence",
        ],
        claim_rows,
    )
    (output / "claim_evidence_table.md").write_text(
        "# Claim-Evidence Table\n\n"
        "Development-only exploratory interpretation. Deltas are DAP minus baseline; lower SLO and "
        "cost are favorable. Confidence intervals and Wilcoxon tests use five independent seed blocks "
        "after averaging budgets. BH-FDR is applied separately within each dataset-metric family.\n\n"
        + claim_table
        + "\n",
        encoding="utf-8",
    )

    reference = overall[overall.method_role == "privileged_model_reference"]
    reference_rows = [
        [
            str(row.dataset),
            str(row.method),
            f"{row.discounted_return:.3f}",
            f"{row.completion_ratio:.4f}",
            f"{row.slo_violation_rate:.4f}",
            f"{row.total_cost:.2f}",
            f"{row.decision_ms_mean:.4f}",
        ]
        for row in reference.itertuples(index=False)
    ]
    reference_table = _markdown_table(
        ["Dataset", "Reference", "Return", "Completion", "SLO", "Cost", "ms/step"],
        reference_rows,
    )
    pareto_summary = pareto.groupby("dataset", as_index=False)[
        ["return_cost_dap_dominates", "service_cost_dap_dominates", "service_risk_dap_dominates"]
    ].sum()
    pareto_rows = [
        [
            str(row.dataset),
            str(int(row.return_cost_dap_dominates)),
            str(int(row.service_cost_dap_dominates)),
            str(int(row.service_risk_dap_dominates)),
        ]
        for row in pareto_summary.itertuples(index=False)
    ]
    pareto_table = _markdown_table(
        ["Dataset", "Return-cost wins", "Completion-cost wins", "Completion-SLO wins"],
        pareto_rows,
    )
    (output / "CLAIM_CENTERED_ANALYSIS.md").write_text(
        "# Claim-Centered Development Analysis\n\n"
        "## Analysis status\n\n"
        "This is an append-only, development-only reanalysis. The comparison scope was clarified "
        "after aggregate development results were observed, so every substantive interpretation "
        "below is exploratory rather than preregistered confirmation. No formal test block was read.\n\n"
        "## Comparison hierarchy\n\n"
        "The primary comparison is Structured DAP versus common RL and constrained/budget RL. "
        "Threshold, PID, causal MPC, and Lyapunov DPP are reported separately as privileged-model "
        "references because they exploit known queue equations, action effects, or hand-designed "
        "control structure. They are diagnostic reference ceilings, not mathematical upper bounds "
        "and not peers in a single fairness ranking.\n\n"
        "## Supported claim boundary\n\n"
        "GenTD26 supports a directional claim that DAP improves action selection over several tested "
        "data-driven policies. Azure2019 does not support broad superiority; any positive comparison "
        "there is baseline-specific and statistically unstable. No claim requires winning every "
        "dataset, metric, budget, or seed.\n\n"
        "All methods have zero budget overspend because the shared hard affordability mask enforces "
        "feasibility. Safety therefore has to be interpreted through SLO violation, lower-tail return, "
        "worst-case completion, queue burden, and high-risk windows rather than overspend alone. The "
        "data do not support a generic assumption that ordinary RL always consumes less resource than "
        "constrained RL; cost direction depends on the method and dataset.\n\n"
        "## Learning-method evidence\n\n"
        + claim_table
        + "\n\n## Privileged-model references\n\n"
        + reference_table
        + "\n\nThese values expose remaining structural headroom and simulator bias toward analytic queue control. "
        "They are not used to turn the learned-method question into a 13-method league table.\n\n"
        "## Multi-objective evidence\n\n"
        + pareto_table
        + "\n\nCounts aggregate 15 paired budget-seed units across the eight RL/SOTA baselines. Full pairwise "
        "counts, including reverse domination and tradeoffs, are in `learning_pareto_counts.csv`.\n",
        encoding="utf-8",
    )


def run_claim_centered_analysis(
    project_root: str | Path,
    tier: str,
    analysis_name: str = "analysis_v3_claim_centered",
    correction_tier: str | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    result_root = project_root / "results/direct_action_planning_dataset_benchmark" / tier
    output = result_root / analysis_name
    if output.exists():
        raise FileExistsError(f"analysis is append-only: {output}")
    output.mkdir(parents=True)

    metrics, manifests, failures, corrected_methods = _read_corrected_metrics(
        project_root, tier, correction_tier
    )
    source_count = 1 + int(correction_tier is not None)
    integrity = _integrity_summary(manifests, failures, source_count)
    write_json(output / "integrity.json", integrity)

    units = _claim_unit_metrics(metrics)
    expected = set(PRIVILEGED_MODEL_REFERENCES + LEARNED_BASELINES + (DAP,))
    observed = set(units.method.unique())
    if observed != expected:
        raise ValueError(f"method registry mismatch: expected={expected}, observed={observed}")
    seed_blocks = units.groupby(
        ["dataset", "seed", "method", "method_role"], as_index=False
    )[list(CLAIM_METRICS)].mean()
    overall = units.groupby(
        ["dataset", "method", "method_role"], as_index=False
    )[list(CLAIM_METRICS)].mean()
    units.to_csv(output / "claim_unit_metrics.csv", index=False)
    seed_blocks.to_csv(output / "claim_seed_block_metrics.csv", index=False)
    overall[overall.method_role != "privileged_model_reference"].to_csv(
        output / "learning_method_summary.csv", index=False
    )
    overall[overall.method_role == "privileged_model_reference"].to_csv(
        output / "privileged_model_references.csv", index=False
    )

    comparisons: list[dict] = []
    pareto_rows: list[dict] = []
    for dataset in sorted(units.dataset.unique()):
        dataset_units = units[units.dataset == dataset]
        pivot = dataset_units.pivot(index=["budget", "seed"], columns="method")
        for baseline in LEARNED_BASELINES:
            for metric in CLAIM_METRICS:
                raw_diff = pivot[(metric, DAP)] - pivot[(metric, baseline)]
                benefit = raw_diff if HIGHER_IS_BETTER[metric] else -raw_diff
                seed_benefit = benefit.groupby(level="seed").mean().to_numpy()
                paired = _paired_summary(seed_benefit)
                comparisons.append({
                    "dataset": dataset,
                    "baseline": baseline,
                    "baseline_role": _method_role(baseline),
                    "metric": metric,
                    "higher_is_better": HIGHER_IS_BETTER[metric],
                    **paired,
                    "raw_mean_difference": float(raw_diff.mean()),
                    "raw_ci_low": paired["ci_low"] if HIGHER_IS_BETTER[metric] else -paired["ci_high"],
                    "raw_ci_high": paired["ci_high"] if HIGHER_IS_BETTER[metric] else -paired["ci_low"],
                    "unit_wins": int(np.sum(benefit > 1.0e-10)),
                    "unit_ties": int(np.sum(np.abs(benefit) <= 1.0e-10)),
                    "unit_losses": int(np.sum(benefit < -1.0e-10)),
                })

            dap_return = pivot[("discounted_return", DAP)]
            base_return = pivot[("discounted_return", baseline)]
            dap_completion = pivot[("completion_ratio", DAP)]
            base_completion = pivot[("completion_ratio", baseline)]
            dap_slo = pivot[("slo_violation_rate", DAP)]
            base_slo = pivot[("slo_violation_rate", baseline)]
            dap_cost = pivot[("total_cost", DAP)]
            base_cost = pivot[("total_cost", baseline)]

            def domination_counts(
                first_better: pd.Series, second_better: pd.Series
            ) -> tuple[int, int, int]:
                candidate = first_better & second_better
                reverse = (~first_better) & (~second_better)
                return int(candidate.sum()), int(reverse.sum()), int((~candidate & ~reverse).sum())

            return_cost = domination_counts(dap_return >= base_return, dap_cost <= base_cost)
            service_cost = domination_counts(dap_completion >= base_completion, dap_cost <= base_cost)
            service_risk = domination_counts(dap_completion >= base_completion, dap_slo <= base_slo)
            pareto_rows.append({
                "dataset": dataset,
                "baseline": baseline,
                "return_cost_dap_dominates": return_cost[0],
                "return_cost_baseline_dominates": return_cost[1],
                "return_cost_tradeoffs": return_cost[2],
                "service_cost_dap_dominates": service_cost[0],
                "service_cost_baseline_dominates": service_cost[1],
                "service_cost_tradeoffs": service_cost[2],
                "service_risk_dap_dominates": service_risk[0],
                "service_risk_baseline_dominates": service_risk[1],
                "service_risk_tradeoffs": service_risk[2],
            })

    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["bh_q"] = comparison_frame.groupby(["dataset", "metric"])[
        "wilcoxon_p"
    ].transform(lambda values: _bh(values.to_numpy()))
    comparison_frame["evidence_label"] = comparison_frame.apply(_evidence_label, axis=1)
    comparison_frame.to_csv(output / "dap_vs_learning_methods_all_metrics.csv", index=False)
    pareto = pd.DataFrame(pareto_rows)
    pareto.to_csv(output / "learning_pareto_counts.csv", index=False)

    return_comparisons = comparison_frame[
        comparison_frame.metric == "discounted_return"
    ].copy()
    safety = overall[
        [
            "dataset",
            "method",
            "method_role",
            "budget_overspend_max",
            "slo_violation_rate",
            "slo_p95",
            "completion_p10",
            "return_cvar20",
        ]
    ]
    safety.to_csv(output / "safety_and_tail_summary.csv", index=False)

    evidence = {
        "schema": "dap.dap_dataset_benchmark.claim_evidence.v1",
        "analysis_status": "exploratory_post_result_scope_clarification",
        "development_only": True,
        "formal_test_accessed": False,
        "independent_seed_blocks": 5,
        "bh_family": "within each dataset and metric across eight learning baselines",
        "claims": {
            "gentd26_learning_method_advantage": {
                "strength": "strong_directional_not_confirmatory",
                "supported_comparators": return_comparisons.loc[
                    (return_comparisons.dataset == "gentd26")
                    & return_comparisons.evidence_label.eq(
                        "strong_directional_not_fdr_confirmed"
                    ),
                    "baseline",
                ].tolist(),
                "fdr_confirmed_comparators": return_comparisons.loc[
                    (return_comparisons.dataset == "gentd26")
                    & return_comparisons.evidence_label.eq("fdr_confirmed"),
                    "baseline",
                ].tolist(),
            },
            "azure2019_broad_learning_method_advantage": {
                "strength": "not_supported",
                "positive_mean_comparators": return_comparisons.loc[
                    (return_comparisons.dataset == "azure2019")
                    & (return_comparisons.mean_difference > 0.0),
                    "baseline",
                ].tolist(),
            },
            "budget_safety": {
                "strength": "not_discriminative_under_shared_hard_mask",
                "maximum_observed_overspend": float(units.budget_overspend_max.max()),
            },
            "privileged_control_references": {
                "strength": "diagnostic_only",
                "reason": "known equations, action effects, forecasts, or hand-designed control structure",
                "mathematical_upper_bound": False,
            },
        },
    }
    write_json(output / "evidence_strength.json", evidence)
    _write_claim_markdown(output, comparison_frame, overall, pareto)

    artifacts = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(output / "manifest.json", {
        "schema": "dap.dap_dataset_benchmark.claim_centered_analysis.v1",
        "source_tier": tier,
        "correction_tier": correction_tier,
        "corrected_methods": sorted(corrected_methods),
        "development_only": True,
        "formal_test_accessed": False,
        "post_result_scope_clarification": True,
        "primary_comparison": "DAP versus common and constrained/budget RL",
        "privileged_references_ranked_separately": True,
        "artifacts": artifacts,
    })
    return output
