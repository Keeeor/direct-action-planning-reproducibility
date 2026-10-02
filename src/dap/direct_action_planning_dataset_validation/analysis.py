from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dap.utils.artifacts import sha256_file, write_json


METRIC_DIRECTIONS = {
    "discounted_return": 1.0,
    "completion_ratio": 1.0,
    "slo_violation_rate": -1.0,
    "total_cost": -1.0,
    "queue_area": -1.0,
}
COMPARATORS = (
    "learned_value_true_transition",
    "original_learned_model",
    "structured_dap",
    "structured_dap_refresh",
)
BASELINE = "b4_budget_state"


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _paired_test(
    differences: np.ndarray,
    *,
    seed: int,
    bootstrap_draws: int,
) -> dict[str, float | int]:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(differences, size=(bootstrap_draws, len(differences)), replace=True)
    bootstrap = sampled.mean(axis=1)
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(differences).pvalue)
    standard_deviation = float(np.std(differences, ddof=1))
    return {
        "n": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "paired_effect_dz": (
            float(np.mean(differences) / standard_deviation)
            if standard_deviation > 0
            else 0.0
        ),
        "wilcoxon_p": p_value,
        "wins": int(np.sum(differences > 1.0e-10)),
        "ties": int(np.sum(np.abs(differences) <= 1.0e-10)),
        "losses": int(np.sum(differences < -1.0e-10)),
    }


def _verify_manifests(result_root: Path) -> dict[str, object]:
    manifests = sorted(result_root.glob("*/*/manifest.json"))
    failures = sorted(result_root.glob("*/*/failure.json"))
    artifact_mismatches: list[str] = []
    code_hashes: set[str] = set()
    input_hashes: dict[str, set[str]] = {}
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        code_hashes.add(str(manifest["code_sha256"]))
        for name, expected in manifest["artifacts"].items():
            actual = sha256_file(path.parent / name)
            if actual != expected:
                artifact_mismatches.append(str(path.parent / name))
        dataset = manifest["run_id"].split("__")[1]
        input_hashes.setdefault(dataset, set()).add(
            json.dumps(manifest["input_sha256"], sort_keys=True)
        )
    ledger = json.loads((result_root / "FINAL_TEST_LEDGER.json").read_text(encoding="utf-8"))
    return {
        "manifest_count": len(manifests),
        "failure_count": len(failures),
        "artifact_mismatches": artifact_mismatches,
        "code_hash_count": len(code_hashes),
        "input_hash_versions": {key: len(value) for key, value in input_hashes.items()},
        "ledger": ledger,
        "passed": (
            len(manifests) == 30
            and not failures
            and not artifact_mismatches
            and len(code_hashes) == 1
            and all(len(value) == 1 for value in input_hashes.values())
            and ledger.get("status") == "finalized"
            and ledger.get("test_execution_count") == 1
        ),
    }


def _pareto_counts(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(units.dataset.unique()):
        frame = units[units.dataset == dataset]
        pivot = frame.pivot(index=["budget", "seed"], columns="method")
        for method in COMPARATORS:
            return_diff = (
                pivot[("discounted_return", method)]
                - pivot[("discounted_return", BASELINE)]
            )
            cost_diff = pivot[("total_cost", method)] - pivot[("total_cost", BASELINE)]
            dominates = (return_diff >= 0) & (cost_diff <= 0) & (
                (return_diff > 0) | (cost_diff < 0)
            )
            dominated = (return_diff <= 0) & (cost_diff >= 0) & (
                (return_diff < 0) | (cost_diff > 0)
            )
            ties = np.isclose(return_diff, 0) & np.isclose(cost_diff, 0)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "dominates_b4": int(dominates.sum()),
                    "dominated_by_b4": int(dominated.sum()),
                    "ties": int(ties.sum()),
                    "tradeoffs": int((~dominates & ~dominated & ~ties).sum()),
                }
            )
    return pd.DataFrame(rows)


def _gate_summary(
    units: pd.DataFrame,
    integrity: dict[str, object],
    selections: pd.DataFrame,
) -> dict[str, object]:
    output: dict[str, object] = {"integrity": integrity, "datasets": {}}
    for dataset in sorted(units.dataset.unique()):
        frame = units[units.dataset == dataset]
        pivot = frame.pivot(index=["budget", "seed"], columns="method")
        dataset_result: dict[str, object] = {}
        for label, method in (
            ("base_structured", "structured_dap"),
            ("validation_selected_final", "structured_dap_refresh"),
        ):
            return_diff = (
                pivot[("discounted_return", method)]
                - pivot[("discounted_return", BASELINE)]
            )
            completion_diff = (
                pivot[("completion_ratio", method)]
                - pivot[("completion_ratio", BASELINE)]
            )
            slo_diff = (
                pivot[("slo_violation_rate", method)]
                - pivot[("slo_violation_rate", BASELINE)]
            )
            cost_diff = pivot[("total_cost", method)] - pivot[("total_cost", BASELINE)]
            checks = {
                "mean_return_noninferior": bool(return_diff.mean() >= 0),
                "at_least_9_of_15_return_wins_or_ties": bool(
                    np.sum(return_diff >= -1.0e-10) >= 9
                ),
                "completion_guardrail": bool(completion_diff.mean() >= -0.01),
                "slo_guardrail": bool(slo_diff.mean() <= 0.01),
                "not_bought_with_more_cost": bool(cost_diff.mean() <= 0),
                "no_budget_overspend": bool(
                    frame.loc[frame.method == method, "budget_overspend"].max() <= 1.0e-10
                ),
            }
            dataset_result[label] = {
                "pass": bool(all(checks.values()) and integrity["passed"]),
                "checks": checks,
                "mean_return_difference": float(return_diff.mean()),
                "wins_ties_losses": [
                    int(np.sum(return_diff > 1.0e-10)),
                    int(np.sum(np.abs(return_diff) <= 1.0e-10)),
                    int(np.sum(return_diff < -1.0e-10)),
                ],
                "mean_completion_difference": float(completion_diff.mean()),
                "mean_slo_difference": float(slo_diff.mean()),
                "mean_cost_difference": float(cost_diff.mean()),
            }
        dataset_result["refresh_selection"] = selections.loc[
            selections.dataset == dataset, "selected"
        ].value_counts().to_dict()
        output["datasets"][dataset] = dataset_result
    output["joint_selected_final_pass"] = bool(
        all(
            result["validation_selected_final"]["pass"]
            for result in output["datasets"].values()
        )
    )
    output["strict_base_structured_pass"] = bool(
        all(
            result["base_structured"]["pass"]
            for result in output["datasets"].values()
        )
    )
    return output


def run_analysis(
    project_root: str | Path,
    *,
    tier: str = "gate_v1",
    analysis_name: str = "analysis_v1",
    bootstrap_draws: int = 10_000,
) -> Path:
    project_root = Path(project_root).resolve()
    result_root = (
        project_root / "results/direct_action_planning_dataset_validation" / tier
    )
    output = result_root / analysis_name
    if output.exists():
        raise FileExistsError(f"analysis is append-only: {output}")
    output.mkdir(parents=True)
    metric_paths = sorted(result_root.glob("*/*/metrics.csv"))
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    test = metrics.loc[metrics.split == "test"].copy()
    unit_columns = [
        "discounted_return",
        "return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "budget_overspend",
        "queue_area",
        "decision_ms_mean",
    ]
    units = (
        test.groupby(["dataset", "budget", "seed", "method"], as_index=False)[unit_columns]
        .mean()
    )
    overall = (
        test.groupby(["dataset", "method"], as_index=False)[unit_columns]
        .agg(["mean", "std"])
    )
    by_domain_budget = (
        test.groupby(["dataset", "domain", "budget", "method"], as_index=False)[unit_columns]
        .mean()
    )
    units.to_csv(output / "unit_metrics.csv", index=False)
    inference_units = (
        units.groupby(["dataset", "seed", "method"], as_index=False)[unit_columns]
        .mean()
    )
    inference_units.to_csv(output / "seed_block_metrics.csv", index=False)
    overall.to_csv(output / "overall_summary.csv", index=False)
    by_domain_budget.to_csv(output / "domain_budget_summary.csv", index=False)

    comparisons = []
    for dataset in sorted(inference_units.dataset.unique()):
        frame = inference_units.loc[inference_units.dataset == dataset]
        pivot = frame.pivot(index="seed", columns="method")
        for method_index, method in enumerate(COMPARATORS):
            for metric_index, (metric, direction) in enumerate(METRIC_DIRECTIONS.items()):
                raw_difference = pivot[(metric, method)] - pivot[(metric, BASELINE)]
                favorable = direction * raw_difference.to_numpy()
                row = _paired_test(
                    favorable,
                    seed=20260804 + method_index * 101 + metric_index,
                    bootstrap_draws=bootstrap_draws,
                )
                row.update(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline": BASELINE,
                        "metric": metric,
                        "direction": "higher_is_better" if direction > 0 else "lower_is_better",
                        "raw_method_minus_b4": float(raw_difference.mean()),
                    }
                )
                comparisons.append(row)
    comparisons_frame = pd.DataFrame(comparisons)
    comparisons_frame["bh_q"] = np.nan
    for dataset, indices in comparisons_frame.groupby("dataset").groups.items():
        comparisons_frame.loc[indices, "bh_q"] = _bh_adjust(
            comparisons_frame.loc[indices, "wilcoxon_p"].to_numpy()
        )
    comparisons_frame.to_csv(output / "paired_comparisons.csv", index=False)

    pareto = _pareto_counts(units)
    pareto.to_csv(output / "pareto_counts.csv", index=False)
    selections = []
    for selection_path in sorted(result_root.glob("*/*/selection.json")):
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        run_id = selection_path.parent.name
        _, dataset, budget_label, seed_label = run_id.split("__")
        selections.append(
            {
                "dataset": dataset,
                "budget": float(budget_label.removeprefix("b").replace("p", ".")),
                "seed": int(seed_label.removeprefix("s")),
                "selected": selection["selected"],
            }
        )
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(output / "refresh_selections.csv", index=False)
    integrity = _verify_manifests(result_root)
    gate = _gate_summary(units, integrity, selection_frame)
    write_json(output / "integrity.json", integrity)
    write_json(output / "continuation_gate.json", gate)

    artifacts = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "schema": "dap.dap_dataset_validation.analysis.v1",
        "source_tier": tier,
        "source_run_count": len(metric_paths),
        "test_episode_rows": int(len(test)),
        "paired_units_per_dataset": 5,
        "inference_unit": "training seed after averaging registered budgets and domains",
        "descriptive_gate_units_per_dataset": 15,
        "bootstrap_draws": bootstrap_draws,
        "bh_family": "20 planned method-metric comparisons separately within each dataset",
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
    }
    write_json(output / "manifest.json", manifest)
    return output
