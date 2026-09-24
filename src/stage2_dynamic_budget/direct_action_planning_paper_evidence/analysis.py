from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import yaml

from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json


CORE_CONTRASTS = {
    "direct_planning": ("dap_full", "dap_policy_distilled"),
    "analytic_structure": ("dap_full", "dap_full_transition"),
    "decision_aware_forecast": ("dap_full", "dap_state_only_forecast"),
    "budget_horizon_conditioning": ("dap_full", "dap_no_budget_horizon"),
    "continuation_value": ("dap_full", "dap_immediate_reward"),
    "anchored_value_refresh": ("dap_refresh_anchor_raw", "dap_direct_base"),
    "anchor_retention": ("dap_refresh_anchor_raw", "dap_refresh_no_anchor"),
    "learned_forecast": ("dap_full", "dap_persistence_forecast"),
}
METRIC_DIRECTIONS = {
    "discounted_return": True,
    "completion_ratio": True,
    "slo_violation_rate": False,
    "total_cost": False,
    "queue_area": False,
    "return_cvar20": True,
    "completion_p10": True,
    "slo_p95": False,
    "decision_ms_mean": False,
    "decision_ms_p95": False,
}
SENSITIVITY_REFERENCE = {
    "hidden_dim": 64.0,
    "collection_episodes_per_domain": 24.0,
    "anchor_weight": 0.05,
    "forecast_multiplier": 1.0,
    "gamma": 0.99,
}


def _hash_bytes(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_manifest(run_dir: Path, expected_schema: str) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"incomplete run: {run_dir}")
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"unexpected schema in {run_dir}")
    if manifest.get("formal_test_accessed") is not False:
        raise ValueError(f"formal test access detected: {run_dir}")
    for name, expected in manifest.get("artifacts", {}).items():
        if _hash_bytes(run_dir / name) != expected:
            raise ValueError(f"artifact hash mismatch: {run_dir / name}")
    return manifest


def _bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _paired_effect(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.std(values, ddof=1))
    return float(np.mean(values) / standard_deviation) if standard_deviation > 0 else 0.0


def _bh(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(p_values) - reverse_rank + 1
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = running
    return np.clip(adjusted, 0.0, 1.0)


def _tail_metrics(frame: pd.DataFrame) -> pd.Series:
    count = max(int(math.ceil(0.2 * len(frame))), 1)
    return pd.Series(
        {
            "return_cvar20": float(np.sort(frame.discounted_return.to_numpy())[:count].mean()),
            "completion_p10": float(frame.completion_ratio.quantile(0.10)),
            "slo_p95": float(frame.slo_violation_rate.quantile(0.95)),
        }
    )


def load_core(
    project_root: Path,
    tier: str,
    *,
    registered_seeds: tuple[int, ...],
    registered_datasets: tuple[str, ...],
    registered_budgets: tuple[float, ...],
    expected_schema: str,
    expected_runs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_root = project_root / "results/direct_action_planning_paper_evidence" / tier
    episodes: list[pd.DataFrame] = []
    steps: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    seen_cells: set[tuple[str, float, int]] = set()
    for run_dir in sorted(path.parent for path in result_root.glob("*/*/manifest.json")):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if int(config["seed"]) not in registered_seeds:
            continue
        dataset = str(config["dataset"])
        budget = float(config["budget"])
        if dataset not in registered_datasets or not any(
            np.isclose(budget, registered) for registered in registered_budgets
        ):
            raise ValueError(f"unregistered core cell in {run_dir}")
        cell = (dataset, budget, int(config["seed"]))
        if cell in seen_cells:
            raise ValueError(f"duplicate registered core cell in {run_dir}: {cell}")
        seen_cells.add(cell)
        _verify_manifest(run_dir, expected_schema)
        episode = pd.read_csv(run_dir / "metrics.csv")
        step = pd.read_csv(run_dir / "steps.csv.gz")
        episode["training_seed"] = int(config["seed"])
        step["training_seed"] = int(config["seed"])
        if episode.select_dtypes("number").isna().any().any():
            raise ValueError(f"non-finite episode metric: {run_dir}")
        episodes.append(episode)
        steps.append(step)
        row = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
        row.update(
            {
                "dataset": config["dataset"],
                "budget": float(config["budget"]),
                "seed": int(config["seed"]),
                "run_dir": str(run_dir.relative_to(project_root)),
            }
        )
        diagnostics.append(row)
    expected_cells = {
        (str(dataset), float(budget), int(seed))
        for dataset in registered_datasets
        for budget in registered_budgets
        for seed in registered_seeds
    }
    if seen_cells != expected_cells or len(seen_cells) != expected_runs:
        raise ValueError(
            "registered core grid mismatch: "
            f"missing={sorted(expected_cells - seen_cells)}, "
            f"extra={sorted(seen_cells - expected_cells)}"
        )
    return (
        pd.concat(episodes, ignore_index=True),
        pd.concat(steps, ignore_index=True),
        pd.DataFrame(diagnostics),
    )


def core_tables(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouping = ["dataset", "budget", "training_seed", "method"]
    means = episodes.groupby(grouping, as_index=False)[
        [
            "discounted_return",
            "completion_ratio",
            "slo_violation_rate",
            "total_cost",
            "budget_overspend",
            "queue_area",
            "decision_ms_mean",
            "decision_ms_p95",
        ]
    ].mean()
    tails = episodes.groupby(grouping).apply(_tail_metrics, include_groups=False).reset_index()
    units = means.merge(tails, on=grouping, validate="one_to_one")
    seed_blocks = units.groupby(["dataset", "training_seed", "method"], as_index=False)[
        list(METRIC_DIRECTIONS)
    ].mean()
    summary = seed_blocks.groupby(["dataset", "method"], as_index=False)[
        list(METRIC_DIRECTIONS)
    ].agg(["mean", "std"])
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return units, seed_blocks, summary


def component_comparisons(seed_blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    decision_rows: list[dict] = []
    seed_counter = 0
    for dataset in sorted(seed_blocks.dataset.unique()):
        subset = seed_blocks[seed_blocks.dataset == dataset]
        full_return_scale = abs(
            float(subset[subset.method == "dap_full"].discounted_return.mean())
        )
        for component, (treatment, control) in CORE_CONTRASTS.items():
            treatment_frame = subset[subset.method == treatment].set_index("training_seed")
            control_frame = subset[subset.method == control].set_index("training_seed")
            for metric, higher_is_better in METRIC_DIRECTIONS.items():
                raw = treatment_frame[metric] - control_frame[metric]
                benefit = raw if higher_is_better else -raw
                ci_low, ci_high = _bootstrap_ci(benefit.to_numpy(), 8100 + seed_counter)
                seed_counter += 1
                rows.append(
                    {
                        "dataset": dataset,
                        "component": component,
                        "treatment": treatment,
                        "control": control,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "n_seed_blocks": len(benefit),
                        "raw_mean_treatment_minus_control": float(raw.mean()),
                        "benefit_mean": float(benefit.mean()),
                        "benefit_ci_low": ci_low,
                        "benefit_ci_high": ci_high,
                        "wilcoxon_p": _wilcoxon(benefit.to_numpy()),
                        "paired_effect": _paired_effect(benefit.to_numpy()),
                        "wins": int((benefit > 1.0e-12).sum()),
                        "ties": int((np.abs(benefit) <= 1.0e-12).sum()),
                        "losses": int((benefit < -1.0e-12).sum()),
                    }
                )
            return_gain = float(
                treatment_frame.discounted_return.mean()
                - control_frame.discounted_return.mean()
            )
            return_wins = int(
                (
                    treatment_frame.discounted_return
                    - control_frame.discounted_return
                    > 1.0e-12
                ).sum()
            )
            completion_delta = float(
                treatment_frame.completion_ratio.mean()
                - control_frame.completion_ratio.mean()
            )
            slo_delta = float(
                treatment_frame.slo_violation_rate.mean()
                - control_frame.slo_violation_rate.mean()
            )
            cost_delta = float(
                treatment_frame.total_cost.mean() - control_frame.total_cost.mean()
            )
            cost_limit = 0.05 * max(float(control_frame.total_cost.mean()), 1.0)
            service_compensation = completion_delta > 0.01 or slo_delta < -0.01
            guardrails = (
                completion_delta >= -0.01
                and slo_delta <= 0.01
                and (cost_delta <= cost_limit or service_compensation)
            )
            relative_effect = return_gain / max(full_return_scale, 1.0e-8)
            supported = (
                dataset == "gentd26"
                and relative_effect >= 0.02
                and return_wins >= 4
                and guardrails
            )
            decision_rows.append(
                {
                    "dataset": dataset,
                    "component": component,
                    "return_gain": return_gain,
                    "relative_return_effect": relative_effect,
                    "return_wins": return_wins,
                    "completion_delta": completion_delta,
                    "slo_delta": slo_delta,
                    "cost_delta": cost_delta,
                    "guardrails_pass": guardrails,
                    "registered_component_support": supported,
                    "decision": "supported" if supported else "not_supported",
                }
            )
    comparison = pd.DataFrame(rows)
    comparison["bh_q"] = _bh(comparison.wilcoxon_p)
    return comparison, pd.DataFrame(decision_rows)


def load_sensitivity(
    project_root: Path,
    tier: str,
    *,
    registered_seeds: tuple[int, ...],
    registered_levels: dict[str, tuple[float, ...]],
    registered_dataset: str,
    registered_budget: float,
    expected_schema: str,
    expected_runs: int,
) -> pd.DataFrame:
    result_root = project_root / "results/direct_action_planning_paper_evidence" / tier
    rows: list[dict] = []
    manifests = list(result_root.glob("*/*/manifest.json"))
    seen_cells: set[tuple[str, float, int]] = set()
    for manifest_path in sorted(manifests):
        run_dir = manifest_path.parent
        _verify_manifest(run_dir, expected_schema)
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if int(config["seed"]) not in registered_seeds:
            raise ValueError(f"unregistered sensitivity seed in {run_dir}")
        factor = str(config["factor"])
        level = float(config["level"])
        if factor not in registered_levels or not any(
            np.isclose(level, registered) for registered in registered_levels[factor]
        ):
            raise ValueError(f"unregistered sensitivity level in {run_dir}")
        if str(config["dataset"]) != registered_dataset or not np.isclose(
            float(config["budget"]), registered_budget
        ):
            raise ValueError(f"unregistered sensitivity dataset/budget in {run_dir}")
        cell = (factor, level, int(config["seed"]))
        if cell in seen_cells:
            raise ValueError(f"duplicate registered sensitivity cell in {run_dir}: {cell}")
        seen_cells.add(cell)
        metrics = pd.read_csv(run_dir / "metrics.csv")
        training = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
        runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
        row = {
            "factor": factor,
            "level": level,
            "seed": int(config["seed"]),
            "discounted_return": float(metrics.discounted_return.mean()),
            "completion_ratio": float(metrics.completion_ratio.mean()),
            "slo_violation_rate": float(metrics.slo_violation_rate.mean()),
            "total_cost": float(metrics.total_cost.mean()),
            "queue_area": float(metrics.queue_area.mean()),
            "decision_ms_mean": float(metrics.decision_ms_mean.mean()),
            "decision_ms_p95": float(metrics.decision_ms_p95.mean()),
            "training_seconds": float(training["training_seconds"]),
            "branch_action_transitions": int(
                training["collection"]["branch_action_transitions"]
            ),
            "parameters": int(
                training["parameters"].get(
                    "full_value", training["parameters"].get("selected_value", 0)
                )
                + training["parameters"]["decision_forecaster"]
            ),
            "elapsed_seconds": float(runtime["elapsed_seconds"]),
            "peak_rss_kib": int(runtime["peak_rss_kib"]),
        }
        rows.append(row)
    expected_cells = {
        (factor, float(level), int(seed))
        for factor, levels in registered_levels.items()
        for level in levels
        for seed in registered_seeds
    }
    if seen_cells != expected_cells or len(seen_cells) != expected_runs:
        raise ValueError(
            "registered sensitivity grid mismatch: "
            f"missing={sorted(expected_cells - seen_cells)}, "
            f"extra={sorted(seen_cells - expected_cells)}"
        )
    return pd.DataFrame(rows)


def sensitivity_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "discounted_return",
        "completion_ratio",
        "slo_violation_rate",
        "total_cost",
        "queue_area",
        "decision_ms_mean",
        "decision_ms_p95",
        "training_seconds",
        "branch_action_transitions",
        "parameters",
        "peak_rss_kib",
    ]
    summary = frame.groupby(["factor", "level"], as_index=False)[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    rows: list[dict] = []
    for factor, reference_level in SENSITIVITY_REFERENCE.items():
        subset = frame[frame.factor == factor]
        reference = subset[np.isclose(subset.level, reference_level)].set_index("seed")
        for level in sorted(subset.level.unique()):
            candidate = subset[np.isclose(subset.level, level)].set_index("seed")
            delta = candidate.discounted_return - reference.discounted_return
            ci_low, ci_high = _bootstrap_ci(delta.to_numpy(), 9200 + len(rows))
            rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "reference_level": reference_level,
                    "return_delta": float(delta.mean()),
                    "return_ci_low": ci_low,
                    "return_ci_high": ci_high,
                    "wins": int((delta > 1.0e-12).sum()),
                    "ties": int((np.abs(delta) <= 1.0e-12).sum()),
                    "losses": int((delta < -1.0e-12).sum()),
                    "completion_delta": float(
                        candidate.completion_ratio.mean()
                        - reference.completion_ratio.mean()
                    ),
                    "slo_delta": float(
                        candidate.slo_violation_rate.mean()
                        - reference.slo_violation_rate.mean()
                    ),
                    "cost_delta": float(
                        candidate.total_cost.mean() - reference.total_cost.mean()
                    ),
                    "training_time_ratio": float(
                        candidate.training_seconds.mean()
                        / max(reference.training_seconds.mean(), 1.0e-9)
                    ),
                }
            )
    return summary, pd.DataFrame(rows)


def cost_tables(
    project_root: Path,
    core_episodes: pd.DataFrame,
    core_steps: pd.DataFrame,
    *,
    core_tier: str,
    registered_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, dict]:
    external_root = (
        project_root
        / "results/direct_action_planning_dataset_benchmark/external_baselines_development_v1"
    )
    training_rows: list[dict] = []
    for path in external_root.glob("*/*/training.json"):
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        for method, values in payload.items():
            if not isinstance(values, dict):
                continue
            elapsed = values.get("elapsed_seconds", values.get("training_seconds"))
            if elapsed is None:
                continue
            training_rows.append(
                {
                    "dataset": config["dataset"],
                    "method": method,
                    "training_seconds": float(elapsed),
                }
            )
    external_training = pd.DataFrame(training_rows).groupby(
        ["dataset", "method"], as_index=False
    ).training_seconds.mean()
    external_latency = pd.read_csv(
        external_root / "analysis_v3_claim_centered/learning_method_summary.csv"
    )[
        ["dataset", "method", "decision_ms_mean", "decision_ms_p95"]
    ]
    external_cost = external_latency.merge(
        external_training, on=["dataset", "method"], how="left"
    )
    external_cost["source"] = "frozen_external_benchmark"

    new_latency = core_episodes.groupby(["dataset", "method"], as_index=False)[
        ["decision_ms_mean", "decision_ms_p95"]
    ].mean()
    new_training_rows: list[dict] = []
    core_root = (
        project_root
        / "results/direct_action_planning_paper_evidence"
        / core_tier
    )
    total_runtime = 0.0
    total_branch_transitions = 0
    peak_rss = 0
    for path in core_root.glob("*/*/training.json"):
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        if int(config["seed"]) not in registered_seeds:
            continue
        training = json.loads(path.read_text(encoding="utf-8"))
        runtime = json.loads((path.parent / "runtime.json").read_text(encoding="utf-8"))
        total_runtime += float(runtime["elapsed_seconds"])
        peak_rss = max(peak_rss, int(runtime["peak_rss_kib"]))
        total_branch_transitions += int(
            training["collection"]["branch_action_transitions"]
        )
        new_training_rows.append(
            {
                "dataset": config["dataset"],
                "method": "dap_full",
                "training_seconds": float(
                    training.get(
                        "full_method_training_seconds", training["training_seconds"]
                    )
                ),
                "parameter_count": int(
                    training["parameters"].get(
                        "full_value", training["parameters"].get("selected_value", 0)
                    )
                    + training["parameters"]["decision_forecaster"]
                ),
                "checkpoint_bytes": int(
                    training["checkpoint_bytes"].get(
                        "full_value",
                        training["checkpoint_bytes"].get("selected_value", 0),
                    )
                    + training["checkpoint_bytes"]["decision_forecaster"]
                ),
            }
        )
    new_training = pd.DataFrame(new_training_rows).groupby(
        ["dataset", "method"], as_index=False
    ).mean(numeric_only=True)
    new_cost = new_latency.merge(new_training, on=["dataset", "method"], how="left")
    new_cost["source"] = core_tier
    combined = pd.concat([external_cost, new_cost], ignore_index=True, sort=False)
    result_root = project_root / "results/direct_action_planning_paper_evidence"
    storage_bytes = sum(path.stat().st_size for path in result_root.rglob("*") if path.is_file())
    resource_summary = {
        "registered_core_runs": int(
            len(registered_seeds) * core_episodes.dataset.nunique() * core_episodes.budget.nunique()
        ),
        "core_cpu_hours_approx": total_runtime / 3600.0,
        "core_branch_action_transitions": total_branch_transitions,
        "peak_worker_rss_kib": peak_rss,
        "paper_evidence_storage_bytes": storage_bytes,
        "formal_test_accessed": False,
    }
    return combined, resource_summary


def _write_claim_table(path: Path, decisions: pd.DataFrame) -> None:
    gentd = decisions[decisions.dataset == "gentd26"].copy()
    lines = [
        "# Component Claim-Evidence Table",
        "",
        "All rows are validation-only exploratory evidence. `not_supported` means the registered",
        "2%-scale/4-of-5-seed/guardrail threshold was not met; it is not an equivalence claim.",
        "",
        "| Component | Return gain | Relative effect | Wins | Guardrail | Decision |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    for row in gentd.itertuples(index=False):
        lines.append(
            f"| {row.component} | {row.return_gain:.4f} | {row.relative_return_effect:.4%} "
            f"| {row.return_wins}/5 | {'pass' if row.guardrails_pass else 'fail'} | {row.decision} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    project_root: str | Path,
    *,
    config_path: str | Path,
    output_name: str = "analysis_v1",
) -> Path:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    core_tier = str(config["tier"])
    sensitivity_tier = str(config["sensitivity_tier"])
    registered_seeds = tuple(int(seed) for seed in config["seeds"])
    expected_core_runs = (
        len(config["datasets"]) * len(config["budgets"]) * len(registered_seeds)
    )
    expected_sensitivity_runs = len(registered_seeds) * sum(
        len(levels) for levels in config["sensitivity"].values()
    )
    output = (
        project_root
        / "results/direct_action_planning_paper_evidence"
        / core_tier
        / output_name
    )
    if (output / "manifest.json").exists():
        raise FileExistsError(f"analysis is append-only: {output}")
    output.mkdir(parents=True, exist_ok=True)
    episodes, steps, diagnostics = load_core(
        project_root,
        core_tier,
        registered_seeds=registered_seeds,
        registered_datasets=tuple(str(dataset) for dataset in config["datasets"]),
        registered_budgets=tuple(float(budget) for budget in config["budgets"]),
        expected_schema=str(config["core_manifest_schema"]),
        expected_runs=expected_core_runs,
    )
    units, seed_blocks, method_summary = core_tables(episodes)
    comparisons, decisions = component_comparisons(seed_blocks)
    sensitivity = load_sensitivity(
        project_root,
        sensitivity_tier,
        registered_seeds=registered_seeds,
        registered_levels={
            str(factor): tuple(float(level) for level in levels)
            for factor, levels in config["sensitivity"].items()
        },
        registered_dataset=str(config["sensitivity_dataset"]),
        registered_budget=float(config["sensitivity_budget"]),
        expected_schema=str(config["sensitivity_manifest_schema"]),
        expected_runs=expected_sensitivity_runs,
    )
    sensitivity_summary, sensitivity_comparisons = sensitivity_tables(sensitivity)
    costs, resource_summary = cost_tables(
        project_root,
        episodes,
        steps,
        core_tier=core_tier,
        registered_seeds=registered_seeds,
    )
    resource_summary["registered_sensitivity_runs"] = expected_sensitivity_runs

    episodes.to_csv(output / "core_episode_metrics.csv.gz", index=False, compression="gzip")
    units.to_csv(output / "core_unit_metrics.csv", index=False)
    seed_blocks.to_csv(output / "core_seed_block_metrics.csv", index=False)
    method_summary.to_csv(output / "core_method_summary.csv", index=False)
    comparisons.to_csv(output / "component_comparisons.csv", index=False)
    decisions.to_csv(output / "component_decisions.csv", index=False)
    diagnostics.to_json(output / "training_diagnostics.jsonl", orient="records", lines=True)
    sensitivity.to_csv(output / "sensitivity_seed_metrics.csv", index=False)
    sensitivity_summary.to_csv(output / "sensitivity_summary.csv", index=False)
    sensitivity_comparisons.to_csv(output / "sensitivity_comparisons.csv", index=False)
    costs.to_csv(output / "cost_summary.csv", index=False)
    write_json(output / "resource_summary.json", resource_summary)
    _write_claim_table(output / "claim_evidence_table.md", decisions)

    evidence = {
        "schema": "stage2.dap_paper_evidence.strength.v1",
        "status": "development_exploratory",
        "formal_test_accessed": False,
        "independent_seed_blocks": 5,
        "component_decisions": decisions.to_dict(orient="records"),
        "allowed_scope": (
            "dataset-specific exploratory method-package evidence; component claims only when "
            "the registered threshold passes"
        ),
        "prohibited_scope": [
            "confirmatory significance",
            "universal scheduling superiority",
            "equal-information comparison with model-free RL",
            "superiority over privileged analytic controllers",
        ],
    }
    write_json(output / "evidence_strength.json", evidence)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(
        output / "manifest.json",
        {
            "schema": "stage2.dap_paper_evidence.analysis.v1",
            "status": "completed",
            "development_only": True,
            "formal_test_accessed": False,
            "registered_core_runs": expected_core_runs,
            "registered_sensitivity_runs": expected_sensitivity_runs,
            "excluded_engineering_runs": [],
            "config_sha256": sha256_file(config_path),
            "artifacts": artifacts,
        },
    )
    return output
