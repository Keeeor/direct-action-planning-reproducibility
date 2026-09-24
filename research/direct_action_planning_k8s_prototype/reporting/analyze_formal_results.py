from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("azure_http", "gentd_inference")
BUDGETS = (512.0, 1024.0, 1664.0)
METHODS = ("static", "threshold", "hpa", "keda", "mpc_4", "dap")
PRIMARY_BASELINES = ("threshold", "hpa", "keda", "mpc_4")
PRIMARY_METRICS = {
    "completion_rate": 1.0,
    "slo_violation_rate": -1.0,
    "ready_replica_seconds": -1.0,
}
COLORS = {
    "dap": "#0072B2",
    "mpc_4": "#E69F00",
    "threshold": "#CC79A7",
    "hpa": "#D55E00",
    "keda": "#009E73",
    "static": "#666666",
}


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def exact_bootstrap_ci(values: Iterable[float]) -> tuple[float, float]:
    array = finite(values)
    if len(array) == 0:
        return math.nan, math.nan
    if len(array) > 7:
        rng = np.random.default_rng(20260808)
        indices = rng.integers(0, len(array), size=(20_000, len(array)))
        means = array[indices].mean(axis=1)
    else:
        indices = np.asarray(list(itertools.product(range(len(array)), repeat=len(array))))
        means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_sign_flip_p(values: Iterable[float]) -> float:
    array = finite(values)
    if len(array) == 0:
        return math.nan
    observed = abs(float(array.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(array))))
    permuted = np.abs((signs * array).mean(axis=1))
    return float(np.mean(permuted >= observed - 1.0e-12))


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return q
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return q


def paired_effect(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    sd = float(np.std(values, ddof=1))
    if sd <= 1.0e-12:
        return math.copysign(math.inf, float(np.mean(values))), math.copysign(math.inf, float(np.mean(values)))
    dz = float(np.mean(values) / sd)
    correction = 1.0 - 3.0 / (4.0 * len(values) - 5.0)
    return dz, dz * correction


def audit_integrity(df: pd.DataFrame, raw_root: Path) -> dict[str, Any]:
    expected = {
        (profile, budget, seed, method)
        for profile in PROFILES
        for budget in BUDGETS
        for seed in (20260901, 20260902, 20260903, 20260904, 20260905)
        for method in METHODS
    }
    observed = {
        (str(row.profile), float(row.budget_seconds), int(row.seed), str(row.method))
        for row in df.itertuples()
    }
    paired_cells = df.groupby(["profile", "budget_seconds", "seed"], dropna=False).agg(
        methods=("method", "nunique"),
        plans=("plan_sha256", "nunique"),
        request_counts=("total_requests", "nunique"),
    )
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw_root.rglob("run_manifest.json"))]
    checks = {
        "registered_rows": 180,
        "observed_rows": int(len(df)),
        "missing_cells": [list(item) for item in sorted(expected - observed)],
        "unexpected_cells": [list(item) for item in sorted(observed - expected)],
        "duplicate_cells": int(df.duplicated(["profile", "budget_seconds", "seed", "method"]).sum()),
        "completed_rows": int((df.run_status == "completed").sum()),
        "failed_rows": int((df.run_status != "completed").sum()),
        "paired_cells": int(len(paired_cells)),
        "paired_cells_with_six_methods": int((paired_cells.methods == 6).sum()),
        "paired_cells_with_one_plan": int((paired_cells.plans == 1).sum()),
        "paired_cells_with_equal_request_count": int((paired_cells.request_counts == 1).sum()),
        "max_budget_violation_seconds": float(df.budget_violation_seconds.max()),
        "max_metric_missing_rate": float(df.metric_missing_rate.max()),
        "total_monitor_failures": int(df.monitor_failures.sum()),
        "total_controller_deadline_misses": int(df.controller_deadline_misses.sum()),
        "manifest_statuses": dict(Counter(str(row.get("status")) for row in manifests)),
        "manifest_source_hashes": sorted({str(row.get("source_tree_sha256")) for row in manifests}),
        "manifest_config_hashes": sorted({str(row.get("config_sha256")) for row in manifests}),
        "manifest_image_digests": sorted({str(row.get("image_digest")) for row in manifests}),
        "manifest_plan_splits": sorted({str(row.get("plan_split")) for row in manifests}),
    }
    checks["passed"] = bool(
        checks["observed_rows"] == 180
        and not checks["missing_cells"]
        and not checks["unexpected_cells"]
        and checks["duplicate_cells"] == 0
        and checks["completed_rows"] == 180
        and checks["paired_cells"] == 30
        and checks["paired_cells_with_six_methods"] == 30
        and checks["paired_cells_with_one_plan"] == 30
        and checks["paired_cells_with_equal_request_count"] == 30
        and checks["max_budget_violation_seconds"] <= 1.0e-9
        and checks["max_metric_missing_rate"] < 0.01
        and checks["total_controller_deadline_misses"] == 0
        and checks["manifest_plan_splits"] == ["test"]
    )
    return checks


def summarize_cells(df: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "total_requests", "completed_requests", "completion_rate", "throughput_rps",
        "mean_latency_seconds", "p95_latency_seconds", "p99_latency_seconds",
        "slo_violation_rate", "timeout_rate", "failure_rate", "queue_area", "final_queue",
        "ready_replica_seconds", "requested_replica_seconds", "estimated_cpu_seconds",
        "mean_replicas", "peak_replicas", "scaling_count", "unused_budget_seconds",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["profile", "budget_seconds", "method"], sort=True):
        row: dict[str, Any] = {"profile": keys[0], "budget_seconds": keys[1], "method": keys[2], "n_runs": len(group)}
        for metric in metrics:
            values = finite(group[metric])
            low, high = exact_bootstrap_ci(values)
            row.update({
                f"{metric}_mean": float(values.mean()) if len(values) else math.nan,
                f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                f"{metric}_median": float(np.median(values)) if len(values) else math.nan,
                f"{metric}_ci95_low": low,
                f"{metric}_ci95_high": high,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_table(df: pd.DataFrame, baselines: tuple[str, ...], family_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["profile", "budget_seconds", "seed"]
    dap = df[df.method == "dap"].set_index(keys)
    for baseline in baselines:
        other = df[df.method == baseline].set_index(keys)
        joined = dap.join(other, lsuffix="_dap", rsuffix="_baseline", how="inner")
        if not (joined.plan_sha256_dap == joined.plan_sha256_baseline).all():
            raise AssertionError(f"request-plan mismatch for {baseline}")
        for (profile, budget), group in joined.groupby(level=["profile", "budget_seconds"], sort=True):
            for metric, direction in PRIMARY_METRICS.items():
                delta = finite(group[f"{metric}_dap"] - group[f"{metric}_baseline"])
                favorable = delta * direction
                low, high = exact_bootstrap_ci(delta)
                favorable_low, favorable_high = exact_bootstrap_ci(favorable)
                dz, gz = paired_effect(favorable)
                rows.append({
                    "family_id": family_id,
                    "profile": profile,
                    "budget_seconds": float(budget),
                    "baseline": baseline,
                    "metric": metric,
                    "direction": "higher_favors_dap" if direction > 0 else "lower_favors_dap",
                    "paired_runs": len(delta),
                    "mean_delta_dap_minus_baseline": float(delta.mean()),
                    "median_delta_dap_minus_baseline": float(np.median(delta)),
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "mean_favorable_effect": float(favorable.mean()),
                    "favorable_effect_ci95_low": favorable_low,
                    "favorable_effect_ci95_high": favorable_high,
                    "dap_favorable_fraction": float(np.mean(favorable > 0)),
                    "exact_sign_flip_p": exact_sign_flip_p(favorable),
                    "cohen_dz_favorable": dz,
                    "hedges_gz_favorable": gz,
                })
    result = pd.DataFrame(rows)
    result["q_bh_within_family"] = bh_fdr(result.exact_sign_flip_p)
    result["ci_direction"] = np.where(
        result.favorable_effect_ci95_low > 0, "favors_dap",
        np.where(result.favorable_effect_ci95_high < 0, "favors_baseline", "crosses_zero"),
    )
    result["evidence_grade"] = np.where(
        (result.q_bh_within_family < 0.05)
        & (result.favorable_effect_ci95_low > 0)
        & (result.paired_runs >= 30),
        "moderate",
        "none",
    )
    return result


def profile_pooled_sensitivity(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (profile, baseline, metric), group in primary.groupby(["profile", "baseline", "metric"], sort=True):
        # The same five request-plan seeds recur at three budgets. Average within
        # seed first so the inferential unit remains the independent seed.
        source = group[["profile", "budget_seconds", "baseline", "metric"]].copy()
        del source
        rows.append({
            "profile": profile,
            "baseline": baseline,
            "metric": metric,
            "note": "computed separately from raw paired runs by seed",
        })
    return pd.DataFrame(rows)


def pooled_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["profile", "budget_seconds", "seed"]
    dap = df[df.method == "dap"].set_index(keys)
    for baseline in PRIMARY_BASELINES:
        other = df[df.method == baseline].set_index(keys)
        joined = dap.join(other, lsuffix="_dap", rsuffix="_baseline", how="inner")
        for profile, profile_group in joined.groupby(level="profile", sort=True):
            for metric, direction in PRIMARY_METRICS.items():
                delta = (profile_group[f"{metric}_dap"] - profile_group[f"{metric}_baseline"]) * direction
                per_seed = delta.groupby(level="seed").mean()
                values = finite(per_seed)
                low, high = exact_bootstrap_ci(values)
                dz, gz = paired_effect(values)
                rows.append({
                    "family_id": "profile_pooled_sensitivity",
                    "profile": profile,
                    "baseline": baseline,
                    "metric": metric,
                    "independent_seed_units": len(values),
                    "mean_favorable_effect": float(values.mean()),
                    "favorable_effect_ci95_low": low,
                    "favorable_effect_ci95_high": high,
                    "dap_favorable_fraction": float(np.mean(values > 0)),
                    "exact_sign_flip_p": exact_sign_flip_p(values),
                    "cohen_dz_favorable": dz,
                    "hedges_gz_favorable": gz,
                })
    result = pd.DataFrame(rows)
    result["q_bh_within_family"] = bh_fdr(result.exact_sign_flip_p)
    return result


def pareto_audit(df: pd.DataFrame) -> pd.DataFrame:
    points = df.groupby(["profile", "method", "budget_seconds"], as_index=False)[list(PRIMARY_METRICS)].mean()

    def dominates(left: pd.Series, right: pd.Series) -> bool:
        weak = (
            left.ready_replica_seconds <= right.ready_replica_seconds
            and left.completion_rate >= right.completion_rate
            and left.slo_violation_rate <= right.slo_violation_rate
        )
        strict = (
            left.ready_replica_seconds < right.ready_replica_seconds
            or left.completion_rate > right.completion_rate
            or left.slo_violation_rate < right.slo_violation_rate
        )
        return bool(weak and strict)

    rows = []
    for _, dap in points[points.method == "dap"].iterrows():
        candidates = points[points.profile == dap.profile]
        dominators = [
            f"{row.method}@{int(row.budget_seconds)}"
            for _, row in candidates.iterrows()
            if row.method != "dap" and dominates(row, dap)
        ]
        rows.append({
            **dap.to_dict(),
            "pareto_nondominated": not dominators,
            "dominator_count": len(dominators),
            "dominators": ";".join(dominators),
        })
    return pd.DataFrame(rows)


def controller_overhead(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.glob("*/controller/controller_actions.jsonl")):
        manifest = json.loads((path.parents[1] / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["method"] not in {"dap", "mpc_4", "threshold"}:
            continue
        for row in read_jsonl(path):
            action_rows.append({"method": manifest["method"], "profile": manifest["profile"], **row})
    actions = pd.DataFrame(action_rows)
    fields = {
        "control_loop_latency_seconds": "loop",
        "state_collection_latency_seconds": "state_collection",
        "kubernetes_api_latency_seconds": "kubernetes_api",
        "dap_inference_latency_seconds": "dap_inference",
    }
    rows = []
    for method in ("dap", "mpc_4", "threshold"):
        group = actions[actions.method == method]
        for field, component in fields.items():
            if field not in group:
                continue
            values = finite(group[field])
            if not len(values):
                continue
            rows.append({
                "method": method,
                "component": component,
                "n_control_steps": len(values),
                "mean_seconds": float(values.mean()),
                "median_seconds": float(np.median(values)),
                "p95_seconds": float(np.quantile(values, 0.95)),
                "p99_seconds": float(np.quantile(values, 0.99)),
                "max_seconds": float(values.max()),
                "deadline_miss_fraction": float(np.mean(values >= 10.0)) if component == "loop" else math.nan,
            })
    return pd.DataFrame(rows), actions


def dap_action_diagnostics(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.glob("*__dap/controller/controller_actions.jsonl")):
        manifest = json.loads((path.parents[1] / "run_manifest.json").read_text(encoding="utf-8"))
        actions = read_jsonl(path)
        targets = np.asarray([int(row["target_replicas"]) for row in actions], dtype=int)
        margins = []
        changed_near_tie = 0
        for index, action in enumerate(actions):
            values = sorted(
                [float(value) for value in action.get("q_values", {}).values() if math.isfinite(float(value))],
                reverse=True,
            )
            margin = values[0] - values[1] if len(values) >= 2 else math.nan
            margins.append(margin)
            if index > 0 and targets[index] != targets[index - 1] and math.isfinite(margin) and margin <= 0.05:
                changed_near_tie += 1
        transitions = list(zip(targets[:-1], targets[1:]))
        counter = Counter(targets.tolist())
        rows.append({
            "profile": manifest["profile"],
            "budget_seconds": float(manifest["budget_seconds"]),
            "seed": int(manifest["seed"]),
            "steps": len(targets),
            "target_changes": int(np.sum(targets[1:] != targets[:-1])),
            "direct_1_to_5": int(sum(left == 1 and right == 5 for left, right in transitions)),
            "direct_5_to_1": int(sum(left == 5 and right == 1 for left, right in transitions)),
            "fraction_target_1": counter[1] / max(len(targets), 1),
            "fraction_target_2": counter[2] / max(len(targets), 1),
            "fraction_target_3": counter[3] / max(len(targets), 1),
            "fraction_target_5": counter[5] / max(len(targets), 1),
            "mean_top1_top2_q_margin": float(np.nanmean(margins)),
            "fraction_q_margin_le_0_01": float(np.mean(np.asarray(margins) <= 0.01)),
            "fraction_q_margin_le_0_05": float(np.mean(np.asarray(margins) <= 0.05)),
            "changes_with_q_margin_le_0_05": changed_near_tie,
        })
    return pd.DataFrame(rows)


def workload_summary(plan_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(plan_root.rglob("*.jsonl.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rates = np.asarray(payload["rate_by_step"], dtype=float)
        rows.append({
            "profile": path.parent.name,
            "seed": int(payload["seed"]),
            "activity_quantile": payload["window_selection"].get("activity_quantile"),
            "window_start": int(payload["window_start"]),
            "request_count": int(payload["request_count"]),
            "mean_rps": float(rates.mean()),
            "std_rps": float(rates.std(ddof=1)),
            "coefficient_of_variation": float(rates.std(ddof=1) / max(rates.mean(), 1.0e-12)),
            "min_rps": float(rates.min()),
            "max_rps": float(rates.max()),
            "p95_rps": float(np.quantile(rates, 0.95)),
            "plan_sha256": payload["plan_sha256"],
        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_service_cost(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex="col")
    for column, profile in enumerate(PROFILES):
        profile_rows = summary[summary.profile == profile]
        for method in METHODS:
            rows = profile_rows[profile_rows.method == method].sort_values("budget_seconds")
            x = rows.ready_replica_seconds_mean.to_numpy()
            for row, metric in enumerate(("completion_rate", "slo_violation_rate")):
                y = rows[f"{metric}_mean"].to_numpy()
                yerr = np.vstack((
                    y - rows[f"{metric}_ci95_low"].to_numpy(),
                    rows[f"{metric}_ci95_high"].to_numpy() - y,
                ))
                axes[row, column].errorbar(
                    x, y, yerr=yerr, marker="o", linewidth=2.2 if method == "dap" else 1.1,
                    markersize=5.5 if method == "dap" else 4.0, capsize=2.0,
                    color=COLORS[method], alpha=1.0 if method == "dap" else 0.78, label=method,
                )
        axes[0, column].set_title(profile.replace("_", " "))
        axes[0, column].set_ylabel("Completion rate")
        axes[1, column].set_ylabel("SLO violation rate")
        axes[1, column].set_xlabel("Ready-replica seconds")
        for row in range(2):
            axes[row, column].grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Formal Kubernetes service-cost outcomes (mean and 95% bootstrap CI, n=5)")
    fig.tight_layout()
    save_figure(fig, output, "formal_service_cost_summary")


def plot_overhead(actions: pd.DataFrame, output: Path) -> None:
    methods = ("threshold", "mpc_4", "dap")
    data = [finite(actions[actions.method == method].control_loop_latency_seconds) for method in methods]
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    box = axis.boxplot(data, tick_labels=methods, showfliers=False, patch_artist=True)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.75)
    axis.axhline(10.0, color="#222222", linestyle="--", linewidth=1.0, label="10 s deadline")
    axis.set_yscale("log")
    axis.set_ylabel("Control-loop latency (s, log scale)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("External-controller latency across all formal control steps")
    fig.tight_layout()
    save_figure(fig, output, "formal_controller_latency")


def plot_action_diagnostics(diagnostics: pd.DataFrame, output: Path) -> None:
    grouped = diagnostics.groupby(["profile", "budget_seconds"], as_index=False).mean(numeric_only=True)
    labels = [f"{row.profile.replace('_', ' ')}\n{int(row.budget_seconds)}" for row in grouped.itertuples()]
    x = np.arange(len(grouped))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    bottom = np.zeros(len(grouped))
    for target, color in zip((1, 2, 3, 5), ("#999999", "#56B4E9", "#E69F00", "#D55E00")):
        values = grouped[f"fraction_target_{target}"].to_numpy()
        axes[0].bar(x, values, bottom=bottom, label=f"{target} Pods", color=color)
        bottom += values
    axes[0].set_ylabel("Fraction of decisions")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_title("DAP target-replica distribution")
    axes[1].bar(x - 0.18, grouped.target_changes, width=0.36, color=COLORS["dap"], label="target changes")
    axes[1].bar(
        x + 0.18, grouped.changes_with_q_margin_le_0_05, width=0.36,
        color="#D55E00", label="changes with Q margin <= 0.05",
    )
    axes[1].set_ylabel("Mean changes per 64-step run")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Action churn and near-tie changes")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    save_figure(fig, output, "formal_dap_action_diagnostics")


def write_claim_evidence(
    output: Path, integrity: dict[str, Any], pareto: pd.DataFrame,
    overhead: pd.DataFrame, diagnostics: pd.DataFrame, primary: pd.DataFrame,
) -> None:
    dap_loop = overhead[(overhead.method == "dap") & (overhead.component == "loop")].iloc[0]
    nondominated = int(pareto.pareto_nondominated.sum())
    near_tie = float(diagnostics.changes_with_q_margin_le_0_05.sum() / max(diagnostics.target_changes.sum(), 1))
    claims = [
        {
            "claim_id": "K8S-C1-DEPLOYABILITY",
            "claim": "DAP executed as an external controller in all 30 registered DAP runs using live Kubernetes metrics and real scaling actions.",
            "grade": "strong_within_testbed",
            "evidence": {"completed_dap_runs": 30, "registered_dap_runs": 30, "integrity_passed": integrity["passed"]},
            "language_ceiling": "demonstrates within the registered local Kubernetes testbed",
        },
        {
            "claim_id": "K8S-C2-HARD-BUDGET",
            "claim": "The complete formal matrix maintained the Ready-replica-second budget without violations or controller deadline misses.",
            "grade": "strong_within_testbed",
            "evidence": {
                "runs": 180,
                "max_budget_violation_seconds": integrity["max_budget_violation_seconds"],
                "deadline_misses": integrity["total_controller_deadline_misses"],
            },
            "language_ceiling": "demonstrates within the registered 180-run matrix",
        },
        {
            "claim_id": "K8S-C3-CONTROL-LATENCY",
            "claim": "DAP control-loop latency remained below the 10-second control interval.",
            "grade": "strong_within_testbed",
            "evidence": {
                "control_steps": int(dap_loop.n_control_steps),
                "p99_seconds": float(dap_loop.p99_seconds),
                "max_seconds": float(dap_loop.max_seconds),
                "deadline_miss_fraction": float(dap_loop.deadline_miss_fraction),
            },
            "language_ceiling": "demonstrates within the registered formal runs",
        },
        {
            "claim_id": "K8S-C4-SERVICE-COST",
            "claim": "DAP provides a favorable service-cost frontier in the high-fidelity prototype.",
            "grade": "none",
            "evidence": {
                "dap_nondominated_points": nondominated,
                "dap_total_points": int(len(pareto)),
                "confirmatory_comparisons_q_lt_0_05": int((primary.q_bh_within_family < 0.05).sum()),
            },
            "language_ceiling": "the formal prototype does not establish this claim",
        },
        {
            "claim_id": "K8S-D1-ACTION-CHURN",
            "claim": "Near-tied action scores are associated with DAP replica-target churn in the formal runs.",
            "grade": "posthoc_association",
            "evidence": {
                "target_changes": int(diagnostics.target_changes.sum()),
                "near_tie_changes": int(diagnostics.changes_with_q_margin_le_0_05.sum()),
                "near_tie_share": near_tie,
            },
            "language_ceiling": "post-hoc diagnostics indicate; not a causal claim",
        },
    ]
    payload = {
        "schema": "dap.k8s.formal_evidence_strength.v1",
        "analysis_unit": "complete Kubernetes run paired by profile, budget, and request-plan seed",
        "multiple_comparison_correction": "BH-FDR within the 72-comparison primary family",
        "claims": claims,
        "source_artifacts": {
            "run_metrics": {"path": "results/analysis/formal/run_metrics.csv", "sha256": sha256(ROOT / "results/analysis/formal/run_metrics.csv")},
            "analysis_script": {"path": "reporting/analyze_formal_results.py", "sha256": sha256(Path(__file__))},
        },
    }
    (output / "evidence_strength.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Formal Prototype Claim-Evidence Table", "",
        "The complete Kubernetes run is the independent unit. Per-cell comparisons use five paired request-plan seeds. "
        "Exact two-sided sign-flip tests therefore have a minimum attainable p-value of 0.0625; significance is assessed only after BH-FDR correction.", "",
        "| Claim ID | Evidence | Grade / permitted wording |", "| --- | --- | --- |",
    ]
    for claim in claims:
        evidence = "; ".join(f"{key}={value}" for key, value in claim["evidence"].items())
        lines.append(f"| {claim['claim_id']} | {evidence} | {claim['grade']}: {claim['language_ceiling']} |")
    lines.extend(["", "All 72 planned DAP-vs-adaptive-baseline comparisons are retained in `pairwise_primary.csv`; no comparison is omitted based on its direction.", ""])
    (output / "claim_evidence_table.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results/analysis/formal/run_metrics.csv")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "results/runs/formal")
    parser.add_argument("--plan-root", type=Path, default=ROOT / "results/request_plans/formal")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/formal/reporting")
    parser.add_argument("--figures", type=Path, default=ROOT / "results/figures/formal_report")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input.resolve())
    integrity = audit_integrity(df, args.raw_root.resolve())
    if not integrity["passed"]:
        raise RuntimeError("formal integrity audit failed; see integrity_summary.json")
    (output / "integrity_summary.json").write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = summarize_cells(df)
    primary = pairwise_table(df, PRIMARY_BASELINES, "primary_dap_vs_adaptive_72")
    static = pairwise_table(df, ("static",), "supplementary_dap_vs_static_18")
    pooled = pooled_from_raw(df)
    pareto = pareto_audit(df)
    overhead, actions = controller_overhead(args.raw_root.resolve())
    diagnostics = dap_action_diagnostics(args.raw_root.resolve())
    workloads = workload_summary(args.plan_root.resolve())
    summary.to_csv(output / "cell_summary.csv", index=False)
    primary.to_csv(output / "pairwise_primary.csv", index=False)
    static.to_csv(output / "pairwise_static_supplement.csv", index=False)
    pooled.to_csv(output / "profile_pooled_sensitivity.csv", index=False)
    pareto.to_csv(output / "dap_pareto_audit.csv", index=False)
    overhead.to_csv(output / "controller_overhead_summary.csv", index=False)
    diagnostics.to_csv(output / "dap_action_diagnostics.csv", index=False)
    workloads.to_csv(output / "workload_summary.csv", index=False)
    plot_service_cost(summary, args.figures.resolve())
    plot_overhead(actions, args.figures.resolve())
    plot_action_diagnostics(diagnostics, args.figures.resolve())
    write_claim_evidence(output, integrity, pareto, overhead, diagnostics, primary)
    manifest = {
        "schema": "dap.k8s.formal_reporting_manifest.v1",
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input.resolve())},
        "analysis_script_sha256": sha256(Path(__file__)),
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
        "figure_outputs": sorted(path.name for path in args.figures.resolve().iterdir() if path.is_file()),
        "primary_comparisons": int(len(primary)),
        "supplementary_comparisons": int(len(static)),
    }
    (output / "reporting_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output), "figures": str(args.figures.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
