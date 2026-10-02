"""Build append-only claim evidence from the locked DAP closure summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("azure2019", "gentd26")
BASELINES = (
    "double_dqn",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "p3o",
    "budgeted_fitted_q",
)
METRICS = (
    "discounted_return",
    "completion_ratio",
    "slo_violation_rate",
    "total_cost",
)
CONTROLS = (
    "dap_immediate_structured",
    "dap_black_box_transition",
    "dap_actor_distilled",
    "dap_no_budget_horizon",
    "mpc_8",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _slug(value: str) -> str:
    return value.upper().replace("-", "_")


def _favourable_difference(frame: pd.DataFrame, method: str, baseline: str, metric: str) -> pd.Series:
    pivot = frame[frame.method.isin((method, baseline))].pivot_table(
        index=("training_seed", "budget"), columns="method", values=metric
    )
    difference = pivot[method] - pivot[baseline]
    if metric in ("slo_violation_rate", "total_cost"):
        difference = -difference
    return difference.groupby(level="training_seed").mean()


def _paired_dz(values: pd.Series) -> float:
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if std > 0 else float("inf")


def _grade(q_value: float, ci_low: float, ci_high: float, n: int) -> tuple[str, str, bool]:
    significant = q_value < 0.05 and (ci_low > 0 or ci_high < 0)
    if significant:
        # The Light evidence contract caps n < 30 at weak even for large effects.
        return "weak", "observed a corrected paired difference in this locked test", True
    return "none", "report the sample direction; no corrected difference was established", True


def _external_evidence(unit: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in paired.itertuples(index=False):
        values = _favourable_difference(
            unit[unit.dataset == record.dataset], "dap_calibrated", record.baseline, record.metric
        )
        grade, wording, hedge = _grade(
            float(record.bh_q_within_dataset_metric), float(record.ci_low), float(record.ci_high), len(values)
        )
        direction = "favourable" if float(record.mean) > 0 else "adverse" if float(record.mean) < 0 else "tie"
        rows.append(
            {
                "claim_id": f"DAP-PC-{_slug(record.dataset)}-{_slug(record.baseline)}-{_slug(record.metric)}",
                "dataset": record.dataset,
                "baseline": record.baseline,
                "metric": record.metric,
                "raw_dap_minus_baseline": float(record.raw_mean_dap_minus_baseline),
                "favourable_effect": float(record.mean),
                "ci_low": float(record.ci_low),
                "ci_high": float(record.ci_high),
                "paired_cohens_dz": _paired_dz(values),
                "wilcoxon_p": float(record.wilcoxon_p),
                "bh_q": float(record.bh_q_within_dataset_metric),
                "n_seed_blocks": int(record.n_seed_blocks),
                "positive_seed_count": int(record.positive_seed_count),
                "negative_seed_count": int(record.negative_seed_count),
                "direction": direction,
                "evidence_grade": grade,
                "allowed_wording": wording,
                "hedge_required": hedge,
            }
        )
    result = pd.DataFrame(rows).sort_values(["dataset", "baseline", "metric"]).reset_index(drop=True)
    expected = len(DATASETS) * len(BASELINES) * len(METRICS)
    if len(result) != expected:
        raise ValueError(f"expected {expected} external comparisons, found {len(result)}")
    return result


def _control_evidence(unit: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    selected = paired[(paired.control.isin(CONTROLS)) & (paired.metric == "discounted_return")].copy()
    rows: list[dict[str, object]] = []
    for record in selected.itertuples(index=False):
        values = _favourable_difference(
            unit[unit.dataset == record.dataset], "dap_full", record.control, "discounted_return"
        )
        grade, wording, hedge = _grade(
            float(record.bh_q_within_dataset_metric), float(record.ci_low), float(record.ci_high), len(values)
        )
        rows.append(
            {
                "claim_id": f"DAP-PC-CONTROL-{_slug(record.dataset)}-{_slug(record.control)}",
                "dataset": record.dataset,
                "control": record.control,
                "metric": "discounted_return",
                "full_minus_control": float(record.raw_mean_full_minus_control),
                "ci_low": float(record.ci_low),
                "ci_high": float(record.ci_high),
                "paired_cohens_dz": _paired_dz(values),
                "wilcoxon_p": float(record.wilcoxon_p),
                "bh_q": float(record.bh_q_within_dataset_metric),
                "n_seed_blocks": int(record.n_seed_blocks),
                "positive_seed_count": int(record.positive_seed_count),
                "negative_seed_count": int(record.negative_seed_count),
                "evidence_grade": grade,
                "allowed_wording": wording,
                "hedge_required": hedge,
            }
        )
    result = pd.DataFrame(rows).sort_values(["dataset", "control"]).reset_index(drop=True)
    expected = len(DATASETS) * len(CONTROLS)
    if len(result) != expected:
        raise ValueError(f"expected {expected} control comparisons, found {len(result)}")
    return result


def _evidence_json(external: pd.DataFrame, controls: pd.DataFrame) -> dict[str, object]:
    claims: list[dict[str, object]] = []
    for row in external.itertuples(index=False):
        claims.append(
            {
                "claim_id": row.claim_id,
                "text": f"Full DAP versus {row.baseline} on {row.dataset}: {row.metric}",
                "q_fdr": row.bh_q,
                "effect_size": row.paired_cohens_dz,
                "effect_kind": "paired_cohens_dz",
                "raw_effect": row.favourable_effect,
                "raw_effect_direction": "positive_is_favourable",
                "ci95": [row.ci_low, row.ci_high],
                "n": row.n_seed_blocks,
                "evidence_grade": row.evidence_grade,
                "grade_level": "low" if row.evidence_grade == "weak" else "insufficient",
                "allowed_verbs": [row.allowed_wording],
                "forbidden_verbs": ["prove", "establish universal superiority", "state-of-the-art"],
                "hedge_required": row.hedge_required,
            }
        )
    for row in controls.itertuples(index=False):
        claims.append(
            {
                "claim_id": row.claim_id,
                "text": f"Full DAP versus {row.control} on {row.dataset}: discounted return",
                "q_fdr": row.bh_q,
                "effect_size": row.paired_cohens_dz,
                "effect_kind": "paired_cohens_dz",
                "raw_effect": row.full_minus_control,
                "raw_effect_direction": "positive_is_favourable",
                "ci95": [row.ci_low, row.ci_high],
                "n": row.n_seed_blocks,
                "evidence_grade": row.evidence_grade,
                "grade_level": "insufficient",
                "allowed_verbs": [row.allowed_wording],
                "forbidden_verbs": ["prove component causality", "significantly outperform"],
                "hedge_required": row.hedge_required,
            }
        )
    return {"schema": "light.evidence_strength.v1", "source": "locked DAP paper-closure analysis", "claims": claims}


def _key_row(external: pd.DataFrame, dataset: str, baseline: str, metric: str) -> pd.Series:
    rows = external[(external.dataset == dataset) & (external.baseline == baseline) & (external.metric == metric)]
    if len(rows) != 1:
        raise ValueError((dataset, baseline, metric, len(rows)))
    return rows.iloc[0]


def _markdown(external: pd.DataFrame, controls: pd.DataFrame) -> str:
    lines = [
        "# Claim-Evidence Table",
        "",
        "The independent unit is the trained seed block after averaging the five registered budgets.",
        "External comparisons use `n=10`; controls use development validation with `n=5`.",
        "BH-FDR is applied within each preregistered dataset-by-metric family.",
        "",
        "## Primary dataset claims",
        "",
        "| Claim | Paired effect (95% CI) | d_z | p / q | Seeds | Allowed conclusion |",
        "|---|---:|---:|---:|---:|---|",
    ]
    key_claims = (
        ("azure2019", "double_dqn", "discounted_return", "Azure return vs Double DQN"),
        ("azure2019", "double_dqn", "slo_violation_rate", "Azure SLO reduction vs Double DQN"),
        ("azure2019", "double_dqn", "total_cost", "Azure cost reduction vs Double DQN"),
        ("azure2019", "cpo", "discounted_return", "Azure return vs CPO"),
        ("azure2019", "cpo", "slo_violation_rate", "Azure SLO reduction vs CPO"),
        ("azure2019", "cpo", "total_cost", "Azure cost reduction vs CPO"),
        ("gentd26", "double_dqn", "discounted_return", "GenTD26 return vs Double DQN"),
        ("gentd26", "double_dqn", "completion_ratio", "GenTD26 completion vs Double DQN"),
        ("gentd26", "double_dqn", "slo_violation_rate", "GenTD26 SLO reduction vs Double DQN"),
        ("gentd26", "double_dqn", "total_cost", "GenTD26 cost reduction vs Double DQN"),
        ("gentd26", "cpo", "discounted_return", "GenTD26 return vs CPO"),
        ("gentd26", "cpo", "total_cost", "GenTD26 cost reduction vs CPO"),
    )
    for dataset, baseline, metric, label in key_claims:
        row = _key_row(external, dataset, baseline, metric)
        lines.append(
            f"| {label} | {row.favourable_effect:+.5g} [{row.ci_low:+.5g}, {row.ci_high:+.5g}] | "
            f"{row.paired_cohens_dz:+.3f} | {row.wilcoxon_p:.5g} / {row.bh_q:.5g} | "
            f"{row.positive_seed_count}/{row.n_seed_blocks} favourable | {row.allowed_wording} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: Azure2019 versus Double DQN is a return/SLO versus cost tradeoff; the completion ratio is also",
            "slightly lower. Azure2019 versus CPO has higher return, lower SLO violation, and lower cost, while the",
            "completion-ratio comparison is not corrected-significant. GenTD26 has joint favourable return, completion,",
            "SLO, and cost directions against both highlighted baselines.",
            "",
            "## Same-information and mechanism controls",
            "",
            "| Dataset | Control | Full-minus-control return | d_z | p / q | Direction | Allowed conclusion |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in controls.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.control} | {row.full_minus_control:+.4f} "
            f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] | {row.paired_cohens_dz:+.3f} | "
            f"{row.wilcoxon_p:.4g} / {row.bh_q:.4g} | {row.positive_seed_count}/{row.n_seed_blocks} | "
            f"{row.allowed_wording} |"
        )
    lines.extend(
        [
            "",
            "No control comparison reaches `q<0.05`; these rows support only directional mechanism discussion.",
            "They do not establish an independently significant causal contribution for every module.",
            "",
            "## Non-inferential guardrails",
            "",
            "- All 100 locked temporal-test cells completed, producing 10,000 episode rows and 640,000 step rows.",
            "- Maximum budget overspend was zero for every method; this demonstrates budget feasibility under the shared",
            "  hard affordability mask, not broader risk safety.",
            "- The test partitions were accessed by an earlier project gate. The current result is a locked temporal",
            "  re-evaluation and must not be called a project-level untouched holdout.",
            "- The full 48-row external table is `external_comparisons.csv`; no external comparison was omitted.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-name", default="evidence_v1")
    args = parser.parse_args()
    root = args.project_root.resolve()
    temporal = root / "results/direct_action_planning_paper_closure/paper_closure_temporal_test_v2_locked/analysis_v1"
    controls = root / "results/direct_action_planning_paper_closure/paper_closure_controls_v3_locked/analysis_v1"
    output = root / "research/direct_action_planning_paper_closure" / args.output_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    temporal_unit = pd.read_csv(temporal / "unit_metrics.csv")
    temporal_paired = pd.read_csv(temporal / "paired_comparisons.csv")
    control_unit = pd.read_csv(controls / "unit_metrics.csv")
    control_paired = pd.read_csv(controls / "paired_control_comparisons.csv")
    external = _external_evidence(temporal_unit, temporal_paired)
    control = _control_evidence(control_unit, control_paired)

    external.to_csv(output / "external_comparisons.csv", index=False)
    control.to_csv(output / "control_comparisons.csv", index=False)
    (output / "claim_evidence_table.md").write_text(_markdown(external, control), encoding="utf-8")
    (output / "evidence_strength.json").write_text(
        json.dumps(_evidence_json(external, control), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    sources = {
        "temporal_unit_metrics": temporal / "unit_metrics.csv",
        "temporal_paired_comparisons": temporal / "paired_comparisons.csv",
        "control_unit_metrics": controls / "unit_metrics.csv",
        "control_paired_comparisons": controls / "paired_control_comparisons.csv",
    }
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema": "dap.dap_paper_closure.claim_evidence.v1",
        "status": "completed",
        "source_artifacts": {name: {"path": str(path.relative_to(root)), "sha256": _sha256(path)} for name, path in sources.items()},
        "coverage": {"external_comparisons": len(external), "control_return_comparisons": len(control)},
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
