from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_dynamic_budget.analysis.aggregation import (
    paired_comparisons,
    pareto_table,
)
from stage2_dynamic_budget.analysis.statistics import benjamini_hochberg
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json


BUDGETS = [110.0, 220.0, 330.0]
SEEDS = list(range(5))
SCENARIOS = ["stable", "early_burst", "late_burst", "periodic"]


def extended_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    identifiers = {"variant", "method", "budget", "seed", "scenario"}
    metrics = [
        column
        for column in seeds.select_dtypes(include=np.number).columns
        if column not in identifiers
    ]
    rows = []
    for keys, group in seeds.groupby(["variant", "method", "budget", "scenario"]):
        for metric in metrics:
            values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if not len(values):
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            half = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else float("nan")
            rows.append(
                {
                    "variant": keys[0],
                    "method": keys[1],
                    "budget": keys[2],
                    "scenario": keys[3],
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                }
            )
    return pd.DataFrame(rows)


def collect_hard(root: Path) -> pd.DataFrame:
    frames = []
    for mode in ("d1", "d2"):
        for budget in BUDGETS:
            label = f"{budget:.6g}".replace(".", "p")
            for seed in SEEDS:
                run = root / f"hard_coupling__v3__{mode}__b{label}__s{seed}"
                manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                if manifest["status"] != "completed" or manifest["completion"]["oracle"] != "PASS":
                    raise RuntimeError(f"ineligible run: {run}")
                frame = pd.read_csv(run / "metrics.csv")
                frame["method"] = f"{mode}_hard_mask"
                frame["variant"] = "hard_coupling_v3"
                frame["run_id"] = run.name
                frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    expected = 2 * len(BUDGETS) * len(SEEDS) * len(SCENARIOS) * 5
    if len(combined) != expected:
        raise RuntimeError(f"expected {expected} hard-coupling episode rows, got {len(combined)}")
    return combined


def collect_frozen_baselines(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "results/summaries/formal_valid_v2_episode_metrics.csv")
    frame = frame.loc[
        frame.method.isin(["cdba", "b4_budget_state"])
        & frame.budget.isin(BUDGETS)
        & frame.seed.isin(SEEDS)
        & frame.scenario.isin(SCENARIOS)
    ].copy()
    action = pd.read_csv(
        root / "results/summaries/final_synthetic_action_reallocation_seed_metrics.csv"
    )
    action = action.loc[
        action.method.isin(["cdba", "b4_budget_state"])
        & action.budget.isin(BUDGETS)
        & action.seed.isin(SEEDS)
        & action.scenario.isin(SCENARIOS)
    ].drop(columns=["episode"])
    grouped = frame.groupby(
        ["variant", "method", "budget", "seed", "scenario"], as_index=False
    ).mean(numeric_only=True)
    grouped = grouped.drop(
        columns=[
            col
            for col in action.columns
            if col not in {"method", "budget", "seed", "scenario"} and col in grouped.columns
        ],
        errors="ignore",
    )
    grouped = grouped.merge(
        action, on=["method", "budget", "seed", "scenario"], validate="one_to_one"
    )
    grouped["variant"] = "frozen_valid_v2"
    return grouped


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results/dynamic_shadow_price/summaries"
    out.mkdir(parents=True, exist_ok=True)
    hard_episodes = collect_hard(root / "results/dynamic_shadow_price/hard_coupling")
    hard_episodes.to_csv(out / "hard_coupling_episode_metrics.csv", index=False)
    numeric = [
        col
        for col in hard_episodes.select_dtypes(include=np.number).columns
        if col not in {"seed", "eval_episode", "eval_seed", "budget"}
    ]
    hard_seed = hard_episodes.groupby(
        ["variant", "method", "budget", "seed", "scenario"], as_index=False
    )[numeric].mean()
    frozen_seed = collect_frozen_baselines(root)
    seeds = pd.concat([hard_seed, frozen_seed], ignore_index=True, sort=False)
    seeds.to_csv(out / "hard_coupling_seed_metrics.csv", index=False)
    extended_summary(seeds).to_csv(out / "hard_coupling_summary.csv", index=False)
    comparisons = []
    metrics = [
        "slo_violation_rate",
        "total_cost",
        "completion_rate",
        "action_reallocation_difference",
    ]
    for candidate in ("d1_hard_mask", "d2_hard_mask"):
        for baseline in ("cdba", "b4_budget_state"):
            comparisons.append(
                paired_comparisons(seeds, candidate, baseline, metrics=metrics)
            )
    tests = pd.concat(comparisons, ignore_index=True)
    tests["q_value_bh_global"] = benjamini_hochberg(tests.p_value)
    tests.to_csv(out / "hard_coupling_paired_tests.csv", index=False)
    pareto_table(seeds).to_csv(out / "hard_coupling_pareto.csv", index=False)
    files = sorted(out.glob("hard_coupling_*.csv"))
    write_json(
        out / "hard_coupling_integrity.json",
        {
            "schema": "dynamic_shadow_price.hard_coupling_summary.v1",
            "eligible_runs": 30,
            "episode_rows": len(hard_episodes),
            "seed_rows": len(seeds),
            "budgets": BUDGETS,
            "seeds": SEEDS,
            "scenarios": SCENARIOS,
            "artifacts": {file.name: sha256_file(file) for file in files},
        },
    )


if __name__ == "__main__":
    main()
