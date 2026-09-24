#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_dynamic_budget.analysis.statistics import (
    benjamini_hochberg,
    paired_bootstrap,
    paired_effect_size,
    paired_wilcoxon,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "summaries"


def analyze(run_directories: list[Path], name: str) -> None:
    episode_rows = []
    for run_dir in run_directories:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        steps = pd.read_csv(
            run_dir / "steps.csv.gz",
            usecols=lambda column: column
            in {"scenario", "episode", "risk_level", "resource_cost"},
        )
        for keys, group in steps.groupby(["scenario", "episode"]):
            low, high = group.risk_level.quantile([0.25, 0.75])
            low_cost = float(group.loc[group.risk_level <= low, "resource_cost"].mean())
            high_cost = float(group.loc[group.risk_level >= high, "resource_cost"].mean())
            episode_rows.append(
                {
                    "method": config["method"],
                    "budget": config["budget"],
                    "seed": config["seed"],
                    "scenario": keys[0],
                    "episode": keys[1],
                    "low_risk_action_cost": low_cost,
                    "high_risk_action_cost": high_cost,
                    "action_reallocation_difference": high_cost - low_cost,
                    "action_reallocation_ratio": (
                        high_cost / low_cost
                        if low_cost > 1e-12
                        else np.inf if high_cost > 1e-12 else np.nan
                    ),
                }
            )
    episodes = pd.DataFrame(episode_rows)
    seeds = episodes.groupby(["method", "budget", "seed", "scenario"], as_index=False).mean(numeric_only=True)
    summary = seeds.groupby(["method", "budget"], as_index=False)[
        ["low_risk_action_cost", "high_risk_action_cost", "action_reallocation_difference"]
    ].agg(["mean", "std"])
    episodes.to_csv(OUT / f"{name}_action_reallocation_episodes.csv", index=False)
    seeds.to_csv(OUT / f"{name}_action_reallocation_seed_metrics.csv", index=False)
    summary.to_csv(OUT / f"{name}_action_reallocation_summary.csv")
    tests = []
    per_seed = seeds.groupby(["method", "seed"], as_index=False).action_reallocation_difference.mean()
    for method, group in per_seed.groupby("method"):
        values = group.action_reallocation_difference.to_numpy(float)
        zero = np.zeros_like(values)
        tests.append(
            {
                "method": method,
                **paired_bootstrap(values, zero),
                "cohens_dz": paired_effect_size(values, zero),
                "p_value": paired_wilcoxon(values, zero),
            }
        )
    tests = pd.DataFrame(tests)
    tests["q_value_bh"] = benjamini_hochberg(tests.p_value)
    tests.to_csv(OUT / f"{name}_action_reallocation_tests.csv", index=False)
    print(f"{name}: runs={len(run_directories)} episodes={len(episodes)}", flush=True)


def main() -> int:
    formal_root = ROOT / "results" / "raw_logs" / "formal"
    synthetic = sorted(
        run_dir
        for run_dir in formal_root.glob("formal__valid_v2__*")
        if run_dir.is_dir()
        and any(token in run_dir.name for token in ("b3_lagrangian", "b4_budget_state", "cdba__", "cdba_discrete"))
    )
    synthetic += sorted(
        run_dir
        for run_dir in formal_root.glob("formal__valid_v2_b5_corrected__b5_fixed_local__*")
        if run_dir.is_dir()
    )
    analyze(synthetic, "final_synthetic")
    trace_root = ROOT / "results" / "raw_logs" / "trace_formal"
    trace = sorted(
        run_dir
        for run_dir in trace_root.glob("trace_formal__train_combined__*")
        if run_dir.is_dir()
        and any(token in run_dir.name for token in ("b3_lagrangian", "b4_budget_state", "b5_fixed_local", "cdba__", "cdba_discrete"))
    )
    analyze(trace, "final_trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
