#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dap.analysis.aggregation import (  # noqa: E402
    collect_runs,
    paired_comparisons,
    pareto_table,
    seed_aggregate,
    summary_table,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="formal")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--candidate", default="cdba")
    parser.add_argument("--baseline", default="b4_budget_state")
    parser.add_argument("--skip-paired", action="store_true")
    args = parser.parse_args()
    run_root = ROOT / "results" / "raw_logs" / args.tier
    output = ROOT / "results" / "summaries"
    output.mkdir(parents=True, exist_ok=True)
    episodes, runtime = collect_runs(run_root, args.prefix)
    seeds = seed_aggregate(episodes)
    summary = summary_table(seeds)
    comparisons = (
        paired_comparisons(seeds, args.candidate, args.baseline)
        if not args.skip_paired
        else None
    )
    pareto = pareto_table(seeds)
    artifacts = {
        "episode_metrics": episodes,
        "seed_metrics": seeds,
        "summary": summary,
        "pareto": pareto,
        "runtime": runtime,
    }
    if comparisons is not None:
        artifacts["paired"] = comparisons
    for suffix, frame in artifacts.items():
        frame.to_csv(output / f"{args.name}_{suffix}.csv", index=False)
    print(
        f"runs={episodes.run_id.nunique()} episodes={len(episodes)} "
        f"seed_cells={len(seeds)} comparisons={len(comparisons) if comparisons is not None else 0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
