#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dap.analysis.statistics import (  # noqa: E402
    benjamini_hochberg,
    paired_bootstrap,
    paired_effect_size,
    paired_wilcoxon,
)


def main() -> int:
    rows = []
    for dataset in ("final_synthetic", "final_trace"):
        frame = pd.read_csv(ROOT / "results" / "summaries" / f"{dataset}_seed_metrics.csv")
        frame["high_low_quota_difference"] = frame.high_risk_budget_mean - frame.low_risk_budget_mean
        for method in ("cdba", "cdba_discrete"):
            subset = frame[frame.method == method]
            for metric in ("risk_budget_correlation", "high_low_quota_difference"):
                values = subset.groupby("seed")[metric].mean().dropna().to_numpy(float)
                zero = np.zeros_like(values)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        **paired_bootstrap(values, zero),
                        "cohens_dz": paired_effect_size(values, zero),
                        "p_value": paired_wilcoxon(values, zero),
                    }
                )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(result.p_value)
    result.to_csv(ROOT / "results" / "summaries" / "allocator_association_tests.csv", index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
