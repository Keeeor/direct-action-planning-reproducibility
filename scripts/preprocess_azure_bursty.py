#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2_dynamic_budget.data.azure_functions import (  # noqa: E402
    aggregate_selected_functions,
    fit_and_scale_chronological,
    select_bursty_functions,
)


def main() -> int:
    raw = ROOT / "data" / "raw" / "azure_functions_2019"
    output = ROOT / "data" / "processed" / "azure_functions_2019"
    paths = [raw / f"invocations_per_function_md.anon.d{day:02d}.csv" for day in range(1, 15)]
    selected, selection_audit = select_bursty_functions(paths[:8], top_k=32, min_total=1_000.0)
    daily = {"http_bursty": [], "async_bursty": []}
    for day, path in enumerate(paths, start=1):
        aggregates = aggregate_selected_functions(path, selected)
        for domain, values in aggregates.items():
            daily[domain].append(values)
        print(f"aggregated selected day={day:02d}", flush=True)
    report = {"schema": "stage2.azure_functions_bursty_selection.v1", "selection": selection_audit, "selected_function_ids": selected, "domains": {}}
    for domain, days in daily.items():
        train = np.concatenate(days[:8])
        validation = np.concatenate(days[8:11])
        test = np.concatenate(days[11:14])
        scaled, scaler = fit_and_scale_chronological(
            train, validation, test, target_mean=3.0, max_scaled_value=24.0
        )
        np.savez_compressed(output / f"{domain}.npz", **scaled)
        report["domains"][domain] = {
            "scaler": scaler,
            "lengths": {key: int(len(value)) for key, value in scaled.items()},
            "raw_totals": {"train": float(train.sum()), "validation": float(validation.sum()), "test": float(test.sum())},
        }
    (output / "bursty_selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
