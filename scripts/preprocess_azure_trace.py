#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2_dynamic_budget.data.azure_functions import (  # noqa: E402
    aggregate_invocation_file,
    fit_and_scale_chronological,
)


def main() -> int:
    raw = ROOT / "data" / "raw" / "azure_functions_2019"
    output = ROOT / "data" / "processed" / "azure_functions_2019"
    output.mkdir(parents=True, exist_ok=True)
    per_domain: dict[str, list[np.ndarray]] = {"all": [], "http": [], "async": []}
    for day in range(1, 15):
        path = raw / f"invocations_per_function_md.anon.d{day:02d}.csv"
        aggregates = aggregate_invocation_file(path)
        for domain, values in aggregates.items():
            per_domain[domain].append(values)
        print(f"aggregated day={day:02d}", flush=True)
    metadata = {
        "schema": "stage2.azure_functions_trace.v1",
        "chronological_split": {"train_days": [1, 8], "validation_days": [9, 11], "test_days": [12, 14]},
        "domains": {},
    }
    for domain, days in per_domain.items():
        train = np.concatenate(days[:8])
        validation = np.concatenate(days[8:11])
        test = np.concatenate(days[11:14])
        scaled, scaler = fit_and_scale_chronological(train, validation, test)
        np.savez_compressed(output / f"{domain}.npz", **scaled)
        metadata["domains"][domain] = {
            "lengths": {key: int(len(value)) for key, value in scaled.items()},
            "scaler": scaler,
            "raw_totals": {
                "train": float(train.sum()),
                "validation": float(validation.sum()),
                "test": float(test.sum()),
            },
        }
    (output / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
