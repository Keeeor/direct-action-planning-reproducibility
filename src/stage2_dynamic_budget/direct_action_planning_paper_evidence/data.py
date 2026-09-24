from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import (
    DATASET_SPECS,
    TraceDataset,
)


DEVELOPMENT_SPLITS = (
    "validation_fit",
    "validation_select",
    "validation_eval",
)


def development_input_hashes(
    project_root: str | Path,
    dataset: str,
) -> dict[str, str]:
    """Hash only arrays used by development runs; the formal test array is omitted."""

    if dataset not in DATASET_SPECS:
        raise ValueError(f"unknown dataset: {dataset}")
    processed = Path(project_root) / str(DATASET_SPECS[dataset]["processed_dir"])
    hashes: dict[str, str] = {}
    for domain in DATASET_SPECS[dataset]["domains"]:
        with np.load(processed / f"{domain}.npz") as bundle:
            for split in ("train", "validation"):
                array = np.asarray(bundle[split], dtype=np.float64)
                digest = hashlib.sha256()
                digest.update(f"{dataset}/{domain}/{split}".encode())
                digest.update(array.tobytes(order="C"))
                hashes[f"{domain}/{split}"] = "sha256:" + digest.hexdigest()
    return hashes


def partition_validation_trace(
    values: np.ndarray,
    *,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Create chronological, non-overlapping development roles.

    The 40/30/30 partition keeps model fitting, candidate selection, and development
    evaluation on different time blocks. Every block must support a complete episode.
    """

    trace = np.asarray(values, dtype=np.float64)
    if trace.ndim != 1 or not np.all(np.isfinite(trace)) or np.any(trace < 0):
        raise ValueError("validation trace must be finite, non-negative, and one-dimensional")
    if horizon <= 0 or len(trace) < 3 * horizon:
        raise ValueError("validation trace must contain at least three complete horizons")

    fit_end = max(horizon, int(np.floor(0.40 * len(trace))))
    select_end = max(fit_end + horizon, int(np.floor(0.70 * len(trace))))
    select_end = min(select_end, len(trace) - horizon)
    if fit_end > select_end - horizon:
        fit_end = select_end - horizon

    parts = {
        "validation_fit": trace[:fit_end].copy(),
        "validation_select": trace[fit_end:select_end].copy(),
        "validation_eval": trace[select_end:].copy(),
    }
    if any(len(part) < horizon for part in parts.values()):
        raise AssertionError("development partition produced an incomplete horizon")
    return parts


def load_development_trace_dataset(
    project_root: str | Path,
    dataset: str,
    *,
    horizon: int,
) -> TraceDataset:
    """Load train plus development roles without materializing the formal test array."""

    if dataset not in DATASET_SPECS:
        raise ValueError(f"unknown dataset: {dataset}")
    project_root = Path(project_root)
    spec = DATASET_SPECS[dataset]
    processed = project_root / str(spec["processed_dir"])
    domains: dict[str, dict[str, np.ndarray]] = {}
    for domain in spec["domains"]:
        with np.load(processed / f"{domain}.npz") as bundle:
            train = np.asarray(bundle["train"], dtype=np.float64).copy()
            validation = np.asarray(bundle["validation"], dtype=np.float64).copy()
        if train.ndim != 1 or len(train) < horizon:
            raise ValueError(f"invalid training trace bundle: {domain}")
        domains[str(domain)] = {
            "train": train,
            **partition_validation_trace(validation, horizon=horizon),
        }
    return TraceDataset(
        name=dataset,
        domains=domains,
        split_contract=(
            f"{spec['split_contract']}; validation is isolated chronologically into "
            "40% fit, 30% select, and 30% development-evaluation blocks; test is not loaded"
        ),
    )
