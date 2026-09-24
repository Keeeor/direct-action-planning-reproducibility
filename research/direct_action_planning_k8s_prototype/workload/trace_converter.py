from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROTOTYPE_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import (  # noqa: E402
    load_trace_dataset,
)
from stage2_dynamic_budget.data.trace_windows import select_trace_window  # noqa: E402


@dataclass(frozen=True)
class RateScale:
    source_split: str
    quantile: float
    quantile_value: float
    target_peak_rps: float
    multiplier: float
    max_rps: float


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def fit_rate_scale(
    training: np.ndarray, *, quantile: float, target_peak_rps: float, max_rps: float
) -> RateScale:
    values = np.asarray(training, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("training trace must be a finite non-negative vector")
    q_value = float(np.quantile(values, quantile))
    if q_value <= 0:
        raise ValueError("training quantile must be positive")
    return RateScale(
        source_split="train",
        quantile=float(quantile),
        quantile_value=q_value,
        target_peak_rps=float(target_peak_rps),
        multiplier=float(target_peak_rps) / q_value,
        max_rps=float(max_rps),
    )


def transform_rate(values: np.ndarray, scale: RateScale) -> np.ndarray:
    return np.minimum(np.asarray(values, dtype=np.float64) * scale.multiplier, scale.max_rps)


def request_offsets(rate: float, interval_seconds: float, rng: np.random.Generator) -> np.ndarray:
    count = int(rng.poisson(max(rate, 0.0) * interval_seconds))
    return np.sort(rng.uniform(0.0, interval_seconds, size=count))


def select_activity_window_start(
    values: np.ndarray, *, horizon: int, activity_quantile: float, seed: int
) -> int:
    """Select a pre-registered activity stratum without consulting a policy.

    This is used only when assembling a benchmark replay episode. It never
    enters rate fitting, model training, checkpoint selection, or online state.
    The source split and the chosen start index are retained in the request-plan
    manifest, making test-window selection auditable.
    """

    trace = np.asarray(values, dtype=np.float64)
    if trace.ndim != 1 or horizon <= 0 or trace.size < horizon:
        raise ValueError("trace must be one-dimensional and at least horizon long")
    if not 0.0 <= activity_quantile <= 1.0:
        raise ValueError("activity_quantile must be in [0, 1]")
    prefix = np.concatenate(([0.0], np.cumsum(trace)))
    activity = prefix[horizon:] - prefix[:-horizon]
    target = float(np.quantile(activity, activity_quantile))
    distance = np.abs(activity - target)
    closest = np.flatnonzero(np.isclose(distance, distance.min(), rtol=0.0, atol=1.0e-12))
    if not len(closest):  # pragma: no cover - defensive numerical guard
        raise AssertionError("activity selection produced no candidate")
    return int(closest[np.random.default_rng(seed).integers(len(closest))])


def build_plan(
    *,
    dataset_name: str,
    domain: str,
    split: str,
    horizon: int,
    interval_seconds: float,
    seed: int,
    quantile: float,
    target_peak_rps: float,
    max_rps: float,
    window_start: int | None = None,
    activity_quantile: float | None = None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict], dict]:
    dataset = load_trace_dataset(project_root, dataset_name)
    if domain not in dataset.domains:
        raise ValueError(f"unknown domain {domain!r} for {dataset_name}")
    scale = fit_rate_scale(
        dataset.domains[domain]["train"],
        quantile=quantile,
        target_peak_rps=target_peak_rps,
        max_rps=max_rps,
    )
    split_values = np.asarray(dataset.domains[domain][split], dtype=np.float64)
    if window_start is not None and activity_quantile is not None:
        raise ValueError("window_start and activity_quantile are mutually exclusive")
    if window_start is not None:
        start = int(window_start)
        if start < 0 or start + horizon > len(split_values):
            raise ValueError("explicit window_start falls outside the requested split")
        trace = split_values[start : start + horizon]
        selection = {"kind": "explicit_start", "window_start": start}
    elif activity_quantile is not None:
        start = select_activity_window_start(
            split_values, horizon=horizon, activity_quantile=float(activity_quantile), seed=seed
        )
        trace = split_values[start : start + horizon]
        selection = {
            "kind": "activity_quantile",
            "activity_quantile": float(activity_quantile),
            "window_start": start,
        }
    else:
        trace, start = select_trace_window(split_values, horizon, seed)
        selection = {"kind": "seeded_random", "window_start": int(start)}
    rates = transform_rate(trace, scale)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    request_id = 0
    for step, rate in enumerate(rates):
        for offset in request_offsets(float(rate), interval_seconds, rng):
            rows.append(
                {
                    "request_id": request_id,
                    "step": step,
                    "scheduled_offset_seconds": float(step * interval_seconds + offset),
                    "target_rps": float(rate),
                    "payload": f"{dataset_name}:{domain}:{split}:{seed}:{request_id}",
                }
            )
            request_id += 1
    manifest = {
        "schema": "dap.k8s.request_plan.v1",
        "dataset": dataset_name,
        "domain": domain,
        "split": split,
        "split_contract": dataset.split_contract,
        "horizon": horizon,
        "interval_seconds": interval_seconds,
        "seed": seed,
        "window_start": int(start),
        "window_selection": selection,
        "source_array_sha256": sha256_array(dataset.domains[domain][split]),
        "training_array_sha256": sha256_array(dataset.domains[domain]["train"]),
        "rate_scale": asdict(scale),
        "request_count": len(rows),
        "rate_by_step": [float(value) for value in rates],
    }
    return rows, manifest


def write_plan(rows: Iterable[dict], manifest: dict, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest["plan_sha256"] = "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("azure2019", "gentd26"), required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--target-peak-rps", type=float, required=True)
    parser.add_argument("--max-rps", type=float, required=True)
    parser.add_argument("--window-start", type=int)
    parser.add_argument("--activity-quantile", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = build_plan(
        dataset_name=args.dataset,
        domain=args.domain,
        split=args.split,
        horizon=args.horizon,
        interval_seconds=args.interval_seconds,
        seed=args.seed,
        quantile=args.quantile,
        target_peak_rps=args.target_peak_rps,
        max_rps=args.max_rps,
        window_start=args.window_start,
        activity_quantile=args.activity_quantile,
    )
    output, metadata = write_plan(rows, manifest, args.output)
    print(json.dumps({"plan": str(output), "manifest": str(metadata), "requests": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
