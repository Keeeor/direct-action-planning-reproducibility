from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .azure_functions import fit_and_scale_chronological


REQUEST_DOMAINS = {
    "txt2img": ("TXT_2_IMG",),
    "image_conditioned": ("IMG_2_IMG", "INPAINTING"),
}


@dataclass(frozen=True)
class GenTD26Grid:
    origin: float
    interval_seconds: float
    size: int

    @property
    def timestamps(self) -> np.ndarray:
        return self.origin + self.interval_seconds * np.arange(self.size, dtype=np.float64)


def infer_native_grid(
    timestamp_tables: list[np.ndarray],
    *,
    expected_interval: float = 57.0,
) -> GenTD26Grid:
    values = [np.asarray(table, dtype=np.float64) for table in timestamp_tables]
    if not values or any(table.ndim != 1 or not len(table) for table in values):
        raise ValueError("timestamp tables must be non-empty vectors")
    unique = np.unique(np.concatenate(values))
    if not np.all(np.isfinite(unique)):
        raise ValueError("timestamps must be finite")
    positive_deltas = np.diff(unique)
    positive_deltas = positive_deltas[positive_deltas > 0]
    interval = float(np.median(positive_deltas))
    if not np.isclose(interval, expected_interval, atol=1.0e-8):
        raise ValueError(f"unexpected GenTD26 sampling interval: {interval}")
    origin = float(unique.min())
    final = float(unique.max())
    size = int(round((final - origin) / interval)) + 1
    return GenTD26Grid(origin=origin, interval_seconds=interval, size=size)


def aggregate_qps_domain(
    frame: pd.DataFrame,
    grid: GenTD26Grid,
    request_type: str,
) -> np.ndarray:
    required = {"timestamp_anon", "value", "request_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing QPS columns: {sorted(missing)}")
    selected = frame.loc[frame.request_type == request_type, ["timestamp_anon", "value"]].copy()
    if selected.empty:
        raise ValueError(f"no rows for request type: {request_type}")
    timestamps = pd.to_numeric(selected.timestamp_anon, errors="raise").to_numpy(np.float64)
    values = pd.to_numeric(selected.value, errors="raise").to_numpy(np.float64)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("QPS must be finite and non-negative")
    positions = (timestamps - grid.origin) / grid.interval_seconds
    indices = np.rint(positions).astype(np.int64)
    if np.max(np.abs(positions - indices)) > 1.0e-8:
        raise ValueError("QPS timestamps do not align to the native grid")
    if np.any(indices < 0) or np.any(indices >= grid.size):
        raise ValueError("QPS timestamp falls outside the monitoring grid")
    aggregated = np.zeros(grid.size, dtype=np.float64)
    np.add.at(aggregated, indices, values)
    return aggregated


def chronological_three_way_split(
    values: np.ndarray,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must lie in (0, 1)")
    train_end = int(np.floor(len(values) * train_fraction))
    validation_end = train_end + int(np.floor(len(values) * validation_fraction))
    if train_end <= 0 or validation_end <= train_end or validation_end >= len(values):
        raise ValueError("split fractions produce an empty partition")
    return {
        "train": values[:train_end].copy(),
        "validation": values[train_end:validation_end].copy(),
        "test": values[validation_end:].copy(),
    }


def aggregate_request_arrivals(
    frame: pd.DataFrame,
    request_types: tuple[str, ...],
    *,
    frequency: str = "10min",
    full_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    required = {"gmt_create", "predict_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing request columns: {sorted(missing)}")
    timestamps = pd.to_datetime(frame.gmt_create, errors="raise")
    if full_index is None:
        full_index = pd.date_range(
            timestamps.min().floor(frequency),
            timestamps.max().floor(frequency),
            freq=frequency,
        )
    selected = frame.loc[frame.predict_type.isin(request_types)].copy()
    selected["time_bin"] = pd.to_datetime(selected.gmt_create, errors="raise").dt.floor(
        frequency
    )
    return selected.groupby("time_bin").size().reindex(full_index, fill_value=0).astype(float)


def preprocess_gentd26(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    target_mean: float = 6.0,
    train_clip_quantile: float = 0.995,
    max_scaled_value: float = 64.0,
) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    qps = pd.read_csv(raw_dir / "qps.csv")
    requests = pd.read_csv(raw_dir / "lora_request_trace.csv")
    monitor_files = [
        "pod_gpu_duty_cycle_anon.csv",
        "queue_size_raw_anon.csv",
        "queue_rt_raw_anon.csv",
        "pipeline_inference_data_anon.csv",
    ]
    timestamp_tables = [qps.timestamp_anon.to_numpy(np.float64)]
    audit_tables: dict[str, dict[str, object]] = {}
    for name in monitor_files:
        frame = pd.read_csv(raw_dir / name)
        timestamps = pd.to_numeric(frame.timestamp_anon, errors="raise").to_numpy(np.float64)
        values = pd.to_numeric(frame.value, errors="coerce")
        timestamp_tables.append(timestamps)
        audit_tables[name] = {
            "rows": int(len(frame)),
            "timestamp_min": float(np.min(timestamps)),
            "timestamp_max": float(np.max(timestamps)),
            "missing_value": int(values.isna().sum()),
            "negative_value": int((values < 0).sum()),
            "container_count": int(frame.container_ip.nunique(dropna=True)),
            "observed_only_not_policy_input": True,
        }
    grid = infer_native_grid(timestamp_tables)
    processed_dir.mkdir(parents=True, exist_ok=True)
    request_times = pd.to_datetime(requests.gmt_create, errors="raise")
    request_index = pd.date_range(
        request_times.min().floor("10min"),
        request_times.max().floor("10min"),
        freq="10min",
    )
    domain_metadata: dict[str, object] = {}
    for domain, request_types in REQUEST_DOMAINS.items():
        raw_trace = aggregate_request_arrivals(
            requests,
            request_types,
            frequency="10min",
            full_index=request_index,
        ).to_numpy(np.float64)
        raw_parts = chronological_three_way_split(raw_trace)
        scaled, fitted = fit_and_scale_chronological(
            raw_parts["train"],
            raw_parts["validation"],
            raw_parts["test"],
            target_mean=target_mean,
            train_clip_quantile=train_clip_quantile,
            max_scaled_value=max_scaled_value,
        )
        np.savez_compressed(
            processed_dir / f"{domain}.npz",
            train=scaled["train"],
            validation=scaled["validation"],
            test=scaled["test"],
        )
        domain_metadata[domain] = {
            "request_types": list(request_types),
            "raw_nonzero_fraction": float(np.mean(raw_trace > 0)),
            "raw_mean": float(np.mean(raw_trace)),
            "raw_max": float(np.max(raw_trace)),
            "partition_lengths": {key: int(len(value)) for key, value in scaled.items()},
            "partition_scaled_means": {
                key: float(np.mean(value)) for key, value in scaled.items()
            },
            "preprocessing": fitted,
        }
    request_size = len(request_index)
    train_end = int(np.floor(request_size * 0.60))
    validation_end = train_end + int(np.floor(request_size * 0.20))
    return {
        "schema": "stage2.gentd26_trace.v1",
        "upstream_repository": "https://github.com/alibaba/clusterdata",
        "upstream_revision": "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71",
        "fitness": "FIT_WITH_LIMITATIONS",
        "redistribution": "do_not_redistribute_raw_files; upstream README permits research/study but no standard LICENSE file was present",
        "monitoring_grid": {
            "origin": grid.origin,
            "interval_seconds": grid.interval_seconds,
            "size": grid.size,
            "duration_hours": (grid.size - 1) * grid.interval_seconds / 3600.0,
        },
        "request_grid": {
            "frequency": "10min",
            "size": request_size,
            "timestamp_min": str(request_index.min()),
            "timestamp_max": str(request_index.max()),
            "duration_hours": float(
                (request_index.max() - request_index.min()).total_seconds() / 3600.0
            ),
            "submission_count": int(len(requests)),
            "final_status_not_used": True,
            "execution_time_not_used": True,
        },
        "split": {
            "contract": "contiguous 60% train, 20% validation, 20% final test",
            "train_indices": [0, train_end],
            "validation_indices": [train_end, validation_end],
            "test_indices": [validation_end, request_size],
            "window_crossing_forbidden": True,
        },
        "domains": domain_metadata,
        "monitoring_audit": audit_tables,
        "qps_monitoring_audit": {
            "rows": int(len(qps)),
            "request_types": sorted(qps.request_type.unique().tolist()),
            "formal_experiment_input": False,
            "reason": "the monitoring tail is nearly idle and would make the scheduling test degenerate",
        },
        "simulator_boundary": "request submission counts are exogenous; observed QPS/queue/resource/latency and request outcomes are audit-only",
    }
