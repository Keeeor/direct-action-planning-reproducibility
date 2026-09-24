from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stage2_dynamic_budget.data.trace_windows import select_trace_window
from stage2_dynamic_budget.envs.synthetic_queue_env import SyntheticQueueConfig
from stage2_dynamic_budget.envs.trace_driven_env import TraceDrivenQueueEnv


DATASET_SPECS = {
    "azure2019": {
        "processed_dir": "data/processed/azure_functions_2019",
        "domains": ("http", "async"),
        "split_contract": "days01-08 train; days09-11 validation; days12-14 test",
    },
    "gentd26": {
        "processed_dir": "data/processed/gentd26",
        "domains": ("txt2img", "image_conditioned"),
        "split_contract": "10-minute bins; contiguous 60% train, 20% validation, 20% test",
    },
}


@dataclass(frozen=True)
class TraceDataset:
    name: str
    domains: dict[str, dict[str, np.ndarray]]
    split_contract: str

    @property
    def domain_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.domains))


def load_trace_dataset(project_root: str | Path, dataset: str) -> TraceDataset:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unknown dataset: {dataset}")
    project_root = Path(project_root)
    spec = DATASET_SPECS[dataset]
    processed = project_root / str(spec["processed_dir"])
    domains: dict[str, dict[str, np.ndarray]] = {}
    for domain in spec["domains"]:
        with np.load(processed / f"{domain}.npz") as bundle:
            parts = {
                split: np.asarray(bundle[split], dtype=np.float64).copy()
                for split in ("train", "validation", "test")
            }
        if any(values.ndim != 1 or len(values) == 0 for values in parts.values()):
            raise ValueError(f"invalid processed trace bundle: {domain}")
        domains[str(domain)] = parts
    return TraceDataset(
        name=dataset,
        domains=domains,
        split_contract=str(spec["split_contract"]),
    )


def make_trace_env(
    dataset: TraceDataset,
    domain: str,
    split: str,
    *,
    horizon: int,
    budget: float,
    window_seed: int,
) -> tuple[TraceDrivenQueueEnv, int]:
    trace, start = select_trace_window(dataset.domains[domain][split], horizon, window_seed)
    config = SyntheticQueueConfig(
        horizon=horizon,
        budget=budget,
        scenario=f"{dataset.name}_{domain}_{split}",
        base_capacity=5.0,
        slo_latency=3.0,
    )
    return TraceDrivenQueueEnv(trace, config), start
