from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Callable

import numpy as np

from dap.direct_action_planning_dataset_validation.data import TraceDataset

from .controllers import (
    CausalMPCController,
    Controller,
    LyapunovDPPController,
    PIDBudgetController,
    ReactiveThresholdController,
)
from .evaluation import evaluate_agent


def controller_candidates() -> dict[str, list[Controller]]:
    return {
        "reactive_threshold": [
            ReactiveThresholdController(queue_threshold=q, load_threshold=l, slo_threshold=s, hysteresis=h)
            for q, l, s, h in product((2.0, 6.0, 12.0), (1.0, 1.25), (0.05, 0.20), (0.02, 0.10))
        ],
        "pid_budget_autoscaler": [
            PIDBudgetController(kp=kp, ki=ki, kd=kd, budget_weight=bw)
            for kp, ki, kd, bw in product((0.25, 0.55, 0.90), (0.0, 0.04), (0.0, 0.15), (0.5, 1.0))
        ],
        "causal_mpc": [
            CausalMPCController(horizon=h, forecast_decay=d, cost_weight=w)
            for h, d, w in product((2, 4, 6), (0.4, 0.75), (0.05, 0.15, 0.35))
        ],
        "lyapunov_dpp": [
            LyapunovDPPController(penalty_weight=p, queue_weight=q)
            for p, q in product((0.1, 0.35, 0.8), (0.5, 1.0))
        ],
    }


def tune_controllers(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    episodes_per_domain: int,
    families: set[str] | None = None,
) -> tuple[dict[str, Controller], list[dict[str, float | int | str | dict]]]:
    """Select each controller on a tuning-window seed disjoint from evaluation windows."""
    selected: dict[str, Controller] = {}
    records: list[dict[str, float | int | str | dict]] = []
    for family, candidates in controller_candidates().items():
        if families is not None and family not in families:
            continue
        family_rows = []
        for candidate_index, candidate in enumerate(candidates):
            episodes, _ = evaluate_agent(
                dataset,
                candidate,
                split="validation",
                horizon=horizon,
                budget=budget,
                seed=seed,
                episodes_per_domain=episodes_per_domain,
            )
            returns = np.asarray([row["discounted_return"] for row in episodes], dtype=np.float64)
            completion = np.asarray([row["completion_ratio"] for row in episodes], dtype=np.float64)
            slo = np.asarray([row["slo_violation_rate"] for row in episodes], dtype=np.float64)
            cost = np.asarray([row["total_cost"] for row in episodes], dtype=np.float64)
            family_rows.append({
                "family": family,
                "candidate_index": candidate_index,
                "parameters": asdict(candidate),
                "mean_discounted_return": float(returns.mean()),
                "mean_completion": float(completion.mean()),
                "mean_slo": float(slo.mean()),
                "mean_cost": float(cost.mean()),
            })
        # Return is primary; completion and SLO provide deterministic tie-breaks.
        best = max(
            family_rows,
            key=lambda row: (
                row["mean_discounted_return"],
                row["mean_completion"],
                -row["mean_slo"],
                -row["mean_cost"],
                -row["candidate_index"],
            ),
        )
        selected[family] = candidates[int(best["candidate_index"])]
        for row in family_rows:
            row["selected"] = bool(row["candidate_index"] == best["candidate_index"])
            records.append(row)
    return selected, records
