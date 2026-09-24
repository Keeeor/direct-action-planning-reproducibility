from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
)
from stage2_dynamic_budget.direct_action_planning.learning import EmpiricalActionModel
from stage2_dynamic_budget.direct_action_planning.planning import BudgetValueTable, one_step_plan

from .model import StructuredActionEffectModel
from .planning import ensemble_one_step_plan, structured_one_step_plan


def _load_value(path: Path) -> BudgetValueTable:
    with np.load(path) as data:
        return BudgetValueTable(data["values"], str(data["source"].item()))


def _load_old_model(path: Path) -> EmpiricalActionModel:
    with np.load(path) as data:
        return EmpiricalActionModel(
            data["next_load_probabilities"],
            data["next_queue_probabilities"],
            data["rewards"],
            data["costs"],
            int(data["samples_per_state_action"]),
            float(data["smoothing"]),
        )


def _load_structured(path: Path) -> StructuredActionEffectModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = StructuredActionEffectModel(
        int(payload["horizon"]),
        int(payload["n_loads"]),
        int(payload["n_actions"]),
        int(payload["hidden_dim"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def run_cold_planning_benchmark(
    project_root: str | Path,
    run_id: str = "minimal_v1",
    states_per_cell: int = 512,
) -> Path:
    root = Path(project_root).resolve()
    source = root / "results/direct_action_planning_repair" / run_id
    output = source / "cold_planning_v1"
    if output.exists():
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed":
            return output
        raise RuntimeError(f"cold benchmark output exists and is incomplete: {output}")
    output.mkdir()
    config = json.loads((source / "config.json").read_text())
    rows = []
    for scenario in config["scenarios"]:
        env = config["environment"]
        mdp = ActionConditionedBudgetMDP(
            ACBADPConfig(
                horizon=int(env["horizon"]),
                max_budget=int(env["max_budget"]),
                max_queue=int(env["max_queue"]),
                scenario=scenario,
                gamma=float(env["gamma"]),
                action_costs=tuple(env["action_costs"]),
                action_capacity=tuple(env["action_capacity"]),
            )
        )
        all_states = list(
            np.ndindex(
                mdp.config.horizon,
                mdp.n_loads,
                mdp.config.max_queue + 1,
                mdp.config.max_budget + 1,
            )
        )
        for seed in config["seeds"]:
            cell = source / "cells" / f"{scenario}__s{seed}"
            value = _load_value(cell / "learned_value.npz")
            old = _load_old_model(cell / "old_model.npz")
            full = _load_structured(cell / "models/full_repair.pt")
            ensemble = [
                _load_structured(path)
                for path in sorted((cell / "models").glob("ensemble_member_*.pt"))
            ]
            rng = np.random.default_rng(20_260_803 + int(seed) + 101 * list(config["scenarios"]).index(scenario))
            selected = rng.choice(len(all_states), size=min(states_per_cell, len(all_states)), replace=False)
            for state_index in selected:
                state = all_states[int(state_index)]
                methods = {
                    "learned_value_branch": lambda: one_step_plan(mdp, value, *state),
                    "original_learned_model": lambda: one_step_plan(
                        mdp, value, *state, learned_model=old
                    ),
                    "full_repair": lambda: structured_one_step_plan(
                        mdp, value, full, *state
                    ),
                    "ensemble_mean": lambda: ensemble_one_step_plan(
                        mdp, value, ensemble, *state
                    ),
                }
                for method, operation in methods.items():
                    started = time.perf_counter_ns()
                    operation()
                    elapsed = (time.perf_counter_ns() - started) / 1.0e6
                    rows.append(
                        {
                            "method": method,
                            "scenario": scenario,
                            "seed": seed,
                            "t": state[0],
                            "load": state[1],
                            "queue": state[2],
                            "remaining_budget": state[3],
                            "cold_planning_ms": elapsed,
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = raw.groupby("method", as_index=False).cold_planning_ms.agg(
        ["count", "mean", "median", lambda values: float(np.quantile(values, 0.95))]
    ).reset_index().rename(columns={"<lambda_0>": "p95"})
    raw.to_csv(output / "cold_latency_samples.csv.gz", index=False, compression="gzip")
    summary.to_csv(output / "cold_latency_summary.csv", index=False)
    files = [output / "cold_latency_samples.csv.gz", output / "cold_latency_summary.csv"]
    payload = {
        "schema": "direct_action_planning_repair.cold_benchmark.v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "states_per_cell": states_per_cell,
        "cells": len(config["scenarios"]) * len(config["seeds"]),
        "timing_note": "single-thread CPU cold uncached calls; descriptive, not inferential",
        "source_manifest_sha256": "sha256:"
        + hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "output_sha256": {
            path.name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
