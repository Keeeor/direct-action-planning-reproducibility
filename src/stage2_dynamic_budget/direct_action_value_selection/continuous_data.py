from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from .continuous import OBS_COLUMNS


def _arrival_tape(env: Any) -> np.ndarray:
    base = env.unwrapped
    return np.asarray(base._arrivals[base.t :], dtype="<f8")


def _tape_hash(env: Any) -> str:
    return "sha256:" + hashlib.sha256(_arrival_tape(env).tobytes()).hexdigest()


def collect_k1_branch_episode(
    env: Any,
    reference_agent: Any,
    *,
    split: str,
    scenario: str,
    budget: float,
    budget_scale: float,
    seed: int,
    trajectory_seed: int,
    episode: int,
    gamma: float,
    action_costs: np.ndarray,
    device: torch.device,
    extra_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Collect all feasible first-action labels from one reference trajectory."""

    observation, _ = env.reset(seed=trajectory_seed)
    if hasattr(reference_agent, "reset_budget_controller"):
        reference_agent.reset_budget_controller()
    rows: list[dict[str, object]] = []
    extra_metadata = extra_metadata or {}
    while True:
        base = env.unwrapped
        t = int(base.t)
        remaining = max(float(observation[-2]) * budget_scale, 0.0)
        feasible = np.flatnonzero(np.asarray(action_costs) <= remaining + 1e-8)
        state_id = (
            f"{split}|{scenario}|b={budget:.6g}|seed={seed}|"
            f"trajectory={trajectory_seed}|episode={episode}|t={t}"
        )
        tape_hash = _tape_hash(env)
        first_exogenous = float(_arrival_tape(env)[0])
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            reference_output = reference_agent.act(
                obs_tensor,
                deterministic=False,
                advance_budget_state=False,
            )
        reference_action = int(reference_output.action.item())
        for action in feasible:
            branch = copy.deepcopy(env)
            next_observation, reward, terminated, truncated, info = branch.step(int(action))
            terminal_value = 0.0
            if not (terminated or truncated):
                with torch.no_grad():
                    value_output = reference_agent.act(
                        torch.as_tensor(
                            next_observation, dtype=torch.float32, device=device
                        ).unsqueeze(0),
                        deterministic=True,
                        advance_budget_state=False,
                    )
                terminal_value = float(value_output.reward_value.item())
            row: dict[str, object] = {
                "state_id": state_id,
                "split": split,
                "scenario": scenario,
                "budget": float(budget),
                "seed": int(seed),
                "trajectory_seed": int(trajectory_seed),
                "episode": int(episode),
                "t": t,
                "remaining_budget": remaining,
                "remaining_horizon": int(base.config.horizon - t),
                "action": int(action),
                "action_cost": float(action_costs[action]),
                "Q_branch": float(reward + gamma * terminal_value),
                "immediate_reward": float(reward),
                "terminal_value": terminal_value,
                "reference_action": reference_action,
                "first_exogenous": first_exogenous,
                "random_tape_sha256": tape_hash,
                "next_queue": float(info["queue_length"]),
                "k": 1,
                **extra_metadata,
            }
            row.update(
                {
                    column: float(observation[index])
                    for index, column in enumerate(OBS_COLUMNS)
                }
            )
            rows.append(row)
        observation, _, terminated, truncated, _ = env.step(reference_action)
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


def validate_continuous_branch_data(
    frame: pd.DataFrame, action_costs: np.ndarray
) -> dict[str, object]:
    required = {
        "state_id",
        "split",
        "remaining_budget",
        "action",
        "Q_branch",
        "first_exogenous",
        "random_tape_sha256",
        *OBS_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"continuous branch data is missing: {sorted(missing)}")
    cross_split = int((frame.groupby("state_id").split.nunique() > 1).sum())
    crn_failures = int(
        (
            frame.groupby("state_id")[["first_exogenous", "random_tape_sha256"]]
            .nunique()
            .gt(1)
            .any(axis=1)
        ).sum()
    )
    incomplete = 0
    costs = np.asarray(action_costs, dtype=float)
    for _, group in frame.groupby("state_id", sort=False):
        budget = float(group.remaining_budget.iloc[0])
        expected = set(np.flatnonzero(costs <= budget + 1e-8).tolist())
        if set(group.action.astype(int)) != expected:
            incomplete += 1
    nonfinite = int((~np.isfinite(frame.Q_branch.to_numpy(dtype=float))).sum())
    checks = {
        "cross_split_state_count": cross_split,
        "common_random_number_failure_count": crn_failures,
        "incomplete_action_group_count": incomplete,
        "nonfinite_label_count": nonfinite,
    }
    return {
        "schema": "direct_action_value_selection.continuous_split_integrity.v1",
        "status": "PASS" if not any(checks.values()) else "FAIL",
        **checks,
        "rows": int(len(frame)),
        "states": int(frame.state_id.nunique()),
        "split_states": {
            key: int(value)
            for key, value in frame.groupby("split").state_id.nunique().items()
        },
    }
