from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from dap.direct_action_planning_dataset_benchmark import (
    budgeted,
    cpo,
    evaluation,
    rl,
)

from .calibrated_protocol import collect_calibrated_branch_dataset
from .environment import make_calibrated_trace_env


class TrainingConstraintEnv:
    """Expose an SLO-rate CMDP cost while preserving the real resource budget."""

    def __init__(self, env, *, slo_constraint_rate: float):
        rate = float(slo_constraint_rate)
        if not 0.0 < rate <= 1.0:
            raise ValueError("slo_constraint_rate must be in (0, 1]")
        self._env = env
        self.slo_constraint_rate = rate
        self._constraint_scale = float(env.config.budget) / (
            float(env.config.horizon) * rate
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self._env.step(action)
        mapped = dict(info)
        mapped["actual_resource_cost"] = float(info["resource_cost"])
        mapped["resource_cost"] = (
            float(info["slo_violation"]) * self._constraint_scale
        )
        mapped["constraint_signal"] = "slo_violation_rate"
        mapped["constraint_target_rate"] = self.slo_constraint_rate
        return observation, reward, terminated, truncated, mapped


@contextmanager
def calibrated_baseline_runtime(
    *,
    quantile: float,
    training_constraint: str = "resource_cost",
    slo_constraint_rate: float | None = None,
) -> Iterator[None]:
    """Temporarily route frozen baseline algorithms through the calibrated env."""

    if training_constraint not in {"resource_cost", "slo_violation_rate"}:
        raise ValueError("unsupported training_constraint")
    if training_constraint == "slo_violation_rate" and slo_constraint_rate is None:
        raise ValueError("slo_constraint_rate is required for the SLO constraint")

    originals = {
        "rl_factory": rl._factory,
        "cpo_factory": cpo._factory,
        "evaluation_factory": evaluation.make_trace_env,
        "budgeted_collection": budgeted.collect_branch_dataset,
    }

    def training_factory(dataset, domain_names, horizon, budget, seed):
        domain = domain_names[abs(int(seed)) % len(domain_names)]
        env, _, _ = make_calibrated_trace_env(
            dataset,
            domain,
            "train",
            horizon=horizon,
            budget=budget,
            window_seed=seed,
            quantile=quantile,
        )
        if training_constraint == "slo_violation_rate":
            return TrainingConstraintEnv(
                env,
                slo_constraint_rate=float(slo_constraint_rate),
            )
        return env

    def evaluation_factory(
        dataset,
        domain,
        split,
        *,
        horizon,
        budget,
        window_seed,
    ):
        env, start, _ = make_calibrated_trace_env(
            dataset,
            domain,
            split,
            horizon=horizon,
            budget=budget,
            window_seed=window_seed,
            quantile=quantile,
        )
        return env, start

    def branch_collection(
        dataset,
        *,
        split,
        horizon,
        budget,
        episodes_per_domain,
        seed,
        planner=None,
    ):
        if planner is not None:
            raise ValueError("calibrated baseline collection does not accept a planner")
        return collect_calibrated_branch_dataset(
            dataset,
            split=split,
            horizon=horizon,
            budget=budget,
            episodes_per_domain=episodes_per_domain,
            seed=seed,
            quantile=quantile,
        )

    rl._factory = training_factory
    cpo._factory = training_factory
    evaluation.make_trace_env = evaluation_factory
    budgeted.collect_branch_dataset = branch_collection
    try:
        yield
    finally:
        rl._factory = originals["rl_factory"]
        cpo._factory = originals["cpo_factory"]
        evaluation.make_trace_env = originals["evaluation_factory"]
        budgeted.collect_branch_dataset = originals["budgeted_collection"]
