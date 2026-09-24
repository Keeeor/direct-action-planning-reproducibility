from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_dynamic_budget.envs.synthetic_queue_env import SyntheticQueueConfig
from stage2_dynamic_budget.envs.trace_driven_env import TraceDrivenQueueEnv

from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.controllers import (
    CausalMPCController,
    LyapunovDPPController,
    PIDBudgetController,
    ReactiveThresholdController,
    feasible_actions,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.analysis import (
    _claim_unit_metrics,
    _method_role,
)
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.evaluation import evaluate_agent
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.cpo import CPOConfig, train_cpo
from stage2_dynamic_budget.direct_action_planning_dataset_benchmark.rl import RLTrainConfig, train_policy
from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import TraceDataset


def _dataset() -> TraceDataset:
    values = np.tile(np.asarray([2.0, 5.0, 8.0, 4.0]), 30)
    return TraceDataset(
        name="benchmark_test",
        domains={
            "a": {"train": values, "validation": values, "test": values},
            "b": {"train": values[::-1], "validation": values[::-1], "test": values[::-1]},
        },
        split_contract="test",
    )


def test_traditional_controllers_respect_hard_budget() -> None:
    dataset = _dataset()
    for controller in (
        ReactiveThresholdController(),
        PIDBudgetController(),
        CausalMPCController(horizon=2),
        LyapunovDPPController(),
    ):
        rows, _ = evaluate_agent(
            dataset,
            controller,
            split="validation",
            horizon=8,
            budget=4.0,
            seed=11,
            episodes_per_domain=1,
        )
        assert rows
        assert all(row["budget_overspend"] <= 1.0e-8 for row in rows)


def test_policy_variants_train_and_mask_actions() -> None:
    result = train_policy(
        _dataset(),
        horizon=8,
        budget=4.0,
        seed=19,
        variant="pid_lagrangian",
        config=RLTrainConfig(total_steps=64, rollout_steps=32, update_epochs=1, minibatch_size=16, hidden_dim=8),
    )
    rows, steps = evaluate_agent(
        _dataset(),
        result.agent,
        split="validation",
        horizon=8,
        budget=4.0,
        seed=29,
        episodes_per_domain=1,
    )
    assert rows and steps
    assert all(row["budget_overspend"] <= 1.0e-8 for row in rows)
    assert np.isfinite([row["return"] for row in rows]).all()


def test_feasible_action_set_always_contains_no_op() -> None:
    dataset = _dataset()
    rows, _ = evaluate_agent(
        dataset,
        ReactiveThresholdController(),
        split="validation",
        horizon=4,
        budget=0.0,
        seed=31,
        episodes_per_domain=1,
    )
    assert rows


def test_cpo_port_respects_trust_region_and_budget() -> None:
    result = train_cpo(
        _dataset(),
        horizon=8,
        budget=4.0,
        seed=37,
        config=CPOConfig(
            total_steps=64,
            rollout_steps=32,
            hidden_dim=8,
            critic_epochs=1,
            cg_iterations=3,
            backtracks=4,
        ),
    )
    assert result.history
    assert max(row["kl"] for row in result.history) <= 0.010001
    rows, _ = evaluate_agent(
        _dataset(),
        result.agent,
        split="validation",
        horizon=8,
        budget=4.0,
        seed=41,
        episodes_per_domain=1,
    )
    assert all(row["budget_overspend"] <= 1.0e-8 for row in rows)


def test_causal_mpc_does_not_read_future_trace_suffix() -> None:
    config = SyntheticQueueConfig(horizon=5, budget=8.0)
    env_a = TraceDrivenQueueEnv(np.asarray([7.0, 0.0, 0.0, 0.0, 0.0]), config)
    env_b = TraceDrivenQueueEnv(np.asarray([7.0, 100.0, 100.0, 100.0, 100.0]), config)
    obs_a, _ = env_a.reset(seed=1)
    obs_b, _ = env_b.reset(seed=1)
    controller = CausalMPCController(horizon=4)
    action_a, scores_a = controller.select(env_a, obs_a)
    action_b, scores_b = controller.select(env_b, obs_b)
    assert action_a == action_b
    np.testing.assert_allclose(scores_a, scores_b)


def test_claim_centered_roles_separate_privileged_references() -> None:
    assert _method_role("causal_mpc") == "privileged_model_reference"
    assert _method_role("ppo") == "common_rl"
    assert _method_role("cpo") == "constrained_or_budget_rl"
    assert _method_role("structured_dap_selected") == "candidate"


def test_claim_unit_metrics_include_tail_safety_outcomes() -> None:
    rows = []
    for episode, value in enumerate([1.0, 2.0, 3.0, 4.0, 100.0]):
        rows.append({
            "dataset": "azure2019",
            "budget": 48.0,
            "seed": 1,
            "method": "ppo",
            "discounted_return": value,
            "completion_ratio": value / 100.0,
            "slo_violation_rate": 1.0 - value / 100.0,
            "total_cost": value,
            "queue_area": value,
            "decision_ms_mean": value,
            "decision_ms_p95": value + 1.0,
            "budget_overspend": 0.0,
            "episode": episode,
        })
    unit = _claim_unit_metrics(pd.DataFrame(rows)).iloc[0]
    assert unit.return_cvar20 == 1.0
    assert unit.budget_overspend_max == 0.0
    assert unit.completion_p10 < unit.completion_ratio
    assert unit.slo_p95 > unit.slo_violation_rate
