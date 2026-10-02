from __future__ import annotations

import numpy as np
import torch

from dap.action_conditioned_budget_advantage.branching import (
    BranchableDiscreteEnv,
    branch_actions,
)
from dap.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from dap.action_conditioned_budget_advantage.model import (
    advantage_adjust_logits,
    pairwise_ranking_loss,
)


def test_actionwise_bellman_q_and_a0_centering() -> None:
    cfg = ACBADPConfig(horizon=3, max_budget=3, max_queue=2, scenario="stable", gamma=1.0)
    mdp = ActionConditionedBudgetMDP(cfg)
    result = solve_action_dp(mdp)
    assert result.q_values.shape == (3, 3, 3, 4, 4)
    assert np.allclose(result.advantages[..., 0], 0.0)
    feasible = np.isfinite(result.q_values)
    recovered = np.nanmax(np.where(feasible, result.q_values, np.nan), axis=-1)
    assert np.allclose(recovered, result.values[:-1])
    assert result.max_bellman_residual <= 1e-12


def test_periodic_dp_is_solvable_and_budget_feasible() -> None:
    cfg = ACBADPConfig(horizon=8, max_budget=4, max_queue=3, scenario="periodic")
    mdp = ActionConditionedBudgetMDP(cfg)
    result = solve_action_dp(mdp)
    assert np.isfinite(result.values).all()
    for t, load, queue, budget in np.ndindex(result.actions.shape):
        action = int(result.actions[t, load, queue, budget])
        assert mdp.action_costs[action] <= budget


def test_snapshot_restore_recovers_state_and_rng_exactly() -> None:
    cfg = ACBADPConfig(horizon=6, max_budget=4, max_queue=3, scenario="early_burst")
    env = BranchableDiscreteEnv(cfg, initial_budget=4)
    env.reset(seed=123)
    env.step(1)
    snapshot = env.snapshot()
    first = env.step_with_uniform(0, 0.314159)
    env.restore(snapshot)
    second = env.step_with_uniform(0, 0.314159)
    assert first[1:] == second[1:]
    assert np.array_equal(first[0], second[0])
    assert env.snapshot() == env.snapshot()


def test_all_candidate_actions_share_future_uniforms() -> None:
    cfg = ACBADPConfig(horizon=6, max_budget=4, max_queue=3, scenario="late_burst")
    env = BranchableDiscreteEnv(cfg, initial_budget=4)
    env.reset(seed=17)
    result = solve_action_dp(env.mdp)
    rows = branch_actions(env, result, k=5, seed=991)
    tapes = {row["random_tape_sha256"] for row in rows}
    assert len(tapes) == 1
    assert len(rows) == 4


def test_k1_branch_return_matches_manual_gold() -> None:
    cfg = ACBADPConfig(horizon=2, max_budget=1, max_queue=2, scenario="stable", gamma=0.5)
    env = BranchableDiscreteEnv(cfg, initial_budget=1)
    env.reset(seed=9)
    result = solve_action_dp(env.mdp)
    rows = branch_actions(env, result, k=1, seed=5)
    row = next(item for item in rows if item["action"] == 0)
    next_queue, immediate, _ = env.mdp.outcome(0, 1, 0)
    probabilities = env.mdp.load_probabilities(0, 1)
    u = row["first_uniform"]
    next_load = int(np.searchsorted(np.cumsum(probabilities), u, side="right"))
    expected = immediate + 0.5 * result.values[1, next_load, next_queue, 1]
    assert np.isclose(row["q_branch"], expected)


def test_acba_a_adds_centered_action_advantage() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    advantages = torch.tensor([[0.0, 2.0, -3.0]])
    adjusted = advantage_adjust_logits(logits, advantages, alpha=0.5)
    assert torch.allclose(adjusted, torch.tensor([[1.0, 1.0, -2.5]]))


def test_acba_b_ranking_loss_rewards_correct_order() -> None:
    target = torch.tensor([[0.0, 2.0, 1.0]])
    correctly_ranked = torch.tensor([[0.0, 3.0, 1.0]], requires_grad=True)
    reversed_rank = torch.tensor([[3.0, 0.0, 1.0]], requires_grad=True)
    correct_loss = pairwise_ranking_loss(correctly_ranked, target, margin=0.1)
    reverse_loss = pairwise_ranking_loss(reversed_rank, target, margin=0.1)
    assert correct_loss < reverse_loss
    reverse_loss.backward()
    assert torch.isfinite(reversed_rank.grad).all()
