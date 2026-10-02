"""Discrete-action LCPO reproduction under the common scheduling contract.

The reservoir, OOD sampling, and locally constrained TRPO update follow the
MIT-licensed official LCPO implementation at commit
``aeb93563cbd6ede13e145381ec3f90e0ac840c41``. The environment adapter and
hard affordability mask are project-specific fair-comparison infrastructure.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from dap.direct_action_planning_dataset_validation.data import (
    TraceDataset,
)
from dap.direct_action_planning_paper_closure.environment import (
    make_calibrated_trace_env,
)


@dataclass(frozen=True)
class LCPOConfig:
    total_steps: int = 102_400
    batch_size: int = 200
    hidden_dim: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.9
    policy_learning_rate: float = 4.0e-4
    value_learning_rate: float = 1.0e-3
    entropy_max: float = 0.03
    auto_target_entropy: float = 0.1
    entropy_learning_rate: float = 1.0e-3
    recent_kl_limit: float = 0.1
    anchor_kl_limit: float = 1.0e-4
    damping: float = 0.1
    cg_steps: int = 15
    max_backtracks: int = 10
    recent_window: int = 200
    reservoir_capacity: int = 1_024
    ood_log_likelihood_threshold: float = -6.0
    context_indices: tuple[int, ...] = (0, 1, 10)
    covariance_ridge: float = 1.0e-3
    solve_dual: bool = False
    capacity_training_quantile: float = 0.95

    def validate(self) -> None:
        if self.total_steps <= 0 or self.batch_size <= 0:
            raise ValueError("total_steps and batch_size must be positive")
        if self.total_steps % self.batch_size:
            raise ValueError("total_steps must be divisible by batch_size")
        if self.hidden_dim <= 0 or self.recent_window <= 1:
            raise ValueError("hidden_dim and recent_window must be positive")
        if self.reservoir_capacity < self.recent_window:
            raise ValueError("reservoir_capacity must cover the recent window")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("invalid discount or GAE lambda")
        positive = (
            self.policy_learning_rate,
            self.value_learning_rate,
            self.entropy_learning_rate,
            self.recent_kl_limit,
            self.anchor_kl_limit,
            self.damping,
            self.covariance_ridge,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("learning rates, KL limits, damping, and ridge must be positive")
        if self.cg_steps <= 0 or self.max_backtracks <= 0:
            raise ValueError("CG and line-search steps must be positive")
        if not self.context_indices or min(self.context_indices) < 0:
            raise ValueError("context_indices must be non-empty and non-negative")
        if self.solve_dual:
            raise ValueError("the preregistered LCPO reproduction disables the dual solver")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class LCPOPolicy(nn.Module):
    def __init__(self, observation_dim: int = 14, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.network = _mlp(self.observation_dim, int(hidden_dim), self.action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class LCPOValue(nn.Module):
    def __init__(self, observation_dim: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.network = _mlp(int(observation_dim), int(hidden_dim), 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


def feasible_action_mask(
    observations: torch.Tensor,
    *,
    budget: float,
    action_costs: torch.Tensor,
) -> torch.Tensor:
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError("observations must be a two-dimensional feature matrix")
    if action_costs.ndim != 1:
        raise ValueError("action_costs must be one-dimensional")
    remaining = torch.clamp(observations[:, -2], min=0.0, max=1.0) * float(budget)
    feasible = action_costs.to(observations).unsqueeze(0) <= remaining.unsqueeze(1) + 1.0e-6
    feasible[:, 0] = True
    return feasible


def _masked_distribution(
    policy: LCPOPolicy,
    observations: torch.Tensor,
    feasible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = policy(observations)
    if feasible.shape != logits.shape:
        raise ValueError("feasible mask must match policy logits")
    masked = logits.masked_fill(~feasible, -1.0e9)
    log_prob = torch.log_softmax(masked, dim=-1)
    probability = torch.exp(log_prob)
    entropy = -(probability * log_prob).sum(dim=-1)
    return log_prob, probability, entropy


def mahalanobis_log_likelihood(
    candidates: np.ndarray,
    recent: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float64)
    recent = np.asarray(recent, dtype=np.float64)
    if candidates.ndim != 2 or recent.ndim != 2 or candidates.shape[1] != recent.shape[1]:
        raise ValueError("candidates and recent must have the same feature width")
    if len(recent) < 2:
        return np.zeros(len(candidates), dtype=np.float64)
    mean = np.mean(recent, axis=0)
    covariance = np.atleast_2d(np.cov(recent, rowvar=False)).astype(np.float64)
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * float(ridge)
    inverse = np.linalg.pinv(covariance, hermitian=True)
    centered = candidates - mean
    squared = np.einsum("bi,ij,bj->b", centered, inverse, centered)
    return -0.5 * squared / max(candidates.shape[1], 1)


class ReservoirOODBuffer:
    """Official-style reservoir plus circular recent-state window."""

    def __init__(
        self,
        observation_dim: int,
        *,
        recent_window: int,
        capacity: int,
        seed: int,
    ) -> None:
        if observation_dim <= 0 or recent_window <= 1 or capacity < recent_window:
            raise ValueError("invalid reservoir dimensions")
        self.observation_dim = int(observation_dim)
        self.recent_window = int(recent_window)
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self._reservoir = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._recent = np.zeros((recent_window, observation_dim), dtype=np.float32)
        self._recent_index = 0
        self.seen = 0

    @property
    def reservoir_states(self) -> np.ndarray:
        return self._reservoir[: min(self.seen, self.capacity)].copy()

    @property
    def recent_states(self) -> np.ndarray:
        count = min(self.seen, self.recent_window)
        if self.seen <= self.recent_window:
            return self._recent[:count].copy()
        return np.concatenate(
            (self._recent[self._recent_index :], self._recent[: self._recent_index]),
            axis=0,
        ).copy()

    def add_many(self, states: np.ndarray) -> None:
        values = np.asarray(states, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.observation_dim:
            raise ValueError("states do not match the reservoir observation width")
        if not np.isfinite(values).all():
            raise ValueError("reservoir states must be finite")
        for state in values:
            if self.seen < self.capacity:
                self._reservoir[self.seen] = state
            else:
                index = int(self.rng.integers(0, self.seen + 1))
                if index < self.capacity:
                    self._reservoir[index] = state
            self._recent[self._recent_index] = state
            self._recent_index = (self._recent_index + 1) % self.recent_window
            self.seen += 1

    def sample_ood(
        self,
        batch_size: int,
        *,
        context_indices: tuple[int, ...],
        threshold: float,
        ridge: float,
        max_resamples: int = 5,
    ) -> np.ndarray:
        if self.seen < self.recent_window:
            return np.empty((0, self.observation_dim), dtype=np.float32)
        reservoir = self.reservoir_states
        recent = self.recent_states
        selected: list[np.ndarray] = []
        indices = np.asarray(context_indices, dtype=np.int64)
        for _ in range(max_resamples):
            sampled = reservoir[self.rng.integers(0, len(reservoir), size=int(batch_size))]
            scores = mahalanobis_log_likelihood(
                sampled[:, indices], recent[:, indices], ridge=float(ridge)
            )
            selected.extend(sampled[scores < float(threshold)])
            if len(selected) >= batch_size:
                return np.asarray(selected[:batch_size], dtype=np.float32)
        return np.empty((0, self.observation_dim), dtype=np.float32)


def _flat_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def _set_flat_parameters(model: nn.Module, values: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(values[offset : offset + count].view_as(parameter))
            offset += count
    if offset != values.numel():
        raise ValueError("flat parameter vector has the wrong length")


def _conjugate_gradient(operator, target: torch.Tensor, steps: int) -> torch.Tensor:
    estimate = torch.zeros_like(target)
    residual = target.clone()
    direction = target.clone()
    squared = torch.dot(residual, residual)
    for _ in range(int(steps)):
        product = operator(direction)
        denominator = torch.dot(direction, product)
        if not torch.isfinite(denominator) or abs(float(denominator)) < 1.0e-12:
            break
        alpha = squared / denominator
        estimate = estimate + alpha * direction
        residual = residual - alpha * product
        new_squared = torch.dot(residual, residual)
        if float(new_squared) < 1.0e-10:
            break
        direction = residual + (new_squared / squared) * direction
        squared = new_squared
    return estimate


def _categorical_kl(
    old_log_prob: torch.Tensor,
    old_probability: torch.Tensor,
    new_log_prob: torch.Tensor,
) -> torch.Tensor:
    return (old_probability * (old_log_prob - new_log_prob)).sum(dim=-1)


def lcpo_policy_step(
    policy: LCPOPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    feasible: torch.Tensor,
    anchors: torch.Tensor,
    anchor_feasible: torch.Tensor,
    *,
    entropy_coef: float,
    recent_kl_limit: float,
    anchor_kl_limit: float,
    damping: float,
    cg_steps: int,
    max_backtracks: int,
) -> dict[str, float]:
    if len(anchors) == 0:
        raise ValueError("LCPO TRPO step requires OOD anchor states")
    with torch.no_grad():
        old_recent_log, old_recent_probability, _ = _masked_distribution(
            policy, observations, feasible
        )
        old_anchor_log, old_anchor_probability, _ = _masked_distribution(
            policy, anchors, anchor_feasible
        )
        old_action_log = old_recent_log.gather(1, actions[:, None]).squeeze(1)

    def loss_function() -> torch.Tensor:
        new_log, _, entropy = _masked_distribution(policy, observations, feasible)
        action_log = new_log.gather(1, actions[:, None]).squeeze(1)
        ratio = torch.exp(action_log - old_action_log)
        return -(advantages * ratio).mean() - float(entropy_coef) * entropy.mean()

    def recent_kl() -> torch.Tensor:
        new_log, _, _ = _masked_distribution(policy, observations, feasible)
        return _categorical_kl(old_recent_log, old_recent_probability, new_log).mean()

    def anchor_kl() -> torch.Tensor:
        new_log, _, _ = _masked_distribution(policy, anchors, anchor_feasible)
        return _categorical_kl(old_anchor_log, old_anchor_probability, new_log).mean()

    before = loss_function()
    gradients = torch.autograd.grad(before, tuple(policy.parameters()))
    flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients]).detach()

    def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
        kl = anchor_kl()
        first = torch.autograd.grad(kl, tuple(policy.parameters()), create_graph=True)
        flat_first = torch.cat([gradient.reshape(-1) for gradient in first])
        directional = torch.dot(flat_first, vector)
        second = torch.autograd.grad(directional, tuple(policy.parameters()))
        flat_second = torch.cat([gradient.contiguous().reshape(-1) for gradient in second]).detach()
        return flat_second + float(damping) * vector

    direction = _conjugate_gradient(fisher_vector_product, -flat_gradient, cg_steps)
    curvature = -torch.dot(flat_gradient, direction)
    old_parameters = _flat_parameters(policy)
    accepted = False
    step_fraction = 0.0
    if torch.isfinite(curvature) and float(curvature) > 1.0e-12:
        full_step = direction * math.sqrt(2.0 * float(anchor_kl_limit) / float(curvature))
        expected = -torch.dot(flat_gradient, full_step)
        for backtrack in range(int(max_backtracks)):
            fraction = 0.5**backtrack
            _set_flat_parameters(policy, old_parameters + fraction * full_step)
            with torch.no_grad():
                candidate = loss_function()
                improvement = before.detach() - candidate
                recent_value = recent_kl()
                anchor_value = anchor_kl()
                expected_value = expected * fraction
                ratio = improvement / torch.clamp(expected_value, min=1.0e-12)
            if (
                torch.isfinite(candidate)
                and float(improvement) > 0.0
                and float(ratio) > 0.1
                and float(recent_value) <= float(recent_kl_limit)
                and float(anchor_value) <= float(anchor_kl_limit)
            ):
                accepted = True
                step_fraction = fraction
                break
    if not accepted:
        _set_flat_parameters(policy, old_parameters)
    with torch.no_grad():
        after = loss_function()
        recent_value = recent_kl()
        anchor_value = anchor_kl()
    diagnostics = {
        "accepted": float(accepted),
        "loss_before": float(before.detach()),
        "loss_after": float(after),
        "recent_kl": float(recent_value),
        "anchor_kl": float(anchor_value),
        "gradient_norm": float(torch.linalg.vector_norm(flat_gradient)),
        "step_fraction": float(step_fraction),
    }
    if not np.isfinite(list(diagnostics.values())).all():
        _set_flat_parameters(policy, old_parameters)
        raise FloatingPointError("LCPO policy update produced non-finite diagnostics")
    return diagnostics


def _a2c_policy_step(
    policy: LCPOPolicy,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    feasible: torch.Tensor,
    entropy_coef: float,
) -> dict[str, float]:
    log_prob, _, entropy = _masked_distribution(policy, observations, feasible)
    selected = log_prob.gather(1, actions[:, None]).squeeze(1)
    loss = -(selected * advantages).mean() - float(entropy_coef) * entropy.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
    optimizer.step()
    return {
        "accepted": 1.0,
        "loss_before": float(loss.detach()),
        "loss_after": float(loss.detach()),
        "recent_kl": 0.0,
        "anchor_kl": 0.0,
        "gradient_norm": float(gradient_norm),
        "step_fraction": 1.0,
    }


def _returns_and_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    running_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_values[index] * nonterminal - values[index]
        running_advantage = (
            delta + gamma * gae_lambda * nonterminal * running_advantage
        )
        advantages[index] = running_advantage
    returns = np.zeros_like(rewards)
    running_return = float(next_values[-1])
    for index in range(len(rewards) - 1, -1, -1):
        if dones[index]:
            running_return = 0.0
        running_return = float(rewards[index]) + gamma * running_return
        returns[index] = running_return
    return returns, advantages


class LCPOAgent:
    name = "lcpo"

    def __init__(self, policy: LCPOPolicy, *, budget: float):
        self.policy = policy.train(False)
        self.budget = float(budget)

    def reset(self) -> None:
        return None

    @torch.no_grad()
    def select(self, env, observation: np.ndarray) -> tuple[int, np.ndarray]:
        tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        logits = self.policy(tensor)[0].cpu().numpy().astype(np.float64)
        remaining = max(self.budget - float(env.cumulative_cost), 0.0)
        logits[np.asarray(env.action_costs) > remaining + 1.0e-8] = -np.inf
        return int(np.argmax(logits)), logits


@dataclass
class LCPOTrainResult:
    agent: LCPOAgent
    policy_state: dict[str, torch.Tensor]
    value_state: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    elapsed_seconds: float
    diagnostics: dict[str, float]


def train_lcpo(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    config: LCPOConfig | None = None,
) -> LCPOTrainResult:
    config = config or LCPOConfig()
    config.validate()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    policy = LCPOPolicy(hidden_dim=config.hidden_dim)
    value = LCPOValue(hidden_dim=config.hidden_dim)
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.policy_learning_rate,
        weight_decay=1.0e-4,
        eps=1.0e-5,
    )
    value_optimizer = torch.optim.Adam(
        value.parameters(),
        lr=config.value_learning_rate,
        weight_decay=1.0e-4,
        eps=1.0e-5,
    )
    log_entropy = torch.zeros(1, requires_grad=True)
    entropy_optimizer = torch.optim.Adam(
        [log_entropy], lr=config.entropy_learning_rate, weight_decay=1.0e-4
    )
    entropy_coef = float(config.entropy_max)
    target_entropy = math.log(4.0) * float(config.auto_target_entropy)
    reservoir = ReservoirOODBuffer(
        14,
        recent_window=config.recent_window,
        capacity=config.reservoir_capacity,
        seed=seed,
    )
    domains = dataset.domain_names
    current_seed = int(seed)

    def make_env(window_seed: int):
        domain = domains[abs(int(window_seed)) % len(domains)]
        env, _, _ = make_calibrated_trace_env(
            dataset,
            domain,
            "train",
            horizon=int(horizon),
            budget=float(budget),
            window_seed=int(window_seed),
            quantile=float(config.capacity_training_quantile),
        )
        return env

    env = make_env(current_seed)
    observation, _ = env.reset(seed=current_seed)
    action_costs = torch.as_tensor(env.action_costs, dtype=torch.float32)
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, float]] = []
    episode_costs: list[float] = []
    episode_cost = 0.0
    ood_updates = 0
    accepted_updates = 0
    started = time.perf_counter()
    updates = config.total_steps // config.batch_size
    for update in range(updates):
        rows: dict[str, list[Any]] = {
            key: [] for key in ("observation", "next_observation", "action", "reward", "done")
        }
        for _ in range(config.batch_size):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                logits = policy(observation_tensor)
                remaining = max(float(budget) - float(env.cumulative_cost), 0.0)
                actual_feasible = torch.as_tensor(
                    np.asarray(env.action_costs) <= remaining + 1.0e-8,
                    dtype=torch.bool,
                ).reshape(1, -1)
                actual_feasible[:, 0] = True
                distribution = Categorical(logits=logits.masked_fill(~actual_feasible, -1.0e9))
                action = int(distribution.sample().item())
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rows["observation"].append(np.asarray(observation, dtype=np.float32))
            rows["next_observation"].append(np.asarray(next_observation, dtype=np.float32))
            rows["action"].append(action)
            rows["reward"].append(float(reward))
            rows["done"].append(float(done))
            episode_cost += float(info["resource_cost"])
            observation = next_observation
            if done:
                episode_costs.append(episode_cost)
                episode_cost = 0.0
                current_seed += 9_973
                env = make_env(current_seed)
                observation, _ = env.reset(seed=current_seed)

        observations_np = np.asarray(rows["observation"], dtype=np.float32)
        next_observations_np = np.asarray(rows["next_observation"], dtype=np.float32)
        observations = torch.as_tensor(observations_np, dtype=torch.float32)
        next_observations = torch.as_tensor(next_observations_np, dtype=torch.float32)
        actions = torch.as_tensor(rows["action"], dtype=torch.long)
        with torch.no_grad():
            values = value(observations).cpu().numpy()
            next_values = value(next_observations).cpu().numpy()
        returns_np, advantages_np = _returns_and_advantages(
            np.asarray(rows["reward"], dtype=np.float32),
            values,
            next_values,
            np.asarray(rows["done"], dtype=np.float32),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
        feasible = feasible_action_mask(
            observations, budget=float(budget), action_costs=action_costs
        )
        reservoir.add_many(observations_np)
        anchor_np = reservoir.sample_ood(
            config.batch_size,
            context_indices=config.context_indices,
            threshold=config.ood_log_likelihood_threshold,
            ridge=config.covariance_ridge,
        )
        if len(anchor_np):
            anchors = torch.as_tensor(anchor_np, dtype=torch.float32)
            anchor_feasible = feasible_action_mask(
                anchors, budget=float(budget), action_costs=action_costs
            )
            policy_diagnostics = lcpo_policy_step(
                policy,
                observations,
                actions,
                advantages,
                feasible,
                anchors,
                anchor_feasible,
                entropy_coef=entropy_coef,
                recent_kl_limit=config.recent_kl_limit,
                anchor_kl_limit=config.anchor_kl_limit,
                damping=config.damping,
                cg_steps=config.cg_steps,
                max_backtracks=config.max_backtracks,
            )
            ood_updates += 1
        else:
            policy_diagnostics = _a2c_policy_step(
                policy,
                policy_optimizer,
                observations,
                actions,
                advantages,
                feasible,
                entropy_coef,
            )
        accepted_updates += int(policy_diagnostics["accepted"] > 0.5)
        value_prediction = value(observations)
        value_target = torch.as_tensor(returns_np, dtype=torch.float32)
        value_loss = nn.functional.mse_loss(value_prediction, value_target)
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), 0.5, error_if_nonfinite=True)
        value_optimizer.step()

        with torch.no_grad():
            _, _, entropy = _masked_distribution(policy, observations, feasible)
            entropy_gap = -entropy + target_entropy
        entropy_loss = -(
            torch.exp(log_entropy) * float(config.entropy_max) * entropy_gap
        ).mean()
        entropy_optimizer.zero_grad(set_to_none=True)
        entropy_loss.backward()
        entropy_optimizer.step()
        entropy_coef = float(torch.exp(log_entropy).item() * config.entropy_max)
        record = {
            "update": float(update + 1),
            "steps": float((update + 1) * config.batch_size),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.mean()),
            "entropy_coef": entropy_coef,
            "ood_anchor_count": float(len(anchor_np)),
            "reservoir_seen": float(reservoir.seen),
            **policy_diagnostics,
        }
        if not np.isfinite(list(record.values())).all():
            raise FloatingPointError("LCPO training history contains non-finite values")
        history.append(record)
        _ = rng  # The explicit RNG remains part of the deterministic training contract.

    elapsed = time.perf_counter() - started
    diagnostics = {
        "updates": float(updates),
        "ood_updates": float(ood_updates),
        "ood_update_fraction": float(ood_updates / updates),
        "accepted_updates": float(accepted_updates),
        "accepted_update_fraction": float(accepted_updates / updates),
        "episodes": float(len(episode_costs)),
        "mean_episode_cost": float(np.mean(episode_costs)) if episode_costs else 0.0,
        "max_episode_cost": float(np.max(episode_costs)) if episode_costs else 0.0,
        "final_entropy_coef": float(entropy_coef),
    }
    return LCPOTrainResult(
        agent=LCPOAgent(copy.deepcopy(policy), budget=float(budget)),
        policy_state=copy.deepcopy(policy.state_dict()),
        value_state=copy.deepcopy(value.state_dict()),
        history=history,
        elapsed_seconds=float(elapsed),
        diagnostics=diagnostics,
    )
