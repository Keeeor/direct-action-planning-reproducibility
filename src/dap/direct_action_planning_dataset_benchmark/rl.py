from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from dap.direct_action_planning_dataset_validation.data import (
    TraceDataset,
    make_trace_env,
)

from .models import ActorCritic, QNetwork, normalize_observation


@dataclass(frozen=True)
class RLTrainConfig:
    total_steps: int = 102_400
    rollout_steps: int = 512
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    hidden_dim: int = 64


@dataclass
class RLTrainResult:
    agent: Any
    history: list[dict[str, float]]
    elapsed_seconds: float
    diagnostics: dict[str, float]


def _gae(rewards, values, dones, gamma, gae_lambda, next_value=0.0):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - dones[index]
        following = next_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * following * nonterminal - values[index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
    return advantages, advantages + values


def _factory(dataset: TraceDataset, domain_names: tuple[str, ...], horizon: int, budget: float, seed: int):
    domain = domain_names[abs(int(seed)) % len(domain_names)]
    env, _ = make_trace_env(dataset, domain, "train", horizon=horizon, budget=budget, window_seed=seed)
    return env


def _masked_logits(logits: torch.Tensor, env, budget: float) -> torch.Tensor:
    remaining = max(float(budget) - float(env.cumulative_cost), 0.0)
    costs = torch.as_tensor(env.action_costs, dtype=logits.dtype, device=logits.device)
    feasible = costs <= remaining + 1.0e-8
    if not bool(feasible.any()):
        feasible[0] = True
    return logits.masked_fill(~feasible.unsqueeze(0), -1.0e9)


class PolicyAgent:
    def __init__(self, model: ActorCritic, name: str, budget: float):
        self.model = model.eval()
        self.name = name
        self.budget = float(budget)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select(self, env, observation: np.ndarray):
        tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        logits, _, _ = self.model(tensor)
        masked = _masked_logits(logits, env, self.budget)
        q = masked.cpu().numpy()[0]
        return int(np.argmax(q)), q


def train_policy(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    variant: str,
    config: RLTrainConfig | None = None,
) -> RLTrainResult:
    """Train PPO/A2C variants under one shared data and masking contract.

    `variant` is one of `a2c`, `ppo`, `ppo_lagrangian`, `pid_lagrangian`, or `p3o`.
    The constraint is the same episodic resource budget for every variant. Hard feasibility is
    retained, so a zero multiplier is an auditable outcome rather than a hidden relaxation.
    """
    if variant not in {"a2c", "ppo", "ppo_lagrangian", "pid_lagrangian", "p3o"}:
        raise ValueError(f"unknown policy variant: {variant}")
    config = config or RLTrainConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ActorCritic(action_dim=4, hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(seed)
    domains = dataset.domain_names
    env = _factory(dataset, domains, horizon, budget, seed)
    observation, _ = env.reset(seed=seed)
    current_seed = int(seed)
    total_steps = 0
    lambda_value = 0.0
    lambda_integral = 0.0
    previous_violation = 0.0
    episode_cost = 0.0
    completed_costs: list[float] = []
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    while total_steps < config.total_steps:
        target = min(config.rollout_steps, config.total_steps - total_steps)
        buffer = {key: [] for key in ("obs", "action", "old_logp", "reward", "cost", "done", "value", "cost_value", "feasible")}
        rollout_completed: list[float] = []
        for _ in range(target):
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            logits, value, cost_value = model(obs_tensor)
            masked = _masked_logits(logits, env, budget)
            distribution = Categorical(logits=masked)
            action_tensor = distribution.sample()
            action = int(action_tensor.item())
            feasible = (masked[0] > -1.0e8).cpu().numpy().astype(bool)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            buffer["obs"].append(np.asarray(observation, dtype=np.float32))
            buffer["action"].append(action)
            buffer["old_logp"].append(float(distribution.log_prob(action_tensor).item()))
            buffer["reward"].append(float(reward))
            buffer["cost"].append(float(info["resource_cost"]))
            buffer["done"].append(float(done))
            buffer["value"].append(float(value.item()))
            buffer["cost_value"].append(float(cost_value.item()))
            buffer["feasible"].append(feasible)
            episode_cost += float(info["resource_cost"])
            observation = next_observation
            if done:
                completed_costs.append(episode_cost)
                rollout_completed.append(episode_cost)
                episode_cost = 0.0
                current_seed += 9973
                env = _factory(dataset, domains, horizon, budget, current_seed)
                observation, _ = env.reset(seed=current_seed)
        total_steps += target
        with torch.no_grad():
            bootstrap = model(torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1))[1].item()
            cost_bootstrap = model(torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1))[2].item()
        reward_adv, reward_returns = _gae(buffer["reward"], buffer["value"], buffer["done"], config.gamma, config.gae_lambda, bootstrap)
        cost_adv, cost_returns = _gae(buffer["cost"], buffer["cost_value"], buffer["done"], config.gamma, config.gae_lambda, cost_bootstrap)
        if variant in {"ppo_lagrangian", "pid_lagrangian"}:
            combined = reward_adv - lambda_value * cost_adv
        else:
            combined = reward_adv
        combined = (combined - combined.mean()) / (combined.std() + 1.0e-8)
        obs = torch.as_tensor(np.asarray(buffer["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(buffer["action"]), dtype=torch.long)
        old_logp = torch.as_tensor(np.asarray(buffer["old_logp"]), dtype=torch.float32)
        advantages = torch.as_tensor(combined, dtype=torch.float32)
        centered_cost_advantages = torch.as_tensor(
            (cost_adv - cost_adv.mean()) / max(float(budget), 1.0e-8),
            dtype=torch.float32,
        )
        reward_targets = torch.as_tensor(reward_returns, dtype=torch.float32)
        cost_targets = torch.as_tensor(cost_returns, dtype=torch.float32)
        feasible = torch.as_tensor(np.asarray(buffer["feasible"]), dtype=torch.bool)
        indices = np.arange(len(actions))
        update_losses: list[float] = []
        epochs = 1 if variant == "a2c" else config.update_epochs
        for _epoch in range(epochs):
            permutation = rng.permutation(indices)
            for start in range(0, len(indices), config.minibatch_size):
                idx = torch.as_tensor(permutation[start : start + config.minibatch_size], dtype=torch.long)
                logits, value, cost_value = model(obs[idx])
                logits = logits.masked_fill(~feasible[idx], -1.0e9)
                distribution = Categorical(logits=logits)
                log_ratio = distribution.log_prob(actions[idx]) - old_logp[idx]
                ratio = torch.exp(log_ratio)
                if variant == "a2c":
                    policy_loss = -(distribution.log_prob(actions[idx]) * advantages[idx]).mean()
                else:
                    unclipped = ratio * advantages[idx]
                    clipped = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * advantages[idx]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    if variant == "p3o":
                        episodic_violation = (
                            (float(np.mean(rollout_completed)) - float(budget)) / max(float(budget), 1.0e-8)
                            if rollout_completed
                            else -1.0
                        )
                        cost_surrogate = episodic_violation + torch.mean(
                            ratio * centered_cost_advantages[idx]
                        )
                        policy_loss = policy_loss + 10.0 * torch.relu(cost_surrogate)
                value_loss = nn.functional.mse_loss(value, reward_targets[idx])
                cost_loss = nn.functional.mse_loss(cost_value, cost_targets[idx])
                loss = policy_loss + config.value_coef * value_loss + config.cost_value_coef * cost_loss - config.entropy_coef * distribution.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                update_losses.append(float(loss.detach()))
        if rollout_completed:
            violation = float(np.mean(rollout_completed) - budget)
            if variant == "ppo_lagrangian":
                lambda_value = max(0.0, lambda_value + 0.05 * violation / max(budget, 1.0e-8))
            elif variant == "pid_lagrangian":
                normalized = violation / max(budget, 1.0e-8)
                lambda_integral = max(0.0, lambda_integral + normalized)
                derivative = normalized - previous_violation
                lambda_value = max(0.0, 0.05 * normalized + 0.01 * lambda_integral + 0.02 * derivative)
                previous_violation = normalized
            elif variant == "p3o":
                # P3O uses a fixed exact-penalty coefficient, not a learned multiplier.
                lambda_value = 0.0
        history.append({
            "steps": float(total_steps),
            "loss": float(np.mean(update_losses)) if update_losses else 0.0,
            "lambda": float(lambda_value),
            "mean_completed_cost": float(np.mean(rollout_completed)) if rollout_completed else float("nan"),
        })
    diagnostics = {
        "episodes": float(len(completed_costs)),
        "mean_episode_cost": float(np.mean(completed_costs)) if completed_costs else float("nan"),
        "max_episode_cost": float(np.max(completed_costs)) if completed_costs else float("nan"),
        "final_lambda": float(lambda_value),
    }
    return RLTrainResult(PolicyAgent(model, variant, budget), history, time.perf_counter() - started, diagnostics)


class DQNAgent:
    def __init__(self, model: QNetwork, name: str, budget: float):
        self.model = model.eval()
        self.name = name
        self.budget = float(budget)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select(self, env, observation: np.ndarray):
        tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        values = self.model(tensor)[0].cpu().numpy().astype(np.float64)
        remaining = max(self.budget - float(env.cumulative_cost), 0.0)
        values[env.action_costs > remaining + 1.0e-8] = -np.inf
        return int(np.argmax(values)), values


def train_double_dqn(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    total_steps: int = 102_400,
    hidden_dim: int = 128,
) -> RLTrainResult:
    """Train a Double-DQN baseline under the same causal trace and hard-mask contract."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    online = QNetwork(action_dim=4, hidden_dim=hidden_dim, dueling=True)
    target = copy.deepcopy(online).eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=3.0e-4)
    replay: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=50_000)
    rng = np.random.default_rng(seed)
    domains = dataset.domain_names
    current_seed = int(seed)
    env = _factory(dataset, domains, horizon, budget, current_seed)
    observation, _ = env.reset(seed=current_seed)
    started = time.perf_counter()
    losses: list[float] = []
    episode_costs: list[float] = []
    episode_cost = 0.0
    for step in range(int(total_steps)):
        epsilon = max(0.05, 1.0 - 0.95 * step / max(total_steps * 0.6, 1))
        feasible = np.flatnonzero(env.action_costs <= max(budget - env.cumulative_cost, 0.0) + 1.0e-8)
        if rng.random() < epsilon:
            action = int(rng.choice(feasible))
        else:
            with torch.no_grad():
                values = online(torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1))[0].numpy()
            values[env.action_costs > max(budget - env.cumulative_cost, 0.0) + 1.0e-8] = -np.inf
            action = int(np.argmax(values))
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        replay.append((observation.copy(), action, float(reward), next_observation.copy(), done))
        episode_cost += float(info["resource_cost"])
        observation = next_observation
        if done:
            episode_costs.append(episode_cost)
            episode_cost = 0.0
            current_seed += 9973
            env = _factory(dataset, domains, horizon, budget, current_seed)
            observation, _ = env.reset(seed=current_seed)
        if len(replay) < 1024 or step % 4 != 0:
            continue
        batch_indices = rng.choice(len(replay), size=128, replace=False)
        batch = [replay[int(index)] for index in batch_indices]
        obs = torch.as_tensor(np.asarray([row[0] for row in batch]), dtype=torch.float32)
        actions = torch.as_tensor([row[1] for row in batch], dtype=torch.long)
        rewards = torch.as_tensor([row[2] for row in batch], dtype=torch.float32)
        next_obs = torch.as_tensor(np.asarray([row[3] for row in batch]), dtype=torch.float32)
        dones = torch.as_tensor([row[4] for row in batch], dtype=torch.float32)
        current = online(obs).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            online_next = online(next_obs)
            remaining = next_obs[:, -2] * float(budget)
            costs = torch.as_tensor(env.action_costs, dtype=torch.float32)
            next_mask = costs[None, :] <= remaining[:, None] + 1.0e-6
            online_next = online_next.masked_fill(~next_mask, -1.0e9)
            next_actions = online_next.argmax(dim=1)
            target_next = target(next_obs).gather(1, next_actions[:, None]).squeeze(1)
            td_target = rewards + 0.99 * (1.0 - dones) * target_next
        loss = nn.functional.smooth_l1_loss(current, td_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 512 == 0:
            target.load_state_dict(online.state_dict())
    return RLTrainResult(
        DQNAgent(online, "double_dqn", budget),
        [{"steps": float(total_steps), "loss": float(np.mean(losses)) if losses else float("nan")}],
        time.perf_counter() - started,
        {"episodes": float(len(episode_costs)), "mean_episode_cost": float(np.mean(episode_costs)) if episode_costs else float("nan")},
    )
