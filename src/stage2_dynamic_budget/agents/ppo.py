from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from stage2_dynamic_budget.models.policy import ConstrainedSchedulingPolicy


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 20_480
    rollout_steps: int = 2_048
    learning_rate: float = 3e-4
    update_epochs: int = 6
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    global_lambda_lr: float = 0.05
    balance_coef: float = 0.1
    smooth_coef: float = 0.01
    allocator_entropy_coef: float = 0.001
    use_balance_loss: bool = True
    use_smooth_loss: bool = True
    use_global_lambda: bool = True
    use_local_constraint: bool = True

    def __post_init__(self):
        if self.total_steps <= 0 or self.rollout_steps <= 0 or self.minibatch_size <= 0:
            raise ValueError("training step counts must be positive")
        if not 0 < self.clip_coef < 1:
            raise ValueError("clip_coef must be in (0, 1)")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount factors must be in [0, 1]")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    next_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must have equal shapes")
    advantages = np.zeros_like(rewards)
    last_advantage = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_nonterminal = 1.0 - dones[t]
        following_value = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * following_value * next_nonterminal - values[t]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


@dataclass
class TrainResult:
    global_lambda: float
    episode_costs: list[float]
    episode_rewards: list[float]
    update_history: list[dict[str, float]]
    elapsed_seconds: float
    config: dict


class PPOTrainer:
    def __init__(
        self,
        policy: ConstrainedSchedulingPolicy,
        config: PPOConfig,
        device: torch.device,
        budget: float,
        seed: int,
    ):
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.budget = float(budget)
        self.seed = int(seed)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.global_lambda = 0.0
        self.rng = np.random.default_rng(seed)

    def train(self, env_factory: Callable[[int], object]) -> TrainResult:
        started = time.perf_counter()
        total_collected = 0
        episode_costs: list[float] = []
        episode_rewards: list[float] = []
        update_history: list[dict[str, float]] = []
        env = env_factory(self.seed)
        obs, _ = env.reset(seed=self.seed)
        current_cost = current_reward = 0.0
        episode_index = 0

        while total_collected < self.config.total_steps:
            target = min(self.config.rollout_steps, self.config.total_steps - total_collected)
            buffer = {key: [] for key in (
                "obs", "action", "log_prob", "reward", "cost", "done",
                "reward_value", "cost_value", "local_budget", "local_lambda",
            )}
            completed_costs: list[float] = []
            for _ in range(target):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    output = self.policy.act(obs_tensor)
                action = int(output.action.item())
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                buffer["obs"].append(np.asarray(obs, dtype=np.float32))
                buffer["action"].append(action)
                buffer["log_prob"].append(float(output.log_prob.item()))
                buffer["reward"].append(float(reward))
                buffer["cost"].append(float(info["resource_cost"]))
                buffer["done"].append(float(done))
                buffer["reward_value"].append(float(output.reward_value.item()))
                buffer["cost_value"].append(float(output.cost_value.item()))
                buffer["local_budget"].append(
                    float(output.local_budget.item()) if output.local_budget is not None else np.nan
                )
                buffer["local_lambda"].append(
                    float(output.local_lambda.item()) if output.local_lambda is not None else 0.0
                )
                current_cost += float(info["resource_cost"])
                current_reward += float(reward)
                obs = next_obs
                if done:
                    completed_costs.append(current_cost)
                    episode_costs.append(current_cost)
                    episode_rewards.append(current_reward)
                    episode_index += 1
                    env = env_factory(self.seed + episode_index * 9973)
                    obs, _ = env.reset(seed=self.seed + episode_index * 9973)
                    self.policy.reset_budget_controller()
                    current_cost = current_reward = 0.0

            total_collected += target
            with torch.no_grad():
                bootstrap = self.policy.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0),
                    deterministic=True,
                    advance_budget_state=False,
                )
            next_reward_value = 0.0 if buffer["done"][-1] else float(bootstrap.reward_value.item())
            next_cost_value = 0.0 if buffer["done"][-1] else float(bootstrap.cost_value.item())
            reward_adv, reward_returns = compute_gae(
                np.asarray(buffer["reward"]), np.asarray(buffer["reward_value"]),
                np.asarray(buffer["done"]), self.config.gamma, self.config.gae_lambda,
                next_reward_value,
            )
            cost_adv, cost_returns = compute_gae(
                np.asarray(buffer["cost"]), np.asarray(buffer["cost_value"]),
                np.asarray(buffer["done"]), self.config.gamma, self.config.gae_lambda,
                next_cost_value,
            )
            combined_adv = reward_adv.copy()
            method = self.policy.config.method
            if method != "ppo" and self.config.use_global_lambda:
                combined_adv -= self.global_lambda * cost_adv
            if method in {"cdba", "cdba_discrete"} and self.config.use_local_constraint:
                # The guidance defines the local term with cost advantage, not
                # raw instantaneous cost. Normalize it to a per-step scale so
                # it is dimensionally comparable with the local budget rate.
                per_step_cost_adv = cost_adv / float(self.policy.config.horizon)
                excess = per_step_cost_adv - np.asarray(buffer["local_budget"])
                combined_adv -= np.asarray(buffer["local_lambda"]) * excess
            combined_adv = (combined_adv - combined_adv.mean()) / (combined_adv.std() + 1e-8)
            stats = self._update(buffer, combined_adv, reward_returns, cost_returns)
            if method != "ppo" and self.config.use_global_lambda and completed_costs:
                violation = (float(np.mean(completed_costs)) - self.budget) / max(self.budget, 1e-8)
                self.global_lambda = max(
                    0.0, self.global_lambda + self.config.global_lambda_lr * violation
                )
            stats.update(
                {
                    "steps": float(total_collected),
                    "global_lambda": self.global_lambda,
                    "mean_completed_cost": float(np.mean(completed_costs)) if completed_costs else np.nan,
                }
            )
            update_history.append(stats)
        return TrainResult(
            global_lambda=self.global_lambda,
            episode_costs=episode_costs,
            episode_rewards=episode_rewards,
            update_history=update_history,
            elapsed_seconds=time.perf_counter() - started,
            config=asdict(self.config),
        )

    def _update(self, buffer, advantages, reward_returns, cost_returns):
        obs = torch.as_tensor(np.asarray(buffer["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buffer["action"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buffer["log_prob"], dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        reward_returns_t = torch.as_tensor(reward_returns, dtype=torch.float32, device=self.device)
        cost_returns_t = torch.as_tensor(cost_returns, dtype=torch.float32, device=self.device)
        costs_t = torch.as_tensor(buffer["cost"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(buffer["done"], dtype=torch.float32, device=self.device)
        n = len(actions)
        aggregate = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "aux_loss": []}
        for _ in range(self.config.update_epochs):
            permutation = self.rng.permutation(n)
            for start in range(0, n, self.config.minibatch_size):
                idx = torch.as_tensor(
                    permutation[start : start + self.config.minibatch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                local_override = None
                if self.policy.config.budget_update_period > 1:
                    local_override = torch.as_tensor(
                        np.asarray(buffer["local_budget"]), dtype=torch.float32, device=self.device
                    )[idx]
                output = self.policy.act(
                    obs[idx], action=actions[idx], local_budget_override=local_override
                )
                log_ratio = output.log_prob - old_log_probs[idx]
                ratio = log_ratio.exp()
                unclipped = -advantages_t[idx] * ratio
                clipped = -advantages_t[idx] * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                policy_loss = torch.maximum(unclipped, clipped).mean()
                reward_value_loss = nn.functional.mse_loss(
                    output.reward_value, reward_returns_t[idx]
                )
                cost_value_loss = nn.functional.mse_loss(
                    output.cost_value, cost_returns_t[idx]
                )
                value_loss = (
                    self.config.value_coef * reward_value_loss
                    + self.config.cost_value_coef * cost_value_loss
                )
                loss = policy_loss + value_loss - self.config.entropy_coef * output.entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                aggregate["policy_loss"].append(float(policy_loss.detach()))
                aggregate["value_loss"].append(float(value_loss.detach()))
                aggregate["entropy"].append(float(output.entropy.mean().detach()))
                aggregate["approx_kl"].append(float(((ratio - 1) - log_ratio).mean().detach()))

            if self.policy.config.method in {"cdba", "cdba_discrete"}:
                output = self.policy.act(obs, action=actions)
                target_rate = self.budget / self.policy.config.horizon
                aux = torch.zeros((), device=self.device)
                if self.config.use_balance_loss and output.local_budget is not None:
                    balance = ((output.local_budget.mean() - target_rate) / max(target_rate, 1e-6)) ** 2
                    aux = aux + self.config.balance_coef * balance
                if self.config.use_smooth_loss and output.local_budget is not None and n > 1:
                    valid = 1.0 - dones_t[:-1]
                    smooth = (((output.local_budget[1:] - output.local_budget[:-1]) ** 2) * valid).sum() / valid.sum().clamp(min=1.0)
                    aux = aux + self.config.smooth_coef * smooth
                if self.config.use_local_constraint and output.local_lambda is not None:
                    local_excess = (costs_t - output.local_budget).detach()
                    aux = aux - (output.local_lambda * local_excess).mean()
                if output.allocator_probabilities is not None:
                    p = output.allocator_probabilities.clamp(min=1e-8)
                    allocator_entropy = -(p * p.log()).sum(dim=-1).mean()
                    aux = aux - self.config.allocator_entropy_coef * allocator_entropy
                self.optimizer.zero_grad(set_to_none=True)
                aux.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                aggregate["aux_loss"].append(float(aux.detach()))
        return {
            key: float(np.mean(values)) if values else 0.0
            for key, values in aggregate.items()
        }
