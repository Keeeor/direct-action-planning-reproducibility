from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from stage2_dynamic_budget.agents.ppo import compute_gae

from .shadow_policy import DynamicShadowPricePolicy, dsp_b_td_target


@dataclass(frozen=True)
class DSPTrainerConfig:
    total_steps: int = 20_480
    rollout_steps: int = 512
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    entropy_coef: float = 0.01
    q_value_coef: float = 0.5
    max_grad_norm: float = 0.5
    global_lambda_lr: float = 0.05
    use_global_lambda: bool = True

    def __post_init__(self) -> None:
        if self.total_steps <= 0 or self.rollout_steps <= 0 or self.minibatch_size <= 0:
            raise ValueError("training counts must be positive")
        if not 0 < self.clip_coef < 1:
            raise ValueError("clip coefficient must be in (0,1)")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount factors must be in [0,1]")


@dataclass
class DSPTrainResult:
    global_lambda: float
    episode_costs: list[float]
    episode_rewards: list[float]
    update_history: list[dict[str, float]]
    elapsed_seconds: float
    config: dict


class DSPTrainer:
    def __init__(
        self,
        policy: DynamicShadowPricePolicy,
        config: DSPTrainerConfig,
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

    def train(self, env_factory: Callable[[int], object]) -> DSPTrainResult:
        started = time.perf_counter()
        total = 0
        episode_index = 0
        episode_cost = 0.0
        episode_reward = 0.0
        episode_costs: list[float] = []
        episode_rewards: list[float] = []
        history: list[dict[str, float]] = []
        env = env_factory(self.seed)
        obs, _ = env.reset(seed=self.seed)
        while total < self.config.total_steps:
            target = min(self.config.rollout_steps, self.config.total_steps - total)
            buffer = {
                key: []
                for key in (
                    "obs",
                    "next_obs",
                    "action",
                    "log_prob",
                    "reward",
                    "cost",
                    "done",
                    "reward_value",
                    "cost_value",
                )
            }
            completed_costs = []
            for _ in range(target):
                tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    output = self.policy.act(tensor)
                action = int(output.action.item())
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                buffer["obs"].append(np.asarray(obs, dtype=np.float32))
                buffer["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
                buffer["action"].append(action)
                buffer["log_prob"].append(float(output.log_prob.item()))
                buffer["reward"].append(float(reward))
                buffer["cost"].append(float(info["resource_cost"]))
                buffer["done"].append(float(done))
                buffer["reward_value"].append(float(output.reward_value.item()))
                buffer["cost_value"].append(float(output.cost_value.item()))
                episode_cost += float(info["resource_cost"])
                episode_reward += float(reward)
                obs = next_obs
                if done:
                    completed_costs.append(episode_cost)
                    episode_costs.append(episode_cost)
                    episode_rewards.append(episode_reward)
                    episode_index += 1
                    env_seed = self.seed + episode_index * 9973
                    env = env_factory(env_seed)
                    obs, _ = env.reset(seed=env_seed)
                    episode_cost = episode_reward = 0.0
            total += target
            with torch.no_grad():
                next_value = self.policy.value(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).item()
                next_cost_value = self.policy.act(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0),
                    deterministic=True,
                ).cost_value.item()
            if buffer["done"][-1]:
                next_value = next_cost_value = 0.0
            reward_adv, reward_returns = compute_gae(
                np.asarray(buffer["reward"]),
                np.asarray(buffer["reward_value"]),
                np.asarray(buffer["done"]),
                self.config.gamma,
                self.config.gae_lambda,
                next_value,
            )
            cost_adv, cost_returns = compute_gae(
                np.asarray(buffer["cost"]),
                np.asarray(buffer["cost_value"]),
                np.asarray(buffer["done"]),
                self.config.gamma,
                self.config.gae_lambda,
                next_cost_value,
            )
            advantages = reward_adv.copy()
            if self.config.use_global_lambda:
                advantages -= self.global_lambda * cost_adv
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            stats = self._update(buffer, advantages, reward_returns, cost_returns)
            if self.config.use_global_lambda and completed_costs:
                violation = (np.mean(completed_costs) - self.budget) / max(self.budget, 1e-8)
                self.global_lambda = max(
                    0.0,
                    self.global_lambda + self.config.global_lambda_lr * float(violation),
                )
            stats.update(
                {
                    "steps": float(total),
                    "global_lambda": float(self.global_lambda),
                    "mean_completed_cost": (
                        float(np.mean(completed_costs)) if completed_costs else np.nan
                    ),
                }
            )
            history.append(stats)
        return DSPTrainResult(
            self.global_lambda,
            episode_costs,
            episode_rewards,
            history,
            time.perf_counter() - started,
            asdict(self.config),
        )

    def _update(self, buffer, advantages, reward_returns, cost_returns):
        obs = torch.as_tensor(np.asarray(buffer["obs"]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(
            np.asarray(buffer["next_obs"]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(buffer["action"], dtype=torch.long, device=self.device)
        old_log = torch.as_tensor(buffer["log_prob"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(buffer["reward"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(buffer["done"], dtype=torch.float32, device=self.device)
        advantage = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        reward_target = torch.as_tensor(reward_returns, dtype=torch.float32, device=self.device)
        cost_target = torch.as_tensor(cost_returns, dtype=torch.float32, device=self.device)
        aggregate = {
            key: []
            for key in (
                "policy_loss",
                "reward_value_loss",
                "cost_value_loss",
                "q_value_loss",
                "monotonic_loss",
                "monotonic_violation_rate",
                "entropy",
                "approx_kl",
                "price_policy_kl",
                "shadow_price_mean",
                "shadow_price_std",
                "raw_negative_mu_rate",
            )
        }
        n = len(actions)
        for _ in range(self.config.update_epochs):
            for start in range(0, n, self.config.minibatch_size):
                permutation = self.rng.permutation(n) if start == 0 else permutation
                idx = torch.as_tensor(
                    permutation[start : start + self.config.minibatch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                output = self.policy.act(obs[idx], action=actions[idx])
                log_ratio = output.log_prob - old_log[idx]
                ratio = log_ratio.exp()
                policy_loss = torch.maximum(
                    -advantage[idx] * ratio,
                    -advantage[idx]
                    * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef),
                ).mean()
                reward_value_loss = nn.functional.mse_loss(
                    output.reward_value, reward_target[idx]
                )
                cost_value_loss = nn.functional.mse_loss(
                    output.cost_value, cost_target[idx]
                )
                monotonic_loss, monotonic_rate = self.policy.monotonic_regularization(obs[idx])
                q_loss = torch.zeros((), device=self.device)
                if output.q_scores is not None:
                    with torch.no_grad():
                        next_values = self.policy.value(next_obs[idx])
                        q_target = dsp_b_td_target(
                            rewards[idx], next_values, dones[idx], self.config.gamma
                        )
                    predicted = output.q_scores.gather(1, actions[idx, None]).squeeze(1)
                    q_loss = nn.functional.mse_loss(predicted, q_target)
                loss = (
                    policy_loss
                    + self.config.value_coef * reward_value_loss
                    + self.config.cost_value_coef * cost_value_loss
                    + self.config.q_value_coef * q_loss
                    + self.policy.config.monotonic_coef * monotonic_loss
                    - self.config.entropy_coef * output.entropy.mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                values = {
                    "policy_loss": policy_loss,
                    "reward_value_loss": reward_value_loss,
                    "cost_value_loss": cost_value_loss,
                    "q_value_loss": q_loss,
                    "monotonic_loss": monotonic_loss,
                    "monotonic_violation_rate": monotonic_rate,
                    "entropy": output.entropy.mean(),
                    "approx_kl": ((ratio - 1) - log_ratio).mean(),
                    "price_policy_kl": output.policy_kl.mean(),
                    "shadow_price_mean": output.shadow_price.mean(),
                    "shadow_price_std": output.shadow_price.std(unbiased=False),
                    "raw_negative_mu_rate": (output.raw_shadow_price < 0).float().mean(),
                }
                for key, value in values.items():
                    aggregate[key].append(float(value.detach()))
        return {key: float(np.mean(values)) for key, values in aggregate.items()}
