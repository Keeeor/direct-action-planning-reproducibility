from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import TraceDataset

from .models import normalize_observation
from .rl import RLTrainResult, _factory, _gae


class CPOActorCritic(nn.Module):
    """Separate actor and critics so critic fitting cannot violate the actor trust region."""

    def __init__(self, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.actor_body = nn.Sequential(nn.Linear(14, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.reward_critic = nn.Sequential(nn.Linear(14, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.cost_critic = nn.Sequential(nn.Linear(14, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def policy_logits(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(observation)
        return self.actor_head(self.actor_body(normalized))

    def values(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = normalize_observation(observation)
        return self.reward_critic(normalized).squeeze(-1), self.cost_critic(normalized).squeeze(-1)

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.actor_body.parameters()) + list(self.actor_head.parameters())


def _flat(values: list[torch.Tensor | None], parameters: list[nn.Parameter]) -> torch.Tensor:
    output = []
    for value, parameter in zip(values, parameters, strict=True):
        output.append(torch.zeros_like(parameter).reshape(-1) if value is None else value.reshape(-1))
    return torch.cat(output)


def _flat_parameters(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def _set_parameters(parameters: list[nn.Parameter], vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter.numel()
            parameter.copy_(vector[offset : offset + size].view_as(parameter))
            offset += size


def conjugate_gradient(operator, vector: torch.Tensor, iterations: int = 10, tolerance: float = 1.0e-10) -> torch.Tensor:
    solution = torch.zeros_like(vector)
    residual = vector.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)
    for _ in range(iterations):
        product = operator(direction)
        alpha = residual_dot / (torch.dot(direction, product) + 1.0e-12)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        new_dot = torch.dot(residual, residual)
        if new_dot <= tolerance:
            break
        direction = residual + (new_dot / (residual_dot + 1.0e-12)) * direction
        residual_dot = new_dot
    return solution


@dataclass(frozen=True)
class CPOConfig:
    total_steps: int = 102_400
    rollout_steps: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_kl: float = 0.01
    damping: float = 0.1
    cg_iterations: int = 10
    backtracks: int = 10
    critic_epochs: int = 4
    critic_learning_rate: float = 1.0e-3
    hidden_dim: int = 64


class CPOAgent:
    name = "cpo"

    def __init__(self, model: CPOActorCritic, budget: float):
        self.model = model.eval()
        self.budget = float(budget)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def select(self, env, observation: np.ndarray):
        tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        logits = self.model.policy_logits(tensor)[0]
        remaining = max(self.budget - float(env.cumulative_cost), 0.0)
        costs = torch.as_tensor(env.action_costs, dtype=torch.float32)
        masked = logits.masked_fill(costs > remaining + 1.0e-8, -1.0e9)
        values = masked.numpy().astype(np.float64)
        return int(np.argmax(values)), values


def projected_cpo_step(
    model: CPOActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    feasible: torch.Tensor,
    old_logits: torch.Tensor,
    reward_advantages: torch.Tensor,
    cost_advantages: torch.Tensor,
    constraint_violation: float,
    *,
    max_kl: float,
    damping: float,
    cg_iterations: int,
    backtracks: int,
) -> dict[str, float]:
    """Natural-gradient CPO step with a linearized cost half-space projection."""
    parameters = model.actor_parameters()
    old_distribution = Categorical(logits=old_logits.detach())

    def surrogates():
        logits = model.policy_logits(observations).masked_fill(~feasible, -1.0e9)
        distribution = Categorical(logits=logits)
        ratio = torch.exp(distribution.log_prob(actions) - old_distribution.log_prob(actions))
        reward_objective = torch.mean(ratio * reward_advantages)
        cost_objective = torch.mean(ratio * cost_advantages)
        kl = torch.mean(kl_divergence(old_distribution, distribution))
        return reward_objective, cost_objective, kl

    reward_objective, cost_objective, kl = surrogates()
    reward_gradient = _flat(torch.autograd.grad(reward_objective, parameters, retain_graph=True, allow_unused=True), parameters).detach()
    cost_gradient = _flat(torch.autograd.grad(cost_objective, parameters, retain_graph=True, allow_unused=True), parameters).detach()

    def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
        _, _, mean_kl = surrogates()
        first = _flat(torch.autograd.grad(mean_kl, parameters, create_graph=True, allow_unused=True), parameters)
        directional = torch.dot(first, vector)
        second = _flat(torch.autograd.grad(directional, parameters, retain_graph=True, allow_unused=True), parameters).detach()
        return second + damping * vector

    natural_reward = conjugate_gradient(fisher_vector_product, reward_gradient, cg_iterations)
    curvature = torch.dot(reward_gradient, natural_reward).clamp(min=1.0e-12)
    step = natural_reward * torch.sqrt(torch.as_tensor(2.0 * max_kl) / curvature)
    predicted_constraint = float(constraint_violation + torch.dot(cost_gradient, step).item())
    projected = False
    if predicted_constraint > 0.0 and torch.linalg.vector_norm(cost_gradient) > 1.0e-10:
        natural_cost = conjugate_gradient(fisher_vector_product, cost_gradient, cg_iterations)
        denominator = torch.dot(cost_gradient, natural_cost).clamp(min=1.0e-12)
        step = step - ((torch.dot(cost_gradient, step) + float(constraint_violation)) / denominator) * natural_cost
        quadratic = torch.dot(step, fisher_vector_product(step)).clamp(min=1.0e-12)
        if quadratic > 2.0 * max_kl:
            step = step * torch.sqrt(torch.as_tensor(2.0 * max_kl) / quadratic)
        projected = True

    old_parameters = _flat_parameters(parameters)
    old_reward = float(reward_objective.detach())
    old_constraint = float(constraint_violation + cost_objective.detach())
    accepted = False
    final_kl = 0.0
    final_constraint = old_constraint
    for index in range(backtracks):
        fraction = 0.5**index
        _set_parameters(parameters, old_parameters + fraction * step)
        new_reward, new_cost, new_kl = surrogates()
        candidate_constraint = float(constraint_violation + new_cost.detach())
        improves_constraint = candidate_constraint <= 0.0 if constraint_violation <= 0.0 else candidate_constraint < constraint_violation
        if float(new_kl.detach()) <= max_kl and float(new_reward.detach()) >= old_reward - 1.0e-8 and improves_constraint:
            accepted = True
            final_kl = float(new_kl.detach())
            final_constraint = candidate_constraint
            break
    if not accepted:
        _set_parameters(parameters, old_parameters)
    return {
        "accepted": float(accepted),
        "projected": float(projected),
        "kl": float(final_kl),
        "constraint_before": float(old_constraint),
        "constraint_after": float(final_constraint),
    }


def train_cpo(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    config: CPOConfig | None = None,
) -> RLTrainResult:
    config = config or CPOConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CPOActorCritic(hidden_dim=config.hidden_dim)
    critic_optimizer = torch.optim.Adam(
        list(model.reward_critic.parameters()) + list(model.cost_critic.parameters()),
        lr=config.critic_learning_rate,
    )
    domains = dataset.domain_names
    current_seed = int(seed)
    env = _factory(dataset, domains, horizon, budget, current_seed)
    observation, _ = env.reset(seed=current_seed)
    total_steps = 0
    episode_cost = 0.0
    all_episode_costs: list[float] = []
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    while total_steps < config.total_steps:
        target_steps = min(config.rollout_steps, config.total_steps - total_steps)
        buffer = {key: [] for key in ("obs", "action", "reward", "cost", "done", "value", "cost_value", "feasible", "old_logits")}
        completed: list[float] = []
        for _ in range(target_steps):
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                logits = model.policy_logits(obs_tensor)
                value, cost_value = model.values(obs_tensor)
            remaining = max(budget - float(env.cumulative_cost), 0.0)
            costs = torch.as_tensor(env.action_costs, dtype=torch.float32)
            feasible = costs <= remaining + 1.0e-8
            masked = logits.masked_fill(~feasible.unsqueeze(0), -1.0e9)
            distribution = Categorical(logits=masked)
            action = int(distribution.sample().item())
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            buffer["obs"].append(observation.copy())
            buffer["action"].append(action)
            buffer["reward"].append(float(reward))
            buffer["cost"].append(float(info["resource_cost"]))
            buffer["done"].append(float(done))
            buffer["value"].append(float(value.item()))
            buffer["cost_value"].append(float(cost_value.item()))
            buffer["feasible"].append(feasible.numpy())
            buffer["old_logits"].append(masked[0].numpy())
            episode_cost += float(info["resource_cost"])
            observation = next_observation
            if done:
                completed.append(episode_cost)
                all_episode_costs.append(episode_cost)
                episode_cost = 0.0
                current_seed += 9973
                env = _factory(dataset, domains, horizon, budget, current_seed)
                observation, _ = env.reset(seed=current_seed)
        total_steps += target_steps
        with torch.no_grad():
            bootstrap_reward, bootstrap_cost = model.values(torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1))
        reward_adv, reward_returns = _gae(buffer["reward"], buffer["value"], buffer["done"], config.gamma, config.gae_lambda, float(bootstrap_reward.item()))
        cost_adv, cost_returns = _gae(buffer["cost"], buffer["cost_value"], buffer["done"], config.gamma, config.gae_lambda, float(bootstrap_cost.item()))
        reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1.0e-8)
        # The linearized CPO constraint is J_C(old)-d + E[ratio * A_C].
        # Centering makes E_old[A_C]=0 despite finite-batch critic error.
        cost_adv = (cost_adv - cost_adv.mean()) / max(float(budget), 1.0e-8)
        observations = torch.as_tensor(np.asarray(buffer["obs"]), dtype=torch.float32)
        actions = torch.as_tensor(buffer["action"], dtype=torch.long)
        feasible = torch.as_tensor(np.asarray(buffer["feasible"]), dtype=torch.bool)
        old_logits = torch.as_tensor(np.asarray(buffer["old_logits"]), dtype=torch.float32)
        violation = (float(np.mean(completed)) - budget) / max(budget, 1.0e-8) if completed else -1.0
        step_info = projected_cpo_step(
            model,
            observations,
            actions,
            feasible,
            old_logits,
            torch.as_tensor(reward_adv, dtype=torch.float32),
            torch.as_tensor(cost_adv, dtype=torch.float32),
            violation,
            max_kl=config.max_kl,
            damping=config.damping,
            cg_iterations=config.cg_iterations,
            backtracks=config.backtracks,
        )
        reward_targets = torch.as_tensor(reward_returns, dtype=torch.float32)
        cost_targets = torch.as_tensor(cost_returns, dtype=torch.float32)
        critic_losses = []
        for _ in range(config.critic_epochs):
            reward_values, cost_values = model.values(observations)
            critic_loss = nn.functional.mse_loss(reward_values, reward_targets) + nn.functional.mse_loss(cost_values, cost_targets)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(model.reward_critic.parameters()) + list(model.cost_critic.parameters()), 5.0)
            critic_optimizer.step()
            critic_losses.append(float(critic_loss.detach()))
        history.append({"steps": float(total_steps), "critic_loss": float(np.mean(critic_losses)), **step_info})
    return RLTrainResult(
        CPOAgent(model, budget),
        history,
        time.perf_counter() - started,
        {
            "episodes": float(len(all_episode_costs)),
            "mean_episode_cost": float(np.mean(all_episode_costs)) if all_episode_costs else float("nan"),
            "accepted_step_rate": float(np.mean([row["accepted"] for row in history])) if history else 0.0,
            "max_kl": float(max((row["kl"] for row in history), default=0.0)),
        },
    )
