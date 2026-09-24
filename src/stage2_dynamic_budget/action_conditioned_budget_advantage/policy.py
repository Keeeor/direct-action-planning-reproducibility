from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from stage2_dynamic_budget.dynamic_shadow_price.hard_coupling import GlobalBudgetMaskedPolicy
from stage2_dynamic_budget.dynamic_shadow_price.shadow_policy import (
    DSPPolicyConfig,
    DynamicShadowPricePolicy,
)
from stage2_dynamic_budget.models.policy import (
    ConstrainedSchedulingPolicy,
    PolicyConfig,
    PolicyOutput,
)

from .model import ActionAdvantageModel, advantage_adjust_logits


BASELINE_METHODS = ("b4_budget_state", "cdba", "dsp_a", "dsp_b")


@dataclass
class ACBAPolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reward_value: torch.Tensor
    cost_value: torch.Tensor
    logits: torch.Tensor
    base_logits: torch.Tensor
    action_advantages: torch.Tensor


class ACBAAPolicy(nn.Module):
    def __init__(
        self,
        base_policy: ConstrainedSchedulingPolicy,
        advantage_model: ActionAdvantageModel,
        action_costs: tuple[float, ...],
        episode_budget: float,
        alpha: float,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.advantage_model = advantage_model
        self.config = base_policy.config
        self.episode_budget = float(episode_budget)
        self.alpha = float(alpha)
        self.register_buffer("action_costs", torch.tensor(action_costs, dtype=torch.float32))

    def reset_budget_controller(self) -> None:
        self.base_policy.reset_budget_controller()

    def act(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        **kwargs,
    ) -> ACBAPolicyOutput:
        base = self.base_policy.act(observation, deterministic=True, **kwargs)
        advantages = self.advantage_model(observation)
        logits = advantage_adjust_logits(base.logits, advantages, self.alpha)
        remaining = observation[..., -2].clamp(min=0.0) * self.episode_budget
        valid = self.action_costs.to(remaining.device).unsqueeze(0) <= remaining.unsqueeze(-1) + 1e-8
        valid[:, int(torch.argmin(self.action_costs).item())] = True
        logits = logits.masked_fill(~valid, -torch.inf)
        distribution = Categorical(logits=logits)
        selected = action
        if selected is None:
            selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ACBAPolicyOutput(
            action=selected,
            log_prob=distribution.log_prob(selected),
            entropy=distribution.entropy(),
            reward_value=base.reward_value,
            cost_value=base.cost_value,
            logits=logits,
            base_logits=base.logits,
            action_advantages=advantages,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def frozen_checkpoint_path(
    project_root: Path, method: str, scenario: str, seed: int
) -> Path:
    if method not in BASELINE_METHODS:
        raise ValueError(f"unknown frozen method: {method}")
    source_scenario = scenario if scenario in {"early_burst", "late_burst"} else "early_burst"
    return (
        project_root
        / "results/dynamic_shadow_price/dp_learning"
        / f"dp_joint__formal__{method}__{source_scenario}__s{seed}"
        / "model.pt"
    )


def load_frozen_baseline(
    project_root: Path,
    method: str,
    scenario: str,
    seed: int,
    action_costs: tuple[float, ...],
    budget_scale: float,
    device: torch.device,
):
    path = frozen_checkpoint_path(project_root, method, scenario, seed)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config_data = dict(checkpoint["config"])
    if method in {"b4_budget_state", "cdba"}:
        config = PolicyConfig(**config_data)
        base = ConstrainedSchedulingPolicy(config).to(device)
        base.load_state_dict(checkpoint["state_dict"], strict=True)
        base.eval()
        return GlobalBudgetMaskedPolicy(base, action_costs, budget_scale).to(device), path
    config_data["action_costs"] = tuple(config_data["action_costs"])
    config = DSPPolicyConfig(**config_data)
    policy = DynamicShadowPricePolicy(config).to(device)
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    policy.eval()
    return policy, path


def load_frozen_b4_base(
    project_root: Path, scenario: str, seed: int, device: torch.device
) -> tuple[ConstrainedSchedulingPolicy, Path, float]:
    path = frozen_checkpoint_path(project_root, "b4_budget_state", scenario, seed)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    base = ConstrainedSchedulingPolicy(PolicyConfig(**dict(checkpoint["config"]))).to(device)
    base.load_state_dict(checkpoint["state_dict"], strict=True)
    return base, path, float(checkpoint.get("global_lambda", 0.0))

