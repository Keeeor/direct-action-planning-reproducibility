from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .metrics import summarize_episode


def evaluate_agent(
    agent: Any,
    env_factory,
    budget: float,
    seed: int,
    episodes: int,
    device: torch.device,
    global_lambda: float = 0.0,
    deterministic: bool = True,
):
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    latencies_ns: list[int] = []
    for episode in range(episodes):
        episode_seed = seed + 100_003 * episode
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)
        env = env_factory(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        if hasattr(agent, "reset_budget_controller"):
            agent.reset_budget_controller()
        current: list[dict[str, Any]] = []
        step = 0
        while True:
            started = time.perf_counter_ns()
            if hasattr(agent, "config"):
                with torch.no_grad():
                    output = agent.act(
                        torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                        deterministic=deterministic,
                    )
                action = int(output.action.item())
                extras = {
                    "local_budget": float(output.local_budget.item()) if output.local_budget is not None else None,
                    "budget_multiplier": float(output.budget_multiplier.item()) if output.budget_multiplier is not None else None,
                    "local_lambda": float(output.local_lambda.item()) if output.local_lambda is not None else None,
                    "allocator_output": float(output.allocator_output.item()) if output.allocator_output is not None else None,
                }
            else:
                action = int(agent.act(obs))
                extras = {}
            latencies_ns.append(time.perf_counter_ns() - started)
            next_obs, reward, terminated, truncated, info = env.step(action)
            row = {
                **info,
                "reward": reward,
                "episode": episode,
                "step": step,
                "global_lambda": global_lambda,
            }
            row.update({key: value for key, value in extras.items() if value is not None})
            current.append(row)
            step_rows.append(row)
            obs = next_obs
            step += 1
            if terminated or truncated:
                break
        summary = summarize_episode(current, budget=budget, horizon=env.config.horizon)
        summary.update({"eval_episode": episode, "eval_seed": episode_seed})
        episode_rows.append(summary)
    latency = np.asarray(latencies_ns, dtype=np.float64) / 1e6
    latency_stats = {
        "decision_latency_ms_mean": float(latency.mean()),
        "decision_latency_ms_p95": float(np.quantile(latency, 0.95)),
    }
    return episode_rows, step_rows, latency_stats
