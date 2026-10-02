from __future__ import annotations

import gymnasium as gym
import numpy as np


class AbsoluteBudgetObservationWrapper(gym.Wrapper):
    """Expose remaining absolute budget on a shared cross-budget scale.

    Queue dynamics, rewards, costs, termination, and the underlying config are
    unchanged. Only canonical observation field 12 is replaced.
    """

    def __init__(self, env: gym.Env, budget_scale: float):
        super().__init__(env)
        if budget_scale <= 0 or budget_scale < float(env.config.budget):
            raise ValueError("budget_scale must be positive and at least the episode budget")
        self.budget_scale = float(budget_scale)
        self.config = env.config

    def _replace(self, observation, remaining_budget: float):
        result = np.asarray(observation, dtype=np.float32).copy()
        result[-2] = float(remaining_budget) / self.budget_scale
        return result

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._replace(observation, self.config.budget), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self._replace(observation, float(info["remaining_budget"]))
        return observation, reward, terminated, truncated, info
