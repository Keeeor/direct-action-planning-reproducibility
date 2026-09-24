from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json

import numpy as np

from .dp import ACBADPConfig, ActionConditionedBudgetMDP, ActionDPResult


@dataclass(frozen=True)
class DiscreteStateSnapshot:
    t: int
    load: int
    queue: int
    previous_queue: int
    previous_capacity: int
    previous_utilization: float
    previous_violation: float
    remaining_budget: int
    cumulative_cost: int
    rng_state_json: str


class BranchableDiscreteEnv:
    """Exact-DP adapter with complete snapshot/restore and exogenous random tapes."""

    def __init__(self, config: ACBADPConfig, initial_budget: int, budget_scale: int | None = None):
        if not 0 <= initial_budget <= config.max_budget:
            raise ValueError("initial_budget outside DP grid")
        self.config = config
        self.initial_budget = int(initial_budget)
        self.budget_scale = int(budget_scale or max(initial_budget, 1))
        if self.budget_scale < self.initial_budget:
            raise ValueError("budget_scale must be at least initial_budget")
        self.mdp = ActionConditionedBudgetMDP(config)
        self.np_random = np.random.default_rng()
        self._reset_state()

    def _reset_state(self) -> None:
        self.t = 0
        self.load = 1
        self.queue = 0
        self.previous_queue = 0
        self.previous_capacity = 1
        self.previous_utilization = 0.0
        self.previous_violation = 0.0
        self.remaining_budget = self.initial_budget
        self.cumulative_cost = 0

    def reset(self, *, seed: int | None = None, options=None):
        if options:
            raise ValueError("options are not supported")
        self.np_random = np.random.default_rng(seed)
        self._reset_state()
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        arrivals = float(self.mdp.load_arrivals[self.load])
        latency = 1.0 + self.queue / max(self.previous_capacity, 1)
        tail = 1.0 + 2.0 * self.queue / max(self.previous_capacity, 1)
        return np.asarray(
            [
                arrivals,
                arrivals,
                float(self.queue),
                float(self.queue - self.previous_queue),
                float(self.previous_capacity),
                float(max(self.previous_capacity - 1, 0) / 3),
                self.previous_utilization,
                latency,
                tail,
                self.previous_violation,
                arrivals / 2.0,
                arrivals / max(self.previous_capacity, 1),
                self.remaining_budget / self.budget_scale,
                max(self.config.horizon - self.t, 0) / self.config.horizon,
            ],
            dtype=np.float32,
        )

    def valid_action_mask(self) -> np.ndarray:
        return self.mdp.action_costs <= self.remaining_budget

    def set_markov_state(self, t: int, load: int, queue: int, budget: int) -> np.ndarray:
        if not 0 <= t < self.config.horizon:
            raise ValueError("time is outside the active horizon")
        if not 0 <= load < self.mdp.n_loads or not 0 <= queue <= self.config.max_queue:
            raise ValueError("load or queue is outside the state grid")
        if not 0 <= budget <= self.config.max_budget:
            raise ValueError("budget is outside the state grid")
        self.t = int(t)
        self.load = int(load)
        self.queue = int(queue)
        self.previous_queue = int(queue)
        self.previous_capacity = 1
        self.previous_utilization = 0.0
        self.previous_violation = float(queue >= max(3, self.config.max_queue // 2))
        self.remaining_budget = int(budget)
        self.cumulative_cost = max(self.initial_budget - int(budget), 0)
        return self._observation()

    def snapshot(self) -> DiscreteStateSnapshot:
        state = copy.deepcopy(self.np_random.bit_generator.state)
        return DiscreteStateSnapshot(
            t=self.t,
            load=self.load,
            queue=self.queue,
            previous_queue=self.previous_queue,
            previous_capacity=self.previous_capacity,
            previous_utilization=float(self.previous_utilization),
            previous_violation=float(self.previous_violation),
            remaining_budget=self.remaining_budget,
            cumulative_cost=self.cumulative_cost,
            rng_state_json=json.dumps(state, sort_keys=True, separators=(",", ":")),
        )

    def restore(self, snapshot: DiscreteStateSnapshot) -> np.ndarray:
        self.t = snapshot.t
        self.load = snapshot.load
        self.queue = snapshot.queue
        self.previous_queue = snapshot.previous_queue
        self.previous_capacity = snapshot.previous_capacity
        self.previous_utilization = snapshot.previous_utilization
        self.previous_violation = snapshot.previous_violation
        self.remaining_budget = snapshot.remaining_budget
        self.cumulative_cost = snapshot.cumulative_cost
        self.np_random.bit_generator.state = json.loads(snapshot.rng_state_json)
        return self._observation()

    def step(self, action: int):
        return self.step_with_uniform(action, float(self.np_random.random()))

    def step_with_uniform(self, action: int, transition_uniform: float):
        if self.t >= self.config.horizon:
            raise RuntimeError("step called after termination")
        if not 0 <= int(action) < self.mdp.n_actions:
            raise ValueError("invalid action")
        if not 0 <= transition_uniform < 1:
            raise ValueError("transition uniform must be in [0, 1)")
        action = int(action)
        cost = int(self.mdp.action_costs[action])
        if cost > self.remaining_budget:
            raise ValueError("action cost exceeds remaining budget")
        prior_queue = self.queue
        current_load = self.load
        next_queue, reward, metrics = self.mdp.outcome(self.queue, self.load, action)
        probabilities = self.mdp.load_probabilities(self.t, self.load)
        next_load = int(
            min(
                np.searchsorted(np.cumsum(probabilities), transition_uniform, side="right"),
                self.mdp.n_loads - 1,
            )
        )
        self.remaining_budget -= cost
        self.cumulative_cost += cost
        self.previous_queue = prior_queue
        self.previous_capacity = int(self.mdp.action_capacity[action])
        self.previous_utilization = metrics["served"] / max(self.previous_capacity, 1)
        self.previous_violation = metrics["slo_violation"]
        self.queue = next_queue
        self.load = next_load
        self.t += 1
        terminated = self.t >= self.config.horizon
        info = {
            "arrivals": float(self.mdp.load_arrivals[current_load]),
            "served": metrics["served"],
            "resource_cost": float(cost),
            "cumulative_cost": float(self.cumulative_cost),
            "remaining_budget": float(self.remaining_budget),
            "slo_violation": metrics["slo_violation"],
            "queue_length": float(self.queue),
            "load": current_load,
            "next_load": next_load,
            "action": action,
        }
        return self._observation(), float(reward), terminated, False, info


def branch_actions(
    env: BranchableDiscreteEnv,
    optimum: ActionDPResult,
    k: int,
    seed: int,
) -> list[dict[str, object]]:
    if k <= 0:
        raise ValueError("k must be positive")
    snapshot = env.snapshot()
    effective_limit = min(k, env.config.horizon - snapshot.t)
    tape = np.random.default_rng(seed).random(effective_limit)
    tape_hash = "sha256:" + hashlib.sha256(tape.tobytes()).hexdigest()
    a0 = int(np.argmin(env.mdp.action_costs))
    rows: list[dict[str, object]] = []
    for candidate in np.flatnonzero(env.valid_action_mask()):
        env.restore(snapshot)
        discounted_reward = 0.0
        service_reward = 0.0
        resource_cost = 0.0
        rewards: list[float] = []
        for step_index in range(effective_limit):
            if step_index == 0:
                action = int(candidate)
            else:
                action = int(
                    optimum.actions[env.t, env.load, env.queue, env.remaining_budget]
                )
            _, reward, terminated, _, info = env.step_with_uniform(
                action, float(tape[step_index])
            )
            discounted_reward += (env.config.gamma**step_index) * float(reward)
            service_reward += float(reward)
            resource_cost += float(info["resource_cost"])
            rewards.append(float(reward))
            if terminated:
                break
        steps = len(rewards)
        bootstrap = 0.0
        if env.t < env.config.horizon:
            bootstrap = float(
                optimum.values[env.t, env.load, env.queue, env.remaining_budget]
            )
        q_branch = discounted_reward + (env.config.gamma**steps) * bootstrap
        rows.append(
            {
                "t": snapshot.t,
                "load": snapshot.load,
                "queue": snapshot.queue,
                "remaining_budget": snapshot.remaining_budget,
                "remaining_horizon": env.config.horizon - snapshot.t,
                "action": int(candidate),
                "action_cost": int(env.mdp.action_costs[candidate]),
                "k_requested": int(k),
                "k_effective": steps,
                "k_step_discounted_service_reward": float(discounted_reward),
                "k_step_service_reward": float(service_reward),
                "k_step_resource_cost": float(resource_cost),
                "terminal_queue": int(env.queue),
                "terminal_load": int(env.load),
                "terminal_remaining_budget": int(env.remaining_budget),
                "terminal_bootstrap_value": bootstrap,
                "q_branch": float(q_branch),
                "random_tape_sha256": tape_hash,
                "first_uniform": float(tape[0]),
                "oracle_bootstrap": True,
            }
        )
    reference = next(float(row["q_branch"]) for row in rows if row["action"] == a0)
    for row in rows:
        row["branch_advantage"] = float(row["q_branch"]) - reference
    env.restore(snapshot)
    return rows
