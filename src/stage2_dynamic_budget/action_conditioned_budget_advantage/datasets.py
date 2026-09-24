from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .branching import BranchableDiscreteEnv, branch_actions
from .dp import ActionConditionedBudgetMDP, ActionDPResult


def _state_seed(seed: int, k: int, t: int, load: int, queue: int, budget: int) -> int:
    payload = f"{seed}:{k}:{t}:{load}:{queue}:{budget}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def generate_branch_dataset(
    mdp: ActionConditionedBudgetMDP,
    optimum: ActionDPResult,
    branch_horizons: list[int] | tuple[int, ...],
    seed: int,
) -> pd.DataFrame:
    env = BranchableDiscreteEnv(
        mdp.config,
        initial_budget=mdp.config.max_budget,
        budget_scale=mdp.config.max_budget,
    )
    env.reset(seed=seed)
    rows: list[dict[str, object]] = []
    for k in branch_horizons:
        for t, load, queue, budget in np.ndindex(optimum.actions.shape):
            observation = env.set_markov_state(t, load, queue, budget)
            branch_seed = _state_seed(seed, int(k), t, load, queue, budget)
            current = branch_actions(env, optimum, int(k), branch_seed)
            for row in current:
                action = int(row["action"])
                q_star = float(optimum.q_values[t, load, queue, budget, action])
                row.update(
                    {
                        "scenario": mdp.config.scenario,
                        "seed": seed,
                        "Q_star": q_star,
                        "A_star": float(
                            optimum.advantages[t, load, queue, budget, action]
                        ),
                        "optimal_action": int(optimum.actions[t, load, queue, budget]),
                    }
                )
                for field, value in enumerate(observation):
                    row[f"obs_{field}"] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_accuracy(group: pd.DataFrame) -> tuple[int, int]:
    q_star = group["Q_star"].to_numpy(dtype=float)
    q_branch = group["q_branch"].to_numpy(dtype=float)
    correct = total = 0
    for left in range(len(group)):
        for right in range(left + 1, len(group)):
            truth = np.sign(q_star[left] - q_star[right])
            if truth == 0:
                continue
            total += 1
            correct += int(np.sign(q_branch[left] - q_branch[right]) == truth)
    return correct, total


def summarize_branch_fidelity(frame: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    group_keys = ["scenario", "seed", "k_requested"]
    state_keys = ["t", "load", "queue", "remaining_budget"]
    for key, subset in frame.groupby(group_keys, sort=True):
        top_correct = states = pair_correct = pair_total = 0
        for _, state in subset.groupby(state_keys, sort=False):
            branch_action = int(state.loc[state.q_branch.idxmax(), "action"])
            optimal_action = int(state["optimal_action"].iloc[0])
            top_correct += int(branch_action == optimal_action)
            states += 1
            correct, total = _pairwise_accuracy(state)
            pair_correct += correct
            pair_total += total
        q_star = subset["Q_star"].to_numpy(dtype=float)
        q_branch = subset["q_branch"].to_numpy(dtype=float)
        summaries.append(
            {
                "scenario": key[0],
                "seed": int(key[1]),
                "K": int(key[2]),
                "rows": len(subset),
                "states": states,
                "pearson_q_branch_q_star": float(pearsonr(q_branch, q_star).statistic),
                "spearman_q_branch_q_star": float(spearmanr(q_branch, q_star).statistic),
                "top_action_accuracy": top_correct / max(states, 1),
                "pairwise_ranking_accuracy": pair_correct / max(pair_total, 1),
            }
        )
    return pd.DataFrame(summaries)


def training_arrays(frame: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = frame[frame.k_requested == k]
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    state_keys = ["scenario", "t", "load", "queue", "remaining_budget"]
    action_dim = int(frame.action.max()) + 1
    for _, state in subset.groupby(state_keys, sort=False):
        observations.append(
            state.iloc[0][[f"obs_{index}" for index in range(14)]].to_numpy(dtype=np.float32)
        )
        target = np.zeros(action_dim, dtype=np.float32)
        mask = np.zeros(action_dim, dtype=bool)
        for row in state.itertuples():
            target[int(row.action)] = float(row.branch_advantage)
            mask[int(row.action)] = True
        targets.append(target)
        masks.append(mask)
    return np.stack(observations), np.stack(targets), np.stack(masks)

