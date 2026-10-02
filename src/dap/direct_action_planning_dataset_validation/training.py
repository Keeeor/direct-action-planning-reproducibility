from __future__ import annotations

import copy
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn

from dap.agents.rules import AggressiveRule, ConservativeRule

from .data import TraceDataset, make_trace_env
from .models import FeatureNormalizer, FullTransitionNetwork, LoadForecaster, ValueNetwork


@dataclass(frozen=True)
class BranchDataset:
    observations: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    feasible: np.ndarray
    done: np.ndarray
    next_load: np.ndarray
    domain: np.ndarray
    window_start: np.ndarray

    @property
    def n_states(self) -> int:
        return int(len(self.observations))

    def concatenate(self, other: "BranchDataset") -> "BranchDataset":
        return BranchDataset(
            **{
                field: np.concatenate([getattr(self, field), getattr(other, field)], axis=0)
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class TrainingArtifacts:
    base_value: ValueNetwork
    refreshed_value: ValueNetwork
    load_forecaster: LoadForecaster
    full_transition: FullTransitionNetwork
    histories: dict[str, list[dict[str, float]]]
    collection: dict[str, float | int]
    training_seconds: float


def feasible_actions(env) -> np.ndarray:
    remaining = max(env.config.budget - env.cumulative_cost, 0.0)
    return np.flatnonzero(env.action_costs <= remaining + 1.0e-8)


def _collection_action(env, observation: np.ndarray, episode: int, rng: np.random.Generator) -> int:
    feasible = feasible_actions(env)
    mode = episode % 4
    if mode == 0:
        return int(rng.choice(feasible))
    if mode == 1:
        proposed = AggressiveRule().act(observation)
    elif mode == 2:
        proposed = ConservativeRule().act(observation)
    else:
        proposed = 0
    return int(proposed if proposed in feasible else feasible[-1])


def collect_branch_dataset(
    dataset: TraceDataset,
    *,
    split: str,
    horizon: int,
    budget: float,
    episodes_per_domain: int,
    seed: int,
    planner=None,
) -> BranchDataset:
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    feasibilities: list[np.ndarray] = []
    dones: list[bool] = []
    next_loads: list[float] = []
    domains: list[str] = []
    starts: list[int] = []
    rng = np.random.default_rng(seed)
    for domain_index, domain in enumerate(dataset.domain_names):
        for episode in range(episodes_per_domain):
            window_seed = seed + domain_index * 1_000_003 + episode * 9_973
            env, start = make_trace_env(
                dataset,
                domain,
                split,
                horizon=horizon,
                budget=budget,
                window_seed=window_seed,
            )
            observation, _ = env.reset(seed=window_seed)
            while True:
                feasible = np.zeros(4, dtype=bool)
                branch_next = np.zeros((4, 14), dtype=np.float32)
                branch_reward = np.full(4, -np.inf, dtype=np.float32)
                for action in feasible_actions(env):
                    branch = copy.deepcopy(env)
                    next_observation, reward, terminated, truncated, _ = branch.step(int(action))
                    feasible[action] = True
                    branch_next[action] = next_observation
                    branch_reward[action] = reward
                    branch_done = bool(terminated or truncated)
                observations.append(observation.copy())
                next_observations.append(branch_next)
                rewards.append(branch_reward)
                feasibilities.append(feasible)
                dones.append(branch_done)
                next_loads.append(float(branch_next[0, 0]) if not branch_done else 0.0)
                domains.append(domain)
                starts.append(start)
                if planner is None:
                    action = _collection_action(env, observation, episode, rng)
                else:
                    action, _ = planner(env, observation)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    return BranchDataset(
        observations=np.asarray(observations, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        feasible=np.asarray(feasibilities, dtype=bool),
        done=np.asarray(dones, dtype=bool),
        next_load=np.asarray(next_loads, dtype=np.float32),
        domain=np.asarray(domains, dtype=str),
        window_start=np.asarray(starts, dtype=np.int64),
    )


def _bellman_targets(
    model: ValueNetwork,
    data: BranchDataset,
    gamma: float,
) -> np.ndarray:
    flat_next = data.next_observations.reshape(-1, 14)
    next_values = model.predict(flat_next).reshape(data.n_states, 4)
    q_values = data.rewards.astype(np.float64) + gamma * (
        ~data.done[:, None]
    ) * next_values
    q_values[~data.feasible] = -np.inf
    return np.max(q_values, axis=1).astype(np.float32)


def train_value_fvi(
    training: BranchDataset,
    validation: BranchDataset,
    *,
    seed: int,
    gamma: float,
    iterations: int,
    epochs_per_iteration: int,
    learning_rate: float,
    hidden_dim: int,
    initial: ValueNetwork | None = None,
    anchors: np.ndarray | None = None,
    anchor_weight: float = 0.0,
) -> tuple[ValueNetwork, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if initial is None:
        normalizer = FeatureNormalizer.fit(training.observations)
        model = ValueNetwork(normalizer, hidden_dim=hidden_dim)
    else:
        model = copy.deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    observations = torch.as_tensor(training.observations, dtype=torch.float32)
    if anchors is not None:
        anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32)
        with torch.no_grad():
            anchor_target = initial(anchor_tensor).detach()
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = np.inf
    for iteration in range(iterations):
        target_model = copy.deepcopy(model).eval()
        targets = torch.as_tensor(
            _bellman_targets(target_model, training, gamma), dtype=torch.float32
        )
        losses: list[float] = []
        for _ in range(epochs_per_iteration):
            for start in range(0, len(observations), 256):
                indices = rng.permutation(len(observations))[start : start + 256]
                index = torch.as_tensor(indices, dtype=torch.long)
                predicted = model(observations[index])
                loss = nn.functional.smooth_l1_loss(predicted, targets[index])
                if anchors is not None and anchor_weight > 0:
                    anchor_prediction = model(anchor_tensor)
                    loss = loss + anchor_weight * nn.functional.mse_loss(
                        anchor_prediction, anchor_target
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
        validation_target = _bellman_targets(model, validation, gamma)
        validation_prediction = model.predict(validation.observations)
        validation_residual = float(np.mean(np.abs(validation_prediction - validation_target)))
        history.append(
            {
                "iteration": float(iteration),
                "training_loss": float(np.mean(losses)),
                "validation_bellman_mae": validation_residual,
            }
        )
        if validation_residual < best_validation:
            best_validation = validation_residual
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history


def train_load_forecaster(
    training: BranchDataset,
    validation: BranchDataset,
    normalizer: FeatureNormalizer,
    *,
    seed: int,
    epochs: int,
) -> tuple[LoadForecaster, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = LoadForecaster(normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = torch.as_tensor(training.observations, dtype=torch.float32)
    y = torch.as_tensor(training.next_load, dtype=torch.float32)
    vx = torch.as_tensor(validation.observations, dtype=torch.float32)
    vy = validation.next_load
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best = np.inf
    for epoch in range(epochs):
        losses = []
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
            prediction = model(x[index])
            loss = nn.functional.smooth_l1_loss(
                torch.log1p(prediction), torch.log1p(y[index])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            validation_prediction = model(vx).numpy()
        mae = float(np.mean(np.abs(validation_prediction - vy)))
        history.append({"epoch": float(epoch), "training_loss": float(np.mean(losses)), "validation_mae": mae})
        if mae < best:
            best = mae
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history


def train_full_transition(
    training: BranchDataset,
    validation: BranchDataset,
    normalizer: FeatureNormalizer,
    *,
    seed: int,
    epochs: int,
) -> tuple[FullTransitionNetwork, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = FullTransitionNetwork(normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    def flatten(data: BranchDataset):
        state_indices, actions = np.nonzero(data.feasible)
        return (
            torch.as_tensor(data.observations[state_indices], dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(data.next_observations[state_indices, actions], dtype=torch.float32),
            torch.as_tensor(data.rewards[state_indices, actions], dtype=torch.float32),
        )

    x, a, y_next, y_reward = flatten(training)
    vx, va, vy_next, vy_reward = flatten(validation)
    scale = torch.as_tensor(normalizer.scale, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best = np.inf
    for epoch in range(epochs):
        losses = []
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = torch.as_tensor(permutation[start : start + 256], dtype=torch.long)
            predicted_next, predicted_reward = model(x[index], a[index])
            state_loss = torch.mean(((predicted_next - y_next[index]) / scale).square())
            reward_loss = nn.functional.smooth_l1_loss(predicted_reward, y_reward[index])
            loss = state_loss + reward_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            predicted_next, predicted_reward = model(vx, va)
            state_mae = float(torch.mean(torch.abs((predicted_next - vy_next) / scale)))
            reward_mae = float(torch.mean(torch.abs(predicted_reward - vy_reward)))
        score = state_mae + reward_mae
        history.append({"epoch": float(epoch), "training_loss": float(np.mean(losses)), "validation_state_nmae": state_mae, "validation_reward_mae": reward_mae})
        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history


def train_dap_components(
    dataset: TraceDataset,
    *,
    horizon: int,
    budget: float,
    seed: int,
    gamma: float,
    collection_episodes: int,
    validation_episodes: int,
    fvi_iterations: int,
    refresh_iterations: int,
    model_epochs: int,
    hidden_dim: int,
) -> TrainingArtifacts:
    started = time.perf_counter()
    training = collect_branch_dataset(
        dataset,
        split="train",
        horizon=horizon,
        budget=budget,
        episodes_per_domain=collection_episodes,
        seed=seed,
    )
    validation = collect_branch_dataset(
        dataset,
        split="validation",
        horizon=horizon,
        budget=budget,
        episodes_per_domain=validation_episodes,
        seed=seed + 10_000_019,
    )
    base_value, base_history = train_value_fvi(
        training,
        validation,
        seed=seed,
        gamma=gamma,
        iterations=fvi_iterations,
        epochs_per_iteration=2,
        learning_rate=1.0e-3,
        hidden_dim=hidden_dim,
    )
    forecaster, load_history = train_load_forecaster(
        training,
        validation,
        base_value.normalizer,
        seed=seed + 1,
        epochs=model_epochs,
    )
    transition, transition_history = train_full_transition(
        training,
        validation,
        base_value.normalizer,
        seed=seed + 2,
        epochs=model_epochs,
    )

    from .planning import make_planner

    base_planner = make_planner(
        "structured_dap", base_value, forecaster, transition, gamma=gamma
    )
    on_policy = collect_branch_dataset(
        dataset,
        split="train",
        horizon=horizon,
        budget=budget,
        episodes_per_domain=max(collection_episodes // 2, 1),
        seed=seed + 20_000_033,
        planner=base_planner,
    )
    refreshed_training = training.concatenate(on_policy)
    refreshed_value, refresh_history = train_value_fvi(
        refreshed_training,
        validation,
        seed=seed + 3,
        gamma=gamma,
        iterations=refresh_iterations,
        epochs_per_iteration=2,
        learning_rate=5.0e-4,
        hidden_dim=hidden_dim,
        initial=base_value,
        anchors=validation.observations[: min(512, validation.n_states)],
        anchor_weight=0.05,
    )
    return TrainingArtifacts(
        base_value=base_value,
        refreshed_value=refreshed_value,
        load_forecaster=forecaster,
        full_transition=transition,
        histories={
            "base_value": base_history,
            "value_refresh": refresh_history,
            "load_forecaster": load_history,
            "full_transition": transition_history,
        },
        collection={
            "training_states": training.n_states,
            "validation_states": validation.n_states,
            "refresh_states": on_policy.n_states,
        },
        training_seconds=time.perf_counter() - started,
    )
