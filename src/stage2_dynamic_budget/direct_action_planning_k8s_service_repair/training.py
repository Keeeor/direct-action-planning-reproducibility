"""Train and select repaired DAP checkpoints on train/validation data only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import yaml

from stage2_dynamic_budget.data.trace_windows import select_trace_window
from stage2_dynamic_budget.direct_action_planning_dataset_validation.data import (
    load_trace_dataset,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from stage2_dynamic_budget.direct_action_planning_dataset_validation.training import (
    BranchDataset,
)
from stage2_dynamic_budget.direct_action_planning_paper_closure.training import (
    compute_value_target_scale,
    train_value_candidates,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.models import (
    EvidenceLoadForecaster,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json

from .planner import RuntimeConsistentPlanner
from .prototype_api import (
    ACTION_ORDER,
    ActionMapper,
    PROJECT_ROOT,
    PROTOTYPE_ROOT,
    save_checkpoint,
)
from .simulation import evaluate_window, initial_observation
from .transition import RuntimeConsistentSystemModel, target_is_feasible


if str(PROTOTYPE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROTOTYPE_ROOT))
from workload.trace_converter import fit_rate_scale, transform_rate  # noqa: E402


@dataclass(frozen=True)
class CandidateCheckpoint:
    value: Any
    forecaster: Any
    gamma: float
    continuation_weight: float
    forecast_strategy: str = "learned"
    forecast_multiplier: float = 1.0


def scaled_windows(
    *, dataset_name: str, domain: str, split: str, horizon: int,
    episodes: int, seed: int, target_peak_rps: float, max_rps: float,
) -> tuple[list[np.ndarray], list[int]]:
    dataset = load_trace_dataset(PROJECT_ROOT, dataset_name)
    scale = fit_rate_scale(
        dataset.domains[domain]["train"], quantile=0.99,
        target_peak_rps=target_peak_rps, max_rps=max_rps,
    )
    windows: list[np.ndarray] = []
    starts: list[int] = []
    for episode in range(int(episodes)):
        window_seed = int(seed) + episode * 9_973
        raw, start = select_trace_window(
            dataset.domains[domain][split], int(horizon), window_seed
        )
        windows.append(transform_rate(raw, scale))
        starts.append(int(start))
    return windows, starts


def collect_branch_dataset(
    *, windows: list[np.ndarray], starts: list[int], profile: str,
    model: RuntimeConsistentSystemModel, mapper: ActionMapper,
    budgets: tuple[float, ...], control_interval_seconds: float, seed: int,
) -> BranchDataset:
    if len(windows) != len(starts) or not windows:
        raise ValueError("windows and starts must be non-empty and aligned")
    horizon = len(windows[0])
    rng = np.random.default_rng(int(seed))
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dones: list[bool] = []
    next_loads: list[float] = []
    domains: list[str] = []
    window_starts: list[int] = []
    for budget in budgets:
        for rates, start in zip(windows, starts, strict=True):
            if len(rates) != horizon:
                raise ValueError("all trace windows must share one horizon")
            observation = initial_observation(model, float(rates[0]), budget, horizon)
            ready = model.base_replicas
            remaining = float(budget)
            for step in range(horizon):
                true_next = float(rates[step + 1]) if step + 1 < horizon else 0.0
                branch_next = np.zeros((4, 14), dtype=np.float32)
                branch_reward = np.full(4, -np.inf, dtype=np.float32)
                branch_cost = np.full(4, np.inf, dtype=np.float64)
                branch_ready = np.full(4, ready, dtype=np.int64)
                feasible = np.zeros(4, dtype=bool)
                for index, action in enumerate(ACTION_ORDER):
                    target = mapper.replicas(action)
                    allowed = target_is_feasible(
                        remaining_budget_seconds=remaining,
                        target_replicas=target, current_ready=ready,
                        base_replicas=model.base_replicas,
                        control_interval_seconds=control_interval_seconds,
                        scale_down_guard_seconds=model.scale_down_guard_seconds,
                    )
                    feasible[index] = allowed
                    if not allowed:
                        continue
                    branch = model.branch(
                        observation=observation, current_ready=ready,
                        action=action, mapper=mapper,
                        forecast_arrival_rps=true_next,
                        total_budget_seconds=budget,
                        remaining_budget_seconds=remaining,
                        remaining_horizon_steps=horizon - step,
                        horizon_steps=horizon,
                        control_interval_seconds=control_interval_seconds,
                    )
                    branch_next[index] = branch.next_observation
                    branch_reward[index] = branch.reward
                    branch_cost[index] = branch.expected_cost_seconds
                    branch_ready[index] = branch.next_ready_replicas
                if not feasible[0] or not feasible.any():
                    raise AssertionError("base action must remain feasible")
                observations.append(observation.copy())
                next_observations.append(branch_next)
                rewards.append(branch_reward)
                masks.append(feasible)
                dones.append(step == horizon - 1)
                next_loads.append(true_next)
                domains.append(profile)
                window_starts.append(start)
                selected = int(rng.choice(np.flatnonzero(feasible)))
                remaining = max(remaining - branch_cost[selected], 0.0)
                ready = int(branch_ready[selected])
                observation = branch_next[selected]
    return BranchDataset(
        observations=np.asarray(observations, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        feasible=np.asarray(masks, dtype=bool),
        done=np.asarray(dones, dtype=bool),
        next_load=np.asarray(next_loads, dtype=np.float32),
        domain=np.asarray(domains, dtype=str),
        window_start=np.asarray(window_starts, dtype=np.int64),
    )


def train_forecaster(
    training: BranchDataset, validation: BranchDataset,
    normalizer: FeatureNormalizer, *, seed: int, epochs: int,
) -> tuple[EvidenceLoadForecaster, list[dict[str, float]]]:
    torch.manual_seed(int(seed))
    model = EvidenceLoadForecaster(normalizer)
    target_median = float(np.median(training.next_load))
    target_scale = max(float(np.std(training.next_load)), 1.0)
    with torch.no_grad():
        final = model.network[-1]
        nn.init.zeros_(final.weight)
        final.bias.fill_(target_median)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = torch.as_tensor(training.observations, dtype=torch.float32)
    y = torch.as_tensor(training.next_load, dtype=torch.float32)
    vx = torch.as_tensor(validation.observations, dtype=torch.float32)
    rng = np.random.default_rng(int(seed))
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_mae = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        losses: list[float] = []
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = torch.as_tensor(order[start : start + 256], dtype=torch.long)
            loss = nn.functional.smooth_l1_loss(
                model(x[index]) / target_scale, y[index] / target_scale
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            prediction = model(vx).cpu().numpy()
        mae = float(np.mean(np.abs(prediction - validation.next_load)))
        history.append({
            "epoch": float(epoch), "training_loss": float(np.mean(losses)),
            "validation_mae": mae, "target_scale_rps": target_scale,
        })
        if mae < best_mae:
            best_mae = mae
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    return model.train(False), history


def _aggregate(metrics: list[Any]) -> dict[str, float]:
    if not metrics:
        raise ValueError("cannot aggregate zero validation runs")
    keys = (
        "completion_ratio", "slo_violation_rate", "ready_cost_seconds",
        "total_reward", "target_changes", "budget_violation_seconds",
    )
    return {
        key: float(np.mean([getattr(metric, key) for metric in metrics]))
        for key in keys
    }


def _safe_action(
    proposed: str, *, observation: np.ndarray, ready: int, remaining: float,
    mapper: ActionMapper, model: RuntimeConsistentSystemModel, interval: float,
) -> str:
    del observation
    if target_is_feasible(
        remaining_budget_seconds=remaining,
        target_replicas=mapper.replicas(proposed), current_ready=ready,
        base_replicas=model.base_replicas, control_interval_seconds=interval,
        scale_down_guard_seconds=model.scale_down_guard_seconds,
    ):
        return proposed
    for action in reversed(ACTION_ORDER):
        if target_is_feasible(
            remaining_budget_seconds=remaining,
            target_replicas=mapper.replicas(action), current_ready=ready,
            base_replicas=model.base_replicas, control_interval_seconds=interval,
            scale_down_guard_seconds=model.scale_down_guard_seconds,
        ):
            return action
    return "no_op"


def threshold_policy(
    *, mapper: ActionMapper, model: RuntimeConsistentSystemModel,
    interval: float, queue_thresholds: tuple[float, float, float],
) -> Callable:
    def policy(
        observation: np.ndarray, ready: int, remaining: float,
        remaining_horizon: int, current_target: int,
    ) -> str:
        del remaining_horizon, current_target
        queue = float(observation[2])
        p95 = float(observation[8])
        if queue >= queue_thresholds[2] or p95 >= model.slo_seconds * 1.5:
            proposed = "scale_large"
        elif queue >= queue_thresholds[1] or p95 >= model.slo_seconds:
            proposed = "scale_medium"
        elif queue >= queue_thresholds[0] or float(observation[3]) > 0:
            proposed = "scale_small"
        else:
            proposed = "no_op"
        return _safe_action(
            proposed, observation=observation, ready=ready, remaining=remaining,
            mapper=mapper, model=model, interval=interval,
        )
    return policy


def mpc_policy(
    *, checkpoint: CandidateCheckpoint, mapper: ActionMapper,
    model: RuntimeConsistentSystemModel, total_budget: float,
    interval: float, depth: int = 4,
) -> Callable:
    def rollout(
        observation: np.ndarray, ready: int, remaining: float,
        remaining_horizon: int, remaining_depth: int,
    ) -> float:
        if remaining_depth <= 0 or remaining_horizon <= 0:
            return 0.0
        forecast = max(float(checkpoint.forecaster.predict(observation)), 0.0)
        best = float("-inf")
        for action in ACTION_ORDER:
            target = mapper.replicas(action)
            if not target_is_feasible(
                remaining_budget_seconds=remaining, target_replicas=target,
                current_ready=ready, base_replicas=model.base_replicas,
                control_interval_seconds=interval,
                scale_down_guard_seconds=model.scale_down_guard_seconds,
            ):
                continue
            branch = model.branch(
                observation=observation, current_ready=ready, action=action,
                mapper=mapper, forecast_arrival_rps=forecast,
                total_budget_seconds=total_budget,
                remaining_budget_seconds=remaining,
                remaining_horizon_steps=remaining_horizon,
                horizon_steps=max(remaining_horizon, 1),
                control_interval_seconds=interval,
            )
            score = branch.reward + checkpoint.gamma * rollout(
                branch.next_observation, branch.next_ready_replicas,
                max(remaining - branch.expected_cost_seconds, 0.0),
                remaining_horizon - 1, remaining_depth - 1,
            )
            best = max(best, score)
        return 0.0 if not np.isfinite(best) else float(best)

    def policy(
        observation: np.ndarray, ready: int, remaining: float,
        remaining_horizon: int, current_target: int,
    ) -> str:
        del current_target
        forecast = max(float(checkpoint.forecaster.predict(observation)), 0.0)
        scores: dict[str, float] = {}
        for action in ACTION_ORDER:
            target = mapper.replicas(action)
            if not target_is_feasible(
                remaining_budget_seconds=remaining, target_replicas=target,
                current_ready=ready, base_replicas=model.base_replicas,
                control_interval_seconds=interval,
                scale_down_guard_seconds=model.scale_down_guard_seconds,
            ):
                scores[action] = float("-inf")
                continue
            branch = model.branch(
                observation=observation, current_ready=ready, action=action,
                mapper=mapper, forecast_arrival_rps=forecast,
                total_budget_seconds=total_budget,
                remaining_budget_seconds=remaining,
                remaining_horizon_steps=remaining_horizon,
                horizon_steps=max(remaining_horizon, 1),
                control_interval_seconds=interval,
            )
            scores[action] = branch.reward + checkpoint.gamma * rollout(
                branch.next_observation, branch.next_ready_replicas,
                max(remaining - branch.expected_cost_seconds, 0.0),
                remaining_horizon - 1, depth - 1,
            )
        return max(ACTION_ORDER, key=lambda action: scores[action])
    return policy


def select_validation_candidate(
    candidates: list[dict[str, Any]], comparator: dict[str, float]
) -> dict[str, Any]:
    """Frozen lexicographic validation rule from PLAN.md section 3."""

    if not candidates:
        raise ValueError("candidate list must not be empty")
    enriched: list[dict[str, Any]] = []
    for raw in candidates:
        row = dict(raw)
        completion_shortfall = max(
            float(comparator["completion_ratio"]) - 0.01
            - float(row["completion_ratio"]),
            0.0,
        )
        slo_excess = max(
            float(row["slo_violation_rate"])
            - (float(comparator["slo_violation_rate"]) + 0.02),
            0.0,
        )
        budget_excess = max(float(row["budget_violation_seconds"]), 0.0)
        row["guard_violation"] = completion_shortfall + slo_excess + budget_excess
        row["passed_service_guard"] = row["guard_violation"] <= 1.0e-12
        row["service_cost_productivity"] = (
            float(row["completion_ratio"]) - float(row["slo_violation_rate"])
        ) / max(float(row["ready_cost_seconds"]), 1.0)
        enriched.append(row)
    survivors = [row for row in enriched if row["passed_service_guard"]]
    if survivors:
        return max(
            survivors,
            key=lambda row: (
                row["service_cost_productivity"],
                -row["ready_cost_seconds"],
                -row["target_changes"],
                -row["iteration"],
                -row["continuation_weight"],
                -row["tie_margin"],
            ),
        )
    return min(
        enriched,
        key=lambda row: (
            row["guard_violation"], row["ready_cost_seconds"],
            row["target_changes"], row["iteration"],
            row["continuation_weight"], row["tie_margin"],
        ),
    )


def _evaluate_candidate(
    *, checkpoint: CandidateCheckpoint, tie_margin: float,
    windows: list[np.ndarray], budgets: tuple[float, ...],
    model: RuntimeConsistentSystemModel, mapper: ActionMapper,
    interval: float,
) -> dict[str, float]:
    metrics = []
    for budget in budgets:
        planner = RuntimeConsistentPlanner(
            checkpoint=checkpoint, system_model=model, mapper=mapper,
            control_interval_seconds=interval, horizon_steps=len(windows[0]),
            total_budget_seconds=budget, tie_margin=tie_margin,
        )
        metrics.extend(
            evaluate_window(
                rates=window, planner=planner, system_model=model, mapper=mapper,
                total_budget_seconds=budget, control_interval_seconds=interval,
            )
            for window in windows
        )
    return _aggregate(metrics)


def _evaluate_policy(
    *, policy_factory: Callable[[float], Callable], windows: list[np.ndarray],
    budgets: tuple[float, ...], model: RuntimeConsistentSystemModel,
    mapper: ActionMapper, interval: float,
) -> dict[str, float]:
    metrics = []
    for budget in budgets:
        policy = policy_factory(budget)
        metrics.extend(
            evaluate_window(
                rates=window, policy=policy, system_model=model, mapper=mapper,
                total_budget_seconds=budget, control_interval_seconds=interval,
            )
            for window in windows
        )
    return _aggregate(metrics)


def verify_development_contract(
    *, contract_path: Path, config_path: Path
) -> dict[str, Any]:
    from .audit import verify_contract

    return verify_contract(
        project_root=PROJECT_ROOT, contract_path=contract_path,
        expected_config_path=config_path,
    )


def train(
    config_path: Path, contract_path: Path, *, only_profile: str | None = None
) -> list[Path]:
    config_path = config_path.resolve()
    contract = verify_development_contract(
        contract_path=contract_path.resolve(), config_path=config_path
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapper = ActionMapper(config["actions"])
    budgets = tuple(float(value) for value in config["budgets_seconds"])
    outputs: list[Path] = []
    selected_profiles = dict(config["profiles"])
    if only_profile is not None:
        if only_profile not in selected_profiles:
            raise ValueError(f"unknown requested profile: {only_profile}")
        selected_profiles = {only_profile: selected_profiles[only_profile]}
    selection_budgets = (float(config["selection_budget_seconds"]),)
    for profile, source in selected_profiles.items():
        model = RuntimeConsistentSystemModel.load(
            PROJECT_ROOT / config["paths"]["system_model"], profile,
            slo_seconds=float(source["slo_seconds"]),
        )
        train_windows, train_starts = scaled_windows(
            dataset_name=source["dataset"], domain=source["domain"], split="train",
            horizon=int(config["horizon_steps"]), episodes=int(config["train_episodes"]),
            seed=int(config["seed"]), target_peak_rps=float(source["target_peak_rps"]),
            max_rps=float(source["max_rps"]),
        )
        val_windows, val_starts = scaled_windows(
            dataset_name=source["dataset"], domain=source["domain"], split="validation",
            horizon=int(config["horizon_steps"]), episodes=int(config["validation_episodes"]),
            seed=int(config["seed"]) + 7_000_003,
            target_peak_rps=float(source["target_peak_rps"]),
            max_rps=float(source["max_rps"]),
        )
        training = collect_branch_dataset(
            windows=train_windows, starts=train_starts, profile=profile,
            model=model, mapper=mapper, budgets=budgets,
            control_interval_seconds=float(config["control_interval_seconds"]),
            seed=int(config["seed"]) + 11,
        )
        validation = collect_branch_dataset(
            windows=val_windows, starts=val_starts, profile=profile,
            model=model, mapper=mapper, budgets=budgets,
            control_interval_seconds=float(config["control_interval_seconds"]),
            seed=int(config["seed"]) + 23,
        )
        normalizer = FeatureNormalizer.fit(training.observations)
        forecaster, forecast_history = train_forecaster(
            training, validation, normalizer,
            seed=int(config["seed"]) + 31,
            epochs=int(config["forecaster_epochs"]),
        )
        values, fvi_history = train_value_candidates(
            training, validation, seed=int(config["seed"]) + 37,
            gamma=float(config["gamma"]),
            iterations=int(config["fvi_iterations"]),
            candidate_iterations=tuple(int(v) for v in config["candidate_iterations"]),
            epochs_per_iteration=int(config["fvi_epochs_per_iteration"]),
            learning_rate=float(config["learning_rate"]),
            hidden_dim=int(config["hidden_dim"]),
            target_scale=compute_value_target_scale(
                training, horizon=int(config["horizon_steps"])
            ),
            zero_initialize_output=True,
        )
        comparison_checkpoint = CandidateCheckpoint(
            value=next(iter(values.values())), forecaster=forecaster,
            gamma=float(config["gamma"]), continuation_weight=0.0,
            forecast_strategy="learned", forecast_multiplier=1.0,
        )
        interval = float(config["control_interval_seconds"])
        if source["primary_comparator"] == "threshold":
            comparator = _evaluate_policy(
                policy_factory=lambda budget: threshold_policy(
                    mapper=mapper, model=model, interval=interval,
                    queue_thresholds=tuple(
                        float(value) for value in config["baseline_parameters"]["threshold"]
                    ),
                ),
                windows=val_windows, budgets=selection_budgets, model=model, mapper=mapper,
                interval=interval,
            )
        elif source["primary_comparator"] == "mpc_4":
            comparator = _evaluate_policy(
                policy_factory=lambda budget: mpc_policy(
                    checkpoint=comparison_checkpoint, mapper=mapper, model=model,
                    total_budget=budget, interval=interval, depth=4,
                ),
                windows=val_windows, budgets=selection_budgets, model=model, mapper=mapper,
                interval=interval,
            )
        else:
            raise ValueError("unknown primary comparator")
        candidate_rows: list[dict[str, Any]] = []
        for iteration, value in values.items():
            for weight in config["continuation_weights"]:
                checkpoint = CandidateCheckpoint(
                    value=value, forecaster=forecaster, gamma=float(config["gamma"]),
                    continuation_weight=float(weight),
                    forecast_strategy=str(config.get("forecast_strategy", "learned")),
                    forecast_multiplier=float(config.get("forecast_multiplier", 1.0)),
                )
                for margin in config["tie_margins"]:
                    row = _evaluate_candidate(
                        checkpoint=checkpoint, tie_margin=float(margin),
                        windows=val_windows, budgets=selection_budgets, model=model,
                        mapper=mapper, interval=interval,
                    )
                    candidate_rows.append({
                        "iteration": int(iteration),
                        "continuation_weight": float(weight),
                        "tie_margin": float(margin), **row,
                    })
        selected = select_validation_candidate(candidate_rows, comparator)
        output_dir = PROJECT_ROOT / config["paths"]["checkpoint_root"] / profile
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "models.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"repair checkpoint is append-only: {checkpoint_path}")
        metadata = {
            "schema": "dap.k8s.service_repair_selection.v1",
            "profile": profile,
            "dataset": source["dataset"],
            "domain": source["domain"],
            "training_split": "train",
            "validation_split": "validation",
            "new_formal_outcomes_accessed": False,
            "development_contract_sha256": sha256_file(contract_path),
            "development_audit_status": contract["status"],
            "config_sha256": sha256_file(config_path),
            "selection": selected,
            "tie_margin": float(selected["tie_margin"]),
            "forecast_strategy": str(config.get("forecast_strategy", "learned")),
            "forecast_multiplier": float(config.get("forecast_multiplier", 1.0)),
        }
        save_checkpoint(
            checkpoint_path, value=values[int(selected["iteration"])],
            forecaster=forecaster, gamma=float(config["gamma"]),
            continuation_weight=float(selected["continuation_weight"]),
            metadata=metadata,
        )
        write_json(output_dir / "diagnostics.json", {
            **metadata,
            "status": "completed" if selected["passed_service_guard"] else "diagnostic_no_guard_survivor",
            "training_states": training.n_states,
            "validation_states": validation.n_states,
            "validation_comparator": comparator,
            "selection_candidates": candidate_rows,
            "forecaster_history": forecast_history,
            "fvi_history": fvi_history,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        outputs.append(checkpoint_path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--profile", choices=("azure_http", "gentd_inference"))
    args = parser.parse_args()
    paths = train(args.config, args.contract, only_profile=args.profile)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
