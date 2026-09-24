from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage2_dynamic_budget.action_conditioned_budget_advantage.dp import (
    ACBADPConfig,
    ActionConditionedBudgetMDP,
    solve_action_dp,
)
from stage2_dynamic_budget.action_conditioned_budget_advantage.evaluation import (
    high_risk_threshold,
)
from stage2_dynamic_budget.direct_action_planning.learning import EmpiricalActionModel
from stage2_dynamic_budget.direct_action_planning.planning import (
    BudgetValueTable,
    one_step_plan,
)


@dataclass(frozen=True)
class LegacyDiagnosticTables:
    actions: pd.DataFrame
    states: pd.DataFrame
    pairs: pd.DataFrame
    first_errors: pd.DataFrame
    budget_divergence: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_model(path: Path) -> EmpiricalActionModel:
    with np.load(path) as data:
        return EmpiricalActionModel(
            next_load_probabilities=data["next_load_probabilities"],
            next_queue_probabilities=data["next_queue_probabilities"],
            rewards=data["rewards"],
            costs=data["costs"],
            samples_per_state_action=int(data["samples_per_state_action"]),
            smoothing=float(data["smoothing"]),
        )


def _load_value(path: Path) -> BudgetValueTable:
    with np.load(path) as data:
        source = str(data["source"].item())
        return BudgetValueTable(values=data["values"], source=source)


def _risk_band(risk: float, threshold: float) -> str:
    return "high" if risk >= threshold else "lower"


def _horizon_band(t: int, horizon: int) -> str:
    fraction = t / max(horizon - 1, 1)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _budget_band(budget: int, maximum: int) -> str:
    fraction = budget / max(maximum, 1)
    if fraction <= 1.0 / 3.0:
        return "tight"
    if fraction <= 2.0 / 3.0:
        return "medium"
    return "ample"


def _gap_band(gap: float) -> str:
    if gap <= 1.0e-9:
        return "tie"
    if gap <= 0.05:
        return "small_(0,0.05]"
    if gap <= 0.25:
        return "medium_(0.05,0.25]"
    return "large_>0.25"


def diagnose_state_grid(
    mdp: ActionConditionedBudgetMDP,
    optimum,
    value: BudgetValueTable,
    model: EmpiricalActionModel,
    scenario: str,
    seed: int,
    tie_tolerance: float = 1.0e-9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    risk_cutoff = high_risk_threshold(mdp)
    cfg = mdp.config
    for t, load, queue, budget in np.ndindex(optimum.actions.shape):
        lv = one_step_plan(mdp, value, t, load, queue, budget)
        lm = one_step_plan(mdp, value, t, load, queue, budget, learned_model=model)
        feasible = np.flatnonzero(np.isfinite(lv.q_values))
        lv_order = feasible[np.argsort(-lv.q_values[feasible], kind="stable")]
        lv_action = int(lv.action)
        lm_action = int(lm.action)
        optimal_action = int(optimum.actions[t, load, queue, budget])
        lv_gap = (
            float(lv.q_values[lv_order[0]] - lv.q_values[lv_order[1]])
            if len(lv_order) > 1
            else np.nan
        )
        exact_order = feasible[
            np.argsort(-optimum.q_values[t, load, queue, budget, feasible], kind="stable")
        ]
        exact_gap = (
            float(
                optimum.q_values[t, load, queue, budget, exact_order[0]]
                - optimum.q_values[t, load, queue, budget, exact_order[1]]
            )
            if len(exact_order) > 1
            else np.nan
        )
        risk = float(load + queue)
        q_star_lv = float(optimum.q_values[t, load, queue, budget, lv_action])
        q_star_lm = float(optimum.q_values[t, load, queue, budget, lm_action])
        q_star_opt = float(optimum.values[t, load, queue, budget])
        state_id = f"{scenario}|s{seed}|t{t}|l{load}|q{queue}|b{budget}"
        state_rows.append(
            {
                "state_id": state_id,
                "scenario": scenario,
                "seed": seed,
                "t": t,
                "load": load,
                "queue": queue,
                "remaining_budget": budget,
                "remaining_horizon": cfg.horizon - t,
                "risk_score": risk,
                "risk_band": _risk_band(risk, risk_cutoff),
                "budget_band": _budget_band(budget, cfg.max_budget),
                "horizon_band": _horizon_band(t, cfg.horizon),
                "learned_value_action": lv_action,
                "learned_model_action": lm_action,
                "optimal_action": optimal_action,
                "ranking_flip": bool(lv_action != lm_action),
                "lv_top1_top2_gap": lv_gap,
                "lv_gap_band": _gap_band(lv_gap) if np.isfinite(lv_gap) else "single_action",
                "q_star_top1_top2_gap": exact_gap,
                "q_star_regret_lv": q_star_opt - q_star_lv,
                "q_star_regret_lm": q_star_opt - q_star_lm,
                "flip_q_star_regret_delta": q_star_lv - q_star_lm,
                "flip_q_star_regret_positive": max(q_star_lv - q_star_lm, 0.0),
                "high_risk_optimal_low_cost": bool(risk >= risk_cutoff and optimal_action == 0),
            }
        )

        true_load = mdp.load_probabilities(t, load)
        true_next_queue_by_action: dict[int, int] = {}
        for action in feasible:
            action = int(action)
            prediction = model.predict(t, load, queue, action)
            next_queue, reward, metrics = mdp.outcome(queue, load, action)
            true_next_queue_by_action[action] = next_queue
            predicted_load = prediction.next_state_probabilities.sum(axis=1)
            predicted_queue = prediction.next_state_probabilities.sum(axis=0)
            load_tv = 0.5 * float(np.abs(predicted_load - true_load).sum())
            load_mean_error = abs(
                float(np.dot(predicted_load - true_load, np.arange(mdp.n_loads)))
            )
            queue_mean = float(np.dot(predicted_queue, np.arange(cfg.max_queue + 1)))
            q_star = float(optimum.q_values[t, load, queue, budget, action])
            action_rows.append(
                {
                    "state_id": state_id,
                    "scenario": scenario,
                    "seed": seed,
                    "t": t,
                    "load": load,
                    "queue": queue,
                    "remaining_budget": budget,
                    "remaining_horizon": cfg.horizon - t,
                    "risk_score": risk,
                    "action": action,
                    "action_cost": int(mdp.action_costs[action]),
                    "q_lv": float(lv.q_values[action]),
                    "q_lm": float(lm.q_values[action]),
                    "q_star": q_star,
                    "q_model_error": float(lm.q_values[action] - lv.q_values[action]),
                    "learned_value_action": lv_action,
                    "learned_model_action": lm_action,
                    "optimal_action": optimal_action,
                    "ranking_flip": bool(lv_action != lm_action),
                    "lv_top1_top2_gap": lv_gap,
                    "true_next_load_mean": float(np.dot(true_load, np.arange(mdp.n_loads))),
                    "predicted_next_load_mean": float(
                        np.dot(predicted_load, np.arange(mdp.n_loads))
                    ),
                    "next_load_mean_abs_error": load_mean_error,
                    "next_load_probability_tv": load_tv,
                    "true_next_queue": next_queue,
                    "predicted_next_queue_mean": queue_mean,
                    "next_queue_mean_abs_error": abs(queue_mean - next_queue),
                    "next_queue_top1_error": float(int(np.argmax(predicted_queue)) != next_queue),
                    "reward_abs_error": abs(float(prediction.reward) - float(reward)),
                    "cost_abs_error": abs(float(prediction.cost) - float(metrics["cost"])),
                }
            )

        for left_index, left_action in enumerate(feasible[:-1]):
            for right_action in feasible[left_index + 1 :]:
                left_action, right_action = int(left_action), int(right_action)
                lv_delta = float(lv.q_values[left_action] - lv.q_values[right_action])
                lm_delta = float(lm.q_values[left_action] - lm.q_values[right_action])
                comparable = abs(lv_delta) > tie_tolerance
                correct = (lm_delta * lv_delta) > 0.0 if comparable else abs(lm_delta) <= tie_tolerance
                pair_rows.append(
                    {
                        "state_id": state_id,
                        "scenario": scenario,
                        "seed": seed,
                        "t": t,
                        "load": load,
                        "queue": queue,
                        "remaining_budget": budget,
                        "remaining_horizon": cfg.horizon - t,
                        "risk_score": risk,
                        "budget_band": _budget_band(budget, cfg.max_budget),
                        "horizon_band": _horizon_band(t, cfg.horizon),
                        "risk_band": _risk_band(risk, risk_cutoff),
                        "action_i": left_action,
                        "action_j": right_action,
                        "q_lv_delta_i_minus_j": lv_delta,
                        "q_lm_delta_i_minus_j": lm_delta,
                        "comparable": comparable,
                        "pair_ranking_correct": bool(correct),
                        "pair_ranking_flip": bool(not correct),
                    }
                )
    return pd.DataFrame(action_rows), pd.DataFrame(state_rows), pd.DataFrame(pair_rows)


def closed_loop_first_errors(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["scenario", "seed", "budget", "episode", "eval_seed", "t"]
    columns = keys + [
        "load",
        "queue",
        "budget_before",
        "remaining_budget",
        "action",
        "Q_star_regret",
    ]
    lv = steps[steps.method == "learned_value_branch"][columns]
    lm = steps[steps.method == "learned_model_branch"][columns]
    paired = lv.merge(lm, on=keys, suffixes=("_lv", "_lm"), validate="one_to_one")
    paired["action_differs"] = paired.action_lv != paired.action_lm
    paired["budget_difference"] = paired.remaining_budget_lm - paired.remaining_budget_lv
    paired["absolute_budget_difference"] = paired.budget_difference.abs()
    episode_keys = keys[:-1]
    first_rows: list[dict[str, object]] = []
    divergence_rows: list[pd.DataFrame] = []
    for episode_key, frame in paired.groupby(episode_keys, sort=False):
        frame = frame.sort_values("t").copy()
        differing = frame[frame.action_differs]
        if differing.empty:
            first_t = np.nan
            post = frame.iloc[0:0].copy()
            first = frame.iloc[0]
        else:
            first = differing.iloc[0]
            first_t = int(first.t)
            post = frame[frame.t >= first_t].copy()
            post["first_error_t"] = first_t
            divergence_rows.append(post)
        payload = dict(zip(episode_keys, episode_key))
        payload.update(
            {
                "first_error_t": first_t,
                "has_action_divergence": bool(not differing.empty),
                "first_load_lv": int(first.load_lv),
                "first_queue_lv": int(first.queue_lv),
                "first_budget_lv": int(first.budget_before_lv),
                "first_action_lv": int(first.action_lv),
                "first_action_lm": int(first.action_lm),
                "first_q_star_regret_lv": float(first.Q_star_regret_lv),
                "first_q_star_regret_lm": float(first.Q_star_regret_lm),
                "post_error_budget_mae": (
                    float(post.absolute_budget_difference.mean()) if len(post) else 0.0
                ),
                "final_budget_difference": (
                    float(post.budget_difference.iloc[-1]) if len(post) else 0.0
                ),
            }
        )
        first_rows.append(payload)
    divergence = pd.concat(divergence_rows, ignore_index=True) if divergence_rows else paired.iloc[0:0]
    return pd.DataFrame(first_rows), divergence


def _rate(frame: pd.DataFrame, group: list[str], column: str) -> pd.DataFrame:
    return (
        frame.groupby(group, dropna=False)[column]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": f"{column}_rate"})
    )


def summarize_diagnostic(tables: LegacyDiagnosticTables) -> dict[str, pd.DataFrame]:
    comparable_pairs = tables.pairs[tables.pairs.comparable]
    action_error = tables.actions.groupby("state_id", as_index=False).agg(
        next_load_tv_mean=("next_load_probability_tv", "mean"),
        next_load_tv_max=("next_load_probability_tv", "max"),
        next_load_mean_abs_error=("next_load_mean_abs_error", "mean"),
        next_queue_mean_abs_error=("next_queue_mean_abs_error", "mean"),
        reward_abs_error=("reward_abs_error", "mean"),
        cost_abs_error=("cost_abs_error", "mean"),
        q_model_error_abs=("q_model_error", lambda x: float(np.mean(np.abs(x)))),
    )
    relation = tables.states[["state_id", "ranking_flip"]].merge(action_error, on="state_id")
    relation_summary = relation.groupby("ranking_flip", as_index=False).mean(numeric_only=True)
    correlations = []
    flip = relation.ranking_flip.astype(float)
    for column in action_error.columns[1:]:
        correlations.append(
            {
                "error_component": column,
                "pearson_with_ranking_flip": float(relation[column].corr(flip)),
                "mean_error": float(relation[column].mean()),
            }
        )
    return {
        "pairwise_flip_rates": _rate(
            comparable_pairs, ["action_i", "action_j"], "pair_ranking_flip"
        ),
        "errors_by_budget": _rate(tables.states, ["remaining_budget"], "ranking_flip"),
        "errors_by_scenario": _rate(tables.states, ["scenario"], "ranking_flip"),
        "errors_by_horizon": _rate(tables.states, ["horizon_band"], "ranking_flip"),
        "errors_by_risk": _rate(tables.states, ["risk_band"], "ranking_flip"),
        "errors_by_gap": _rate(tables.states, ["lv_gap_band"], "ranking_flip"),
        "prediction_error_by_flip": relation_summary,
        "prediction_error_flip_correlations": pd.DataFrame(correlations),
    }


def run_legacy_diagnostic(
    project_root: str | Path,
    run_id: str = "legacy_error_diagnostic_v1",
    tie_tolerance: float = 1.0e-9,
) -> Path:
    root = Path(project_root).resolve()
    frozen = root / "results/direct_action_planning/minimal_v1"
    output = root / "results/direct_action_planning_repair" / run_id
    if output.exists():
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed":
            return output
        raise RuntimeError(f"diagnostic output already exists and is incomplete: {output}")
    output.mkdir(parents=True)
    config = json.loads((frozen / "config.json").read_text(encoding="utf-8"))
    action_frames, state_frames, pair_frames = [], [], []
    for scenario in config["scenarios"]:
        env = config["environment"]
        mdp = ActionConditionedBudgetMDP(
            ACBADPConfig(
                horizon=int(env["horizon"]),
                max_budget=int(env["max_budget"]),
                max_queue=int(env["max_queue"]),
                scenario=scenario,
                gamma=float(env["gamma"]),
                action_costs=tuple(int(x) for x in env["action_costs"]),
                action_capacity=tuple(int(x) for x in env["action_capacity"]),
            )
        )
        optimum = solve_action_dp(mdp)
        for seed in config["seeds"]:
            model = _load_model(frozen / "models" / f"model__{scenario}__s{seed}.npz")
            value = _load_value(frozen / "models" / f"value__{scenario}__s{seed}.npz")
            actions, states, pairs = diagnose_state_grid(
                mdp, optimum, value, model, scenario, int(seed), tie_tolerance
            )
            action_frames.append(actions)
            state_frames.append(states)
            pair_frames.append(pairs)
    steps = pd.read_csv(frozen / "steps.csv.gz")
    first_errors, divergence = closed_loop_first_errors(steps)
    tables = LegacyDiagnosticTables(
        actions=pd.concat(action_frames, ignore_index=True),
        states=pd.concat(state_frames, ignore_index=True),
        pairs=pd.concat(pair_frames, ignore_index=True),
        first_errors=first_errors,
        budget_divergence=divergence,
    )
    tables.actions.to_csv(output / "action_q_diagnostics.csv.gz", index=False, compression="gzip")
    tables.states.to_csv(output / "state_flip_diagnostics.csv.gz", index=False, compression="gzip")
    tables.pairs.to_csv(output / "pairwise_ranking.csv.gz", index=False, compression="gzip")
    tables.first_errors.to_csv(output / "closed_loop_first_errors.csv", index=False)
    tables.budget_divergence.to_csv(
        output / "closed_loop_budget_divergence.csv.gz", index=False, compression="gzip"
    )
    summaries = summarize_diagnostic(tables)
    for name, frame in summaries.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    inputs = [
        frozen / "config.json",
        frozen / "manifest.json",
        frozen / "steps.csv.gz",
        *sorted((frozen / "models").glob("model__*.npz")),
        *sorted((frozen / "models").glob("value__*.npz")),
    ]
    manifest = {
        "schema": "direct_action_planning_repair.legacy_diagnostic.v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_run": "results/direct_action_planning/minimal_v1",
        "read_only_source": True,
        "tie_tolerance": tie_tolerance,
        "rows": {
            "actions": len(tables.actions),
            "states": len(tables.states),
            "pairs": len(tables.pairs),
            "closed_loop_episodes": len(tables.first_errors),
            "post_divergence_steps": len(tables.budget_divergence),
        },
        "input_sha256": {str(path.relative_to(root)): _sha256(path) for path in inputs},
        "output_sha256": {},
    }
    for path in sorted(output.glob("*.csv*")):
        manifest["output_sha256"][path.name] = _sha256(path)
    _write_json(output / "manifest.json", manifest)
    return output
