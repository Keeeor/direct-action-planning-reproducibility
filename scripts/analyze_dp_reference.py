from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stage2_dynamic_budget.dynamic_shadow_price.dp_reference import (
    DiscreteBudgetMDP,
    DiscreteDPConfig,
    simulate_optimal_policy,
    solve_backward_dp,
)
from stage2_dynamic_budget.utils.artifacts import sha256_file, write_json


SCENARIOS = ("early_burst", "late_burst")
BUDGETS = (4, 8, 12)
SEEDS = range(5)


def state_rows(scenario: str):
    mdp = DiscreteBudgetMDP(
        DiscreteDPConfig(horizon=16, max_budget=12, scenario=scenario)
    )
    result = solve_backward_dp(mdp)
    rows = []
    for t in range(mdp.config.horizon):
        for load in range(mdp.n_loads):
            for queue in range(mdp.config.max_queue + 1):
                for budget in range(mdp.config.max_budget + 1):
                    rows.append(
                        {
                            "scenario": scenario,
                            "t": t,
                            "remaining_horizon": mdp.config.horizon - t,
                            "load": load,
                            "queue": queue,
                            "risk_score": load + queue,
                            "remaining_budget": budget,
                            "optimal_action": int(result.actions[t, load, queue, budget]),
                            "action_cost": int(
                                mdp.action_costs[result.actions[t, load, queue, budget]]
                            ),
                            "optimal_value": float(result.values[t, load, queue, budget]),
                            "shadow_price": (
                                float(result.shadow_prices[t, load, queue, budget])
                                if budget > 0
                                else np.nan
                            ),
                        }
                    )
    return mdp, result, pd.DataFrame(rows)


def plot_policy_heatmap(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    for row, scenario in enumerate(SCENARIOS):
        for col, t in enumerate((0, 12)):
            selected = table[
                (table.scenario == scenario) & (table.t == t) & (table.load == 2)
            ]
            matrix = selected.pivot(
                index="queue", columns="remaining_budget", values="optimal_action"
            ).sort_index(ascending=False)
            image = axes[row, col].imshow(
                matrix.to_numpy(), aspect="auto", vmin=0, vmax=3, cmap="viridis"
            )
            axes[row, col].set_title(
                f"{scenario.replace('_', ' ')}, remaining horizon={16-t}"
            )
            axes[row, col].set_xlabel("Remaining budget")
            axes[row, col].set_ylabel("Queue level")
            axes[row, col].set_xticks(range(len(matrix.columns)), matrix.columns)
            axes[row, col].set_yticks(range(len(matrix.index)), matrix.index)
    colorbar = fig.colorbar(image, ax=axes, ticks=[0, 1, 2, 3], shrink=0.85)
    colorbar.set_label("Optimal action")
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_shadow_prices(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    subset = table[(table.load == 1) & (table.queue == 0) & table.remaining_budget.isin(BUDGETS)]
    for scenario in SCENARIOS:
        data = subset[subset.scenario == scenario]
        grouped = data.groupby("remaining_horizon").shadow_price.mean()
        axes[0].plot(grouped.index, grouped.values, marker="o", label=scenario)
    axes[0].set(xlabel="Remaining horizon", ylabel="Mean shadow price", title="Time dependence")
    axes[0].legend(frameon=False)
    data = table[(table.scenario == "late_burst") & (table.t == 0)]
    for risk in (0, 4, 8):
        candidates = data[data.risk_score == risk]
        grouped = candidates.groupby("remaining_budget").shadow_price.mean()
        axes[1].plot(grouped.index, grouped.values, marker="o", label=f"risk={risk}")
    axes[1].set(xlabel="Remaining budget", ylabel="Mean shadow price", title="Budget dependence")
    axes[1].legend(frameon=False)
    data = table[(table.scenario == "late_burst") & (table.t == 8) & (table.remaining_budget == 8)]
    pivot = data.pivot(index="queue", columns="load", values="shadow_price")
    image = axes[2].imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    axes[2].set(
        xlabel="Load level", ylabel="Queue level", title="State dependence (b=8, h=8)"
    )
    axes[2].set_xticks(range(3), range(3))
    axes[2].set_yticks(range(7), range(7))
    fig.colorbar(image, ax=axes[2], label="Shadow price")
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_trajectories(trajectories: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True, constrained_layout=True)
    selected = trajectories[(trajectories.initial_budget == 8) & (trajectories.seed == 0)]
    for scenario in SCENARIOS:
        data = selected[selected.scenario == scenario]
        axes[0].step(data.t, data.cumulative_cost, where="post", label=scenario)
        axes[1].step(data.t, data.action, where="post", label=scenario)
    axes[0].axhline(8, color="black", linestyle="--", linewidth=1, label="budget")
    axes[0].set(ylabel="Cumulative cost", title="Exact optimal budget-use trajectories")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="Decision step", ylabel="Optimal action", yticks=[0, 1, 2, 3])
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary_dir = root / "results/dynamic_shadow_price/dp_reference"
    figure_dir = root / "results/dynamic_shadow_price/figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    results = {}
    mdps = {}
    trajectories = []
    for scenario in SCENARIOS:
        mdp, result, table = state_rows(scenario)
        tables.append(table)
        results[scenario] = result
        mdps[scenario] = mdp
        for budget in BUDGETS:
            for seed in SEEDS:
                for row in simulate_optimal_policy(mdp, result, budget, seed):
                    row.update(
                        {"scenario": scenario, "initial_budget": budget, "seed": seed}
                    )
                    trajectories.append(row)
    states = pd.concat(tables, ignore_index=True)
    paths = pd.DataFrame(trajectories)
    states.to_csv(summary_dir / "dp_state_policy_value.csv.gz", index=False, compression="gzip")
    paths.to_csv(summary_dir / "dp_optimal_trajectories.csv", index=False)
    shadow_summary = states.groupby(
        ["scenario", "remaining_horizon", "remaining_budget", "load", "queue"],
        as_index=False,
    ).shadow_price.mean()
    shadow_summary.to_csv(summary_dir / "dp_shadow_price_table.csv", index=False)
    high_risk_no_action = states[
        (states.risk_score >= 6)
        & (states.remaining_budget >= 3)
        & (states.optimal_action == 0)
    ]
    counterexamples = 0
    for _, group in states.groupby(["scenario", "t", "remaining_budget"]):
        values = group[["risk_score", "optimal_action"]].to_numpy()
        counterexamples += int(
            sum(x[0] > y[0] and x[1] < y[1] for x in values for y in values)
        )
    budget_change = float(
        np.mean(
            states.sort_values(["scenario", "t", "load", "queue", "remaining_budget"])
            .groupby(["scenario", "t", "load", "queue"])
            .optimal_action.diff()
            .fillna(0)
            != 0
        )
    )
    horizon_change = float(
        np.mean(
            states.sort_values(["scenario", "load", "queue", "remaining_budget", "t"])
            .groupby(["scenario", "load", "queue", "remaining_budget"])
            .optimal_action.diff()
            .fillna(0)
            != 0
        )
    )
    finite = states.dropna(subset=["shadow_price"])
    questions = {
        "high_risk_always_more_resource": False,
        "risk_action_counterexample_pairs": counterexamples,
        "high_risk_no_action_states_with_budget_at_least_3": int(len(high_risk_no_action)),
        "optimal_action_changes_across_adjacent_budgets_fraction": budget_change,
        "optimal_action_changes_across_adjacent_horizons_fraction": horizon_change,
        "shadow_price_min": float(finite.shadow_price.min()),
        "shadow_price_max": float(finite.shadow_price.max()),
        "shadow_price_risk_spearman": float(
            spearmanr(finite.risk_score, finite.shadow_price).statistic
        ),
        "b4_cdba_deviation": "pending learned-policy DP benchmark",
        "max_bellman_residual": max(
            result.max_bellman_residual for result in results.values()
        ),
    }
    write_json(summary_dir / "dp_questions.json", questions)
    plot_policy_heatmap(states, figure_dir / "dp_optimal_action_heatmap")
    plot_shadow_prices(states, figure_dir / "dp_shadow_price_patterns")
    plot_trajectories(paths, figure_dir / "dp_optimal_trajectories")
    artifacts = sorted(summary_dir.glob("dp_*")) + sorted(figure_dir.glob("dp_*"))
    write_json(
        summary_dir / "dp_reference_integrity.json",
        {
            "schema": "dynamic_shadow_price.dp_reference.v1",
            "state_rows": len(states),
            "trajectory_rows": len(paths),
            "scenarios": list(SCENARIOS),
            "budgets": list(BUDGETS),
            "seeds": list(SEEDS),
            "artifacts": {path.name: sha256_file(path) for path in artifacts},
        },
    )


if __name__ == "__main__":
    main()
