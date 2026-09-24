from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stage2_dynamic_budget.direct_action_planning_repair.aggregation import (
    aggregate_branch_labels,
)


ACTION_KEYS = ["t", "load", "queue", "action"]
PROBABILITY_COLUMNS = ["next_load_prob_0", "next_load_prob_1", "next_load_prob_2"]


@dataclass(frozen=True)
class ReplayResult:
    labels: pd.DataFrame
    source_mix: pd.DataFrame
    state_targets: pd.DataFrame | None = None


def _sample_probability_table(samples: pd.DataFrame, source: str) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=[*ACTION_KEYS, *PROBABILITY_COLUMNS, "source"])
    counts = samples.groupby(ACTION_KEYS + ["true_next_load"], as_index=False).size()
    total = counts.groupby(ACTION_KEYS, as_index=False)["size"].sum().rename(
        columns={"size": "source_count"}
    )
    probability = counts.pivot_table(
        index=ACTION_KEYS, columns="true_next_load", values="size", fill_value=0
    ).reset_index()
    for load in (0, 1, 2):
        if load not in probability:
            probability[load] = 0
    probability = probability.merge(total, on=ACTION_KEYS, validate="one_to_one")
    for load in (0, 1, 2):
        probability[f"next_load_prob_{load}"] = probability[load] / probability.source_count
    probability["source"] = source
    return probability[[*ACTION_KEYS, *PROBABILITY_COLUMNS, "source_count", "source"]]


def _d0_probability_table(base_labels: pd.DataFrame) -> pd.DataFrame:
    columns = [*ACTION_KEYS, *PROBABILITY_COLUMNS]
    table = base_labels[base_labels.split == "train"][columns].drop_duplicates(ACTION_KEYS)
    table["source_count"] = 1.0
    table["source"] = "D0"
    return table


def balanced_replay_labels(
    base_labels: pd.DataFrame,
    historical_best_samples: list[pd.DataFrame],
    current_samples: pd.DataFrame,
    hard_samples: pd.DataFrame,
    source_weights: dict[str, float],
) -> ReplayResult:
    """Mix source-specific transition targets with exact full-batch source weights."""

    sources = {
        "D0": _d0_probability_table(base_labels),
        "historical_best": _sample_probability_table(
            pd.concat(historical_best_samples, ignore_index=True)
            if historical_best_samples
            else pd.DataFrame(),
            "historical_best",
        ),
        "current": _sample_probability_table(current_samples, "current"),
        "hard": _sample_probability_table(hard_samples, "hard"),
    }
    keys = sources["D0"][ACTION_KEYS].drop_duplicates().reset_index(drop=True)
    denominator = np.zeros(len(keys), dtype=np.float64)
    numerator = np.zeros((len(keys), 3), dtype=np.float64)
    contribution_rows: list[dict[str, object]] = []
    available_weight = sum(
        float(source_weights[name]) for name, table in sources.items() if not table.empty
    )
    state_target_frames: list[pd.DataFrame] = []
    for source_name, table in sources.items():
        weight = float(source_weights[source_name])
        renamed = table.rename(
            columns={column: f"{source_name}_{column}" for column in PROBABILITY_COLUMNS}
        )
        joined = keys.merge(renamed, on=ACTION_KEYS, how="left", validate="one_to_one")
        available = joined[f"{source_name}_{PROBABILITY_COLUMNS[0]}"].notna().to_numpy()
        denominator += weight * available
        values = joined[
            [f"{source_name}_{column}" for column in PROBABILITY_COLUMNS]
        ].fillna(0.0).to_numpy(dtype=np.float64)
        numerator += weight * values
        contribution_rows.append(
            {
                "source": source_name,
                "configured_weight": weight,
                "available_state_action_fraction": float(np.mean(available)),
                "raw_rows": int(
                    len(current_samples)
                    if source_name == "current"
                    else len(hard_samples)
                    if source_name == "hard"
                    else sum(len(frame) for frame in historical_best_samples)
                    if source_name == "historical_best"
                    else len(base_labels[base_labels.split == "train"])
                ),
            }
        )
        if not table.empty:
            target = table[[*ACTION_KEYS, *PROBABILITY_COLUMNS]].copy()
            target["replay_source"] = source_name
            target["state_loss_weight"] = (
                weight / available_weight / max(len(target), 1)
            )
            state_target_frames.append(target)
    if np.any(denominator <= 0.0):
        raise RuntimeError("D0 must provide every state-action transition target")
    mixed = numerator / denominator[:, None]
    for index, column in enumerate(PROBABILITY_COLUMNS):
        keys[column] = mixed[:, index]
    for row in contribution_rows:
        source = str(row["source"])
        table = sources[source]
        available = keys[ACTION_KEYS].merge(
            table[ACTION_KEYS].assign(available=True),
            on=ACTION_KEYS,
            how="left",
        ).available.fillna(False).to_numpy(dtype=bool)
        row["realized_full_grid_loss_mass"] = (
            float(row["configured_weight"]) / available_weight
            if len(table)
            else 0.0
        )
    train = base_labels[base_labels.split == "train"].drop(columns=PROBABILITY_COLUMNS)
    train = train.merge(keys, on=ACTION_KEYS, validate="many_to_one")
    validation = base_labels[base_labels.split != "train"].copy()
    labels = pd.concat([train, validation], ignore_index=True, sort=False)
    labels["aggregation_protocol"] = "balanced_replay"
    return ReplayResult(
        labels,
        pd.DataFrame(contribution_rows),
        pd.concat(state_target_frames, ignore_index=True),
    )


def unconstrained_replay_labels(
    base_labels: pd.DataFrame,
    base_samples: pd.DataFrame,
    samples: list[pd.DataFrame],
    protocol: str,
) -> ReplayResult:
    labels = aggregate_branch_labels(base_labels, base_samples, samples)
    labels["aggregation_protocol"] = protocol
    row_count = sum(len(frame) for frame in samples)
    mix = pd.DataFrame(
        [
            {
                "source": "D0_label_grid",
                "configured_weight": 1.0,
                "available_state_action_fraction": 1.0,
                "raw_rows": len(base_labels[base_labels.split == "train"]),
                "realized_full_grid_loss_mass": 1.0,
            },
            {
                "source": "aggregated_target_only",
                "configured_weight": np.nan,
                "available_state_action_fraction": np.nan,
                "raw_rows": row_count,
                "realized_full_grid_loss_mass": 0.0,
            },
        ]
    )
    return ReplayResult(labels, mix, None)
