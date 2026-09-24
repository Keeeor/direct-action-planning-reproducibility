from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalHistory:
    arrivals: tuple[float, ...]
    queues: tuple[float, ...]
    actions: tuple[float, ...] = ()
    capacities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.arrivals) != len(self.queues):
            raise ValueError("arrival and queue histories must align")
        if len(self.actions) > max(len(self.arrivals) - 1, 0):
            raise ValueError("actions cannot extend beyond observed transitions")
        if len(self.actions) != len(self.capacities):
            raise ValueError("action and capacity histories must align")

    def with_future_for_test(
        self, arrivals: tuple[float, ...], queues: tuple[float, ...]
    ) -> "CausalHistory":
        """Append a suffix for leakage tests; production code should not call this."""

        if len(arrivals) != len(queues):
            raise ValueError("future suffixes must align")
        return CausalHistory(
            arrivals=self.arrivals + arrivals,
            queues=self.queues + queues,
            actions=self.actions,
            capacities=self.capacities,
        )

    def advance(
        self,
        action: int | float,
        capacity: int | float,
        next_arrival: int | float,
        next_queue: int | float,
    ) -> "CausalHistory":
        return CausalHistory(
            arrivals=(*self.arrivals, float(next_arrival)),
            queues=(*self.queues, float(next_queue)),
            actions=(*self.actions, float(action)),
            capacities=(*self.capacities, float(capacity)),
        )


def _window(values: tuple[float, ...], window: int, cutoff: int) -> np.ndarray:
    if window <= 0 or cutoff <= 0:
        raise ValueError("window and cutoff must be positive")
    return np.asarray(values[:cutoff][-window:], dtype=np.float64)


def _trend(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.sum(centered * centered))
    return float(np.sum(centered * (values - values.mean())) / max(denominator, 1.0e-12))


def _autocorrelation(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values[:-1]) <= 1.0e-12 or np.std(values[1:]) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _change_point_distance(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    differences = np.abs(np.diff(values))
    threshold = max(1.0, float(np.mean(differences) + np.std(differences)))
    positions = np.flatnonzero(differences >= threshold)
    return float(len(values) - 1 - positions[-1]) if len(positions) else float(len(values) - 1)


def history_features(history: CausalHistory, window: int, cutoff: int | None = None) -> np.ndarray:
    cutoff = len(history.arrivals) if cutoff is None else int(cutoff)
    if cutoff > len(history.arrivals):
        raise ValueError("cutoff exceeds available observations")
    arrivals = _window(history.arrivals, window, cutoff)
    queues = _window(history.queues, window, cutoff)
    action_cutoff = min(max(cutoff - 1, 0), len(history.actions))
    actions = np.asarray(history.actions[:action_cutoff][-window:], dtype=np.float64)
    capacities = np.asarray(history.capacities[:action_cutoff][-window:], dtype=np.float64)
    first = np.diff(arrivals)
    second = np.diff(first)
    queue_delta = np.diff(queues)

    def moments(values: np.ndarray) -> tuple[float, float, float, float]:
        if not len(values):
            return 0.0, 0.0, 0.0, 0.0
        return float(values[-1]), float(values.mean()), float(values.var()), float(values.max())

    arrival_last, arrival_mean, arrival_var, arrival_peak = moments(arrivals)
    queue_last, queue_mean, queue_var, queue_peak = moments(queues)
    action_last, action_mean, action_var, _ = moments(actions)
    capacity_last, capacity_mean, capacity_var, _ = moments(capacities)
    features = np.asarray(
        [
            arrival_last,
            arrival_mean,
            arrival_var,
            arrival_peak,
            float(first[-1]) if len(first) else 0.0,
            float(first.mean()) if len(first) else 0.0,
            float(second[-1]) if len(second) else 0.0,
            float(second.mean()) if len(second) else 0.0,
            _trend(arrivals),
            _autocorrelation(arrivals),
            _change_point_distance(arrivals) / max(window - 1, 1),
            queue_last,
            queue_mean,
            queue_var,
            queue_peak,
            float(queue_delta[-1]) if len(queue_delta) else 0.0,
            _trend(queues),
            action_last,
            action_mean,
            action_var,
            capacity_last,
            capacity_mean,
            capacity_var,
            min(len(arrivals), window) / float(window),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise ValueError("history features contain non-finite values")
    return features


def history_sequence(
    history: CausalHistory, window: int, cutoff: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    cutoff = len(history.arrivals) if cutoff is None else int(cutoff)
    arrivals = _window(history.arrivals, window, cutoff)
    queues = _window(history.queues, window, cutoff)
    action_cutoff = min(max(cutoff - 1, 0), len(history.actions))
    actions = np.asarray(history.actions[:action_cutoff][-window:], dtype=np.float64)
    capacities = np.asarray(history.capacities[:action_cutoff][-window:], dtype=np.float64)
    sequence = np.zeros((window, 4), dtype=np.float64)
    mask = np.zeros(window, dtype=bool)
    valid = len(arrivals)
    sequence[-valid:, 0] = arrivals
    sequence[-valid:, 1] = queues
    mask[-valid:] = True
    action_valid = min(len(actions), max(valid - 1, 0))
    if action_valid:
        sequence[-action_valid - 1 : -1, 2] = actions[-action_valid:]
        sequence[-action_valid - 1 : -1, 3] = capacities[-action_valid:]
    return sequence, mask


def _encode(values: tuple[float, ...]) -> str:
    return "|".join(f"{value:.12g}" for value in values)


def _decode(value: object) -> tuple[float, ...]:
    text = str(value)
    if not text or text.lower() == "nan":
        return ()
    return tuple(float(item) for item in text.split("|") if item != "")


def serialize_history(history: CausalHistory) -> dict[str, str]:
    return {
        "arrivals_history": _encode(history.arrivals),
        "queues_history": _encode(history.queues),
        "actions_history": _encode(history.actions),
        "capacities_history": _encode(history.capacities),
    }


def deserialize_history(row: object) -> CausalHistory:
    def value(name: str) -> object:
        if isinstance(row, dict):
            return row[name]
        return getattr(row, name)

    return CausalHistory(
        arrivals=_decode(value("arrivals_history")),
        queues=_decode(value("queues_history")),
        actions=_decode(value("actions_history")),
        capacities=_decode(value("capacities_history")),
    )


def leakage_audit(frame) -> dict[str, object]:
    required = {
        "t",
        "arrivals_history",
        "queues_history",
        "actions_history",
        "capacities_history",
    }
    missing = required - set(frame.columns)
    violations: list[dict[str, object]] = []
    if missing:
        return {
            "schema": "direct_action_planning_context_value.leakage_audit.v1",
            "passed": False,
            "rows": int(len(frame)),
            "violations": [{"kind": "missing_columns", "columns": sorted(missing)}],
        }
    prohibited = [
        column
        for column in frame.columns
        if "future" in column.lower()
        or column.lower() in {"scenario_id", "true_load_phase", "oracle_phase"}
    ]
    if prohibited:
        violations.append({"kind": "prohibited_columns", "columns": prohibited})
    for index, row in enumerate(frame.itertuples(index=False)):
        try:
            history = deserialize_history(row)
        except (ValueError, KeyError) as error:
            violations.append({"row": index, "kind": "parse_error", "error": str(error)})
            continue
        t = int(row.t)
        lengths = {
            "arrivals": len(history.arrivals),
            "queues": len(history.queues),
            "actions": len(history.actions),
            "capacities": len(history.capacities),
        }
        expected = {"arrivals": t + 1, "queues": t + 1, "actions": t, "capacities": t}
        if lengths != expected:
            violations.append(
                {"row": index, "kind": "causal_length", "observed": lengths, "expected": expected}
            )
        values = (*history.arrivals, *history.queues, *history.actions, *history.capacities)
        if values and not np.isfinite(np.asarray(values, dtype=float)).all():
            violations.append({"row": index, "kind": "non_finite"})
    return {
        "schema": "direct_action_planning_context_value.leakage_audit.v1",
        "passed": not violations,
        "rows": int(len(frame)),
        "violations": violations[:100],
        "violation_count": len(violations),
    }
