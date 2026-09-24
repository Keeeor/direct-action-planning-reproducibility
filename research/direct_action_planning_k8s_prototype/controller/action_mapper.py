from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTION_ORDER = ("no_op", "scale_small", "scale_medium", "scale_large")


@dataclass(frozen=True)
class ActionMapper:
    targets: Mapping[str, int]

    def __post_init__(self) -> None:
        if tuple(self.targets) != ACTION_ORDER:
            raise ValueError(f"actions must be ordered as {ACTION_ORDER}")
        values = tuple(int(self.targets[name]) for name in ACTION_ORDER)
        if values[0] < 1 or any(value <= 0 for value in values):
            raise ValueError("replica targets must be positive")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("replica targets must be strictly increasing")

    def replicas(self, action: str | int) -> int:
        name = ACTION_ORDER[action] if isinstance(action, int) else action
        if name not in self.targets:
            raise KeyError(name)
        return int(self.targets[name])

    def action_index(self, replicas: int) -> int:
        values = [self.replicas(name) for name in ACTION_ORDER]
        return values.index(int(replicas))

