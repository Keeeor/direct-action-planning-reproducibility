from __future__ import annotations

from dataclasses import asdict, dataclass
import time


@dataclass(frozen=True)
class BudgetSample:
    monotonic_seconds: float
    wall_time: str
    ready_replicas: int
    desired_replicas: int
    interval_seconds: float
    ready_increment: float
    requested_increment: float
    cumulative_ready_cost: float
    cumulative_requested_cost: float


class BudgetTracker:
    """Integrate actual and requested additional-replica lifetime."""

    def __init__(self, total_budget: float, *, base_replicas: int = 1):
        if total_budget < 0:
            raise ValueError("total_budget must be non-negative")
        if base_replicas < 1:
            raise ValueError("base_replicas must be positive")
        self.total_budget = float(total_budget)
        self.base_replicas = int(base_replicas)
        self.ready_cost = 0.0
        self.requested_cost = 0.0
        self._last_monotonic: float | None = None
        self._last_ready = self.base_replicas
        self._last_desired = self.base_replicas
        self.samples: list[BudgetSample] = []

    @property
    def remaining(self) -> float:
        return max(self.total_budget - self.ready_cost, 0.0)

    def update(
        self,
        *,
        ready_replicas: int,
        desired_replicas: int,
        monotonic_seconds: float | None = None,
        wall_time: str = "",
    ) -> BudgetSample:
        now = time.monotonic() if monotonic_seconds is None else float(monotonic_seconds)
        if ready_replicas < 0 or desired_replicas < 0:
            raise ValueError("replica counts must be non-negative")
        interval = 0.0 if self._last_monotonic is None else now - self._last_monotonic
        if interval < 0:
            raise ValueError("budget timestamps must be monotonic")
        ready_increment = max(self._last_ready - self.base_replicas, 0) * interval
        requested_increment = max(self._last_desired - self.base_replicas, 0) * interval
        self.ready_cost += ready_increment
        self.requested_cost += requested_increment
        sample = BudgetSample(
            monotonic_seconds=now,
            wall_time=wall_time,
            ready_replicas=int(ready_replicas),
            desired_replicas=int(desired_replicas),
            interval_seconds=interval,
            ready_increment=ready_increment,
            requested_increment=requested_increment,
            cumulative_ready_cost=self.ready_cost,
            cumulative_requested_cost=self.requested_cost,
        )
        self.samples.append(sample)
        self._last_monotonic = now
        self._last_ready = int(ready_replicas)
        self._last_desired = int(desired_replicas)
        return sample

    def can_target(
        self,
        target_replicas: int,
        *,
        current_ready: int,
        control_interval: float,
        scale_down_guard_seconds: float,
    ) -> bool:
        if target_replicas < self.base_replicas:
            return False
        if target_replicas == self.base_replicas:
            # Returning to the base footprint never creates a new resource
            # commitment. Existing Ready Pods may still incur measured
            # termination latency, but masking this action would remove the
            # controller's only safe recovery action near budget exhaustion.
            return True
        current_extra = max(current_ready - self.base_replicas, 0)
        target_extra = max(target_replicas - self.base_replicas, 0)
        # An action remains in force for the whole next interval. At that
        # boundary the controller may request base replicas, but actual Ready
        # Pods can still accrue cost during measured termination latency. The
        # reserve therefore covers the larger present/target footprint rather
        # than treating scale-down as instantaneous.
        projected = (
            self.ready_cost
            + target_extra * max(control_interval, 0.0)
            + max(current_extra, target_extra) * max(scale_down_guard_seconds, 0.0)
        )
        return projected <= self.total_budget + 1e-9

    def as_records(self) -> list[dict]:
        return [asdict(sample) for sample in self.samples]
