from __future__ import annotations

import numpy as np

from .synthetic_queue_env import DynamicBudgetSchedulingEnv, SyntheticQueueConfig


class TraceDrivenQueueEnv(DynamicBudgetSchedulingEnv):
    """Queue simulator whose arrivals are supplied by a chronological trace."""

    def __init__(self, trace: np.ndarray, config: SyntheticQueueConfig):
        trace = np.asarray(trace, dtype=np.float64)
        if trace.ndim != 1 or len(trace) != config.horizon:
            raise ValueError("trace must be one-dimensional and match config.horizon")
        if np.any(trace < 0) or not np.all(np.isfinite(trace)):
            raise ValueError("trace values must be finite and non-negative")
        self.trace = trace.copy()
        super().__init__(config)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("TraceDrivenQueueEnv owns its fixed trace")
        return super().reset(seed=seed, options={"arrival_trace": self.trace})
