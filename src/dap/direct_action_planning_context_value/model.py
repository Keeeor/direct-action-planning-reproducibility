from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dap.direct_action_planning.planning import BudgetValueTable

from .history import CausalHistory, history_features, history_sequence


class ContextResidualValueModel(nn.Module):
    """Residual value model with no scenario or privileged phase input."""

    def __init__(
        self,
        mode: str,
        horizon: int,
        n_loads: int,
        max_queue: int,
        max_budget: int,
        feature_dim: int,
        hidden_dim: int,
        gru_hidden_dim: int,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ):
        super().__init__()
        if mode not in {"current", "feature", "gru"}:
            raise ValueError("mode must be current, feature, or gru")
        self.mode = mode
        self.horizon = int(horizon)
        self.n_loads = int(n_loads)
        self.max_queue = int(max_queue)
        self.max_budget = int(max_budget)
        self.feature_dim = int(feature_dim)
        mean = np.zeros(feature_dim) if feature_mean is None else np.asarray(feature_mean)
        std = np.ones(feature_dim) if feature_std is None else np.asarray(feature_std)
        if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
            raise ValueError("feature normalization shape mismatch")
        self.register_buffer("feature_mean", torch.as_tensor(mean, dtype=torch.float64))
        self.register_buffer(
            "feature_std", torch.as_tensor(np.maximum(std, 1.0e-6), dtype=torch.float64)
        )
        if mode == "gru":
            self.gru = nn.GRU(input_size=4, hidden_size=gru_hidden_dim, batch_first=True)
            head_input = 4 + gru_hidden_dim
        else:
            self.gru = None
            head_input = 4 + (feature_dim if mode == "feature" else 0)
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.double()

    def _state(
        self,
        remaining_horizon: torch.Tensor,
        load: torch.Tensor,
        queue: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                remaining_horizon.to(torch.float64) / max(self.horizon, 1),
                load.to(torch.float64) / max(self.n_loads - 1, 1),
                queue.to(torch.float64) / max(self.max_queue, 1),
                budget.to(torch.float64) / max(self.max_budget, 1),
            ],
            dim=1,
        )

    def forward(
        self,
        remaining_horizon: torch.Tensor,
        load: torch.Tensor,
        queue: torch.Tensor,
        budget: torch.Tensor,
        base_value: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        sequence: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self._state(remaining_horizon, load, queue, budget)
        if self.mode == "feature":
            if features is None or features.shape[-1] != self.feature_dim:
                raise ValueError("feature mode requires matching history features")
            normalized = (features.to(torch.float64) - self.feature_mean) / self.feature_std
            encoded = torch.cat([state, normalized], dim=1)
        elif self.mode == "gru":
            if sequence is None or mask is None:
                raise ValueError("gru mode requires a causal sequence and mask")
            normalized_sequence = sequence.to(torch.float64).clone()
            scales = torch.as_tensor(
                [4.0, max(self.max_queue, 1), 3.0, 4.0],
                dtype=torch.float64,
                device=normalized_sequence.device,
            )
            normalized_sequence = normalized_sequence / scales
            normalized_sequence = normalized_sequence * mask.unsqueeze(-1).to(torch.float64)
            output, _ = self.gru(normalized_sequence)  # type: ignore[arg-type]
            encoded = torch.cat([state, output[:, -1]], dim=1)
        else:
            encoded = state
        residual = self.head(encoded).squeeze(1)
        residual = residual * (remaining_horizon > 0).to(torch.float64)
        return base_value.to(torch.float64) + residual


class ContextValuePredictor:
    def __init__(
        self,
        model: ContextResidualValueModel,
        base_value: BudgetValueTable,
        window: int,
    ):
        self.model = model.eval()
        self.base_value = base_value
        self.window = int(window)

    @torch.no_grad()
    def predict_many(
        self,
        loads: np.ndarray,
        queues: np.ndarray,
        budgets: np.ndarray,
        remaining_horizons: np.ndarray,
        histories: list[CausalHistory],
    ) -> np.ndarray:
        loads = np.asarray(loads, dtype=np.int64)
        queues = np.asarray(queues, dtype=np.int64)
        budgets = np.asarray(budgets, dtype=np.int64)
        remaining_horizons = np.asarray(remaining_horizons, dtype=np.int64)
        size = len(loads)
        if not (
            len(queues) == len(budgets) == len(remaining_horizons) == len(histories) == size
        ):
            raise ValueError("batched context inputs must align")
        device = next(self.model.parameters()).device
        kwargs: dict[str, torch.Tensor] = {}
        if self.model.mode == "feature":
            kwargs["features"] = torch.as_tensor(
                np.stack([history_features(history, self.window) for history in histories]),
                dtype=torch.float64,
                device=device,
            )
        elif self.model.mode == "gru":
            pairs = [history_sequence(history, self.window) for history in histories]
            kwargs["sequence"] = torch.as_tensor(
                np.stack([pair[0] for pair in pairs]), dtype=torch.float64, device=device
            )
            kwargs["mask"] = torch.as_tensor(
                np.stack([pair[1] for pair in pairs]), dtype=torch.bool, device=device
            )
        base = np.asarray(
            [
                self.base_value.predict(int(load), int(queue), int(budget), int(horizon))
                for load, queue, budget, horizon in zip(
                    loads, queues, budgets, remaining_horizons
                )
            ],
            dtype=np.float64,
        )
        value = self.model(
            torch.as_tensor(remaining_horizons, device=device),
            torch.as_tensor(loads, device=device),
            torch.as_tensor(queues, device=device),
            torch.as_tensor(budgets, device=device),
            torch.as_tensor(base, dtype=torch.float64, device=device),
            **kwargs,
        )
        return value.cpu().numpy()

    @torch.no_grad()
    def predict(
        self,
        load: int,
        queue: int,
        budget: int,
        remaining_horizon: int,
        history: CausalHistory,
    ) -> float:
        values = self.predict_many(
            np.asarray([load]),
            np.asarray([queue]),
            np.asarray([budget]),
            np.asarray([remaining_horizon]),
            [history],
        )
        return float(values[0])
