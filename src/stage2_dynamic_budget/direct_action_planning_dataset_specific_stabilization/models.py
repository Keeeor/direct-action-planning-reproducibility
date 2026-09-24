from __future__ import annotations

import numpy as np
import torch

from stage2_dynamic_budget.direct_action_planning_dataset_validation.models import (
    FeatureNormalizer,
)
from stage2_dynamic_budget.direct_action_planning_paper_evidence.models import (
    EvidenceValueNetwork,
)


class ScaledEvidenceValueNetwork(EvidenceValueNetwork):
    """Expose values in reward units while fitting a normalized output head."""

    def __init__(
        self,
        normalizer: FeatureNormalizer,
        hidden_dim: int = 64,
        feature_mask: np.ndarray | None = None,
        *,
        output_scale: float = 1.0,
        zero_initialize_output: bool = False,
    ) -> None:
        scale = float(output_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("output_scale must be finite and positive")
        super().__init__(normalizer, hidden_dim=hidden_dim, feature_mask=feature_mask)
        self.output_scale = scale
        self.zero_initialize_output = bool(zero_initialize_output)
        if self.zero_initialize_output:
            torch.nn.init.zeros_(self.network[-1].weight)
            torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return super().forward(observation) * self.output_scale
