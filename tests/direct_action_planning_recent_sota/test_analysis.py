from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_direct_action_planning_recent_sota import (
    benjamini_hochberg,
    classify_pareto,
    paired_effect,
)


def test_bh_adjustment_is_monotone_in_p_order() -> None:
    values = np.asarray([0.04, 0.001, 0.03, 0.2])
    adjusted = benjamini_hochberg(values)
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= -1.0e-12)
    assert np.all(adjusted >= values)
    assert np.all(adjusted <= 1.0)


def test_pareto_classification_handles_dominance_and_tradeoff() -> None:
    assert classify_pareto(5.0, 2.0, 4.0, 3.0) == "dap_dominates"
    assert classify_pareto(4.0, 3.0, 5.0, 2.0) == "baseline_dominates"
    assert classify_pareto(5.0, 3.0, 4.0, 2.0) == "tradeoff"


def test_paired_effect_uses_seed_blocks_not_episode_rows() -> None:
    frame = pd.DataFrame(
        {
            "training_seed": [1, 2, 3, 1, 2, 3],
            "method": ["dap_calibrated"] * 3 + ["lcpo"] * 3,
            "metric": [3.0, 4.0, 5.0, 1.0, 2.0, 3.0],
        }
    )
    result = paired_effect(frame, metric="metric", baseline="lcpo", favorable="higher")
    assert result["n"] == 3
    assert result["mean_effect"] == 2.0
