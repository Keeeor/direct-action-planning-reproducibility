"""Read-only imports from the frozen Kubernetes prototype."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE_ROOT = PROJECT_ROOT / "research/direct_action_planning_k8s_prototype"
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from controller.action_mapper import ACTION_ORDER, ActionMapper  # noqa: E402
from controller.budget_tracker import BudgetTracker  # noqa: E402
from controller.checkpoint_loader import (  # noqa: E402
    DAPCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from controller.config import ControllerConfig  # noqa: E402
from controller.kube_client import KubectlClient  # noqa: E402
from controller.state_collector import (  # noqa: E402
    FieldEvidence,
    StateCollector,
    StateSnapshot,
)


__all__ = [
    "ACTION_ORDER", "ActionMapper", "BudgetTracker", "ControllerConfig",
    "DAPCheckpoint", "FieldEvidence", "KubectlClient", "PROJECT_ROOT",
    "PROTOTYPE_ROOT", "StateCollector", "StateSnapshot", "load_checkpoint",
    "save_checkpoint",
]

