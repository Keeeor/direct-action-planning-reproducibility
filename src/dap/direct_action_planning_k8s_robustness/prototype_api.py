"""Read-only import boundary to the frozen Kubernetes prototype."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE_ROOT = PROJECT_ROOT / "research/direct_action_planning_k8s_prototype"
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from controller.state_collector import FieldEvidence, StateSnapshot  # noqa: E402


__all__ = ["FieldEvidence", "StateSnapshot", "PROJECT_ROOT", "PROTOTYPE_ROOT"]
