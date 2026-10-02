"""Code-identity guard for formal paper-closure development runs."""

from __future__ import annotations

from pathlib import Path

from dap.utils.artifacts import sha256_tree


def closure_code_hash(project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    return sha256_tree(root / "src/dap/direct_action_planning_paper_closure")


def verify_code_lock(project_root: str | Path, config: dict) -> str:
    """Reject a registered run if its optional pre-locked source hash changed."""

    observed = closure_code_hash(project_root)
    expected = config.get("expected_code_tree_sha256")
    if expected is not None and str(expected) != observed:
        raise ValueError(
            "closure source does not match the pre-locked code hash: "
            f"expected={expected}, observed={observed}"
        )
    return observed
