from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from worldquant_brain.optimization_policy import (
    AlphaPerformance,
    ObjectivePolicy,
    policy_snapshot,
)


def build_evidence_context(
    *,
    before: AlphaPerformance | None,
    after: AlphaPerformance,
    policy: ObjectivePolicy,
    action: str | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = policy_snapshot(after, policy)
    return {
        "optimization_objective": policy.objective.value,
        "before_metrics": asdict(before) if before else None,
        "after_metrics": asdict(after),
        "turnover_efficiency": snapshot["turnover_efficiency"],
        "diagnoses": snapshot["diagnoses"],
        "selected_action": action,
        "recommended_actions": snapshot["actions"],
        "extra": dict(extra or {}),
    }
