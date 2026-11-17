"""Canonical Alpha lifecycle and fail-closed transition rules."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class AlphaState(str, Enum):
    GENERATED = "GENERATED"
    SYNTAX_VALIDATED = "SYNTAX_VALIDATED"
    DEDUPED = "DEDUPED"
    SIMULATION_QUEUED = "SIMULATION_QUEUED"
    SIMULATION_RUNNING = "SIMULATION_RUNNING"
    BRAIN_COMPLETE = "BRAIN_COMPLETE"
    IS_CHECK_PASSED = "IS_CHECK_PASSED"
    CORRELATION_PASSED = "CORRELATION_PASSED"
    SUBMISSION_CANDIDATE = "SUBMISSION_CANDIDATE"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"
    LOCAL_VERIFIED = "LOCAL_VERIFIED"


DISCOVERED_STATES = {
    AlphaState.BRAIN_COMPLETE,
    AlphaState.IS_CHECK_PASSED,
    AlphaState.CORRELATION_PASSED,
    AlphaState.SUBMISSION_CANDIDATE,
    AlphaState.SUBMITTING,
    AlphaState.SUBMITTED,
}

_ALLOWED: dict[AlphaState, set[AlphaState]] = {
    AlphaState.GENERATED: {AlphaState.SYNTAX_VALIDATED, AlphaState.REJECTED},
    AlphaState.SYNTAX_VALIDATED: {AlphaState.DEDUPED, AlphaState.REJECTED},
    AlphaState.DEDUPED: {AlphaState.SIMULATION_QUEUED, AlphaState.ARCHIVED, AlphaState.REJECTED},
    AlphaState.SIMULATION_QUEUED: {AlphaState.SIMULATION_RUNNING, AlphaState.REJECTED},
    AlphaState.SIMULATION_RUNNING: {AlphaState.BRAIN_COMPLETE, AlphaState.REJECTED},
    AlphaState.BRAIN_COMPLETE: {AlphaState.IS_CHECK_PASSED, AlphaState.REJECTED, AlphaState.ARCHIVED},
    AlphaState.IS_CHECK_PASSED: {AlphaState.CORRELATION_PASSED, AlphaState.REJECTED},
    AlphaState.CORRELATION_PASSED: {AlphaState.SUBMISSION_CANDIDATE, AlphaState.REJECTED},
    AlphaState.SUBMISSION_CANDIDATE: {AlphaState.SUBMITTING, AlphaState.REJECTED, AlphaState.ARCHIVED},
    AlphaState.SUBMITTING: {AlphaState.SUBMITTED, AlphaState.SUBMISSION_CANDIDATE, AlphaState.REJECTED},
    AlphaState.SUBMITTED: {AlphaState.ARCHIVED},
    AlphaState.LOCAL_VERIFIED: {AlphaState.SIMULATION_QUEUED, AlphaState.ARCHIVED, AlphaState.REJECTED},
    AlphaState.REJECTED: {AlphaState.ARCHIVED, AlphaState.SIMULATION_QUEUED},
    AlphaState.ARCHIVED: set(),
}


def normalize_legacy_state(value: str | None, *, has_official_simulation: bool = False) -> AlphaState:
    raw = (value or "").strip().upper()
    if raw == "VERIFIED":
        return AlphaState.BRAIN_COMPLETE if has_official_simulation else AlphaState.LOCAL_VERIFIED
    if raw in {"COMPLETE", "COMPLETED"}:
        return AlphaState.BRAIN_COMPLETE
    try:
        return AlphaState(raw)
    except ValueError:
        return AlphaState.GENERATED


def can_transition(current: AlphaState, target: AlphaState) -> bool:
    return target == current or target in _ALLOWED[current]


def require_transition(current: AlphaState, target: AlphaState) -> None:
    if not can_transition(current, target):
        raise ValueError(f"Invalid Alpha lifecycle transition: {current.value} -> {target.value}")


def is_discovered(alpha: Mapping[str, Any]) -> bool:
    state = normalize_legacy_state(
        str(alpha.get("status") or ""),
        has_official_simulation=bool(alpha.get("brain_simulation_id") or alpha.get("simulation_id")),
    )
    return state in DISCOVERED_STATES


@dataclass(frozen=True)
class SubmissionEligibility:
    ready: bool
    reasons: tuple[str, ...]


def submission_eligibility(alpha: Mapping[str, Any]) -> SubmissionEligibility:
    """Evaluate formal submission readiness without using a daily quota."""
    reasons: list[str] = []
    state = normalize_legacy_state(str(alpha.get("status") or ""), has_official_simulation=True)
    if state != AlphaState.SUBMISSION_CANDIDATE:
        reasons.append("STATUS_NOT_SUBMISSION_CANDIDATE")
    for key, threshold in (("sharpe", 1.25), ("fitness", 1.0)):
        value = alpha.get(key)
        if value is None or float(value) < threshold:
            reasons.append(f"{key.upper()}_GATE_FAILED")
    turnover = alpha.get("turnover")
    if turnover is None or not 0.01 <= float(turnover) <= 0.70:
        reasons.append("TURNOVER_GATE_FAILED")
    margin_bps = alpha.get("margin_bps")
    if margin_bps is None or float(margin_bps) <= 10.0:
        reasons.append("MARGIN_GATE_FAILED")
    drawdown = alpha.get("drawdown")
    if drawdown is None or float(drawdown) >= 0.10:
        reasons.append("DRAWDOWN_GATE_FAILED")
    if alpha.get("is_check_status") != "PASS":
        reasons.append("IS_CHECK_NOT_PASSED")
    corr = alpha.get("self_correlation")
    if corr is None or float(corr) >= 0.70:
        reasons.append("CORRELATION_GATE_FAILED")
    if not alpha.get("brain_simulation_id"):
        reasons.append("SIMULATION_ID_MISSING")
    if alpha.get("brain_alpha_id") or alpha.get("submitted_at"):
        reasons.append("ALREADY_SUBMITTED")
    if alpha.get("rejection_code"):
        reasons.append("REJECTED")
    return SubmissionEligibility(not reasons, tuple(reasons))
