from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping


class GateVerdict(StrEnum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    WAIT = "WAIT"


@dataclass(frozen=True, slots=True)
class SubmissionGatePolicy:
    minimum_sharpe: float = 1.25
    minimum_fitness: float = 1.00
    minimum_turnover: float = 0.01
    maximum_turnover: float = 0.70
    maximum_self_correlation: float = 0.70
    preferred_self_correlation: float = 0.45
    maximum_instrument_weight: float = 0.10
    require_official_source: bool = True
    require_official_checks: bool = True
    require_good_ready_for_automatic_submit: bool = False


@dataclass(frozen=True, slots=True)
class SubmissionGateResult:
    verdict: GateVerdict
    eligible: bool
    reasons: tuple[str, ...]
    alpha_id: str | None
    simulation_id: str | None
    policy: SubmissionGatePolicy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_pre_submission(
    record: Mapping[str, Any],
    *,
    policy: SubmissionGatePolicy | None = None,
    automatic: bool = False,
) -> SubmissionGateResult:
    p = policy or SubmissionGatePolicy()
    reasons: list[str] = []
    alpha_id = str(record.get("brain_alpha_id") or record.get("alpha_id") or "").strip() or None
    simulation_id = str(record.get("official_simulation_id") or record.get("brain_simulation_id") or record.get("simulation_id") or "").strip() or None

    if record.get("submitted_at") or str(record.get("status") or "").upper() == "SUBMITTED":
        reasons.append("ALREADY_SUBMITTED")
    if not alpha_id:
        reasons.append("MISSING_BRAIN_ALPHA_ID")
    if not simulation_id:
        reasons.append("MISSING_OFFICIAL_SIMULATION_ID")
    if p.require_official_source and record.get("metrics_source") and record.get("metrics_source") != "WORLDQUANT_BRAIN":
        reasons.append("UNVERIFIED_METRICS_SOURCE")
    if record.get("duplicate_expression") or record.get("duplicate"):
        reasons.append("DUPLICATE_EXPRESSION")

    is_pass = str(record.get("is_check_status") or "").upper() == "PASS"
    sharpe = _number(record, "sharpe")
    fitness = _number(record, "fitness")
    turnover = _number(record, "turnover")
    self_corr = _number(record, "self_correlation")
    max_weight = _number(record, "max_instrument_weight")

    if sharpe is None or sharpe <= p.minimum_sharpe:
        reasons.append("SHARPE_NOT_PASSED")
    if fitness is None or fitness <= p.minimum_fitness:
        reasons.append("FITNESS_NOT_PASSED")
    if turnover is None or not (p.minimum_turnover < turnover < p.maximum_turnover):
        reasons.append("TURNOVER_NOT_PASSED")
    if self_corr is None:
        reasons.append("SELF_CORRELATION_NOT_VERIFIED")
    elif self_corr >= p.maximum_self_correlation:
        reasons.append("SELF_CORRELATION_NOT_PASSED")
    if max_weight is None and not is_pass:
        reasons.append("MAX_WEIGHT_NOT_VERIFIED")
    elif max_weight is not None and max_weight >= p.maximum_instrument_weight:
        reasons.append("MAX_WEIGHT_NOT_PASSED")

    if p.require_official_checks:
        is_pass = str(record.get("is_check_status") or "").upper() == "PASS"
        for key, code in (
            ("weight_test_passed", "WEIGHT_TEST_NOT_PASSED"),
            ("sub_universe_passed", "SUB_UNIVERSE_TEST_NOT_PASSED"),
            ("bias_test_passed", "BIAS_TEST_NOT_PASSED"),
        ):
            if not is_pass and record.get(key) is not True:
                reasons.append(code)

    if automatic:
        if not bool(record.get("submit_ready")):
            reasons.append("SUBMIT_READY_FALSE")
        if p.require_good_ready_for_automatic_submit and not bool(record.get("good_ready")):
            reasons.append("GOOD_READY_FALSE")

    eligible = not reasons
    return SubmissionGateResult(
        verdict=GateVerdict.PASS if eligible else GateVerdict.BLOCK,
        eligible=eligible,
        reasons=tuple(dict.fromkeys(reasons)),
        alpha_id=alpha_id,
        simulation_id=simulation_id,
        policy=p,
    )


def guarded_submit(
    record: Mapping[str, Any],
    submit_fn,
    *,
    policy: SubmissionGatePolicy | None = None,
    automatic: bool = False,
) -> dict[str, Any]:
    gate = evaluate_pre_submission(record, policy=policy, automatic=automatic)
    if not gate.eligible:
        return {"success": False, "status": "BLOCKED", "error": "PRE_SUBMISSION_GATE_FAILED", "gate": gate.to_dict()}
    response = dict(submit_fn(gate.alpha_id))
    accepted = bool(response.get("success")) or str(response.get("status") or "").upper() in {"SUBMITTED", "SUCCESS"}
    return {"success": accepted, "status": "SUBMITTED" if accepted else "SUBMISSION_FAILED", "brain_alpha_id": gate.alpha_id, "simulation_id": gate.simulation_id, "gate": gate.to_dict(), "raw_response": response}
