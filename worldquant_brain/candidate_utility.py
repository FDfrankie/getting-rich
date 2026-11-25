"""Transparent cold-start Utility V1 scoring for WorldQuant BRAIN Alpha candidates."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

UTILITY_MODEL_VERSION = "utility-v1.0"


def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class UtilityResult:
    sharpe_score: float
    fitness_score: float
    novelty_score: float
    margin_score: float
    turnover_quality: float
    stability_score: float
    exploration_score: float

    correlation_penalty: float
    missing_evidence_penalty: float
    complexity_penalty: float
    duplicate_penalty: float

    gross_utility: float
    submission_utility: float

    eligible: bool
    decision: str
    reasons: tuple[str, ...]
    model_version: str
    calculated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def turnover_quality_score(
    turnover: float | None,
    ideal_turnover: float = 0.20,
    sigma: float = 0.65,
) -> float:
    turnover_value = finite_number(turnover)
    if turnover_value is None or turnover_value <= 0:
        return 0.0
    if turnover_value > 1.5:
        turnover_value = turnover_value / 100.0
    if turnover_value <= 0:
        return 0.0
    distance = math.log(turnover_value) - math.log(ideal_turnover)
    return clip(math.exp(-(distance * distance) / (2.0 * sigma * sigma)))


def correlation_penalty(maximum_correlation: float | None) -> float:
    corr = finite_number(maximum_correlation)
    if corr is None:
        return 0.0
    corr = abs(corr)
    if corr < 0.70:
        return 0.0
    if corr >= 0.90:
        return 20.0
    return 20.0 * (corr - 0.70) / 0.20


def normalize_margin_bps(candidate: Mapping[str, Any]) -> float | None:
    margin_bps = finite_number(candidate.get("margin_bps"))
    if margin_bps is not None:
        return margin_bps
    margin = finite_number(candidate.get("margin"))
    if margin is None:
        return None
    unit = str(candidate.get("margin_unit") or "").lower()
    if unit == "bps":
        return margin
    if unit in {"fraction", "return_fraction"}:
        return margin * 10_000.0
    if 0.0 <= margin <= 0.1:
        return margin * 10_000.0
    return margin


def evaluate_candidate(candidate: Mapping[str, Any]) -> UtilityResult:
    reasons: list[str] = []

    expression_hash = candidate.get("expression_hash")
    duplicate_expression = bool(candidate.get("duplicate_expression"))
    already_submitted = bool(candidate.get("already_submitted") or candidate.get("submitted_at"))

    duplicate_penalty = 0.0
    if duplicate_expression:
        duplicate_penalty = 100.0
        reasons.append("EXACT_EXPRESSION_DUPLICATE")
    if already_submitted:
        duplicate_penalty = 100.0
        reasons.append("ALREADY_SUBMITTED")
    if not expression_hash:
        reasons.append("EXPRESSION_HASH_MISSING")

    sharpe = finite_number(candidate.get("sharpe"))
    fitness = finite_number(candidate.get("fitness"))
    turnover = finite_number(candidate.get("turnover"))

    sharpe_score = clip((sharpe - 1.0) / 1.5) if sharpe is not None else 0.0
    fitness_score = clip((fitness - 0.7) / 1.3) if fitness is not None else 0.0

    max_corr = finite_number(candidate.get("maximum_correlation"))
    if max_corr is None:
        max_corr = finite_number(candidate.get("self_correlation"))

    if max_corr is None:
        novelty_score = 0.30
        missing_correlation_penalty = 5.0
        reasons.append("CORRELATION_MISSING")
    else:
        novelty_score = clip(1.0 - abs(max_corr))
        missing_correlation_penalty = 0.0

    margin_bps = normalize_margin_bps(candidate)
    if margin_bps is None:
        margin_score = 0.30
        missing_margin_penalty = 5.0
        reasons.append("MARGIN_MISSING")
    else:
        margin_score = clip((margin_bps - 5.0) / 20.0)
        missing_margin_penalty = 0.0

    turnover_quality = turnover_quality_score(turnover)

    supplied_stability = finite_number(candidate.get("stability_score"))
    if supplied_stability is None:
        components = [
            finite_number(candidate.get("subperiod_stability")),
            finite_number(candidate.get("regime_stability")),
            finite_number(candidate.get("setting_stability")),
            finite_number(candidate.get("family_persistence")),
        ]
        defaults = [value if value is not None else 0.30 for value in components]
        stability_score = clip(
            0.35 * defaults[0] + 0.25 * defaults[1] + 0.20 * defaults[2] + 0.20 * defaults[3]
        )
    else:
        stability_score = clip(supplied_stability)

    exploration_score = clip(finite_number(candidate.get("exploration_score")) or 0.0)

    missing_is_penalty = 0.0
    if candidate.get("is_check_status") not in {"PASS", "PASSED", True}:
        missing_is_penalty = 8.0
        reasons.append("IS_CHECK_NOT_PASSED")

    maximum_corr_penalty = correlation_penalty(max_corr)
    if max_corr is not None and abs(max_corr) >= 0.90:
        reasons.append("VERY_HIGH_CORRELATION")
    elif max_corr is not None and abs(max_corr) >= 0.70:
        reasons.append("HIGH_CORRELATION")

    complexity = finite_number(candidate.get("expression_complexity"))
    if complexity is None or complexity <= 20:
        complexity_penalty = 0.0
    elif complexity >= 80:
        complexity_penalty = 10.0
    else:
        complexity_penalty = 10.0 * (complexity - 20.0) / 60.0

    near_duplicate_penalty = clip(finite_number(candidate.get("near_duplicate_score")) or 0.0) * 25.0

    subperiod_only_penalty = 15.0 if bool(candidate.get("one_subperiod_only")) else 0.0
    if subperiod_only_penalty > 0:
        reasons.append("ONE_SUBPERIOD_ONLY")

    missing_evidence_penalty = (
        missing_correlation_penalty + missing_margin_penalty + missing_is_penalty + subperiod_only_penalty
    )

    gross_utility = 100.0 * (
        0.30 * sharpe_score
        + 0.22 * fitness_score
        + 0.18 * novelty_score
        + 0.12 * margin_score
        + 0.08 * turnover_quality
        + 0.07 * stability_score
        + 0.03 * exploration_score
    )

    total_penalty = (
        maximum_corr_penalty
        + missing_evidence_penalty
        + complexity_penalty
        + near_duplicate_penalty
        + duplicate_penalty
    )
    final_utility = max(0.0, gross_utility - total_penalty)
    has_blocking_corr = (max_corr is not None and abs(max_corr) >= 0.70)
    is_failed = str(candidate.get("is_check_status") or "").upper() in {"FAIL", "FAILED", "ERROR"}
    eligible = not (duplicate_expression or already_submitted or has_blocking_corr or is_failed)

    if duplicate_expression or already_submitted or is_failed:
        decision = "REJECT_DUPLICATE"
    elif max_corr is not None and abs(max_corr) >= 0.90:
        decision = "NEAR_MISS_CORRELATION"
    elif not eligible:
        decision = "REJECT_DUPLICATE"
    elif missing_is_penalty > 0:
        decision = "WAIT_FOR_IS_CHECK"
    elif final_utility >= 70:
        decision = "SUBMIT_HIGH_PRIORITY"
    elif final_utility >= 50:
        decision = "SUBMIT_STANDARD"
    elif exploration_score >= 0.70:
        decision = "EXPLORE"
    else:
        decision = "HOLD"

    return UtilityResult(
        sharpe_score=round(sharpe_score, 6),
        fitness_score=round(fitness_score, 6),
        novelty_score=round(novelty_score, 6),
        margin_score=round(margin_score, 6),
        turnover_quality=round(turnover_quality, 6),
        stability_score=round(stability_score, 6),
        exploration_score=round(exploration_score, 6),
        correlation_penalty=round(maximum_corr_penalty, 6),
        missing_evidence_penalty=round(missing_evidence_penalty, 6),
        complexity_penalty=round(complexity_penalty + near_duplicate_penalty, 6),
        duplicate_penalty=round(duplicate_penalty, 6),
        gross_utility=round(gross_utility, 6),
        submission_utility=round(final_utility, 6),
        eligible=eligible,
        decision=decision,
        reasons=tuple(reasons),
        model_version=UTILITY_MODEL_VERSION,
        calculated_at=datetime.now(timezone.utc).isoformat(),
    )
