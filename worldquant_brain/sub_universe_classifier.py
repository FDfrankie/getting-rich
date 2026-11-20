"""Sub-universe Root-cause Classifier for WorldQuant BRAIN Alpha Failures.

Classifies SUB_UNIVERSE failures into specific economic/data root causes
and maps each diagnosis to its strictly allowed repair action set so the
miner avoids blind window/rank/decay churn.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class SubUniverseRootCause(StrEnum):
    LOW_COVERAGE = "LOW_COVERAGE"
    ILLIQUID_DEPENDENCY = "ILLIQUID_DEPENDENCY"
    SMALL_CAP_BIAS = "SMALL_CAP_BIAS"
    GROUP_FRAGMENTATION = "GROUP_FRAGMENTATION"
    EXTREME_WEIGHT_DEPENDENCY = "EXTREME_WEIGHT_DEPENDENCY"
    DATASET_UNIVERSE_MISMATCH = "DATASET_UNIVERSE_MISMATCH"
    PARAMETER_INSTABILITY = "PARAMETER_INSTABILITY"
    GENUINE_IDEA_FAILURE = "GENUINE_IDEA_FAILURE"


ALLOWED_REPAIRS_BY_ROOT_CAUSE: dict[SubUniverseRootCause, tuple[str, ...]] = {
    SubUniverseRootCause.LOW_COVERAGE: (
        "ADD_CAUSAL_BACKFILL",
        "SWITCH_HIGHER_COVERAGE_FIELD",
        "RELAX_EVENT_CONDITION",
    ),
    SubUniverseRootCause.ILLIQUID_DEPENDENCY: (
        "APPLY_LIQUIDITY_FILTER",
        "TEST_TOP1000_OR_TOP500_UNIVERSE",
        "ADD_VOLATILITY_OR_ADV_SCALING",
    ),
    SubUniverseRootCause.SMALL_CAP_BIAS: (
        "ADD_CAP_OR_SIZE_NEUTRALIZATION",
        "TEST_TOP1000_UNIVERSE",
        "ADD_WINSORIZE",
    ),
    SubUniverseRootCause.GROUP_FRAGMENTATION: (
        "BROADEN_NEUTRALIZATION_GROUP",
        "TEST_SECTOR_OR_MARKET_NEUTRALIZATION",
    ),
    SubUniverseRootCause.EXTREME_WEIGHT_DEPENDENCY: (
        "ADD_TRUNCATION",
        "ADD_WINSORIZE",
        "CLIP_OUTLIER_ZSCORE",
    ),
    SubUniverseRootCause.DATASET_UNIVERSE_MISMATCH: (
        "SWITCH_UNIVERSE_TO_DATASET_NATIVE",
        "ABANDON_HYPOTHESIS_FOR_UNIVERSE",
    ),
    SubUniverseRootCause.PARAMETER_INSTABILITY: (
        "RUN_PARAMETER_NEIGHBORHOOD_ABLATION",
        "SMOOTH_HORIZON_WITH_CAUSAL_MEAN",
    ),
    SubUniverseRootCause.GENUINE_IDEA_FAILURE: (
        "RETIRE_HYPOTHESIS",
        "SURVEY_NEW_ECONOMIC_DOMAIN",
    ),
}


@dataclass(frozen=True)
class SubUniverseDiagnosis:
    root_cause: SubUniverseRootCause
    confidence: float
    rationale: str
    allowed_repairs: tuple[str, ...]
    disallowed_blind_edits: tuple[str, ...]


def classify_sub_universe_failure(
    *,
    main_sharpe: float,
    sub_universe_sharpe: float | None,
    coverage: float | None = None,
    max_instrument_weight: float | None = None,
    turnover: float | None = None,
    neutralization: str = "SUBINDUSTRY",
    universe: str = "TOP3000",
    dataset_domain: str = "",
    small_cap_pnl_share: float | None = None,
    illiquid_pnl_share: float | None = None,
    neighbor_sharpe_std: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SubUniverseDiagnosis:
    """Diagnose the most likely root cause of a sub-universe check failure."""
    sub_s = float(sub_universe_sharpe) if sub_universe_sharpe is not None else 0.0
    domain = (dataset_domain or "").upper()
    neut = (neutralization or "SUBINDUSTRY").upper()
    univ = (universe or "TOP3000").upper()

    if max_instrument_weight is not None and max_instrument_weight >= 0.08:
        rc = SubUniverseRootCause.EXTREME_WEIGHT_DEPENDENCY
        return SubUniverseDiagnosis(
            root_cause=rc,
            confidence=0.85,
            rationale=f"max_instrument_weight={max_instrument_weight:.3f} >= 0.08 indicates concentration driving sub-universe mismatch.",
            allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
            disallowed_blind_edits=("WINDOW_RESIZE", "CHANGE_DECAY"),
        )

    if coverage is not None and coverage < 0.55:
        rc = SubUniverseRootCause.LOW_COVERAGE
        return SubUniverseDiagnosis(
            root_cause=rc,
            confidence=0.82,
            rationale=f"coverage={coverage:.2f} < 0.55 creates sparse sub-universe cross-sections.",
            allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
            disallowed_blind_edits=("ADD_RANK", "WINDOW_RESIZE"),
        )

    if illiquid_pnl_share is not None and illiquid_pnl_share >= 0.55:
        rc = SubUniverseRootCause.ILLIQUID_DEPENDENCY
        return SubUniverseDiagnosis(
            root_cause=rc,
            confidence=0.80,
            rationale=f"illiquid_pnl_share={illiquid_pnl_share:.2f} indicates alpha concentrates in less liquid names.",
            allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
            disallowed_blind_edits=("WINDOW_RESIZE", "ADD_RANK"),
        )

    if small_cap_pnl_share is not None and small_cap_pnl_share >= 0.60:
        rc = SubUniverseRootCause.SMALL_CAP_BIAS
        return SubUniverseDiagnosis(
            root_cause=rc,
            confidence=0.80,
            rationale=f"small_cap_pnl_share={small_cap_pnl_share:.2f} indicates signal collapses in liquid large-cap sub-universe.",
            allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
            disallowed_blind_edits=("WINDOW_RESIZE",),
        )

    if domain in ("SENTIMENT", "NEWS", "SOCIAL", "ALTERNATIVE") and univ == "TOP3000" and main_sharpe >= 1.25 and sub_s < 0.70:
        rc = SubUniverseRootCause.DATASET_UNIVERSE_MISMATCH
        return SubUniverseDiagnosis(
            root_cause=rc,
            confidence=0.72,
            rationale=f"Alternative/sentiment domain {domain} with strong TOP3000 Sharpe ({main_sharpe:.2f}) but weak sub-universe ({sub_s:.2f}) suggests coverage/universe mismatch.",
            allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
            disallowed_blind_edits=("ADD_RANK", "PRIME_WINDOW"),
        )

    if neut == "SUBINDUSTRY" and main_sharpe >= 1.25 and sub_s > 0.25 and (main_sharpe - sub_s) >= 0.50:
        rc = SubUniverseRootCause.GROUP_FRAGMENTATION
        return SubUniverseDiagnosis(
            root_cause=rc,
            confidence=0.68,
            rationale="SUBINDUSTRY neutralization can fragment small sub-universe groups; test broader SECTOR/INDUSTRY neutralization.",
            allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
            disallowed_blind_edits=("PRIME_WINDOW",),
        )

    if neighbor_sharpe_std is not None and neighbor_sharpe_std >= 0.35:
        rc = SubUniverseRootCause.PARAMETER_INSTABILITY
        return SubUniverseDiagnosis(
            root_cause=rc,
            confidence=0.75,
            rationale=f"neighbor_sharpe_std={neighbor_sharpe_std:.2f} indicates fragile parameter spike.",
            allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
            disallowed_blind_edits=("PRIME_WINDOW",),
        )

    rc = SubUniverseRootCause.GENUINE_IDEA_FAILURE
    return SubUniverseDiagnosis(
        root_cause=rc,
        confidence=0.65,
        rationale=f"Sub-universe Sharpe ({sub_s:.2f}) fails without a remediable structural artifact (main={main_sharpe:.2f}).",
        allowed_repairs=ALLOWED_REPAIRS_BY_ROOT_CAUSE[rc],
        disallowed_blind_edits=("WINDOW_RESIZE", "PRIME_WINDOW", "ADD_RANK", "CHANGE_DECAY"),
    )


def compute_sub_universe_sharpe_floor(
    alpha_sharpe: float,
    *,
    alpha_universe_size: int = 3000,
    sub_universe_size: int = 1000,
) -> float:
    """Compute WorldQuant BRAIN / IQC minimum required sub-universe Sharpe.

    Source: WorldQuant BRAIN / IQC Sub-Universe Check Guidelines
    (0.75 * sqrt(sub_universe_size / alpha_universe_size) * alpha_sharpe).
    """
    import math

    if alpha_universe_size <= 0 or sub_universe_size <= 0:
        raise ValueError("Universe sizes must be positive")
    return round(0.75 * math.sqrt(float(sub_universe_size) / float(alpha_universe_size)) * float(alpha_sharpe), 4)


def diagnose_turnover_and_cost_stress(
    *,
    sharpe: float,
    fitness: float,
    turnover: float,
    margin: float | None = None,
    delay: int = 1,
) -> dict[str, Any]:
    """Diagnose turnover, fitness floor (0.125), and cost-stress margin resilience.

    Sources:
    - WorldQuant BRAIN Fitness & Turnover Guidelines (1% < turnover < 70%, 0.125 floor)
    - QuantML-Research / wq-alpha-research (https://github.com/QuantML-Research/wq-alpha-research)
    """
    to = float(turnover)
    sh = float(sharpe)
    fit = float(fitness)
    norm_margin = (float(margin) / 10000.0) if (margin is not None and float(margin) > 0.5) else (float(margin) if margin is not None else None)
    min_sharpe = 2.00 if int(delay) == 0 else 1.25

    issues: list[str] = []
    repairs: list[str] = []
    if to <= 0.01:
        issues.append("TURNOVER_TOO_LOW")
        repairs.extend(["DECREASE_DECAY", "SHORTEN_TS_WINDOW"])
    elif to >= 0.70:
        issues.append("TURNOVER_ABOVE_70PCT_LIMIT")
        repairs.extend(["ADD_HUMP", "INCREASE_DECAY", "ADD_TS_MEAN_SMOOTHING"])
    elif to > 0.25 and fit < 1.0 and sh >= min_sharpe:
        issues.append("HIGH_TURNOVER_FITNESS_DRAG")
        repairs.extend(["ADD_HUMP", "INCREASE_DECAY"])

    if norm_margin is not None and norm_margin < 0.0010:
        issues.append("LOW_MARGIN_COST_STRESS_RISK")
        for r in ("ADD_HUMP", "REDUCE_TRUNCATION", "INCREASE_DECAY"):
            if r not in repairs:
                repairs.append(r)

    return {
        "healthy": len(issues) == 0 and sh >= min_sharpe and fit >= 1.0,
        "issues": issues,
        "recommended_repairs": repairs,
        "normalized_margin": norm_margin,
        "at_fitness_sweet_spot": 0.01 < to <= 0.15,
    }
