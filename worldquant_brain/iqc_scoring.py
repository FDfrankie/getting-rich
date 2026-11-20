"""IQC 2026 Score-Aware Alpha Architecture: Three-Layer Scoring, Utility, & Daily Submission Planner.

Internal candidate ranking and daily planning tool — NOT an official IQC score predictor.
All hard gates, official metrics, correlation checks, and robustness metrics MUST come from
trusted backend sources (never from unverified UI payloads), and this scorer NEVER bypasses
PreSubmissionGate.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

IQC_POLICY_VERSION = "2026.1.0"
TURNOVER_SCORE_FLOOR = 0.125
MAX_DEFAULT_CONDITIONS = 3
SELF_CORRELATION_HARD_LIMIT = 0.70
MAX_INSTRUMENT_WEIGHT_LIMIT = 0.10
EST_TZ = ZoneInfo("America/New_York")

@dataclass(frozen=True, slots=True)
class IQCScoringPolicy:
    policy_version: str = IQC_POLICY_VERSION
    minimum_sharpe: float = 1.25
    minimum_fitness: float = 1.00
    minimum_turnover: float = 0.01
    maximum_turnover: float = 0.70
    maximum_self_correlation: float = SELF_CORRELATION_HARD_LIMIT
    maximum_instrument_weight: float = MAX_INSTRUMENT_WEIGHT_LIMIT
    turnover_score_floor: float = TURNOVER_SCORE_FLOOR
    maximum_daily_submissions: int = 3
    maximum_same_family_per_day: int = 1
    merged_score_negative_blocks: bool = True

    @classmethod
    def for_delay(cls, delay: int) -> "IQCScoringPolicy":
        """Official BRAIN Sharpe floor: 2.0 for Delay 0, 1.25 for Delay 1."""
        if int(delay) == 0:
            return cls(minimum_sharpe=2.0)
        return cls(minimum_sharpe=1.25)


class IQCCandidateDecision(StrEnum):
    BLOCKED = "BLOCKED"
    SUBMIT_NOW = "SUBMIT_NOW"
    SUBMIT_IF_SLOT_AVAILABLE = "SUBMIT_IF_SLOT_AVAILABLE"
    HOLD_FOR_BETTER_DAY = "HOLD_FOR_BETTER_DAY"
    HOLD_OR_REFINE = "HOLD_OR_REFINE"
    TARGETED_REPAIR = "TARGETED_REPAIR"
    STRUCTURAL_RESEARCH = "STRUCTURAL_RESEARCH"
    ARCHIVE = "ARCHIVE"


class IQCGrade(StrEnum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    BLOCKED = "BLOCKED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def est_day_key(dt: datetime | None = None) -> str:
    now = dt or datetime.now(EST_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(EST_TZ).strftime("%Y-%m-%d")



def count_conditional_ops(expression: str) -> int:
    """Count conditional operators (if_else, trade_when, ternary ?) in an Alpha expression."""
    expr = (expression or "").lower()
    return expr.count("if_else(") + expr.count("trade_when(") + expr.count("?")


def compute_official_fitness(sharpe: float, returns: float, turnover: float) -> float:
    """Official BRAIN Fitness formula: Sharpe * sqrt(abs(Returns) / max(Turnover, 0.125))."""
    denom = max(float(turnover), TURNOVER_SCORE_FLOOR)
    return float(sharpe) * math.sqrt(abs(float(returns)) / denom)


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(val)))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_snapshot_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IQCHardGates:
    official_provenance: bool
    simulation_id_present: bool
    alpha_id_present: bool
    sharpe_passed: bool
    fitness_passed: bool
    turnover_passed: bool
    self_correlation_verified: bool
    self_correlation_passed: bool
    max_weight_passed: bool
    bias_test_passed: bool
    weight_test_passed: bool
    sub_universe_passed: bool
    duplicate_free: bool
    not_parameter_only_variant: bool
    clean_test_period: bool
    not_already_submitted: bool
    submit_ready: bool

    @property
    def passed(self) -> bool:
        return all(getattr(self, f) for f in self.__dataclass_fields__)

    def failure_reasons(self) -> tuple[str, ...]:
        mapping = (
            ("official_provenance", "UNVERIFIED_METRICS_SOURCE"),
            ("simulation_id_present", "MISSING_OFFICIAL_SIMULATION_ID"),
            ("alpha_id_present", "MISSING_BRAIN_ALPHA_ID"),
            ("sharpe_passed", "SHARPE_NOT_PASSED"),
            ("fitness_passed", "FITNESS_NOT_PASSED"),
            ("turnover_passed", "TURNOVER_NOT_PASSED"),
            ("self_correlation_verified", "SELF_CORRELATION_NOT_VERIFIED"),
            ("self_correlation_passed", "SELF_CORRELATION_NOT_PASSED"),
            ("max_weight_passed", "MAX_WEIGHT_NOT_PASSED"),
            ("bias_test_passed", "BIAS_TEST_NOT_PASSED"),
            ("weight_test_passed", "WEIGHT_TEST_NOT_PASSED"),
            ("sub_universe_passed", "SUB_UNIVERSE_TEST_NOT_PASSED"),
            ("duplicate_free", "DUPLICATE_OR_PUBLIC_FORMULA"),
            ("not_parameter_only_variant", "PARAMETER_ONLY_VARIANT"),
            ("clean_test_period", "TEST_PERIOD_CONTAMINATED"),
            ("not_already_submitted", "ALREADY_SUBMITTED"),
            ("submit_ready", "SUBMIT_READY_FALSE"),
        )
        return tuple(code for attr, code in mapping if not getattr(self, attr))


@dataclass(frozen=True, slots=True)
class IQCScoreComponents:
    official_quality: float
    os_robustness: float
    merged_contribution: float
    correlation_uniqueness: float
    return_drawdown: float
    turnover_efficiency: float
    dataset_idea_diversity: float
    delay_universe_value: float
    dataset_pyramid_value: float
    simplicity: float

    def total(self) -> float:
        return round(
            self.official_quality
            + self.os_robustness
            + self.merged_contribution
            + self.correlation_uniqueness
            + self.return_drawdown
            + self.turnover_efficiency
            + self.dataset_idea_diversity
            + self.delay_universe_value
            + self.dataset_pyramid_value
            + self.simplicity,
            4,
        )


@dataclass(frozen=True, slots=True)
class IQCPenalties:
    test_contamination: float = 0.0
    parameter_cliff: float = 0.0
    single_year_dominance: float = 0.0
    idea_crowding: float = 0.0
    excessive_conditions: float = 0.0
    field_crowding: float = 0.0
    missing_cost_stress: float = 0.0
    missing_merged_analysis: float = 0.0
    complexity_delay: float = 0.0
    missing_overfitting_checks: float = 0.0
    data_quality_risk: float = 0.0

    def total(self) -> float:
        return round(
            self.test_contamination
            + self.parameter_cliff
            + self.single_year_dominance
            + self.idea_crowding
            + self.excessive_conditions
            + self.field_crowding
            + self.missing_cost_stress
            + self.missing_merged_analysis
            + self.complexity_delay
            + self.missing_overfitting_checks
            + self.data_quality_risk,
            4,
        )


@dataclass(frozen=True, slots=True)
class IQCTimingComponents:
    daily_slot_value: float
    portfolio_timing: float
    diversity_timing: float
    competition_estimate: float
    operational_readiness: float

    def total(self) -> float:
        return round(
            self.daily_slot_value
            + self.portfolio_timing
            + self.diversity_timing
            + self.competition_estimate
            + self.operational_readiness,
            4,
        )


@dataclass(frozen=True, slots=True)
class IQCCandidateInput:
    alpha_id: str
    expression: str = ""
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    neutralization: str = "SUBINDUSTRY"
    dataset_category: str = "PRICE_VOLUME"
    idea_family: str = "GENERAL"

    # Trusted official provenance
    metrics_source: str = "UNKNOWN"
    official_simulation_id: str | None = None
    brain_alpha_id: str | None = None

    # Official metrics
    sharpe: float | None = None
    fitness: float | None = None
    returns: float | None = None
    turnover: float | None = None
    margin: float | None = None
    drawdown: float | None = None
    self_correlation: float | None = None
    prod_correlation: float | None = None
    max_instrument_weight: float | None = None

    # Official checks
    bias_test_passed: bool = False
    weight_test_passed: bool = False
    sub_universe_passed: bool = False
    duplicate_or_public: bool = False
    parameter_only_variant: bool = False
    test_period_contaminated: bool = False
    already_submitted: bool = False
    submit_ready: bool = False

    # Robustness (0..1)
    temporal_stability: float | None = 0.5
    parameter_stability: float | None = 0.5
    universe_stability: float | None = 0.5
    regime_stability: float | None = 0.5
    dsr_probability: float | None = None
    pbo: float | None = None
    cpcv_positive_ratio: float | None = None
    max_single_year_pnl_share: float = 0.25

    # Merged portfolio marginal impact
    delta_merged_sharpe: float = 0.0
    delta_merged_return: float = 0.0
    delta_merged_drawdown: float = 0.0
    delta_return_drawdown_ratio: float = 0.0
    delta_merged_turnover: float = 0.0
    official_before_after_improves: bool | None = None
    merged_harm_exception_approved: bool = False
    merged_harm_exception_reason: str | None = None

    # Diversity & strategic gap (0..1)
    structural_novelty: float = 0.5
    dataset_gap_value: float = 0.5
    delay_universe_gap_value: float = 0.5
    dataset_value_score: float = 0.0  # Under-used dataset score (NOT Value Factor)
    value_factor: float = 0.0         # Alpha portfolio uniqueness factor (NOT Dataset Value Score)
    pyramid_gap_value: float = 0.0
    family_submission_count: int = 0
    condition_count: int = 0
    condition_complexity_exception: bool = False

    # Operational / penalty flags
    cost_stress_completed: bool = False
    merged_analysis_completed: bool = False
    complexity_delay_detected: bool = False
    data_quality_verified: bool = True
    parameter_cliff_detected: bool = False
    field_crowding_level: float = 0.0

    # Timing context (only real data; relative_competition_estimate MUST remain None if unknown)
    submitted_today_count: int = 0
    same_family_submitted_today: bool = False
    same_slice_submitted_today: bool = False
    relative_competition_estimate: float | None = None
    operational_ready: bool = True

    # Percentile hints (0..1, optional from internal history)
    sharpe_percentile: float | None = None
    fitness_percentile: float | None = None
    returns_percentile: float | None = None
    margin_percentile: float | None = None
    return_drawdown_percentile: float | None = None

    # Special consultant tags (isolated from standard gates)
    is_atom_alpha: bool = False
    is_power_pool_alpha: bool = False


@dataclass(frozen=True, slots=True)
class IQCCandidateReport:
    alpha_id: str
    idea_family: str
    dataset_category: str
    universe: str
    delay: int
    hard_gates: IQCHardGates
    components: IQCScoreComponents
    penalties: IQCPenalties
    timing: IQCTimingComponents
    submission_utility: float
    os_promotion_score: float
    relative_competition_estimate: float | None
    policy_version: str
    input_snapshot_hash: str
    calculated_at: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def candidate_score(self) -> float:
        raw = self.components.total() - self.penalties.total()
        return round(max(0.0, min(100.0, raw)), 2)

    @property
    def timing_score(self) -> float:
        return round(max(0.0, min(20.0, self.timing.total())), 2)

    @property
    def grade(self) -> IQCGrade:
        if not self.hard_gates.passed:
            return IQCGrade.BLOCKED
        s = self.candidate_score
        if s >= 85.0:
            return IQCGrade.S
        if s >= 75.0:
            return IQCGrade.A
        if s >= 65.0:
            return IQCGrade.B
        if s >= 50.0:
            return IQCGrade.C
        return IQCGrade.D

    @property
    def merged_harm_blocked(self) -> bool:
        return "MERGED_SCORE_NEGATIVE" in self.blockers

    def decision(self) -> IQCCandidateDecision:
        if not self.hard_gates.passed:
            return IQCCandidateDecision.BLOCKED

        score = self.candidate_score
        timing = self.timing_score

        if self.merged_harm_blocked:
            if score < 65.0:
                return IQCCandidateDecision.STRUCTURAL_RESEARCH
            return IQCCandidateDecision.HOLD_OR_REFINE

        if score >= 85.0:
            if timing >= 14.0 and self.submission_utility > 0.0:
                return IQCCandidateDecision.SUBMIT_NOW
            return IQCCandidateDecision.HOLD_FOR_BETTER_DAY

        if score >= 75.0:
            if timing >= 16.0 and self.submission_utility > 0.0:
                return IQCCandidateDecision.SUBMIT_IF_SLOT_AVAILABLE
            return IQCCandidateDecision.HOLD_OR_REFINE

        if score >= 65.0:
            return IQCCandidateDecision.TARGETED_REPAIR

        if score >= 50.0:
            return IQCCandidateDecision.STRUCTURAL_RESEARCH

        return IQCCandidateDecision.ARCHIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_id": self.alpha_id,
            "idea_family": self.idea_family,
            "dataset_category": self.dataset_category,
            "universe": self.universe,
            "delay": self.delay,
            "candidate_score": self.candidate_score,
            "timing_score": self.timing_score,
            "submission_utility": self.submission_utility,
            "os_promotion_score": self.os_promotion_score,
            "grade": self.grade.value,
            "decision": self.decision().value,
            "hard_gates_passed": self.hard_gates.passed,
            "hard_gates": asdict(self.hard_gates),
            "components": {**asdict(self.components), "total": self.components.total()},
            "penalties": {**asdict(self.penalties), "total": self.penalties.total()},
            "timing": {**asdict(self.timing), "total": self.timing.total()},
            "relative_competition_estimate": self.relative_competition_estimate,
            "policy_version": self.policy_version,
            "input_snapshot_hash": self.input_snapshot_hash,
            "calculated_at": self.calculated_at,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "recommended_action": self.decision().value,
        }


def compute_submission_utility(inp: IQCCandidateInput, official_quality_norm: float) -> float:
    merged_gain = _clamp((inp.delta_merged_sharpe + 0.05) / 0.15, -1.0, 1.0)
    os_rob = _clamp(
        0.35 * (inp.temporal_stability if inp.temporal_stability is not None else 0.0)
        + 0.25 * (inp.parameter_stability if inp.parameter_stability is not None else 0.0)
        + 0.20 * (inp.universe_stability if inp.universe_stability is not None else 0.0)
        + 0.20 * (inp.regime_stability if inp.regime_stability is not None else 0.0)
    )
    iq_pot = _clamp(
        0.4 * (1.0 if str(inp.universe).upper() in {"TOP1000", "TOP500", "TOPSP500"} else 0.5)
        + 0.6 * (1.0 - _clamp((inp.self_correlation or 0.5) / 0.70))
    )
    div_gain = _clamp(0.5 * inp.structural_novelty + 0.5 * inp.dataset_gap_value)
    low_corr = 1.0 - _clamp((inp.self_correlation or 0.7) / 0.70)
    gap = _clamp(inp.delay_universe_gap_value)

    turnover_pen = _clamp(max(0.0, inp.delta_merged_turnover) / 0.05)
    dd_pen = _clamp(max(0.0, inp.delta_merged_drawdown) / 0.03)
    crowding_pen = _clamp(1.0 - (1.0 / math.sqrt(max(0, inp.family_submission_count) + 1)))

    utility = (
        0.28 * merged_gain
        + 0.20 * os_rob
        + 0.14 * iq_pot
        + 0.12 * div_gain
        + 0.10 * low_corr
        + 0.08 * gap
        + 0.08 * _clamp(official_quality_norm)
        - 0.16 * turnover_pen
        - 0.14 * dd_pen
        - 0.12 * crowding_pen
    )
    return round(utility, 4)


def compute_os_promotion_score(inp: IQCCandidateInput) -> float:
    t_stab = inp.temporal_stability if inp.temporal_stability is not None else 0.0
    p_stab = inp.parameter_stability if inp.parameter_stability is not None else 0.0
    u_stab = inp.universe_stability if inp.universe_stability is not None else 0.0
    r_stab = inp.regime_stability if inp.regime_stability is not None else 0.0
    clean_test = 0.0 if inp.test_period_contaminated else _clamp(0.5 * t_stab + 0.5 * r_stab)
    dsr = _clamp(inp.dsr_probability if inp.dsr_probability is not None else 0.5)
    cpcv = _clamp(inp.cpcv_positive_ratio if inp.cpcv_positive_ratio is not None else 0.5)
    pbo_risk = _clamp(inp.pbo if inp.pbo is not None else 0.3)
    simplicity = 1.0 if inp.condition_count <= MAX_DEFAULT_CONDITIONS else 0.2
    contam = 1.0 if inp.test_period_contaminated else 0.0

    score = (
        0.25 * clean_test
        + 0.20 * _clamp(t_stab)
        + 0.15 * _clamp(p_stab)
        + 0.10 * _clamp(u_stab)
        + 0.10 * _clamp(r_stab)
        + 0.08 * dsr
        + 0.07 * cpcv
        + 0.05 * simplicity
        - 0.15 * pbo_risk
        - 0.10 * contam
    )
    return round(_clamp(score, 0.0, 1.0), 4)


def evaluate_iqc_candidate(inp: IQCCandidateInput, policy: IQCScoringPolicy | None = None) -> IQCCandidateReport:
    pol = policy or IQCScoringPolicy.for_delay(inp.delay)
    sharpe_val = inp.sharpe
    fitness_val = inp.fitness
    turnover_val = inp.turnover

    hard_gates = IQCHardGates(
        official_provenance=(str(inp.metrics_source).upper() == "WORLDQUANT_BRAIN"),
        simulation_id_present=bool(str(inp.official_simulation_id or "").strip()),
        alpha_id_present=bool(str(inp.brain_alpha_id or "").strip()),
        sharpe_passed=(sharpe_val is not None and float(sharpe_val) > pol.minimum_sharpe),
        fitness_passed=(fitness_val is not None and float(fitness_val) > pol.minimum_fitness),
        turnover_passed=(turnover_val is not None and pol.minimum_turnover < float(turnover_val) < pol.maximum_turnover),
        self_correlation_verified=(inp.self_correlation is not None),
        self_correlation_passed=(
            inp.self_correlation is not None and float(inp.self_correlation) < pol.maximum_self_correlation
        ),
        max_weight_passed=(
            inp.max_instrument_weight is not None and float(inp.max_instrument_weight) < pol.maximum_instrument_weight
        ),
        bias_test_passed=bool(inp.bias_test_passed),
        weight_test_passed=bool(inp.weight_test_passed),
        sub_universe_passed=bool(inp.sub_universe_passed),
        duplicate_free=not bool(inp.duplicate_or_public),
        not_parameter_only_variant=not bool(inp.parameter_only_variant),
        clean_test_period=not bool(inp.test_period_contaminated),
        not_already_submitted=not bool(inp.already_submitted),
        submit_ready=bool(inp.submit_ready),
    )

    # A. Official Quality (18)
    s_pct = inp.sharpe_percentile if inp.sharpe_percentile is not None else _clamp(((sharpe_val or 0.0) - 1.0) / 1.5)
    f_pct = inp.fitness_percentile if inp.fitness_percentile is not None else _clamp(((fitness_val or 0.0) - 0.8) / 1.5)
    r_pct = inp.returns_percentile if inp.returns_percentile is not None else _clamp((inp.returns or 0.0) / 0.25)
    m_pct = inp.margin_percentile if inp.margin_percentile is not None else _clamp((inp.margin or 0.0) / 0.002)
    official_quality = round(7.0 * _clamp(s_pct) + 5.0 * _clamp(f_pct) + 4.0 * _clamp(r_pct) + 2.0 * _clamp(m_pct), 2)

    # B. OS Robustness (18)
    clean_test_pts = 0.0 if inp.test_period_contaminated else 5.0
    temp_pts = 4.0 * _clamp(inp.temporal_stability if inp.temporal_stability is not None else 0.0)
    param_pts = 3.0 * _clamp(inp.parameter_stability if inp.parameter_stability is not None else 0.0)
    univ_pts = 2.0 * _clamp(inp.universe_stability if inp.universe_stability is not None else 0.0)
    reg_pts = 2.0 * _clamp(inp.regime_stability if inp.regime_stability is not None else 0.0)
    dsr_v = _clamp(inp.dsr_probability) if inp.dsr_probability is not None else 0.4
    pbo_v = 1.0 - _clamp(inp.pbo) if inp.pbo is not None else 0.4
    cpcv_v = _clamp(inp.cpcv_positive_ratio) if inp.cpcv_positive_ratio is not None else 0.4
    stat_pts = 2.0 * ((dsr_v + pbo_v + cpcv_v) / 3.0)
    os_robustness = round(clean_test_pts + temp_pts + param_pts + univ_pts + reg_pts + stat_pts, 2)
    if inp.test_period_contaminated:
        os_robustness = min(os_robustness, 5.0)

    # C. Expected Merged Performance Contribution (16)
    d_sharpe = _clamp((inp.delta_merged_sharpe + 0.02) / 0.12)
    d_r2d = _clamp((inp.delta_return_drawdown_ratio + 0.05) / 0.25)
    d_ret = _clamp((inp.delta_merged_return + 0.005) / 0.025)
    d_dd = _clamp((-inp.delta_merged_drawdown + 0.002) / 0.015)
    d_to = _clamp((-inp.delta_merged_turnover + 0.01) / 0.03)
    merged_contribution = round(5.0 * d_sharpe + 5.0 * d_r2d + 2.0 * d_ret + 2.0 * d_dd + 2.0 * d_to, 2)

    # D. Correlation & Uniqueness (12)
    sc = inp.self_correlation if inp.self_correlation is not None else 1.0
    if sc < 0.30:
        corr_pts = 8.0
    elif sc < 0.40:
        corr_pts = 7.0
    elif sc < 0.50:
        corr_pts = 6.0
    elif sc < 0.60:
        corr_pts = 4.0
    elif sc < 0.65:
        corr_pts = 2.0
    elif sc < 0.70:
        corr_pts = 1.0
    else:
        corr_pts = 0.0
    nov_pts = 0.0 if inp.parameter_only_variant else 4.0 * _clamp(inp.structural_novelty)
    correlation_uniqueness = round(corr_pts + nov_pts, 2)

    # E. Return-to-Drawdown (10)
    if inp.return_drawdown_percentile is not None:
        r2d_pct = _clamp(inp.return_drawdown_percentile)
    else:
        ret_v = float(inp.returns or 0.0)
        dd_v = max(abs(float(inp.drawdown or 0.1)), 1e-4)
        r2d_pct = _clamp((ret_v / dd_v) / 3.0)
    return_drawdown = round(10.0 * r2d_pct, 2)

    # F. Turnover Efficiency (8) — turnover <= 0.125 does NOT grant extra points automatically
    to_v = float(turnover_val if turnover_val is not None else 0.5)
    margin_v = float(inp.margin or 0.0)
    if to_v <= TURNOVER_SCORE_FLOOR:
        # Below floor: scored by margin & merged turnover effect, not lower turnover itself
        to_eff = 5.0 * _clamp(margin_v / 0.0015) + 3.0 * _clamp((-inp.delta_merged_turnover + 0.005) / 0.015)
        turnover_efficiency = round(min(7.5, to_eff), 2)
    else:
        to_eff = (
            4.0 * _clamp(1.0 - (to_v - TURNOVER_SCORE_FLOOR) / 0.55)
            + 2.5 * _clamp(margin_v / 0.0015)
            + 1.5 * _clamp((-inp.delta_merged_turnover + 0.005) / 0.015)
        )
        turnover_efficiency = round(to_eff, 2)

    # G. Dataset / Idea Diversity (7)
    marginal_weight = 1.0 / math.sqrt(max(0, inp.family_submission_count) + 1)
    if inp.parameter_only_variant:
        dataset_idea_diversity = 0.0
    else:
        dataset_idea_diversity = round(
            7.0 * _clamp(0.6 * inp.dataset_gap_value + 0.4 * inp.structural_novelty) * marginal_weight,
            2,
        )

    # H. Delay / Universe Strategic Value (5)
    univ_bonus = 0.2 if str(inp.universe).upper() in {"TOP1000", "TOP500", "TOPSP500"} else 0.0
    delay_universe_value = round(5.0 * _clamp(0.8 * inp.delay_universe_gap_value + univ_bonus), 2)

    # I. Dataset Value / Pyramid Value (4) — strictly keep dataset_value_score vs value_factor distinct
    dataset_pyramid_value = round(
        4.0 * _clamp(0.45 * inp.dataset_value_score + 0.35 * inp.pyramid_gap_value + 0.20 * inp.value_factor),
        2,
    )

    # J. Simplicity (2)
    if inp.condition_count > MAX_DEFAULT_CONDITIONS and not inp.condition_complexity_exception:
        simplicity = 0.0
    elif inp.condition_count <= MAX_DEFAULT_CONDITIONS and not inp.parameter_only_variant:
        simplicity = 2.0
    else:
        simplicity = 1.0

    components = IQCScoreComponents(
        official_quality=official_quality,
        os_robustness=os_robustness,
        merged_contribution=merged_contribution,
        correlation_uniqueness=correlation_uniqueness,
        return_drawdown=return_drawdown,
        turnover_efficiency=turnover_efficiency,
        dataset_idea_diversity=dataset_idea_diversity,
        delay_universe_value=delay_universe_value,
        dataset_pyramid_value=dataset_pyramid_value,
        simplicity=simplicity,
    )

    penalties = IQCPenalties(
        test_contamination=15.0 if inp.test_period_contaminated else 0.0,
        parameter_cliff=10.0 if inp.parameter_cliff_detected else 0.0,
        single_year_dominance=10.0 if inp.max_single_year_pnl_share > 0.40 else 0.0,
        idea_crowding=round(min(10.0, max(0, inp.family_submission_count - 1) * 1.5), 2),
        excessive_conditions=(
            min(8.0, float(inp.condition_count - MAX_DEFAULT_CONDITIONS) * 3.0)
            if inp.condition_count > MAX_DEFAULT_CONDITIONS and not inp.condition_complexity_exception
            else 0.0
        ),
        field_crowding=round(5.0 * _clamp(inp.field_crowding_level), 2),
        missing_cost_stress=5.0 if not inp.cost_stress_completed else 0.0,
        missing_merged_analysis=5.0 if not inp.merged_analysis_completed else 0.0,
        complexity_delay=3.0 if inp.complexity_delay_detected else 0.0,
        missing_overfitting_checks=3.0 if (inp.dsr_probability is None or inp.pbo is None) else 0.0,
        data_quality_risk=10.0 if not inp.data_quality_verified else 0.0,
    )

    # Timing Score (20)
    if inp.submitted_today_count <= 0:
        slot_val = 6.0
    elif inp.submitted_today_count == 1:
        slot_val = 5.0
    elif inp.submitted_today_count == 2:
        slot_val = 3.0
    else:
        slot_val = 0.0

    if inp.official_before_after_improves is False or inp.delta_merged_sharpe < 0.0:
        port_timing = 0.0
    elif inp.delta_merged_sharpe >= 0.05:
        port_timing = 5.0
    elif inp.delta_merged_sharpe > 0.0:
        port_timing = 3.5
    else:
        port_timing = 1.5

    if inp.same_family_submitted_today:
        div_timing = 0.0
    elif inp.same_slice_submitted_today:
        div_timing = 2.0
    else:
        div_timing = 4.0

    comp_est = (
        round(3.0 * _clamp(inp.relative_competition_estimate), 2)
        if inp.relative_competition_estimate is not None
        else 0.0
    )
    op_ready = 2.0 if (hard_gates.passed and inp.operational_ready) else 0.0

    timing = IQCTimingComponents(
        daily_slot_value=slot_val,
        portfolio_timing=port_timing,
        diversity_timing=div_timing,
        competition_estimate=comp_est,
        operational_readiness=op_ready,
    )

    blockers = list(hard_gates.failure_reasons())
    warnings: list[str] = []

    if pol.merged_score_negative_blocks and (inp.official_before_after_improves is False or inp.delta_merged_sharpe < 0.0) and not inp.merged_harm_exception_approved:
        blockers.append("MERGED_SCORE_NEGATIVE")
    if inp.condition_count > MAX_DEFAULT_CONDITIONS and not inp.condition_complexity_exception:
        warnings.append("EXCESSIVE_CONDITIONS_OVER_3")
    if str(inp.neutralization).upper() == "NONE" and "group_neutralize" not in inp.expression.lower() and "group_normalize" not in inp.expression.lower():
        warnings.append("UNBALANCED_NEUTRALIZATION_NONE")
    if not inp.merged_analysis_completed:
        warnings.append("MERGED_ANALYSIS_MISSING")
    if turnover_val is not None and float(turnover_val) >= 0.40:
        warnings.append("TURNOVER_ABOVE_PREFERRED_RANGE")
    if turnover_val is not None and float(turnover_val) >= 0.30:
        warnings.append("TRANSACTION_COST_STRESS_REQUIRED")
    if any(v is None for v in (inp.temporal_stability, inp.parameter_stability, inp.universe_stability, inp.regime_stability)):
        warnings.append("MISSING_ROBUSTNESS_DATA")
    if inp.relative_competition_estimate is None:
        warnings.append("RELATIVE_COMPETITION_UNAVAILABLE")
    if inp.is_atom_alpha or inp.is_power_pool_alpha:
        warnings.append("CONSULTANT_SPECIAL_LANE_ISOLATED")

    sub_utility = compute_submission_utility(inp, official_quality / 18.0)
    os_promo = compute_os_promotion_score(inp)
    snap_hash = compute_snapshot_hash(asdict(inp))

    return IQCCandidateReport(
        alpha_id=inp.alpha_id,
        idea_family=inp.idea_family,
        dataset_category=inp.dataset_category,
        universe=inp.universe,
        delay=inp.delay,
        hard_gates=hard_gates,
        components=components,
        penalties=penalties,
        timing=timing,
        submission_utility=sub_utility,
        os_promotion_score=os_promo,
        relative_competition_estimate=inp.relative_competition_estimate,
        policy_version=pol.policy_version,
        input_snapshot_hash=snap_hash,
        calculated_at=utc_now_iso(),
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def plan_daily_submissions(
    candidates: Sequence[IQCCandidateInput | IQCCandidateReport],
    *,
    max_daily: int = 3,
    max_per_family: int = 1,
    submitted_today: int = 0,
) -> list[IQCCandidateReport]:
    """Select up to max_daily (default 3) candidates for today's EST submission budget."""
    reports = [
        c if isinstance(c, IQCCandidateReport) else evaluate_iqc_candidate(c)
        for c in candidates
    ]
    eligible = [
        r for r in reports
        if r.hard_gates.passed
        and r.decision() in {IQCCandidateDecision.SUBMIT_NOW, IQCCandidateDecision.SUBMIT_IF_SLOT_AVAILABLE}
        and r.submission_utility > 0.0
    ]
    eligible.sort(
        key=lambda r: (
            1 if r.decision() == IQCCandidateDecision.SUBMIT_NOW else 0,
            r.submission_utility,
            r.candidate_score,
            r.timing_score,
        ),
        reverse=True,
    )

    remaining_slots = max(0, min(max_daily, 3) - max(0, submitted_today))
    selected: list[IQCCandidateReport] = []
    family_counts: dict[str, int] = {}
    slice_counts: dict[tuple[str, str, int], int] = {}

    for r in eligible:
        if len(selected) >= remaining_slots:
            break
        fam_count = family_counts.get(r.idea_family, 0)
        if fam_count >= max_per_family:
            continue
        sl_key = (r.dataset_category, r.universe, r.delay)
        if slice_counts.get(sl_key, 0) >= max_daily - 1 and max_daily >= 2:
            # Prevent one dataset/universe/delay slice from occupying all slots
            continue
        selected.append(r)
        family_counts[r.idea_family] = fam_count + 1
        slice_counts[sl_key] = slice_counts.get(sl_key, 0) + 1

    return selected


def build_iqc_input_from_trusted_record(
    record: Mapping[str, Any],
    *,
    submitted_today_count: int = 0,
) -> IQCCandidateInput:
    """Build IQCCandidateInput strictly from trusted backend vault/evidence records."""
    aid = str(record.get("brain_alpha_id") or record.get("id") or record.get("alpha_id") or "").strip()
    sim_id = str(record.get("official_simulation_id") or record.get("brain_simulation_id") or "").strip() or None
    brain_id = str(record.get("brain_alpha_id") or "").strip() or None
    src = str(record.get("metrics_source") or ("WORLDQUANT_BRAIN" if brain_id else "LOCAL_PROXY")).upper()
    status_val = str(record.get("status") or "").upper()

    def _f(key: str, default: float | None = None) -> float | None:
        v = record.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    return IQCCandidateInput(
        alpha_id=aid or "unknown",
        expression=str(record.get("expression") or ""),
        region=str(record.get("region") or "USA"),
        universe=str(record.get("universe") or "TOP3000"),
        delay=int(record.get("delay") if record.get("delay") is not None else 1),
        neutralization=str(record.get("neutralization") or "SUBINDUSTRY").upper(),
        dataset_category=str(record.get("dataset_category") or record.get("domain") or "PRICE_VOLUME"),
        idea_family=str(record.get("idea_family") or record.get("hypothesis_family") or record.get("domain") or "GENERAL"),
        metrics_source=src,
        official_simulation_id=sim_id,
        brain_alpha_id=brain_id,
        sharpe=_f("sharpe"),
        fitness=_f("fitness"),
        returns=_f("returns"),
        turnover=_f("turnover"),
        margin=_f("margin"),
        drawdown=_f("drawdown"),
        self_correlation=_f("self_correlation"),
        prod_correlation=_f("prod_correlation"),
        max_instrument_weight=_f("max_instrument_weight"),
        bias_test_passed=bool(record.get("bias_test_passed", False)),
        weight_test_passed=bool(record.get("weight_test_passed", False)),
        sub_universe_passed=bool(record.get("sub_universe_passed", False)),
        duplicate_or_public=bool(record.get("duplicate_expression") or record.get("duplicate") or False),
        parameter_only_variant=bool(record.get("parameter_only_variant", False)),
        test_period_contaminated=bool(record.get("test_period_contaminated", False)),
        already_submitted=bool(record.get("submitted_at") or status_val == "SUBMITTED"),
        submit_ready=bool(record.get("submit_ready", False)),
        temporal_stability=float(_f("temporal_stability", 0.5) or 0.5),
        parameter_stability=float(_f("parameter_stability", 0.5) or 0.5),
        universe_stability=float(_f("universe_stability", 0.5) or 0.5),
        regime_stability=float(_f("regime_stability", 0.5) or 0.5),
        dsr_probability=_f("dsr_probability"),
        pbo=_f("pbo"),
        cpcv_positive_ratio=_f("cpcv_positive_ratio"),
        delta_merged_sharpe=float(_f("delta_merged_sharpe", 0.0) or 0.0),
        delta_merged_return=float(_f("delta_merged_return", 0.0) or 0.0),
        delta_merged_drawdown=float(_f("delta_merged_drawdown", 0.0) or 0.0),
        delta_return_drawdown_ratio=float(_f("delta_return_drawdown_ratio", 0.0) or 0.0),
        delta_merged_turnover=float(_f("delta_merged_turnover", 0.0) or 0.0),
        official_before_after_improves=record.get("official_before_after_improves"),
        structural_novelty=float(_f("structural_novelty", 0.5) or 0.5),
        dataset_gap_value=float(_f("dataset_gap_value", 0.5) or 0.5),
        delay_universe_gap_value=float(_f("delay_universe_gap_value", 0.5) or 0.5),
        dataset_value_score=float(_f("dataset_value_score", 0.0) or 0.0),
        value_factor=float(_f("value_factor", 0.0) or 0.0),
        pyramid_gap_value=float(_f("pyramid_gap_value", 0.0) or 0.0),
        family_submission_count=int(record.get("family_submission_count") or 0),
        condition_count=int(record["condition_count"]) if record.get("condition_count") is not None else count_conditional_ops(str(record.get("expression") or "")),
        cost_stress_completed=bool(record.get("cost_stress_completed", False)),
        merged_analysis_completed=bool(record.get("merged_analysis_completed", False)),
        complexity_delay_detected=bool(record.get("complexity_delay_detected", False)),
        data_quality_verified=bool(record.get("data_quality_verified", True)),
        submitted_today_count=submitted_today_count,
        relative_competition_estimate=None,  # Never fabricate unless real historical data exists
    )


IQC_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS iqc_candidate_scores (
    alpha_id TEXT PRIMARY KEY,
    candidate_score REAL NOT NULL,
    timing_score REAL NOT NULL,
    submission_utility REAL NOT NULL,
    grade TEXT NOT NULL,
    decision TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    input_snapshot_hash TEXT NOT NULL,
    calculated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iqc_calculation_audits (
    audit_id TEXT PRIMARY KEY,
    alpha_id TEXT NOT NULL,
    candidate_score REAL NOT NULL,
    timing_score REAL NOT NULL,
    submission_utility REAL NOT NULL,
    grade TEXT NOT NULL,
    decision TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    input_snapshot_hash TEXT NOT NULL,
    calculated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_iqc_audits_alpha_time
ON iqc_calculation_audits(alpha_id, calculated_at DESC);
"""


class IQCScoringRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.database_path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def migrate(self) -> None:
        with self.connection() as con:
            con.executescript(IQC_SCHEMA_SQL)
            ensure_schema_from_ddl(con, IQC_SCHEMA_SQL)

    def save_report_with_audit(self, report: IQCCandidateReport) -> str:
        self.migrate()
        payload = report.to_dict()
        breakdown_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        audit_id = f"iqc_audit_{uuid.uuid4().hex[:16]}"
        with self.connection() as con:
            con.execute(
                """
                INSERT INTO iqc_candidate_scores (
                    alpha_id, candidate_score, timing_score, submission_utility,
                    grade, decision, breakdown_json, policy_version,
                    input_snapshot_hash, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alpha_id) DO UPDATE SET
                    candidate_score = excluded.candidate_score,
                    timing_score = excluded.timing_score,
                    submission_utility = excluded.submission_utility,
                    grade = excluded.grade,
                    decision = excluded.decision,
                    breakdown_json = excluded.breakdown_json,
                    policy_version = excluded.policy_version,
                    input_snapshot_hash = excluded.input_snapshot_hash,
                    calculated_at = excluded.calculated_at
                """,
                (
                    report.alpha_id,
                    report.candidate_score,
                    report.timing_score,
                    report.submission_utility,
                    report.grade.value,
                    report.decision().value,
                    breakdown_json,
                    report.policy_version,
                    report.input_snapshot_hash,
                    report.calculated_at,
                ),
            )
            con.execute(
                """
                INSERT INTO iqc_calculation_audits (
                    audit_id, alpha_id, candidate_score, timing_score, submission_utility,
                    grade, decision, breakdown_json, policy_version,
                    input_snapshot_hash, calculated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    report.alpha_id,
                    report.candidate_score,
                    report.timing_score,
                    report.submission_utility,
                    report.grade.value,
                    report.decision().value,
                    breakdown_json,
                    report.policy_version,
                    report.input_snapshot_hash,
                    report.calculated_at,
                ),
            )
        return audit_id

    def get_score(self, alpha_id: str) -> dict[str, Any] | None:
        self.migrate()
        with self.connection() as con:
            row = con.execute(
                "SELECT * FROM iqc_candidate_scores WHERE alpha_id = ?",
                (alpha_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_audits(self, alpha_id: str) -> list[dict[str, Any]]:
        self.migrate()
        with self.connection() as con:
            rows = con.execute(
                "SELECT * FROM iqc_calculation_audits WHERE alpha_id = ? ORDER BY calculated_at DESC",
                (alpha_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_report_if_changed(self, report: IQCCandidateReport) -> tuple[str | None, bool]:
        """Idempotent audit persistence: reuse existing record if policy_version and input_snapshot_hash match."""
        existing = self.get_score(report.alpha_id)
        if (
            existing is not None
            and existing.get("policy_version") == report.policy_version
            and existing.get("input_snapshot_hash") == report.input_snapshot_hash
        ):
            return None, False
        audit_id = self.save_report_with_audit(report)
        return audit_id, True
