from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence


class EvidenceSource(StrEnum):
    BRAIN_OFFICIAL = "BRAIN_OFFICIAL"
    BRAIN_LABS = "BRAIN_LABS"
    LOCAL_REPLICATION = "LOCAL_REPLICATION"
    INFERRED_PROXY = "INFERRED_PROXY"
    UNKNOWN = "UNKNOWN"


class PnLJumpDiagnosis(StrEnum):
    NAN_TRANSITION = "NAN_TRANSITION"
    RAPID_ALPHA_CHANGE = "RAPID_ALPHA_CHANGE"
    WEIGHT_CONCENTRATION = "WEIGHT_CONCENTRATION"
    MARKET_EXPOSURE = "MARKET_EXPOSURE"
    REGIME_FAILURE = "REGIME_FAILURE"
    BACKFILL_STALENESS = "BACKFILL_STALENESS"
    MULTI_SIGNAL_OVERFIT = "MULTI_SIGNAL_OVERFIT"
    TRANSACTION_COST_RISK = "TRANSACTION_COST_RISK"
    VECTOR_NOT_OBSERVED = "VECTOR_NOT_OBSERVED"


class ImprovementPriority(StrEnum):
    P0_BLOCKER = "P0_BLOCKER"
    P1_HIGH = "P1_HIGH"
    P2_MEDIUM = "P2_MEDIUM"
    P3_RESEARCH = "P3_RESEARCH"
    P6_ARCHIVE = "P6_ARCHIVE"


class TurnoverBand(StrEnum):
    FAILED = "FAILED"
    BELOW_FITNESS_FLOOR = "BELOW_FITNESS_FLOOR"
    PREFERRED = "PREFERRED"
    ACCEPTABLE = "ACCEPTABLE"
    HIGH_BUT_ELIGIBLE = "HIGH_BUT_ELIGIBLE"


@dataclass(frozen=True, slots=True)
class PnLJumpPolicy:
    delay_0_minimum_sharpe: float = 2.0
    delay_1_minimum_sharpe: float = 1.25
    minimum_fitness: float = 1.0
    minimum_turnover: float = 0.01
    fitness_turnover_floor: float = 0.125
    preferred_turnover: float = 0.30
    acceptable_turnover: float = 0.40
    maximum_turnover: float = 0.70
    maximum_instrument_weight: float = 0.10
    preferred_truncation_minimum: float = 0.05
    preferred_truncation_maximum: float = 0.10
    nan_transition_warning: float = 0.05
    rapid_alpha_change_warning: float = 0.30
    market_beta_warning: float = 0.30
    largest_year_pnl_share_maximum: float = 0.40
    maximum_backfill_staleness_days: int = 60
    maximum_conditions: int = 3
    maximum_primary_signals: int = 1
    maximum_confirmation_signals: int = 1
    maximum_risk_control_signals: int = 1

    def minimum_sharpe(self, delay: int) -> float:
        if delay == 0:
            return self.delay_0_minimum_sharpe
        if delay == 1:
            return self.delay_1_minimum_sharpe
        raise ValueError(f"Unsupported delay: {delay}")


@dataclass(frozen=True, slots=True)
class PnLJumpInput:
    alpha_id: str
    delay: int
    sharpe: float | None
    fitness: float | None
    returns: float | None
    turnover: float | None
    margin: float | None
    drawdown: float | None
    self_correlation: float | None
    max_instrument_weight: float | None
    truncation: float | None

    vector_evidence_source: EvidenceSource = EvidenceSource.UNKNOWN
    alpha_vector_observed: bool = False
    nan_transition_rate: float | None = None
    alpha_change_rate: float | None = None
    market_beta: float | None = None
    largest_year_pnl_share: float | None = None

    field_coverage: float | None = None
    backfilled_coverage: float | None = None
    backfill_window_days: int | None = None
    field_staleness_days: int | None = None
    point_in_time_verified: bool | None = None

    primary_signal_count: int = 1
    confirmation_signal_count: int = 0
    risk_control_signal_count: int = 0
    condition_count: int = 0

    cost_stress_completed: bool = False
    cost_stress_passed: bool | None = None
    neutralization_tested: bool = False
    neutralized_return_retention: float | None = None


@dataclass(frozen=True, slots=True)
class ImprovementAction:
    action: str
    priority: ImprovementPriority
    score: float
    reason: str
    required_validation: tuple[str, ...]
    prohibited_shortcuts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PnLJumpReport:
    alpha_id: str
    hard_blockers: tuple[str, ...]
    diagnoses: tuple[PnLJumpDiagnosis, ...]
    turnover_band: TurnoverBand | None
    smoothness_risk_score: float
    improvement_score: float
    actions: tuple[ImprovementAction, ...]
    warnings: tuple[str, ...]
    evidence_source: EvidenceSource

    @property
    def blocked(self) -> bool:
        return bool(self.hard_blockers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def classify_turnover(value: float | None, policy: PnLJumpPolicy) -> TurnoverBand | None:
    turnover = _finite(value)
    if turnover is None:
        return None
    if not (policy.minimum_turnover < turnover < policy.maximum_turnover):
        return TurnoverBand.FAILED
    if turnover < policy.fitness_turnover_floor:
        return TurnoverBand.BELOW_FITNESS_FLOOR
    if turnover < policy.preferred_turnover:
        return TurnoverBand.PREFERRED
    if turnover < policy.acceptable_turnover:
        return TurnoverBand.ACCEPTABLE
    return TurnoverBand.HIGH_BUT_ELIGIBLE


def evaluate_hard_blockers(x: PnLJumpInput, p: PnLJumpPolicy) -> tuple[str, ...]:
    blockers: list[str] = []
    minimum_sharpe = p.minimum_sharpe(x.delay)
    if x.sharpe is None or x.sharpe <= minimum_sharpe:
        blockers.append("SHARPE_NOT_PASSED")
    if x.fitness is None or x.fitness <= p.minimum_fitness:
        blockers.append("FITNESS_NOT_PASSED")
    if classify_turnover(x.turnover, p) in {None, TurnoverBand.FAILED}:
        blockers.append("TURNOVER_NOT_PASSED")
    if x.self_correlation is None:
        blockers.append("SELF_CORRELATION_NOT_VERIFIED")
    elif x.self_correlation >= 0.70:
        blockers.append("SELF_CORRELATION_NOT_PASSED")
    return tuple(blockers)


def diagnose(x: PnLJumpInput, p: PnLJumpPolicy) -> set[PnLJumpDiagnosis]:
    result: set[PnLJumpDiagnosis] = set()
    if not x.alpha_vector_observed:
        result.add(PnLJumpDiagnosis.VECTOR_NOT_OBSERVED)
    if x.nan_transition_rate is not None and x.nan_transition_rate > p.nan_transition_warning:
        result.add(PnLJumpDiagnosis.NAN_TRANSITION)
    if x.alpha_change_rate is not None and x.alpha_change_rate > p.rapid_alpha_change_warning:
        result.add(PnLJumpDiagnosis.RAPID_ALPHA_CHANGE)
    if x.max_instrument_weight is not None and x.max_instrument_weight >= p.maximum_instrument_weight:
        result.add(PnLJumpDiagnosis.WEIGHT_CONCENTRATION)
    if x.market_beta is not None and abs(x.market_beta) > p.market_beta_warning:
        result.add(PnLJumpDiagnosis.MARKET_EXPOSURE)
    if x.largest_year_pnl_share is not None and x.largest_year_pnl_share > p.largest_year_pnl_share_maximum:
        result.add(PnLJumpDiagnosis.REGIME_FAILURE)
    if (
        x.backfill_window_days is not None
        and x.field_staleness_days is not None
        and x.field_staleness_days > p.maximum_backfill_staleness_days
    ):
        result.add(PnLJumpDiagnosis.BACKFILL_STALENESS)
    if (
        x.primary_signal_count > p.maximum_primary_signals
        or x.confirmation_signal_count > p.maximum_confirmation_signals
        or x.risk_control_signal_count > p.maximum_risk_control_signals
        or x.condition_count > p.maximum_conditions
    ):
        result.add(PnLJumpDiagnosis.MULTI_SIGNAL_OVERFIT)
    if x.turnover is not None and x.turnover >= p.preferred_turnover and not x.cost_stress_completed:
        result.add(PnLJumpDiagnosis.TRANSACTION_COST_RISK)
    return result


def _risk_score(x: PnLJumpInput, diagnoses: set[PnLJumpDiagnosis], p: PnLJumpPolicy) -> float:
    score = 0.0
    score += 0.20 * _clamp((x.nan_transition_rate or 0.0) / max(p.nan_transition_warning, 1e-9))
    score += 0.20 * _clamp((x.alpha_change_rate or 0.0) / max(p.rapid_alpha_change_warning, 1e-9))
    score += 0.20 * _clamp((x.max_instrument_weight or 0.0) / p.maximum_instrument_weight)
    score += 0.15 * _clamp(abs(x.market_beta or 0.0) / p.market_beta_warning)
    score += 0.15 * _clamp((x.largest_year_pnl_share or 0.0) / p.largest_year_pnl_share_maximum)
    score += 0.10 if PnLJumpDiagnosis.MULTI_SIGNAL_OVERFIT in diagnoses else 0.0
    return round(_clamp(score) * 100.0, 3)


def _improvement_score(x: PnLJumpInput, risk_score: float) -> float:
    quality = 0.0
    if x.sharpe is not None:
        quality += 0.30 * _clamp(x.sharpe / 2.5)
    if x.fitness is not None:
        quality += 0.25 * _clamp(x.fitness / 2.0)
    if x.returns is not None:
        quality += 0.20 * _clamp(abs(x.returns) / 0.20)
    if x.margin is not None:
        quality += 0.15 * _clamp(x.margin / 0.0015)
    if x.self_correlation is not None:
        quality += 0.10 * _clamp(1.0 - x.self_correlation / 0.70)
    return round(_clamp(quality - 0.35 * risk_score / 100.0) * 100.0, 3)


def recommend_actions(x: PnLJumpInput, diagnoses: set[PnLJumpDiagnosis], p: PnLJumpPolicy) -> list[ImprovementAction]:
    actions: list[ImprovementAction] = []
    if PnLJumpDiagnosis.WEIGHT_CONCENTRATION in diagnoses:
        actions.append(ImprovementAction(
            "TEST_TRUNCATION_RANK_WINSORIZE", ImprovementPriority.P0_BLOCKER, 0.99,
            "单一股票权重可能放大 PnL 跳变。", ("WEIGHT_TEST", "MAX_WEIGHT_LT_0_10", "RETURN_NON_DEGRADATION"),
            ("DO_NOT_INCREASE_CONCENTRATION",),
        ))
    if PnLJumpDiagnosis.BACKFILL_STALENESS in diagnoses:
        actions.append(ImprovementAction(
            "REDUCE_OR_REMOVE_BACKFILL", ImprovementPriority.P0_BLOCKER, 0.98,
            "Backfill 使资料超过允许新鲜度。", ("POINT_IN_TIME", "FIELD_FRESHNESS", "COVERAGE_ABLATION"),
            ("DO_NOT_HIDE_LOW_COVERAGE_WITH_LONG_BACKFILL",),
        ))
    if PnLJumpDiagnosis.NAN_TRANSITION in diagnoses:
        actions.append(ImprovementAction(
            "TEST_CAUSAL_BACKFILL", ImprovementPriority.P1_HIGH, 0.94,
            "频繁 NaN/非 NaN 切换可能造成权重与 PnL 跳变。", ("POINT_IN_TIME", "BACKFILL_WINDOW", "STALENESS", "SUB_UNIVERSE"),
            ("NO_FORWARD_FILL_FROM_FUTURE",),
        ))
    if PnLJumpDiagnosis.RAPID_ALPHA_CHANGE in diagnoses:
        actions.extend([
            ImprovementAction("TEST_DECAY", ImprovementPriority.P1_HIGH, 0.88,
                "Alpha 随时间变化过快，测试 BRAIN Decay 设置。", ("RETURN", "SHARPE", "TURNOVER", "MARGIN")),
            ImprovementAction("TEST_SIGNAL_AVERAGE", ImprovementPriority.P2_MEDIUM, 0.81,
                "测试可解释的时间平均以降低噪音。", ("PARAMETER_STABILITY", "INFORMATION_DELAY")),
        ])
    if PnLJumpDiagnosis.MARKET_EXPOSURE in diagnoses:
        actions.append(ImprovementAction(
            "TEST_NEUTRALIZATION", ImprovementPriority.P1_HIGH, 0.91,
            "市场方向曝险可能导致特定年份大回撤。", ("SHARPE", "DRAWDOWN", "RETURN_RETENTION"),
            ("DO_NOT_ASSUME_NEUTRALIZATION_ALWAYS_HELPS",),
        ))
    if PnLJumpDiagnosis.REGIME_FAILURE in diagnoses:
        actions.append(ImprovementAction(
            "RUN_REGIME_AND_YEAR_ATTRIBUTION", ImprovementPriority.P2_MEDIUM, 0.86,
            "单一年份 PnL 贡献过高，需确认市场期或事件集中。", ("YEARLY_SHARPE", "REGIME_STABILITY", "SINGLE_YEAR_SHARE"),
        ))
    if PnLJumpDiagnosis.MULTI_SIGNAL_OVERFIT in diagnoses:
        actions.append(ImprovementAction(
            "SPLIT_INTO_ATOMIC_ALPHAS", ImprovementPriority.P0_BLOCKER, 0.97,
            "过多讯号或条件提高 IS 过拟合风险，应拆成原子 Alpha 并在组合层处理。", ("ABLATION", "CLEAN_TEST", "MERGED_PERFORMANCE"),
            ("DO_NOT_ADD_MORE_CONDITIONS",),
        ))
    if PnLJumpDiagnosis.TRANSACTION_COST_RISK in diagnoses:
        actions.append(ImprovementAction(
            "RUN_COST_STRESS", ImprovementPriority.P1_HIGH, 0.89,
            "Turnover 高于成本偏好区，必须验证 1x/1.5x/2x 成本。", ("COST_1X", "COST_1_5X", "COST_2X", "MARGIN"),
        ))
    if PnLJumpDiagnosis.VECTOR_NOT_OBSERVED in diagnoses:
        actions.append(ImprovementAction(
            "USE_PROXIES_AND_LABEL_EVIDENCE", ImprovementPriority.P3_RESEARCH, 0.70,
            "BRAIN 结果未显示完整 Alpha vector，不得声称精确知道每日权重变化。", ("EVIDENCE_SOURCE", "MAX_WEIGHT", "COVERAGE", "LONG_SHORT_COUNT"),
        ))
    return sorted(actions, key=lambda item: item.score, reverse=True)


def evaluate_pnl_jump(x: PnLJumpInput, policy: PnLJumpPolicy | None = None) -> PnLJumpReport:
    p = policy or PnLJumpPolicy()
    diagnoses = diagnose(x, p)
    risks = _risk_score(x, diagnoses, p)
    warnings: list[str] = []
    if x.turnover is not None and x.turnover >= p.acceptable_turnover:
        warnings.append("TURNOVER_ABOVE_ACCEPTABLE_RANGE")
    elif x.turnover is not None and x.turnover >= p.preferred_turnover:
        warnings.append("TURNOVER_ABOVE_PREFERRED_RANGE")
    if not x.alpha_vector_observed:
        warnings.append("ALPHA_VECTOR_NOT_OFFICIALLY_OBSERVED")
    if x.point_in_time_verified is not True and x.backfill_window_days:
        warnings.append("BACKFILL_POINT_IN_TIME_NOT_VERIFIED")
    return PnLJumpReport(
        alpha_id=x.alpha_id,
        hard_blockers=evaluate_hard_blockers(x, p),
        diagnoses=tuple(sorted(diagnoses, key=lambda item: item.value)),
        turnover_band=classify_turnover(x.turnover, p),
        smoothness_risk_score=risks,
        improvement_score=_improvement_score(x, risks),
        actions=tuple(recommend_actions(x, diagnoses, p)),
        warnings=tuple(dict.fromkeys(warnings)),
        evidence_source=x.vector_evidence_source,
    )


def build_pnl_jump_input_from_trusted_record(record: Mapping[str, Any]) -> PnLJumpInput:
    """Build PnLJumpInput strictly from trusted backend Vault/Evidence records."""
    aid = str(record.get("brain_alpha_id") or record.get("id") or record.get("alpha_id") or "unknown").strip()
    expr = str(record.get("expression") or "")
    def _f(k: str) -> float | None:
        v = record.get(k)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    cond_cnt = record.get("condition_count")
    if cond_cnt is None:
        el = expr.lower()
        cond_cnt = el.count("if_else(") + el.count("trade_when(") + el.count("?")
    vec_obs = bool(record.get("alpha_vector_observed", False))
    raw_src = str(record.get("vector_evidence_source") or ("BRAIN_OFFICIAL" if vec_obs else "INFERRED_PROXY")).upper()
    try:
        vec_src = EvidenceSource(raw_src)
    except ValueError:
        vec_src = EvidenceSource.UNKNOWN
    return PnLJumpInput(
        alpha_id=aid,
        delay=int(record.get("delay") if record.get("delay") is not None else 1),
        sharpe=_f("sharpe"),
        fitness=_f("fitness"),
        returns=_f("returns"),
        turnover=_f("turnover"),
        margin=_f("margin"),
        drawdown=_f("drawdown"),
        self_correlation=_f("self_correlation"),
        max_instrument_weight=_f("max_instrument_weight"),
        truncation=_f("truncation"),
        vector_evidence_source=vec_src,
        alpha_vector_observed=vec_obs,
        nan_transition_rate=_f("nan_transition_rate"),
        alpha_change_rate=_f("alpha_change_rate"),
        market_beta=_f("market_beta"),
        largest_year_pnl_share=_f("largest_year_pnl_share"),
        field_coverage=_f("field_coverage"),
        backfilled_coverage=_f("backfilled_coverage"),
        backfill_window_days=int(record["backfill_window_days"]) if record.get("backfill_window_days") is not None else None,
        field_staleness_days=int(record["field_staleness_days"]) if record.get("field_staleness_days") is not None else None,
        point_in_time_verified=bool(record.get("point_in_time_verified", True)),
        condition_count=int(cond_cnt),
        primary_signal_count=int(record.get("primary_signal_count") or 1),
        confirmation_signal_count=int(record.get("confirmation_signal_count") or 0),
        risk_control_signal_count=int(record.get("risk_control_signal_count") or 0),
        cost_stress_completed=bool(record.get("cost_stress_completed", False)),
        cost_stress_passed=record.get("cost_stress_passed"),
    )
