"""Neutralization-Aware Architecture: Expression vs Setting Layer, Diagnostics, and Multi-Objective Ablation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Sequence


class NeutralizationMode(StrEnum):
    NONE = "NONE"
    MARKET = "MARKET"
    SECTOR = "SECTOR"
    INDUSTRY = "INDUSTRY"
    SUBINDUSTRY = "SUBINDUSTRY"
    SLOW_FACTORS = "SLOW_FACTORS"
    FAST_FACTORS = "FAST_FACTORS"
    SLOW_FAST_FACTORS = "SLOW_FAST_FACTORS"


class NeutralizationLayer(StrEnum):
    EXPRESSION = "EXPRESSION"
    SIMULATION_SETTING = "SIMULATION_SETTING"
    BOTH = "BOTH"
    UNKNOWN = "UNKNOWN"


class NeutralizationDiagnosis(StrEnum):
    MISSING_FINAL_NEUTRALIZATION = "MISSING_FINAL_NEUTRALIZATION"
    LONG_SHORT_IMBALANCE = "LONG_SHORT_IMBALANCE"
    MARKET_BETA_EXPOSURE = "MARKET_BETA_EXPOSURE"
    DOUBLE_NEUTRALIZATION = "DOUBLE_NEUTRALIZATION"
    OVER_NEUTRALIZED = "OVER_NEUTRALIZED"
    RISK_FACTOR_DEPENDENCY = "RISK_FACTOR_DEPENDENCY"
    CONSTANT_SIGNAL_COLLAPSE = "CONSTANT_SIGNAL_COLLAPSE"
    GROUP_CHOICE_UNVERIFIED = "GROUP_CHOICE_UNVERIFIED"


@dataclass(frozen=True, slots=True)
class NeutralizationPolicy:
    maximum_long_short_imbalance: float = 0.20
    maximum_market_beta: float = 0.30
    minimum_return_retention: float = 0.70
    risk_factor_dependency_retention: float = 0.25
    maximum_sharpe_degradation: float = 0.10
    maximum_drawdown_increase: float = 0.0
    minimum_os_stability: float = 0.60
    double_neutralization_penalty: float = 0.05


@dataclass(frozen=True, slots=True)
class NeutralizationConfiguration:
    simulation_mode: NeutralizationMode
    expression_group_neutralize_count: int = 0
    expression_groups: tuple[str, ...] = ()
    outermost_group_neutralize: bool = False
    post_neutralization_weight_transform: bool = False
    decay: int = 0
    truncation: float = 0.0
    layer: NeutralizationLayer = NeutralizationLayer.UNKNOWN
    long_count: int | None = None
    short_count: int | None = None
    market_beta: float | None = None
    sector_exposure: float | None = None
    industry_exposure: float | None = None


@dataclass(frozen=True, slots=True)
class NeutralizationMetrics:
    sharpe: float
    returns: float
    drawdown: float
    fitness: float
    turnover: float
    margin: float
    self_correlation: float | None
    sub_universe_passed: bool | None
    os_stability: float | None
    market_beta: float | None
    long_count: int | None = None
    short_count: int | None = None


@dataclass(frozen=True, slots=True)
class NeutralizationTrial:
    trial_id: str
    config: NeutralizationConfiguration
    metrics: NeutralizationMetrics
    official: bool = True


@dataclass(frozen=True, slots=True)
class NeutralizationScoreBreakdown:
    sharpe: float
    return_retention: float
    return_drawdown: float
    fitness: float
    os_stability: float
    low_correlation: float
    sub_universe: float
    exposure_reduction: float
    penalties: float

    @property
    def total(self) -> float:
        return max(
            0.0,
            min(
                100.0,
                self.sharpe
                + self.return_retention
                + self.return_drawdown
                + self.fitness
                + self.os_stability
                + self.low_correlation
                + self.sub_universe
                + self.exposure_reduction
                - self.penalties,
            ),
        )


@dataclass(frozen=True, slots=True)
class NeutralizationTrialResult:
    trial_id: str
    mode: NeutralizationMode
    diagnoses: tuple[NeutralizationDiagnosis, ...]
    score: NeutralizationScoreBreakdown
    succeeded: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeutralizationAblationReport:
    baseline_trial_id: str
    ranked_trials: tuple[NeutralizationTrialResult, ...]
    selected_trial_id: str | None
    selected_mode: NeutralizationMode | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def has_effective_final_neutralization(config: NeutralizationConfiguration) -> bool:
    if config.simulation_mode != NeutralizationMode.NONE:
        return True
    return bool(
        config.outermost_group_neutralize
        and config.expression_group_neutralize_count > 0
        and not config.post_neutralization_weight_transform
    )


def expression_setting_equivalent(config: NeutralizationConfiguration) -> bool:
    return bool(
        config.simulation_mode == NeutralizationMode.NONE
        and config.outermost_group_neutralize
        and config.expression_group_neutralize_count == 1
        and not config.post_neutralization_weight_transform
        and config.decay == 0
        and config.truncation == 0
        and len(config.expression_groups) == 1
    )


def long_short_imbalance(
    config: NeutralizationConfiguration,
    metrics: NeutralizationMetrics | None = None,
) -> float | None:
    long_count = metrics.long_count if metrics and metrics.long_count is not None else config.long_count
    short_count = metrics.short_count if metrics and metrics.short_count is not None else config.short_count
    if long_count is None or short_count is None:
        return None
    return abs(long_count - short_count) / max(long_count + short_count, 1)


def neutralized_vector(values: Sequence[float], groups: Sequence[str]) -> list[float]:
    if len(values) != len(groups):
        raise ValueError("values and groups must have equal length")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for value, group in zip(values, groups):
        sums[group] = sums.get(group, 0.0) + value
        counts[group] = counts.get(group, 0) + 1
    return [value - sums[group] / counts[group] for value, group in zip(values, groups)]


def diagnose_neutralization(
    config: NeutralizationConfiguration,
    baseline: NeutralizationMetrics,
    after: NeutralizationMetrics,
    policy: NeutralizationPolicy | None = None,
    *,
    constant_expression: bool = False,
) -> set[NeutralizationDiagnosis]:
    p = policy or NeutralizationPolicy()
    diagnoses: set[NeutralizationDiagnosis] = set()
    if not has_effective_final_neutralization(config):
        diagnoses.add(NeutralizationDiagnosis.MISSING_FINAL_NEUTRALIZATION)
    imbalance = long_short_imbalance(config, after)
    if imbalance is not None and imbalance > p.maximum_long_short_imbalance:
        diagnoses.add(NeutralizationDiagnosis.LONG_SHORT_IMBALANCE)
    beta = after.market_beta if after.market_beta is not None else config.market_beta
    if beta is not None and abs(beta) > p.maximum_market_beta:
        diagnoses.add(NeutralizationDiagnosis.MARKET_BETA_EXPOSURE)
    if config.expression_group_neutralize_count > 0 and config.simulation_mode != NeutralizationMode.NONE:
        diagnoses.add(NeutralizationDiagnosis.DOUBLE_NEUTRALIZATION)
    retention = after.returns / baseline.returns if baseline.returns > 0 else 0.0
    if retention < p.risk_factor_dependency_retention:
        diagnoses.add(NeutralizationDiagnosis.RISK_FACTOR_DEPENDENCY)
    if after.sharpe < baseline.sharpe - p.maximum_sharpe_degradation:
        diagnoses.add(NeutralizationDiagnosis.OVER_NEUTRALIZED)
    if constant_expression and has_effective_final_neutralization(config):
        diagnoses.add(NeutralizationDiagnosis.CONSTANT_SIGNAL_COLLAPSE)
    if config.simulation_mode == NeutralizationMode.NONE and not expression_setting_equivalent(config):
        diagnoses.add(NeutralizationDiagnosis.GROUP_CHOICE_UNVERIFIED)
    return diagnoses


def neutralization_succeeded(
    baseline: NeutralizationMetrics,
    after: NeutralizationMetrics,
    policy: NeutralizationPolicy | None = None,
) -> bool:
    p = policy or NeutralizationPolicy()
    if baseline.returns <= 0:
        return False
    retention = after.returns / baseline.returns
    beta_ok = after.market_beta is None or abs(after.market_beta) <= p.maximum_market_beta
    return bool(
        after.sharpe >= baseline.sharpe - p.maximum_sharpe_degradation
        and retention >= p.minimum_return_retention
        and after.drawdown <= baseline.drawdown + p.maximum_drawdown_increase
        and beta_ok
        and after.sub_universe_passed is True
        and (after.os_stability or 0.0) >= p.minimum_os_stability
    )


def score_neutralization_trial(
    trial: NeutralizationTrial,
    baseline: NeutralizationTrial,
    policy: NeutralizationPolicy | None = None,
) -> NeutralizationTrialResult:
    p = policy or NeutralizationPolicy()
    after, before = trial.metrics, baseline.metrics
    diagnoses = diagnose_neutralization(trial.config, before, after, p)
    retention = after.returns / before.returns if before.returns > 0 else 0.0
    before_rdd = before.returns / max(abs(before.drawdown), 1e-9)
    after_rdd = after.returns / max(abs(after.drawdown), 1e-9)
    beta_before = abs(before.market_beta or 0.0)
    beta_after = abs(after.market_beta or 0.0)
    exposure_reduction = _clamp((beta_before - beta_after) / max(beta_before, 0.01))
    penalties = 0.0
    warnings: list[str] = []
    blockers: list[str] = []
    if NeutralizationDiagnosis.MISSING_FINAL_NEUTRALIZATION in diagnoses:
        blockers.append("MISSING_FINAL_NEUTRALIZATION")
    if NeutralizationDiagnosis.LONG_SHORT_IMBALANCE in diagnoses:
        blockers.append("LONG_SHORT_IMBALANCE")
    if NeutralizationDiagnosis.RISK_FACTOR_DEPENDENCY in diagnoses:
        penalties += 20.0
    if NeutralizationDiagnosis.OVER_NEUTRALIZED in diagnoses:
        penalties += 12.0
    if NeutralizationDiagnosis.DOUBLE_NEUTRALIZATION in diagnoses:
        penalties += 100 * p.double_neutralization_penalty
        warnings.append("DOUBLE_NEUTRALIZATION_REQUIRES_ABLATION")
    if not trial.official:
        blockers.append("UNOFFICIAL_METRICS")
    breakdown = NeutralizationScoreBreakdown(
        sharpe=24 * _clamp(after.sharpe / max(before.sharpe, 0.01)),
        return_retention=18 * _clamp(retention),
        return_drawdown=18 * _clamp(after_rdd / max(before_rdd, 0.01)),
        fitness=12 * _clamp(after.fitness / max(before.fitness, 0.01)),
        os_stability=10 * _clamp(after.os_stability or 0.0),
        low_correlation=8 * _clamp(1.0 - (after.self_correlation or 0.70) / 0.70),
        sub_universe=5 if after.sub_universe_passed is True else 0,
        exposure_reduction=5 * exposure_reduction,
        penalties=penalties,
    )
    return NeutralizationTrialResult(
        trial.trial_id,
        trial.config.simulation_mode,
        tuple(sorted(diagnoses, key=lambda item: item.value)),
        breakdown,
        neutralization_succeeded(before, after, p) and not blockers,
        tuple(blockers),
        tuple(warnings),
    )


def evaluate_neutralization_ablation(
    baseline: NeutralizationTrial,
    trials: Sequence[NeutralizationTrial],
    policy: NeutralizationPolicy | None = None,
) -> NeutralizationAblationReport:
    if not trials:
        return NeutralizationAblationReport(baseline.trial_id, (), None, None, "NO_TRIALS")
    results = [score_neutralization_trial(trial, baseline, policy) for trial in trials]
    ranked = tuple(sorted(results, key=lambda item: item.score.total, reverse=True))
    successful = [item for item in ranked if item.succeeded]
    if not successful:
        return NeutralizationAblationReport(baseline.trial_id, ranked, None, None, "NO_SUCCESSFUL_NEUTRALIZATION")
    best = successful[0]
    return NeutralizationAblationReport(baseline.trial_id, ranked, best.trial_id, best.mode, "BEST_MULTI_OBJECTIVE_UTILITY")
