from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from worldquant_brain.alpha_blueprint import AlphaBlueprint, RobustnessPlan


class RobustnessStatus(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    FAIL = "FAIL"
    FRAGILE = "FRAGILE"
    ROBUST = "ROBUST"


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    run_id: str
    variant_type: str
    variant_value: str
    sharpe: float | None
    fitness: float | None
    turnover: float | None
    returns: float | None = None
    drawdown: float | None = None
    pnl: float | None = None
    passed: bool | None = None
    metrics_source: str = "WORLDQUANT_BRAIN"
    simulation_id: str | None = None
    year: int | None = None
    regime: str | None = None
    universe: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DimensionReport:
    name: str
    status: RobustnessStatus
    score: float
    passed: int
    total: int
    details: dict[str, Any]
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    blueprint_id: str
    blueprint_hash: str
    baseline_run_id: str
    status: RobustnessStatus
    score: float
    dimensions: tuple[DimensionReport, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    promotion_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _relative_degradation(baseline: float, observed: float) -> float:
    if baseline <= 0:
        return 1.0 if observed < baseline else 0.0
    return max(0.0, (baseline - observed) / baseline)


def _run_passes(run: EvaluationRun, *, minimum_sharpe: float, minimum_fitness: float, maximum_turnover: float) -> bool:
    sharpe, fitness, turnover = _finite(run.sharpe), _finite(run.fitness), _finite(run.turnover)
    provenance = run.metrics_source == "WORLDQUANT_BRAIN" and bool(run.simulation_id)
    return bool(provenance and sharpe is not None and fitness is not None and turnover is not None and sharpe >= minimum_sharpe and fitness >= minimum_fitness and 0.01 < turnover <= maximum_turnover)


def _dimension(name: str, runs: Sequence[EvaluationRun], *, baseline: EvaluationRun, required_ratio: float, plan: RobustnessPlan, minimum_sharpe: float, minimum_fitness: float, maximum_turnover: float) -> DimensionReport:
    if not runs:
        return DimensionReport(name, RobustnessStatus.NOT_EVALUATED, 0.0, 0, 0, {}, (f"{name} has no runs",))
    baseline_sharpe = _finite(baseline.sharpe) or 0.0
    passed = 0
    degradations: list[float] = []
    failures: list[str] = []
    for run in runs:
        hard_pass = _run_passes(run, minimum_sharpe=minimum_sharpe, minimum_fitness=minimum_fitness, maximum_turnover=maximum_turnover)
        observed = _finite(run.sharpe) or -999.0
        degradation = _relative_degradation(baseline_sharpe, observed)
        degradations.append(degradation)
        stable = degradation <= plan.maximum_sharpe_degradation
        if hard_pass and stable:
            passed += 1
        else:
            failures.append(run.run_id)
    pass_ratio = _ratio(passed, len(runs))
    median_degradation = median(degradations) if degradations else 1.0
    score = max(0.0, min(100.0, 100 * (0.75 * pass_ratio + 0.25 * (1 - min(median_degradation, 1.0)))))
    status = RobustnessStatus.ROBUST if pass_ratio >= required_ratio else RobustnessStatus.FRAGILE if passed else RobustnessStatus.FAIL
    blockers = () if status == RobustnessStatus.ROBUST else (f"{name} pass ratio {pass_ratio:.1%} < required {required_ratio:.1%}",)
    return DimensionReport(name, status, round(score, 3), passed, len(runs), {"pass_ratio": pass_ratio, "median_sharpe_degradation": median_degradation, "failed_run_ids": failures}, blockers)


def _year_dimension(runs: Sequence[EvaluationRun], *, plan: RobustnessPlan) -> DimensionReport:
    yearly = [r for r in runs if r.year is not None]
    if not yearly:
        return DimensionReport("TEMPORAL_STABILITY", RobustnessStatus.NOT_EVALUATED, 0.0, 0, 0, {}, ("No yearly runs",))
    positive = [r for r in yearly if (_finite(r.pnl) or 0.0) > 0 and (_finite(r.sharpe) or -999.0) > 0]
    positive_ratio = _ratio(len(positive), len(yearly))
    absolute_pnl = [abs(_finite(r.pnl) or 0.0) for r in yearly]
    total_abs = sum(absolute_pnl)
    largest_share = max(absolute_pnl) / total_abs if total_abs else 1.0
    ok = positive_ratio >= plan.minimum_positive_year_ratio and largest_share <= plan.maximum_single_year_pnl_share
    status = RobustnessStatus.ROBUST if ok else RobustnessStatus.FRAGILE if positive else RobustnessStatus.FAIL
    score = max(0.0, min(100.0, 100 * (0.65 * positive_ratio + 0.35 * (1 - min(largest_share, 1.0)))))
    blockers: list[str] = []
    if positive_ratio < plan.minimum_positive_year_ratio:
        blockers.append(f"Positive-year ratio {positive_ratio:.1%} below {plan.minimum_positive_year_ratio:.1%}")
    if largest_share > plan.maximum_single_year_pnl_share:
        blockers.append(f"Single-year PnL share {largest_share:.1%} exceeds {plan.maximum_single_year_pnl_share:.1%}")
    return DimensionReport("TEMPORAL_STABILITY", status, round(score, 3), len(positive), len(yearly), {"positive_year_ratio": positive_ratio, "largest_absolute_pnl_share": largest_share}, tuple(blockers))


def evaluate_robustness(*, blueprint: AlphaBlueprint, baseline: EvaluationRun, runs: Sequence[EvaluationRun], minimum_sharpe: float = 1.25, minimum_fitness: float = 1.0, maximum_turnover: float = 0.70) -> RobustnessReport:
    validation_errors = blueprint.validate()
    if validation_errors:
        raise ValueError(f"Invalid blueprint: {validation_errors}")
    if not _run_passes(baseline, minimum_sharpe=minimum_sharpe, minimum_fitness=minimum_fitness, maximum_turnover=maximum_turnover):
        raise ValueError("Baseline must be an official, passing BRAIN run")
    plan = blueprint.robustness_plan
    groups: dict[str, list[EvaluationRun]] = {"PARAMETER_STABILITY": [], "UNIVERSE_STABILITY": [], "NEUTRALIZATION_STABILITY": [], "COST_STRESS": [], "REGIME_STABILITY": []}
    for run in runs:
        kind = run.variant_type.upper()
        if kind in {"WINDOW", "DECAY", "TRUNCATION", "PARAMETER"}:
            groups["PARAMETER_STABILITY"].append(run)
        elif kind == "UNIVERSE":
            groups["UNIVERSE_STABILITY"].append(run)
        elif kind == "NEUTRALIZATION":
            groups["NEUTRALIZATION_STABILITY"].append(run)
        elif kind == "COST":
            groups["COST_STRESS"].append(run)
        elif kind == "REGIME" or run.regime:
            groups["REGIME_STABILITY"].append(run)
    dimensions = [
        _year_dimension(runs, plan=plan),
        _dimension("PARAMETER_STABILITY", groups["PARAMETER_STABILITY"], baseline=baseline, required_ratio=plan.minimum_parameter_pass_ratio, plan=plan, minimum_sharpe=minimum_sharpe, minimum_fitness=minimum_fitness, maximum_turnover=maximum_turnover),
        _dimension("UNIVERSE_STABILITY", groups["UNIVERSE_STABILITY"], baseline=baseline, required_ratio=plan.minimum_universe_pass_ratio, plan=plan, minimum_sharpe=minimum_sharpe, minimum_fitness=minimum_fitness, maximum_turnover=maximum_turnover),
        _dimension("NEUTRALIZATION_STABILITY", groups["NEUTRALIZATION_STABILITY"], baseline=baseline, required_ratio=0.50, plan=plan, minimum_sharpe=minimum_sharpe, minimum_fitness=minimum_fitness, maximum_turnover=maximum_turnover),
        _dimension("COST_STRESS", groups["COST_STRESS"], baseline=baseline, required_ratio=0.67, plan=plan, minimum_sharpe=minimum_sharpe, minimum_fitness=minimum_fitness, maximum_turnover=maximum_turnover),
        _dimension("REGIME_STABILITY", groups["REGIME_STABILITY"], baseline=baseline, required_ratio=plan.minimum_regime_pass_ratio, plan=plan, minimum_sharpe=minimum_sharpe, minimum_fitness=minimum_fitness, maximum_turnover=maximum_turnover),
    ]
    evaluated = [d for d in dimensions if d.status != RobustnessStatus.NOT_EVALUATED]
    blockers = tuple(blocker for d in dimensions for blocker in d.blockers)
    warnings = tuple(f"{d.name} not evaluated" for d in dimensions if d.status == RobustnessStatus.NOT_EVALUATED)
    score = round(sum(d.score for d in evaluated) / len(evaluated), 3) if evaluated else 0.0
    required_names = {"TEMPORAL_STABILITY", "PARAMETER_STABILITY", "UNIVERSE_STABILITY", "COST_STRESS", "REGIME_STABILITY"}
    required = [d for d in dimensions if d.name in required_names]
    promotion_ready = all(d.status == RobustnessStatus.ROBUST for d in required) and score >= 70.0
    if promotion_ready:
        status = RobustnessStatus.ROBUST
    elif any(d.status == RobustnessStatus.FAIL for d in required):
        status = RobustnessStatus.FAIL
    else:
        status = RobustnessStatus.FRAGILE
    return RobustnessReport(blueprint.blueprint_id, blueprint.sha256(), baseline.run_id, status, score, tuple(dimensions), blockers, warnings, promotion_ready)
