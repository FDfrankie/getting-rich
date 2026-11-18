from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

Delay = Literal[0, 1]
MetricSource = Literal["WORLDQUANT_BRAIN", "LOCAL_BACKTEST", "LOCAL_PROXY", "UNKNOWN"]


class AlphaQuality(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    AVERAGE = "AVERAGE"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"
    SPECTACULAR = "SPECTACULAR"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    NOT_VERIFIED = "NOT_VERIFIED"


@dataclass(frozen=True, slots=True)
class AlphaMetrics:
    alpha_id: str
    expression: str
    delay: Delay
    sharpe: float | None
    fitness: float | None
    returns: float | None
    turnover: float | None
    margin: float | None = None
    drawdown: float | None = None
    self_correlation: float | None = None
    production_correlation: float | None = None
    max_instrument_weight: float | None = None
    sub_universe_passed: bool | None = None
    weight_test_passed: bool | None = None
    ranked_sharpe_passed: bool | None = None
    bias_test_passed: bool | None = None
    long_count: int | None = None
    short_count: int | None = None
    official_simulation_id: str | None = None
    metrics_source: MetricSource = "UNKNOWN"
    region: str | None = None
    universe: str | None = None
    neutralization: str | None = None
    decay: int | None = None
    truncation: float | None = None


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    minimum_turnover: float
    maximum_turnover: float
    maximum_good_turnover: float
    maximum_self_correlation: float
    maximum_weight: float
    average_sharpe: float
    good_sharpe: float
    excellent_sharpe: float
    spectacular_sharpe: float
    average_fitness: float
    good_fitness: float
    excellent_fitness: float
    spectacular_fitness: float


@dataclass(frozen=True, slots=True)
class QualityGate:
    name: str
    status: GateStatus
    actual: Any
    required: Any
    reason: str


@dataclass(frozen=True, slots=True)
class AlphaQualityReport:
    alpha_id: str
    quality: AlphaQuality
    officially_verified: bool
    submit_ready: bool
    good_ready: bool
    gates: tuple[QualityGate, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    optimization_actions: tuple[str, ...]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def thresholds_for_delay(delay: Delay) -> QualityThresholds:
    if delay == 1:
        return QualityThresholds(0.01, 0.70, 0.60, 0.70, 0.10, 1.25, 1.50, 2.00, 2.50, 1.00, 1.50, 2.00, 2.50)
    return QualityThresholds(0.01, 0.70, 0.60, 0.70, 0.10, 2.00, 2.40, 2.60, 3.20, 1.30, 1.95, 2.60, 3.25)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def official_metrics_verified(metrics: AlphaMetrics) -> bool:
    return (
        metrics.metrics_source == "WORLDQUANT_BRAIN"
        and bool(metrics.official_simulation_id)
        and all(_finite(v) is not None for v in (metrics.sharpe, metrics.fitness, metrics.turnover))
    )


def _numeric_gate(name: str, actual: float | None, required: str, predicate) -> QualityGate:
    value = _finite(actual)
    if value is None:
        return QualityGate(name, GateStatus.NOT_AVAILABLE, None, required, f"{name} 尚無官方結果")
    passed = bool(predicate(value))
    return QualityGate(name, GateStatus.PASS if passed else GateStatus.FAIL, value, required, "符合門檻" if passed else f"{name} 未達門檻")


def _boolean_gate(name: str, actual: bool | None) -> QualityGate:
    if actual is None:
        return QualityGate(name, GateStatus.NOT_VERIFIED, None, True, f"{name} 尚無官方檢查結果")
    return QualityGate(name, GateStatus.PASS if actual else GateStatus.FAIL, actual, True, "符合門檻" if actual else f"{name} 未通過")


def classify_quality(metrics: AlphaMetrics, t: QualityThresholds) -> AlphaQuality:
    if not official_metrics_verified(metrics):
        return AlphaQuality.UNVERIFIED
    sharpe, fitness = float(metrics.sharpe), float(metrics.fitness)
    if sharpe >= t.spectacular_sharpe and fitness >= t.spectacular_fitness:
        return AlphaQuality.SPECTACULAR
    if sharpe >= t.excellent_sharpe and fitness >= t.excellent_fitness:
        return AlphaQuality.EXCELLENT
    if sharpe >= t.good_sharpe and fitness >= t.good_fitness:
        return AlphaQuality.GOOD
    if sharpe > t.average_sharpe and fitness > t.average_fitness:
        return AlphaQuality.AVERAGE
    return AlphaQuality.NEEDS_IMPROVEMENT


def quality_score(metrics: AlphaMetrics, t: QualityThresholds) -> float:
    sharpe = max(_finite(metrics.sharpe) or 0.0, 0.0)
    fitness = max(_finite(metrics.fitness) or 0.0, 0.0)
    turnover = _finite(metrics.turnover)
    corr = _finite(metrics.self_correlation)
    weight = _finite(metrics.max_instrument_weight)
    margin = max(_finite(metrics.margin) or 0.0, 0.0)
    drawdown = abs(_finite(metrics.drawdown) or 0.0)
    score = min(sharpe / t.good_sharpe, 1.5) * 25 + min(fitness / t.good_fitness, 1.5) * 25
    if turnover is not None:
        score += 15 if t.minimum_turnover < turnover <= 0.40 else 11 if turnover <= t.maximum_good_turnover else 5 if turnover < t.maximum_turnover else 0
    if corr is not None:
        score += max(0.0, 15 * (1 - corr / t.maximum_self_correlation))
    if weight is not None:
        score += max(0.0, 10 * (1 - weight / t.maximum_weight))
    score += min(margin / 0.001, 1.0) * 5
    score -= min(drawdown * 50, 10)
    if not official_metrics_verified(metrics):
        score *= 0.45
    return round(max(0.0, min(100.0, score)), 3)


def optimization_actions(metrics: AlphaMetrics, t: QualityThresholds) -> list[str]:
    actions: list[str] = []
    if not official_metrics_verified(metrics):
        actions.append("先取得 BRAIN 官方 simulation_id 與完整 Metrics，本地結果不得標示 GOOD。")
    if (_finite(metrics.sharpe) or 0.0) < t.good_sharpe:
        actions.extend(["比較 raw/rank/group_rank 訊號。", "檢查分年 Sharpe，不接受單一年度主導。", "測試 INDUSTRY/SUBINDUSTRY neutralization。"])
    if (_finite(metrics.fitness) or 0.0) < t.good_fitness:
        actions.append("提高成本後報酬或降低換手率，將 Fitness 提升至 GOOD 緩衝區。")
    turnover = _finite(metrics.turnover)
    if turnover is not None and turnover > t.maximum_good_turnover:
        actions.extend(["測試 Decay +1/+3/+7。", "用 trade_when 或 hump 抑制微小換倉。"])
    if _finite(metrics.self_correlation) is not None and float(metrics.self_correlation) >= t.maximum_self_correlation:
        actions.append("Self-correlation 過高：更換資料域或經濟假說，不要僅微調視窗。")
    if metrics.sub_universe_passed is False:
        actions.append("檢查低流動性與小型股偏誤，分層比較 TOP3000/TOP2000/TOP1000。")
    if metrics.weight_test_passed is False or ((_finite(metrics.max_instrument_weight) or 0.0) >= t.maximum_weight):
        actions.append("使用 rank/winsorize 與 0.05/0.06/0.08 truncation 降低權重集中。")
    if metrics.bias_test_passed is False:
        actions.append("Bias Test 失敗：停止優化，先修正未來資料洩漏或 Delay 語意。")
    return list(dict.fromkeys(actions))


def evaluate_alpha(metrics: AlphaMetrics) -> AlphaQualityReport:
    t = thresholds_for_delay(metrics.delay)
    official = official_metrics_verified(metrics)
    gates = [
        QualityGate("OFFICIAL_PROVENANCE", GateStatus.PASS if official else GateStatus.NOT_VERIFIED, {"source": metrics.metrics_source, "simulation_id": metrics.official_simulation_id}, {"source": "WORLDQUANT_BRAIN", "simulation_id": "required"}, "官方來源已驗證" if official else "缺少官方 provenance"),
        _numeric_gate("SHARPE_SUBMISSION", metrics.sharpe, f"> {t.average_sharpe}", lambda v: v > t.average_sharpe),
        _numeric_gate("SHARPE_GOOD", metrics.sharpe, f">= {t.good_sharpe}", lambda v: v >= t.good_sharpe),
        _numeric_gate("FITNESS_SUBMISSION", metrics.fitness, f"> {t.average_fitness}", lambda v: v > t.average_fitness),
        _numeric_gate("FITNESS_GOOD", metrics.fitness, f">= {t.good_fitness}", lambda v: v >= t.good_fitness),
        _numeric_gate("TURNOVER_SUBMISSION", metrics.turnover, f"{t.minimum_turnover} < turnover < {t.maximum_turnover}", lambda v: t.minimum_turnover < v < t.maximum_turnover),
        _numeric_gate("TURNOVER_GOOD_BUFFER", metrics.turnover, f"{t.minimum_turnover} < turnover <= {t.maximum_good_turnover}", lambda v: t.minimum_turnover < v <= t.maximum_good_turnover),
        _numeric_gate("SELF_CORRELATION", metrics.self_correlation, f"< {t.maximum_self_correlation}", lambda v: v < t.maximum_self_correlation),
        _numeric_gate("MAX_INSTRUMENT_WEIGHT", metrics.max_instrument_weight, f"< {t.maximum_weight}", lambda v: v < t.maximum_weight),
        _boolean_gate("WEIGHT_TEST", metrics.weight_test_passed),
        _boolean_gate("SUB_UNIVERSE_TEST", metrics.sub_universe_passed),
        _boolean_gate("RANKED_SHARPE_TEST", metrics.ranked_sharpe_passed),
        _boolean_gate("BIAS_TEST", metrics.bias_test_passed),
    ]
    quality = classify_quality(metrics, t)
    submission_names = {"OFFICIAL_PROVENANCE", "SHARPE_SUBMISSION", "FITNESS_SUBMISSION", "TURNOVER_SUBMISSION", "SELF_CORRELATION", "MAX_INSTRUMENT_WEIGHT", "WEIGHT_TEST", "SUB_UNIVERSE_TEST", "BIAS_TEST"}
    submit_ready = all(g.status == GateStatus.PASS for g in gates if g.name in submission_names)
    good_ready = submit_ready and quality in {AlphaQuality.GOOD, AlphaQuality.EXCELLENT, AlphaQuality.SPECTACULAR} and next(g for g in gates if g.name == "TURNOVER_GOOD_BUFFER").status == GateStatus.PASS
    return AlphaQualityReport(
        alpha_id=metrics.alpha_id,
        quality=quality,
        officially_verified=official,
        submit_ready=submit_ready,
        good_ready=good_ready,
        gates=tuple(gates),
        blockers=tuple(f"{g.name}: {g.reason}" for g in gates if g.status == GateStatus.FAIL),
        warnings=tuple(f"{g.name}: {g.reason}" for g in gates if g.status in {GateStatus.NOT_AVAILABLE, GateStatus.NOT_VERIFIED}),
        optimization_actions=tuple(optimization_actions(metrics, t)),
        score=quality_score(metrics, t),
    )
