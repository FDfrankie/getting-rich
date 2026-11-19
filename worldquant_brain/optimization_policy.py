from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class OptimizationObjective(StrEnum):
    RETURN = "RETURN"
    SHARPE = "SHARPE"
    FITNESS = "FITNESS"
    CAPACITY = "CAPACITY"
    BALANCED = "BALANCED"


class AlphaDiagnosis(StrEnum):
    HIGH_TURNOVER_LOW_RETURN = "HIGH_TURNOVER_LOW_RETURN"
    HIGH_TURNOVER_PRODUCTIVE = "HIGH_TURNOVER_PRODUCTIVE"
    LOW_TURNOVER_STALE_SIGNAL = "LOW_TURNOVER_STALE_SIGNAL"
    LOW_RETURN_LOW_INFORMATION = "LOW_RETURN_LOW_INFORMATION"
    HIGH_VOLATILITY_LOW_RETURN = "HIGH_VOLATILITY_LOW_RETURN"
    HIGH_VOLATILITY_PRODUCTIVE = "HIGH_VOLATILITY_PRODUCTIVE"
    HIGH_MARKET_EXPOSURE = "HIGH_MARKET_EXPOSURE"
    EVENT_SIGNAL_TOO_SPARSE = "EVENT_SIGNAL_TOO_SPARSE"
    WASTEFUL_WEIGHT_CHANGES = "WASTEFUL_WEIGHT_CHANGES"


@dataclass(frozen=True, slots=True)
class AlphaPerformance:
    returns: float
    net_returns: float
    sharpe: float
    fitness: float
    turnover: float
    margin: float
    drawdown: float
    volatility: float | None = None
    coverage: float | None = None
    event_frequency: float | None = None
    market_beta: float | None = None
    previous_net_returns: float | None = None
    previous_turnover: float | None = None
    previous_drawdown: float | None = None
    previous_volatility: float | None = None


@dataclass(frozen=True, slots=True)
class OptimizationAction:
    action: str
    priority: float
    reason: str
    target_metric: str
    required_validation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObjectivePolicy:
    objective: OptimizationObjective
    minimum_net_return_gain: float = 0.01
    maximum_drawdown_increase: float = 0.02
    high_turnover: float = 0.55
    low_turnover: float = 0.08
    minimum_turnover_efficiency: float = 0.05
    sparse_coverage: float = 0.20
    high_market_beta: float = 0.30


def turnover_efficiency(metrics: AlphaPerformance) -> float | None:
    if metrics.previous_net_returns is None or metrics.previous_turnover is None:
        return None
    delta_turnover = metrics.turnover - metrics.previous_turnover
    if abs(delta_turnover) < 1e-12:
        return None
    return (metrics.net_returns - metrics.previous_net_returns) / delta_turnover


def diagnose_alpha(metrics: AlphaPerformance, policy: ObjectivePolicy) -> set[AlphaDiagnosis]:
    diagnoses: set[AlphaDiagnosis] = set()
    efficiency = turnover_efficiency(metrics)

    if metrics.turnover > policy.high_turnover:
        if efficiency is None or efficiency <= policy.minimum_turnover_efficiency:
            diagnoses.add(AlphaDiagnosis.HIGH_TURNOVER_LOW_RETURN)
            diagnoses.add(AlphaDiagnosis.WASTEFUL_WEIGHT_CHANGES)
        else:
            diagnoses.add(AlphaDiagnosis.HIGH_TURNOVER_PRODUCTIVE)

    if metrics.turnover < policy.low_turnover and metrics.net_returns < 0.05:
        diagnoses.add(AlphaDiagnosis.LOW_TURNOVER_STALE_SIGNAL)

    if metrics.net_returns < 0.05 and metrics.sharpe < 1.0:
        diagnoses.add(AlphaDiagnosis.LOW_RETURN_LOW_INFORMATION)

    if metrics.volatility is not None and metrics.drawdown > 0.12:
        if metrics.net_returns < 0.08:
            diagnoses.add(AlphaDiagnosis.HIGH_VOLATILITY_LOW_RETURN)
        else:
            diagnoses.add(AlphaDiagnosis.HIGH_VOLATILITY_PRODUCTIVE)

    if metrics.market_beta is not None and abs(metrics.market_beta) > policy.high_market_beta:
        diagnoses.add(AlphaDiagnosis.HIGH_MARKET_EXPOSURE)

    if metrics.coverage is not None and metrics.coverage < policy.sparse_coverage:
        diagnoses.add(AlphaDiagnosis.EVENT_SIGNAL_TOO_SPARSE)

    return diagnoses


def recommend_actions(metrics: AlphaPerformance, policy: ObjectivePolicy) -> list[OptimizationAction]:
    diagnoses = diagnose_alpha(metrics, policy)
    actions: list[OptimizationAction] = []

    if AlphaDiagnosis.HIGH_TURNOVER_LOW_RETURN in diagnoses:
        actions.extend([
            OptimizationAction("APPLY_HUMP", 0.98, "Turnover 高但成本后收益边际贡献低，移除微小权重变化。", "TURNOVER_EFFICIENCY", ("COST_STRESS", "SHARPE_NON_DEGRADATION")),
            OptimizationAction("ADD_TRADE_WHEN", 0.92, "只在高资讯事件发生时更新仓位。", "NET_RETURN_PER_TURNOVER", ("COVERAGE", "EVENT_FREQUENCY", "LONG_SHORT_COUNT")),
            OptimizationAction("INCREASE_DECAY", 0.76, "平滑 Alpha 输出权重，而非修改原始输入字段。", "WEIGHT_VARIANCE", ("RETURN_NON_DEGRADATION", "MARGIN")),
        ])

    if AlphaDiagnosis.HIGH_TURNOVER_PRODUCTIVE in diagnoses:
        actions.append(OptimizationAction("PRESERVE_TURNOVER_AND_STRESS_COST", 0.96, "较高换手带来显著成本后收益，不应机械压低。", "NET_RETURN", ("COST_1X", "COST_1_5X", "COST_2X", "DRAWDOWN")))

    if AlphaDiagnosis.LOW_TURNOVER_STALE_SIGNAL in diagnoses:
        actions.extend([
            OptimizationAction("LOWER_DECAY", 0.94, "讯号可能被过度平滑，测试更低 Decay 提高反应速度。", "NET_RETURN", ("TURNOVER", "DRAWDOWN", "CORRELATION")),
            OptimizationAction("SHORTEN_HORIZON", 0.72, "验证预测周期是否过长。", "INFORMATION_DECAY", ("OOS_SHARPE", "PARAMETER_STABILITY")),
        ])

    if AlphaDiagnosis.LOW_RETURN_LOW_INFORMATION in diagnoses:
        actions.extend([
            OptimizationAction("ADD_ANALYST_NEWS_OR_FUNDAMENTAL", 0.99, "低收益优先改善资讯，而不是只修改执行参数。", "PREDICTIVE_POWER", ("FIELD_COVERAGE", "POINT_IN_TIME", "SUB_UNIVERSE")),
            OptimizationAction("CHANGE_ECONOMIC_HYPOTHESIS", 0.95, "现有讯号预测能力不足。", "FALSIFIABLE_CLAIM", ("BLUEPRINT_REVIEW", "NOVELTY")),
        ])

    if AlphaDiagnosis.HIGH_VOLATILITY_LOW_RETURN in diagnoses:
        actions.extend([
            OptimizationAction("TEST_NEUTRALIZATION", 0.96, "降低市场、产业或群组风险曝险。", "VOLATILITY", ("RETURN", "RANKED_SHARPE", "SUB_UNIVERSE")),
            OptimizationAction("APPLY_VOLATILITY_SCALING", 0.88, "按个股波动缩放权重。", "RISK_ADJUSTED_RETURN", ("WEIGHT", "CAPACITY", "DRAWDOWN")),
        ])

    if AlphaDiagnosis.HIGH_MARKET_EXPOSURE in diagnoses:
        actions.append(OptimizationAction("MARKET_OR_GROUP_NEUTRALIZE", 0.93, "高市场曝险可能压低 IR。", "BETA", ("SHARPE", "RETURN_NON_DEGRADATION")))

    if AlphaDiagnosis.EVENT_SIGNAL_TOO_SPARSE in diagnoses:
        actions.append(OptimizationAction("RELAX_EVENT_CONDITION", 0.85, "事件过稀会损害 Coverage 与权重分散。", "COVERAGE", ("WEIGHT", "LONG_SHORT_COUNT", "TURNOVER")))

    if policy.objective == OptimizationObjective.RETURN:
        actions.append(OptimizationAction("TEST_LOWER_DECAY_CONTROLLED", 0.68, "Return Lane 可测试更低 Decay，但必须同时监控成本、回撤与相关性。", "NET_RETURN", ("COST_STRESS", "DRAWDOWN", "SELF_CORRELATION")))
    elif policy.objective == OptimizationObjective.CAPACITY:
        actions.append(OptimizationAction("REDUCE_WASTEFUL_TURNOVER", 0.97, "Capacity Lane 优先消除无收益贡献交易。", "CAPACITY", ("LIQUIDITY", "CORRELATION", "MARGIN")))
    elif policy.objective == OptimizationObjective.SHARPE:
        actions.append(OptimizationAction("DECOMPOSE_RETURN_AND_VOLATILITY", 0.97, "先判断 Sharpe 低来自 Return 不足或 Volatility 过高。", "SHARPE", ("RETURN_ATTRIBUTION", "RISK_ATTRIBUTION")))

    unique: dict[str, OptimizationAction] = {}
    for action in actions:
        current = unique.get(action.action)
        if current is None or action.priority > current.priority:
            unique[action.action] = action
    return sorted(unique.values(), key=lambda item: item.priority, reverse=True)


def policy_snapshot(metrics: AlphaPerformance, policy: ObjectivePolicy) -> dict[str, Any]:
    return {
        "objective": policy.objective.value,
        "metrics": asdict(metrics),
        "turnover_efficiency": turnover_efficiency(metrics),
        "diagnoses": sorted(item.value for item in diagnose_alpha(metrics, policy)),
        "actions": [asdict(item) for item in recommend_actions(metrics, policy)],
    }
