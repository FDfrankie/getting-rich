from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from worldquant_brain.alpha_quality import AlphaMetrics, evaluate_alpha
from worldquant_brain.brain_result_parser import ParsedBrainSimulation, parse_brain_simulation_result


def parsed_to_alpha_metrics(parsed: ParsedBrainSimulation, *, expression: str, settings: Mapping[str, Any] | None = None) -> AlphaMetrics:
    settings = dict(settings or {})
    delay = int(settings.get("delay", 1))
    if delay not in (0, 1):
        raise ValueError(f"Unsupported BRAIN delay: {delay}")
    return AlphaMetrics(
        alpha_id=parsed.alpha_id or parsed.simulation_id or "UNKNOWN",
        expression=expression,
        delay=delay,
        sharpe=parsed.sharpe,
        fitness=parsed.fitness,
        returns=parsed.returns,
        turnover=parsed.turnover,
        margin=parsed.margin,
        drawdown=parsed.drawdown,
        self_correlation=parsed.self_correlation,
        production_correlation=parsed.production_correlation,
        max_instrument_weight=parsed.max_instrument_weight,
        sub_universe_passed=parsed.sub_universe_passed,
        weight_test_passed=parsed.weight_test_passed,
        ranked_sharpe_passed=parsed.ranked_sharpe_passed,
        bias_test_passed=parsed.bias_test_passed,
        long_count=parsed.long_count,
        short_count=parsed.short_count,
        official_simulation_id=parsed.simulation_id,
        metrics_source=parsed.metrics_source,
        region=settings.get("region"), universe=settings.get("universe"), neutralization=settings.get("neutralization"), decay=settings.get("decay"), truncation=settings.get("truncation"),
    )


def parse_and_evaluate_brain_result(payload: Mapping[str, Any], *, expression: str, settings: Mapping[str, Any] | None = None, location: str | None = None) -> dict[str, Any]:
    parsed = parse_brain_simulation_result(payload, location=location)
    metrics = parsed_to_alpha_metrics(parsed, expression=expression, settings=settings)
    report = evaluate_alpha(metrics)
    return {"parsed": parsed.to_dict(), "metrics": metrics, "quality_report": report.to_dict()}
