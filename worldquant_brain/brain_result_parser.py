from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

MetricSource = Literal["WORLDQUANT_BRAIN", "UNKNOWN"]
_STATUS_ALIASES = {
    "COMPLETE": "COMPLETED",
    "COMPLETED": "COMPLETED",
    "DONE": "COMPLETED",
    "SUCCESS": "COMPLETED",
    "RUNNING": "RUNNING",
    "IN_PROGRESS": "RUNNING",
    "PENDING": "RUNNING",
    "QUEUED": "RUNNING",
    "SUBMITTED": "RUNNING",
    "ERROR": "FAILED",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELED",
    "CANCELED": "CANCELED",
    "TIMEOUT": "TIMEOUT",
}

@dataclass(frozen=True, slots=True)
class BrainCheckResult:
    name: str
    status: str
    value: float | str | bool | None = None
    limit: float | str | bool | None = None
    message: str | None = None

@dataclass(frozen=True, slots=True)
class ParsedBrainSimulation:
    status: str
    success: bool
    terminal: bool
    simulation_id: str | None
    alpha_id: str | None
    location: str | None
    sharpe: float | None
    fitness: float | None
    returns: float | None
    turnover: float | None
    margin: float | None
    drawdown: float | None
    self_correlation: float | None
    production_correlation: float | None
    max_instrument_weight: float | None
    long_count: int | None
    short_count: int | None
    sub_universe_passed: bool | None
    weight_test_passed: bool | None
    ranked_sharpe_passed: bool | None
    bias_test_passed: bool | None
    checks: tuple[BrainCheckResult, ...] = field(default_factory=tuple)
    metrics_source: MetricSource = "UNKNOWN"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100.0
            except ValueError:
                return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    return int(number) if number is not None else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value or "").strip().upper()
    if normalized in {"PASS", "PASSED", "TRUE", "YES", "OK", "SUCCESS"}:
        return True
    if normalized in {"FAIL", "FAILED", "FALSE", "NO", "ERROR"}:
        return False
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _walk_mappings(value: Any, *, max_depth: int = 7) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    seen: set[int] = set()

    def visit(node: Any, depth: int) -> None:
        if depth > max_depth or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Mapping):
            found.append(node)
            for child in node.values():
                visit(child, depth + 1)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child, depth + 1)

    visit(value, 0)
    return found


def _lookup(sources: Sequence[Mapping[str, Any]], *aliases: str) -> Any:
    wanted = {_key(alias) for alias in aliases}
    for source in sources:
        normalized = {_key(k): v for k, v in source.items()}
        for alias in wanted:
            if alias in normalized and normalized[alias] is not None:
                return normalized[alias]
    return None


def _id_from_location(location: str | None) -> str | None:
    if not location:
        return None
    path = urlparse(location).path.rstrip("/")
    return path.rsplit("/", 1)[-1] or None


def _normalize_status(value: Any) -> str:
    raw = str(value or "UNKNOWN").strip().upper().replace("-", "_").replace(" ", "_")
    return _STATUS_ALIASES.get(raw, raw)


def _parse_checks(payload: Any) -> tuple[BrainCheckResult, ...]:
    candidates: list[Any] = []
    for source in _walk_mappings(payload):
        for key, value in source.items():
            if _key(key) in {"checks", "checkresults", "ischecks", "submittests", "submissionchecks"}:
                candidates.append(value)

    parsed: list[BrainCheckResult] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            iterable = []
            for name, body in candidate.items():
                if isinstance(body, Mapping):
                    iterable.append({"name": name, **body})
                else:
                    iterable.append({"name": name, "status": body})
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            iterable = list(candidate)
        else:
            continue

        for item in iterable:
            if not isinstance(item, Mapping):
                continue
            sources = [item]
            name = str(_lookup(sources, "name", "check", "type", "id") or "UNKNOWN_CHECK")
            status = _normalize_status(_lookup(sources, "status", "result", "outcome", "passed"))
            parsed.append(BrainCheckResult(
                name=name,
                status=status,
                value=_lookup(sources, "value", "actual", "metric"),
                limit=_lookup(sources, "limit", "threshold", "required", "cutoff"),
                message=str(_lookup(sources, "message", "description", "reason") or "") or None,
            ))

    unique: dict[str, BrainCheckResult] = {}
    for check in parsed:
        unique[_key(check.name)] = check
    return tuple(unique.values())


def _check_boolean(checks: Sequence[BrainCheckResult], *names: str) -> bool | None:
    wanted = {_key(name) for name in names}
    for check in checks:
        if _key(check.name) in wanted:
            explicit = _boolean(check.status)
            if explicit is not None:
                return explicit
            return _boolean(check.value)
    return None


def parse_brain_simulation_result(
    payload: Mapping[str, Any],
    *,
    location: str | None = None,
    submitted: bool = True,
) -> ParsedBrainSimulation:
    if not isinstance(payload, Mapping):
        raise TypeError("BRAIN simulation payload must be a mapping")

    all_sources = _walk_mappings(payload)
    preferred: list[Mapping[str, Any]] = []
    for key in ("is", "metrics", "result", "simulation", "alpha", "data"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            preferred.extend(_walk_mappings(value))
    sources = [*preferred, payload, *all_sources]

    status = _normalize_status(_lookup(sources, "status", "state", "simulationStatus"))
    location = location or str(_lookup(sources, "location", "simulationLocation", "url") or "") or None
    simulation_id = _lookup(sources, "simulation_id", "simulationId", "simulation", "id")
    alpha_id = _lookup(sources, "alpha_id", "alphaId", "alpha")
    if isinstance(alpha_id, Mapping):
        alpha_id = _lookup([alpha_id], "id", "alphaId")
    if isinstance(simulation_id, Mapping):
        simulation_id = _lookup([simulation_id], "id", "simulationId")
    simulation_id = str(simulation_id) if simulation_id not in (None, "") else _id_from_location(location)
    alpha_id = str(alpha_id) if alpha_id not in (None, "") else None

    checks = _parse_checks(payload)
    sharpe = _finite_number(_lookup(sources, "sharpe", "sharpeRatio"))
    fitness = _finite_number(_lookup(sources, "fitness"))
    returns = _finite_number(_lookup(sources, "returns", "return", "annualReturn", "annualizedReturn"))
    turnover = _finite_number(_lookup(sources, "turnover", "turnoverRatio"))
    margin = _finite_number(_lookup(sources, "margin"))
    drawdown = _finite_number(_lookup(sources, "drawdown", "maxDrawdown", "maximumDrawdown"))
    self_correlation = _finite_number(_lookup(sources, "selfCorrelation", "selfCorr", "self_correlation"))
    production_correlation = _finite_number(_lookup(sources, "productionCorrelation", "prodCorrelation", "production_correlation"))
    max_weight = _finite_number(_lookup(sources, "maxWeight", "maxInstrumentWeight", "maximumWeight"))
    long_count = _integer(_lookup(sources, "longCount", "long_count"))
    short_count = _integer(_lookup(sources, "shortCount", "short_count"))

    terminal = status in {"COMPLETED", "FAILED", "CANCELED", "TIMEOUT"}
    valid_core = sharpe is not None and fitness is not None and turnover is not None
    success = status == "COMPLETED" and valid_core and bool(simulation_id or alpha_id or location)

    warnings: list[str] = []
    if status == "COMPLETED" and not valid_core:
        warnings.append("Simulation completed but core metrics are incomplete")
    if submitted and terminal and not (simulation_id or alpha_id or location):
        warnings.append("Terminal result lacks simulation/alpha provenance")
    if turnover is not None and turnover > 1.0:
        warnings.append("Turnover is greater than 1.0; verify whether API returned percent or ratio")

    return ParsedBrainSimulation(
        status=status,
        success=success,
        terminal=terminal,
        simulation_id=simulation_id,
        alpha_id=alpha_id,
        location=location,
        sharpe=sharpe,
        fitness=fitness,
        returns=returns,
        turnover=turnover,
        margin=margin,
        drawdown=drawdown,
        self_correlation=self_correlation,
        production_correlation=production_correlation,
        max_instrument_weight=max_weight,
        long_count=long_count,
        short_count=short_count,
        sub_universe_passed=_check_boolean(checks, "SUB_UNIVERSE", "SUB_UNIVERSE_TEST", "LOW_SUB_UNIVERSE_SHARPE"),
        weight_test_passed=_check_boolean(checks, "WEIGHT", "WEIGHT_TEST", "CONCENTRATED_WEIGHT"),
        ranked_sharpe_passed=_check_boolean(checks, "RANKED_SHARPE", "RANKED_SHARPE_TEST"),
        bias_test_passed=_check_boolean(checks, "BIAS", "BIAS_TEST"),
        checks=checks,
        metrics_source="WORLDQUANT_BRAIN" if success else "UNKNOWN",
        warnings=tuple(warnings),
    )
