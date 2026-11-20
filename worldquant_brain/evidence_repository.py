from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    experiment_id: str
    candidate_id: str
    blueprint_id: str
    blueprint_hash: str
    hypothesis_family: str
    data_domain: str
    expression: str
    expression_hash: str
    normalized_expression_hash: str
    structural_expression_hash: str
    settings: Mapping[str, Any]
    settings_hash: str
    region: str
    universe: str
    delay: int
    experiment_type: str
    parent_experiment_id: str | None = None
    alpha_id: str | None = None


@dataclass(frozen=True, slots=True)
class OfficialMetrics:
    metrics_source: str
    official_simulation_id: str | None
    sharpe: float | None
    fitness: float | None
    returns: float | None
    turnover: float | None
    margin: float | None = None
    drawdown: float | None = None
    self_correlation: float | None = None
    production_correlation: float | None = None
    max_instrument_weight: float | None = None
    long_count: int | None = None
    short_count: int | None = None
    weight_test_passed: bool | None = None
    sub_universe_passed: bool | None = None
    ranked_sharpe_passed: bool | None = None
    bias_test_passed: bool | None = None
    submit_ready: bool = False
    good_ready: bool = False
    robust_good: bool = False
    quality_class: str = "UNVERIFIED"
    quality_score: float | None = None
    raw_metrics: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class FailureObservation:
    gate_name: str
    severity: str
    failure_code: str
    actual_value: float | None = None
    required_value: float | None = None
    context: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AppliedFix:
    fix_name: str
    fix_category: str
    intended_gate: str
    structural_change: bool = False
    before_value: Any = None
    after_value: Any = None


@dataclass(frozen=True, slots=True)
class EvidenceAggregate:
    evidence_id: str
    hypothesis_family: str
    data_domain: str
    failure_gate: str
    fix_name: str
    fix_category: str
    region: str
    universe: str
    delay: int
    attempts: int
    successful_attempts: int
    failed_attempts: int
    mean_sharpe_delta: float
    mean_fitness_delta: float
    mean_turnover_delta: float
    mean_correlation_delta: float
    mean_margin_delta: float
    mean_drawdown_delta: float
    sharpe_delta_m2: float
    fitness_delta_m2: float
    confidence: float
    effectiveness_score: float


@dataclass(frozen=True, slots=True)
class PolicyRecommendation:
    fix_name: str
    fix_category: str
    score: float
    confidence: float
    attempts: int
    expected_sharpe_delta: float
    expected_fitness_delta: float
    expected_turnover_delta: float
    expected_correlation_delta: float
    reason: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(prefix: str, *parts: Any) -> str:
    payload = canonical_json(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:32]}"


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bool_int(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _delta(after: float | None, before: float | None) -> float:
    a, b = _finite(after), _finite(before)
    return 0.0 if a is None or b is None else a - b


def _welford(count: int, mean: float, m2: float, value: float) -> tuple[int, float, float]:
    new_count = count + 1
    step = value - mean
    new_mean = mean + step / new_count
    new_m2 = m2 + step * (value - new_mean)
    return new_count, new_mean, new_m2


def _confidence(attempts: int, variance: float, maximum_attempts: int = 12) -> float:
    sample = min(attempts / maximum_attempts, 1.0)
    stability = 1.0 / (1.0 + max(variance, 0.0))
    return max(0.0, min(1.0, 0.70 * sample + 0.30 * stability))


def _effectiveness(*, success_rate: float, sharpe_delta: float, fitness_delta: float, turnover_delta: float, correlation_delta: float, confidence: float) -> float:
    metric_value = 0.35 * sharpe_delta + 0.25 * fitness_delta - 0.20 * max(turnover_delta, 0.0) - 0.20 * max(correlation_delta, 0.0)
    return round(0.55 * success_rate + 0.30 * metric_value + 0.15 * confidence, 6)


def fix_succeeded(before: OfficialMetrics, after: OfficialMetrics, target_gate: str) -> bool:
    if after.metrics_source != "WORLDQUANT_BRAIN" or not after.official_simulation_id:
        return False
    gate = target_gate.upper()
    if gate in {"HIGH_TURNOVER", "TURNOVER_SUBMISSION", "TURNOVER_GOOD_BUFFER"}:
        return bool(after.turnover is not None and before.turnover is not None and after.turnover < before.turnover and (after.fitness or -999) >= (before.fitness or -999) - 0.05 and (after.sharpe or -999) >= (before.sharpe or -999) - 0.10)
    if gate in {"HIGH_SELF_CORRELATION", "SELF_CORRELATION"}:
        return bool(after.self_correlation is not None and before.self_correlation is not None and after.self_correlation < before.self_correlation and after.self_correlation < 0.70 and (after.sharpe or -999) >= (before.sharpe or -999) - 0.10)
    if gate in {"LOW_SHARPE", "SHARPE_SUBMISSION", "SHARPE_GOOD"}:
        return bool(after.sharpe is not None and before.sharpe is not None and after.sharpe >= before.sharpe + 0.10 and after.turnover is not None and after.turnover < 0.70)
    if gate in {"WEIGHT_FAILURE", "WEIGHT_TEST", "MAX_INSTRUMENT_WEIGHT"}:
        return after.weight_test_passed is True and after.max_instrument_weight is not None and after.max_instrument_weight < 0.10
    if gate in {"SUB_UNIVERSE_FAILURE", "SUB_UNIVERSE_TEST"}:
        return after.sub_universe_passed is True and after.submit_ready
    if gate in {"BIAS_FAILURE", "BIAS_TEST"}:
        return after.bias_test_passed is True
    return after.submit_ready and not before.submit_ready


class EvidenceRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.database_path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
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
            con.executescript(SCHEMA_SQL)

    def create_experiment(self, context: ExperimentContext) -> bool:
        now = utc_now()
        with self.connection() as con:
            cursor = con.execute(
                """
                INSERT OR IGNORE INTO alpha_research_experiments (
                    experiment_id, alpha_id, candidate_id, parent_experiment_id,
                    blueprint_id, blueprint_hash, hypothesis_family, data_domain,
                    expression, expression_hash, normalized_expression_hash,
                    structural_expression_hash, settings_json, settings_hash,
                    region, universe, delay, experiment_type, experiment_status,
                    metrics_source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CREATED', 'UNKNOWN', ?)
                """,
                (
                    context.experiment_id, context.alpha_id, context.candidate_id,
                    context.parent_experiment_id, context.blueprint_id,
                    context.blueprint_hash, context.hypothesis_family,
                    context.data_domain, context.expression, context.expression_hash,
                    context.normalized_expression_hash, context.structural_expression_hash,
                    canonical_json(dict(context.settings)), context.settings_hash,
                    context.region, context.universe, context.delay,
                    context.experiment_type, now,
                ),
            )
            return cursor.rowcount == 1

    def mark_started(self, experiment_id: str, simulation_id: str | None = None) -> None:
        with self.connection() as con:
            con.execute(
                """UPDATE alpha_research_experiments
                   SET experiment_status='RUNNING', official_simulation_id=COALESCE(?, official_simulation_id),
                       started_at=COALESCE(started_at, ?)
                   WHERE experiment_id=?""",
                (simulation_id, utc_now(), experiment_id),
            )

    def record_metrics(self, experiment_id: str, metrics: OfficialMetrics) -> None:
        if metrics.good_ready and (metrics.metrics_source != "WORLDQUANT_BRAIN" or not metrics.official_simulation_id):
            raise ValueError("GOOD metrics require official BRAIN provenance")
        if metrics.self_correlation is not None and not (-1.0 <= metrics.self_correlation <= 1.0):
            raise ValueError("self_correlation out of range")
        now = utc_now()
        with self.connection() as con:
            con.execute(
                """
                INSERT INTO alpha_experiment_metrics (
                    experiment_id, sharpe, fitness, returns, turnover, margin, drawdown,
                    self_correlation, production_correlation, max_instrument_weight,
                    long_count, short_count, weight_test_passed, sub_universe_passed,
                    ranked_sharpe_passed, bias_test_passed, submit_ready, good_ready,
                    robust_good, quality_class, quality_score, raw_metrics_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    sharpe=excluded.sharpe, fitness=excluded.fitness, returns=excluded.returns,
                    turnover=excluded.turnover, margin=excluded.margin, drawdown=excluded.drawdown,
                    self_correlation=excluded.self_correlation,
                    production_correlation=excluded.production_correlation,
                    max_instrument_weight=excluded.max_instrument_weight,
                    long_count=excluded.long_count, short_count=excluded.short_count,
                    weight_test_passed=excluded.weight_test_passed,
                    sub_universe_passed=excluded.sub_universe_passed,
                    ranked_sharpe_passed=excluded.ranked_sharpe_passed,
                    bias_test_passed=excluded.bias_test_passed,
                    submit_ready=excluded.submit_ready, good_ready=excluded.good_ready,
                    robust_good=excluded.robust_good, quality_class=excluded.quality_class,
                    quality_score=excluded.quality_score,
                    raw_metrics_json=excluded.raw_metrics_json,
                    recorded_at=excluded.recorded_at
                """,
                (
                    experiment_id, metrics.sharpe, metrics.fitness, metrics.returns,
                    metrics.turnover, metrics.margin, metrics.drawdown,
                    metrics.self_correlation, metrics.production_correlation,
                    metrics.max_instrument_weight, metrics.long_count, metrics.short_count,
                    _bool_int(metrics.weight_test_passed), _bool_int(metrics.sub_universe_passed),
                    _bool_int(metrics.ranked_sharpe_passed), _bool_int(metrics.bias_test_passed),
                    int(metrics.submit_ready), int(metrics.good_ready), int(metrics.robust_good),
                    metrics.quality_class, metrics.quality_score,
                    canonical_json(dict(metrics.raw_metrics or {})), now,
                ),
            )
            con.execute(
                """UPDATE alpha_research_experiments
                   SET metrics_source=?, official_simulation_id=COALESCE(?, official_simulation_id),
                       experiment_status='COMPLETED', completed_at=COALESCE(completed_at, ?)
                   WHERE experiment_id=?""",
                (metrics.metrics_source, metrics.official_simulation_id, now, experiment_id),
            )

    def record_failures(self, experiment_id: str, failures: Sequence[FailureObservation]) -> None:
        now = utc_now()
        with self.connection() as con:
            for failure in failures:
                failure_id = stable_id("fail", experiment_id, failure.gate_name, failure.failure_code)
                con.execute(
                    """INSERT OR IGNORE INTO alpha_experiment_failures
                       (failure_id, experiment_id, gate_name, severity, actual_value,
                        required_value, failure_code, failure_context_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (failure_id, experiment_id, failure.gate_name, failure.severity,
                     failure.actual_value, failure.required_value, failure.failure_code,
                     canonical_json(dict(failure.context or {})), now),
                )

    def record_fixes(self, experiment_id: str, fixes: Sequence[AppliedFix]) -> None:
        now = utc_now()
        with self.connection() as con:
            for fix in fixes:
                fix_id = stable_id("fix", experiment_id, fix.fix_name, fix.intended_gate)
                con.execute(
                    """INSERT OR IGNORE INTO alpha_experiment_fixes
                       (experiment_fix_id, experiment_id, fix_name, fix_category,
                        before_value_json, after_value_json, intended_gate,
                        structural_change, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (fix_id, experiment_id, fix.fix_name, fix.fix_category,
                     canonical_json(fix.before_value), canonical_json(fix.after_value),
                     fix.intended_gate, int(fix.structural_change), now),
                )

    def get_metrics(self, experiment_id: str) -> OfficialMetrics:
        with self.connection() as con:
            row = con.execute(
                """SELECT e.metrics_source, e.official_simulation_id, m.*
                   FROM alpha_research_experiments e
                   JOIN alpha_experiment_metrics m USING(experiment_id)
                   WHERE e.experiment_id=?""", (experiment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return OfficialMetrics(
            metrics_source=row["metrics_source"], official_simulation_id=row["official_simulation_id"],
            sharpe=row["sharpe"], fitness=row["fitness"], returns=row["returns"], turnover=row["turnover"],
            margin=row["margin"], drawdown=row["drawdown"], self_correlation=row["self_correlation"],
            production_correlation=row["production_correlation"], max_instrument_weight=row["max_instrument_weight"],
            long_count=row["long_count"], short_count=row["short_count"],
            weight_test_passed=_to_bool(row["weight_test_passed"]), sub_universe_passed=_to_bool(row["sub_universe_passed"]),
            ranked_sharpe_passed=_to_bool(row["ranked_sharpe_passed"]), bias_test_passed=_to_bool(row["bias_test_passed"]),
            submit_ready=bool(row["submit_ready"]), good_ready=bool(row["good_ready"]), robust_good=bool(row["robust_good"]),
            quality_class=row["quality_class"], quality_score=row["quality_score"], raw_metrics=json.loads(row["raw_metrics_json"]),
        )

    def update_fix_evidence(self, *, child_experiment_id: str) -> list[EvidenceAggregate]:
        """Consume each child/fix evidence exactly once.

        A unique row in evidence_consumption_log is inserted in the same
        transaction as the aggregate update. Replayed queue messages, process
        restarts, repeated polling, and duplicate webhook delivery therefore do
        not increment attempts twice.
        """
        with self.connection() as con:
            child = con.execute(
                "SELECT * FROM alpha_research_experiments WHERE experiment_id=?",
                (child_experiment_id,),
            ).fetchone()
            if child is None or not child["parent_experiment_id"]:
                raise ValueError("Child experiment with parent is required")
            fixes = con.execute(
                "SELECT * FROM alpha_experiment_fixes WHERE experiment_id=? ORDER BY experiment_fix_id",
                (child_experiment_id,),
            ).fetchall()

        before = self.get_metrics(child["parent_experiment_id"])
        after = self.get_metrics(child_experiment_id)
        results: list[EvidenceAggregate] = []
        for fix in fixes:
            aggregate = self._consume_and_update_one_evidence(child, fix, before, after)
            if aggregate is not None:
                results.append(aggregate)
        return results

    def _consume_and_update_one_evidence(
        self,
        child: sqlite3.Row,
        fix: sqlite3.Row,
        before: OfficialMetrics,
        after: OfficialMetrics,
    ) -> EvidenceAggregate | None:
        context = (
            child["hypothesis_family"], child["data_domain"],
            fix["intended_gate"], fix["fix_name"], child["region"],
            child["universe"], child["delay"],
        )
        evidence_id = stable_id("evidence", *context)
        consumption_id = stable_id(
            "consume", child["experiment_id"], fix["experiment_fix_id"], evidence_id
        )
        now = utc_now()

        # BEGIN IMMEDIATE serializes competing consumers before either can read
        # and update the aggregate. The consumption row and aggregate update are
        # committed atomically.
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            inserted = con.execute(
                """INSERT OR IGNORE INTO evidence_consumption_log (
                       consumption_id, experiment_id, experiment_fix_id,
                       evidence_id, source_simulation_id, consumed_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    consumption_id, child["experiment_id"],
                    fix["experiment_fix_id"], evidence_id,
                    after.official_simulation_id, now,
                ),
            )
            if inserted.rowcount == 0:
                return None
            return self._update_one_evidence_in_transaction(
                con, child, fix, before, after, evidence_id=evidence_id, now=now
            )

    def _update_one_evidence_in_transaction(
        self,
        con: sqlite3.Connection,
        child: sqlite3.Row,
        fix: sqlite3.Row,
        before: OfficialMetrics,
        after: OfficialMetrics,
        *,
        evidence_id: str,
        now: str,
    ) -> EvidenceAggregate:
        success = fix_succeeded(before, after, fix["intended_gate"])
        deltas = {
            "sharpe": _delta(after.sharpe, before.sharpe),
            "fitness": _delta(after.fitness, before.fitness),
            "turnover": _delta(after.turnover, before.turnover),
            "correlation": _delta(after.self_correlation, before.self_correlation),
            "margin": _delta(after.margin, before.margin),
            "drawdown": _delta(after.drawdown, before.drawdown),
        }
        row = con.execute(
            "SELECT * FROM research_fix_evidence WHERE evidence_id=?",
            (evidence_id,),
        ).fetchone()
        if row is None:
            attempts, successes, failures = 0, 0, 0
            means = {key: 0.0 for key in deltas}
            sharpe_m2 = fitness_m2 = 0.0
            first_at = now
        else:
            attempts, successes, failures = row["attempts"], row["successful_attempts"], row["failed_attempts"]
            means = {"sharpe": row["mean_sharpe_delta"], "fitness": row["mean_fitness_delta"], "turnover": row["mean_turnover_delta"], "correlation": row["mean_correlation_delta"], "margin": row["mean_margin_delta"], "drawdown": row["mean_drawdown_delta"]}
            sharpe_m2, fitness_m2, first_at = row["sharpe_delta_m2"], row["fitness_delta_m2"], row["first_evidence_at"]
        old_count = attempts
        _, means["sharpe"], sharpe_m2 = _welford(old_count, means["sharpe"], sharpe_m2, deltas["sharpe"])
        _, means["fitness"], fitness_m2 = _welford(old_count, means["fitness"], fitness_m2, deltas["fitness"])
        for key in ("turnover", "correlation", "margin", "drawdown"):
            _, means[key], _ = _welford(old_count, means[key], 0.0, deltas[key])
        attempts += 1
        successes += int(success)
        failures += int(not success)
        variance = sharpe_m2 / (attempts - 1) if attempts > 1 else 0.0
        confidence = _confidence(attempts, variance)
        success_rate = (successes + 1) / (attempts + 2)
        score = _effectiveness(success_rate=success_rate, sharpe_delta=means["sharpe"], fitness_delta=means["fitness"], turnover_delta=means["turnover"], correlation_delta=means["correlation"], confidence=confidence)
        con.execute(
            """INSERT INTO research_fix_evidence (
                evidence_id, hypothesis_family, data_domain, failure_gate, fix_name, fix_category,
                region, universe, delay, attempts, successful_attempts, failed_attempts,
                mean_sharpe_delta, mean_fitness_delta, mean_turnover_delta, mean_correlation_delta,
                mean_margin_delta, mean_drawdown_delta, sharpe_delta_m2, fitness_delta_m2,
                confidence, effectiveness_score, first_evidence_at, last_evidence_at,
                last_experiment_id, evidence_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(evidence_id) DO UPDATE SET
                attempts=excluded.attempts, successful_attempts=excluded.successful_attempts,
                failed_attempts=excluded.failed_attempts, mean_sharpe_delta=excluded.mean_sharpe_delta,
                mean_fitness_delta=excluded.mean_fitness_delta, mean_turnover_delta=excluded.mean_turnover_delta,
                mean_correlation_delta=excluded.mean_correlation_delta, mean_margin_delta=excluded.mean_margin_delta,
                mean_drawdown_delta=excluded.mean_drawdown_delta, sharpe_delta_m2=excluded.sharpe_delta_m2,
                fitness_delta_m2=excluded.fitness_delta_m2, confidence=excluded.confidence,
                effectiveness_score=excluded.effectiveness_score, last_evidence_at=excluded.last_evidence_at,
                last_experiment_id=excluded.last_experiment_id, evidence_version=research_fix_evidence.evidence_version+1
            """,
            (evidence_id, child["hypothesis_family"], child["data_domain"], fix["intended_gate"], fix["fix_name"], fix["fix_category"], child["region"], child["universe"], child["delay"], attempts, successes, failures, means["sharpe"], means["fitness"], means["turnover"], means["correlation"], means["margin"], means["drawdown"], sharpe_m2, fitness_m2, confidence, score, first_at, now, child["experiment_id"]),
        )
        saved = con.execute("SELECT * FROM research_fix_evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        return _row_to_evidence(saved)

    def evidence_was_consumed(self, *, experiment_id: str, evidence_id: str) -> bool:
        with self.connection() as con:
            row = con.execute(
                "SELECT 1 FROM evidence_consumption_log WHERE experiment_id=? AND evidence_id=? LIMIT 1",
                (experiment_id, evidence_id),
            ).fetchone()
        return row is not None

    def list_consumptions(self, *, experiment_id: str) -> list[dict[str, Any]]:
        with self.connection() as con:
            rows = con.execute(
                "SELECT * FROM evidence_consumption_log WHERE experiment_id=? ORDER BY consumed_at",
                (experiment_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recommend_fixes(self, *, hypothesis_family: str, data_domain: str, failure_gate: str, region: str, universe: str, delay: int, limit: int = 3, exploration_weight: float = 0.15) -> list[PolicyRecommendation]:
        with self.connection() as con:
            rows = con.execute(
                """SELECT * FROM research_fix_evidence
                   WHERE hypothesis_family=? AND data_domain=? AND failure_gate=?
                     AND region=? AND universe=? AND delay=?
                   ORDER BY effectiveness_score DESC, confidence DESC LIMIT ?""",
                (hypothesis_family, data_domain, failure_gate, region, universe, delay, limit * 3),
            ).fetchall()
        total_attempts = sum(row["attempts"] for row in rows)
        recommendations: list[PolicyRecommendation] = []
        for row in rows:
            exploration = exploration_weight * math.sqrt(math.log(total_attempts + 1) / (row["attempts"] + 1)) if total_attempts else exploration_weight
            score = row["effectiveness_score"] + exploration
            recommendations.append(PolicyRecommendation(
                fix_name=row["fix_name"], fix_category=row["fix_category"], score=round(score, 6),
                confidence=row["confidence"], attempts=row["attempts"], expected_sharpe_delta=row["mean_sharpe_delta"],
                expected_fitness_delta=row["mean_fitness_delta"], expected_turnover_delta=row["mean_turnover_delta"],
                expected_correlation_delta=row["mean_correlation_delta"],
                reason=f"attempts={row['attempts']}, success={row['successful_attempts']}, confidence={row['confidence']:.3f}",
            ))
        return sorted(recommendations, key=lambda item: item.score, reverse=True)[:limit]

    def record_policy_decision(self, *, candidate_id: str, blueprint_id: str, hypothesis_family: str, data_domain: str, detected_failure_gate: str | None, selected_action: str, selected_fix: str | None, considered_evidence_ids: Sequence[str], rejected_actions: Sequence[str], expected_utility: float | None, exploration_probability: float, policy_version: str, decision_context: Mapping[str, Any] | None = None) -> str:
        decision_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO research_policy_decisions
                   (decision_id, candidate_id, blueprint_id, hypothesis_family, data_domain,
                    detected_failure_gate, considered_lesson_ids_json, considered_evidence_ids_json,
                    selected_action, selected_fix, rejected_actions_json, expected_utility,
                    exploration_probability, policy_version, decision_context_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (decision_id, candidate_id, blueprint_id, hypothesis_family, data_domain,
                 detected_failure_gate, canonical_json(list(considered_evidence_ids)), selected_action,
                 selected_fix, canonical_json(list(rejected_actions)), expected_utility,
                 exploration_probability, policy_version, canonical_json(dict(decision_context or {})), utc_now()),
            )
        return decision_id


def _to_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _row_to_evidence(row: sqlite3.Row) -> EvidenceAggregate:
    return EvidenceAggregate(**{field: row[field] for field in EvidenceAggregate.__dataclass_fields__})


SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS research_memory_versions (
    policy_version TEXT PRIMARY KEY, description TEXT NOT NULL,
    config_json TEXT NOT NULL, config_sha256 TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alpha_research_experiments (
    experiment_id TEXT PRIMARY KEY, alpha_id TEXT, candidate_id TEXT NOT NULL,
    parent_experiment_id TEXT, blueprint_id TEXT NOT NULL, blueprint_hash TEXT NOT NULL,
    hypothesis_family TEXT NOT NULL, data_domain TEXT NOT NULL, expression TEXT NOT NULL,
    expression_hash TEXT NOT NULL, normalized_expression_hash TEXT NOT NULL,
    structural_expression_hash TEXT NOT NULL, settings_json TEXT NOT NULL,
    settings_hash TEXT NOT NULL, region TEXT NOT NULL, universe TEXT NOT NULL,
    delay INTEGER NOT NULL, experiment_type TEXT NOT NULL, experiment_status TEXT NOT NULL,
    metrics_source TEXT NOT NULL DEFAULT 'UNKNOWN', official_simulation_id TEXT,
    started_at TEXT, completed_at TEXT, created_at TEXT NOT NULL, error_code TEXT, safe_error TEXT,
    FOREIGN KEY(parent_experiment_id) REFERENCES alpha_research_experiments(experiment_id),
    UNIQUE(expression_hash, settings_hash, region, universe, delay)
);
CREATE TABLE IF NOT EXISTS alpha_experiment_metrics (
    experiment_id TEXT PRIMARY KEY, sharpe REAL, fitness REAL, returns REAL, turnover REAL,
    margin REAL, drawdown REAL, self_correlation REAL, production_correlation REAL,
    max_instrument_weight REAL, long_count INTEGER, short_count INTEGER,
    weight_test_passed INTEGER, sub_universe_passed INTEGER, ranked_sharpe_passed INTEGER,
    bias_test_passed INTEGER, submit_ready INTEGER NOT NULL DEFAULT 0,
    good_ready INTEGER NOT NULL DEFAULT 0, robust_good INTEGER NOT NULL DEFAULT 0,
    quality_class TEXT NOT NULL DEFAULT 'UNVERIFIED', quality_score REAL,
    raw_metrics_json TEXT NOT NULL DEFAULT '{}', recorded_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES alpha_research_experiments(experiment_id) ON DELETE CASCADE,
    CHECK(self_correlation IS NULL OR self_correlation BETWEEN -1.0 AND 1.0)
);
CREATE TABLE IF NOT EXISTS alpha_experiment_failures (
    failure_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, gate_name TEXT NOT NULL,
    severity TEXT NOT NULL, actual_value REAL, required_value REAL, failure_code TEXT NOT NULL,
    failure_context_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES alpha_research_experiments(experiment_id) ON DELETE CASCADE,
    UNIQUE(experiment_id, gate_name, failure_code)
);
CREATE TABLE IF NOT EXISTS alpha_experiment_fixes (
    experiment_fix_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, fix_name TEXT NOT NULL,
    fix_category TEXT NOT NULL, before_value_json TEXT, after_value_json TEXT,
    intended_gate TEXT NOT NULL, structural_change INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES alpha_research_experiments(experiment_id) ON DELETE CASCADE,
    UNIQUE(experiment_id, fix_name, intended_gate)
);
CREATE TABLE IF NOT EXISTS research_fix_evidence (
    evidence_id TEXT PRIMARY KEY, hypothesis_family TEXT NOT NULL, data_domain TEXT NOT NULL,
    failure_gate TEXT NOT NULL, fix_name TEXT NOT NULL, fix_category TEXT NOT NULL,
    region TEXT NOT NULL, universe TEXT NOT NULL, delay INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, successful_attempts INTEGER NOT NULL DEFAULT 0,
    failed_attempts INTEGER NOT NULL DEFAULT 0, mean_sharpe_delta REAL NOT NULL DEFAULT 0,
    mean_fitness_delta REAL NOT NULL DEFAULT 0, mean_turnover_delta REAL NOT NULL DEFAULT 0,
    mean_correlation_delta REAL NOT NULL DEFAULT 0, mean_margin_delta REAL NOT NULL DEFAULT 0,
    mean_drawdown_delta REAL NOT NULL DEFAULT 0, sharpe_delta_m2 REAL NOT NULL DEFAULT 0,
    fitness_delta_m2 REAL NOT NULL DEFAULT 0, confidence REAL NOT NULL DEFAULT 0,
    effectiveness_score REAL NOT NULL DEFAULT 0, first_evidence_at TEXT NOT NULL,
    last_evidence_at TEXT NOT NULL, last_experiment_id TEXT, evidence_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE(hypothesis_family,data_domain,failure_gate,fix_name,region,universe,delay)
);
CREATE TABLE IF NOT EXISTS evidence_consumption_log (
    consumption_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    experiment_fix_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    source_simulation_id TEXT,
    consumed_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES alpha_research_experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY(experiment_fix_id) REFERENCES alpha_experiment_fixes(experiment_fix_id) ON DELETE CASCADE,
    FOREIGN KEY(evidence_id) REFERENCES research_fix_evidence(evidence_id) ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
    UNIQUE(experiment_id, experiment_fix_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_consumption_experiment
ON evidence_consumption_log(experiment_id, consumed_at);
CREATE INDEX IF NOT EXISTS idx_consumption_evidence
ON evidence_consumption_log(evidence_id, consumed_at);
CREATE TABLE IF NOT EXISTS research_policy_decisions (
    decision_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, blueprint_id TEXT NOT NULL,
    hypothesis_family TEXT NOT NULL, data_domain TEXT NOT NULL, detected_failure_gate TEXT,
    considered_lesson_ids_json TEXT NOT NULL DEFAULT '[]', considered_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    selected_action TEXT NOT NULL, selected_fix TEXT, rejected_actions_json TEXT NOT NULL DEFAULT '[]',
    expected_utility REAL, exploration_probability REAL NOT NULL DEFAULT 0,
    policy_version TEXT NOT NULL, decision_context_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_lookup ON research_fix_evidence(hypothesis_family,data_domain,failure_gate,effectiveness_score DESC);
CREATE INDEX IF NOT EXISTS idx_experiment_context ON alpha_research_experiments(hypothesis_family,data_domain,experiment_status);
"""


def build_evidence_context(
    *,
    before: Any,
    after: Any,
    policy: Any,
    action: str | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from dataclasses import asdict
    from worldquant_brain.optimization_policy import policy_snapshot

    snapshot = policy_snapshot(after, policy)
    return {
        "optimization_objective": policy.objective.value,
        "before_metrics": asdict(before) if before else None,
        "after_metrics": asdict(after),
        "turnover_efficiency": snapshot["turnover_efficiency"],
        "diagnoses": snapshot["diagnoses"],
        "selected_action": action,
        "recommended_actions": snapshot["actions"],
        "extra": dict(extra or {}),
    }
