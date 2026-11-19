"""Durable Alpha experiment and experience-memory store."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_expression(expression: str) -> str:
    from worldquant_brain.canonical_ast import parse_fastexpr

    return parse_fastexpr(expression).canonical_expression


def expression_hash(expression: str) -> str:
    from worldquant_brain.canonical_ast import parse_fastexpr

    return parse_fastexpr(expression).exact_hash


class AlphaExperimentStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS alpha_experiments (
                experiment_id TEXT PRIMARY KEY,
                expression_hash TEXT NOT NULL,
                normalized_expression TEXT NOT NULL,
                source TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 20,
                operator_features_json TEXT NOT NULL DEFAULT '{}',
                data_fields_json TEXT NOT NULL DEFAULT '[]',
                settings_json TEXT NOT NULL DEFAULT '{}',
                simulation_status TEXT NOT NULL,
                brain_simulation_id TEXT,
                brain_alpha_id TEXT,
                sharpe REAL,
                fitness REAL,
                turnover REAL,
                margin REAL,
                drawdown REAL,
                self_correlation REAL,
                production_correlation REAL,
                is_check_status TEXT,
                failure_code TEXT,
                safe_error TEXT,
                simulation_latency_seconds REAL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_alpha_experiments_hash
                ON alpha_experiments(expression_hash);
            CREATE INDEX IF NOT EXISTS ix_alpha_experiments_completed
                ON alpha_experiments(completed_at);
            CREATE TABLE IF NOT EXISTS alpha_experience_memory (
                pattern_hash TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                feature_json TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                mean_sharpe REAL,
                mean_fitness REAL,
                mean_margin REAL,
                last_failure_code TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            """)

    def start_experiment(
        self,
        experiment_id: str,
        expression: str,
        *, source: str = "AUTO_MINING",
        priority: int = 20,
        settings: Mapping[str, Any] | None = None,
        operator_features: Mapping[str, Any] | None = None,
        data_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_expression(expression)
        ehash = expression_hash(expression)
        now = utc_now()
        with self.connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO alpha_experiments (
                    experiment_id, expression_hash, normalized_expression, source, priority,
                    operator_features_json, data_fields_json, settings_json,
                    simulation_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'QUEUED', ?)
            """, (
                experiment_id, ehash, normalized, source, int(priority),
                json.dumps(operator_features or {}, sort_keys=True),
                json.dumps(data_fields or []), json.dumps(dict(settings or {}), sort_keys=True), now,
            ))
        return {"experiment_id": experiment_id, "expression_hash": ehash, "created_at": now}

    def complete_experiment(self, experiment_id: str, result: Mapping[str, Any]) -> None:
        status = str(result.get("status") or "COMPLETED")
        success = status in {"COMPLETED", "BRAIN_COMPLETE", "IS_CHECK_PASSED", "SUBMITTED"}
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT expression_hash, operator_features_json FROM alpha_experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            if not row:
                raise KeyError(experiment_id)
            conn.execute("""
                UPDATE alpha_experiments SET
                    simulation_status=?, brain_simulation_id=?, brain_alpha_id=?,
                    sharpe=?, fitness=?, turnover=?, margin=?, drawdown=?,
                    self_correlation=?, production_correlation=?, is_check_status=?,
                    failure_code=?, safe_error=?, simulation_latency_seconds=?, completed_at=?
                WHERE experiment_id=?
            """, (
                status, result.get("brain_simulation_id") or result.get("simulation_id"),
                result.get("brain_alpha_id") or result.get("alpha_id"), result.get("sharpe"),
                result.get("fitness"), result.get("turnover"), result.get("margin"),
                result.get("drawdown"), result.get("self_correlation"),
                result.get("production_correlation"), result.get("is_check_status"),
                result.get("failure_code"), result.get("safe_error"),
                result.get("simulation_latency_seconds"), now, experiment_id,
            ))
            old = conn.execute(
                "SELECT * FROM alpha_experience_memory WHERE pattern_hash=?", (row["expression_hash"],)
            ).fetchone()
            n_success = (old["success_count"] if old else 0) + int(success)
            n_failure = (old["failure_count"] if old else 0) + int(not success)
            def updated_mean(old_mean: float | None, value: Any, count: int) -> float | None:
                if value is None:
                    return old_mean
                return float(value) if count <= 1 or old_mean is None else old_mean + (float(value) - old_mean) / count
            conn.execute("""
                INSERT INTO alpha_experience_memory (
                    pattern_hash, pattern_type, feature_json, success_count, failure_count,
                    mean_sharpe, mean_fitness, mean_margin, last_failure_code,
                    first_seen_at, last_seen_at
                ) VALUES (?, 'EXPRESSION', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_hash) DO UPDATE SET
                    success_count=excluded.success_count,
                    failure_count=excluded.failure_count,
                    mean_sharpe=excluded.mean_sharpe,
                    mean_fitness=excluded.mean_fitness,
                    mean_margin=excluded.mean_margin,
                    last_failure_code=excluded.last_failure_code,
                    last_seen_at=excluded.last_seen_at
            """, (
                row["expression_hash"], row["operator_features_json"], n_success, n_failure,
                updated_mean(old["mean_sharpe"] if old else None, result.get("sharpe"), n_success),
                updated_mean(old["mean_fitness"] if old else None, result.get("fitness"), n_success),
                updated_mean(old["mean_margin"] if old else None, result.get("margin"), n_success),
                result.get("failure_code"), old["first_seen_at"] if old else now, now,
            ))

    def export_training_rows(self) -> list[dict[str, Any]]:
        """Rows for a separate BRAIN Alpha meta-model, ordered chronologically."""
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT * FROM alpha_experiments
                WHERE completed_at IS NOT NULL
                ORDER BY completed_at, expression_hash
            """).fetchall()
        return [dict(r) for r in rows]
