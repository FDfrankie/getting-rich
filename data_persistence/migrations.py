"""Versioned schema migration framework for all 20 required Antigravity Quant tables (SQLite & PostgreSQL)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

REQUIRED_TABLES: tuple[str, ...] = (
    "alphas",
    "alpha_experiments",
    "alpha_experience_memory",
    "brain_jobs",
    "brain_candidate_pool",
    "brain_scheduler_runs",
    "brain_submissions",
    "brain_score_snapshots",
    "brain_submission_score_attribution",
    "audit_events",
    "dataset_manifests",
    "model_manifests",
    "artifact_registry",
    "taiwan_universe_snapshots",
    "taiwan_security_history",
    "shadow_predictions",
    "order_intents",
    "account_snapshots",
    "reconciliation_events",
    "schema_migrations",
    "alpha_blueprints",
    "alpha_robustness_runs",
    "alpha_robustness_reports",
    "research_lessons",
)

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alphas (
    alpha_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    expression TEXT NOT NULL,
    expression_hash TEXT NOT NULL UNIQUE,
    ast_family_hash TEXT,
    status TEXT NOT NULL,
    brain_simulation_id TEXT,
    brain_alpha_id TEXT,
    sharpe REAL,
    fitness REAL,
    turnover REAL,
    margin_bps REAL,
    drawdown REAL,
    is_check_status TEXT,
    self_correlation REAL,
    submission_utility REAL,
    utility_components_json TEXT,
    submitted_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alpha_experiments (
    experiment_id TEXT PRIMARY KEY,
    expression TEXT NOT NULL,
    expression_hash TEXT NOT NULL,
    ast_family_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 20,
    settings_json TEXT NOT NULL,
    metrics_json TEXT,
    failure_code TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS alpha_experience_memory (
    memory_id TEXT PRIMARY KEY,
    expression_hash TEXT NOT NULL UNIQUE,
    normalized_expression TEXT NOT NULL,
    ast_family_hash TEXT NOT NULL,
    operators_json TEXT NOT NULL,
    data_fields_json TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT 'USA',
    universe TEXT NOT NULL DEFAULT 'TOP3000',
    delay INTEGER NOT NULL DEFAULT 1,
    decay INTEGER NOT NULL DEFAULT 4,
    truncation REAL NOT NULL DEFAULT 0.08,
    neutralization TEXT NOT NULL DEFAULT 'SUBINDUSTRY',
    simulation_status TEXT NOT NULL,
    simulation_latency_sec REAL,
    sharpe REAL,
    fitness REAL,
    turnover REAL,
    margin_bps REAL,
    drawdown REAL,
    is_checks_json TEXT,
    self_correlation REAL,
    production_correlation REAL,
    failure_code TEXT,
    official_response_hash TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS brain_jobs (
    job_id TEXT PRIMARY KEY,
    simulation_key TEXT NOT NULL UNIQUE,
    expression TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    status TEXT NOT NULL,
    simulation_location TEXT,
    alpha_id TEXT,
    metrics_json TEXT,
    safe_error TEXT,
    priority INTEGER NOT NULL DEFAULT 20,
    source TEXT NOT NULL DEFAULT 'AUTO_MINING',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS brain_candidate_pool (
    alpha_id TEXT PRIMARY KEY,
    expression TEXT NOT NULL,
    expression_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    sharpe REAL,
    fitness REAL,
    turnover REAL,
    margin_bps REAL,
    self_correlation REAL,
    is_check_status TEXT,
    submission_utility REAL,
    utility_json TEXT,
    candidate_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    submitted_at TEXT
);

CREATE TABLE IF NOT EXISTS brain_scheduler_runs (
    taipei_date TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    scheduled_at TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    claimed_by TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    candidates_considered INTEGER NOT NULL DEFAULT 0,
    candidates_submitted INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS brain_submissions (
    submission_id TEXT PRIMARY KEY,
    alpha_id TEXT NOT NULL,
    brain_alpha_id TEXT,
    brain_simulation_id TEXT,
    status TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    submit_acknowledgement TEXT,
    submit_response_hash TEXT,
    official_status TEXT,
    score_before REAL,
    score_after REAL,
    attribution_status TEXT
);

CREATE TABLE IF NOT EXISTS brain_score_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    observed_at TEXT NOT NULL,
    taipei_date TEXT NOT NULL,
    total_score REAL,
    daily_score REAL,
    daily_cap REAL,
    rank_value INTEGER,
    submission_days_60d INTEGER,
    uniqueness REAL,
    is_score REAL,
    source TEXT NOT NULL,
    cap_evidence TEXT NOT NULL,
    raw_payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS brain_submission_score_attribution (
    attribution_id TEXT PRIMARY KEY,
    brain_alpha_id TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    score_before REAL,
    score_after REAL,
    score_delta REAL,
    score_observed_at TEXT,
    attribution_status TEXT NOT NULL,
    batch_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    entity_id TEXT,
    previous_state TEXT,
    next_state TEXT,
    reason TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_manifests (
    dataset_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    market_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    expected_rows INTEGER NOT NULL,
    actual_rows INTEGER NOT NULL,
    missing_timestamps INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    invalid_ohlc_count INTEGER NOT NULL DEFAULT 0,
    closed_candle_status TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    validated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_manifests (
    model_id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    deployment_stage TEXT NOT NULL,
    approved_for_inference INTEGER NOT NULL DEFAULT 0,
    approved_for_order_entry INTEGER NOT NULL DEFAULT 0,
    manifest_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_registry (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS taiwan_universe_snapshots (
    as_of_date TEXT PRIMARY KEY,
    rows_count INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    markets_json TEXT NOT NULL,
    source_urls_json TEXT NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS taiwan_security_history (
    symbol TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    name TEXT NOT NULL,
    market TEXT NOT NULL,
    industry TEXT NOT NULL,
    active_state INTEGER NOT NULL DEFAULT 1,
    trading_halt INTEGER NOT NULL DEFAULT 0,
    disposition_status TEXT,
    warning_status TEXT,
    close_price REAL,
    volume INTEGER,
    pe_ratio REAL,
    dividend_yield REAL,
    pb_ratio REAL,
    foreign_net_lots REAL,
    trust_net_lots REAL,
    dealer_net_lots REAL,
    margin_balance_lots REAL,
    short_balance_lots REAL,
    PRIMARY KEY (symbol, as_of_date)
);

CREATE TABLE IF NOT EXISTS shadow_predictions (
    prediction_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    model_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    features_hash TEXT NOT NULL,
    signal REAL NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_intents (
    intent_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    model_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    features_hash TEXT NOT NULL,
    signal REAL NOT NULL,
    confidence REAL NOT NULL,
    target_position REAL NOT NULL,
    risk_decision TEXT NOT NULL,
    cost_estimate_bps REAL NOT NULL,
    expected_fill_price REAL,
    counterfactual_pnl REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    execution_mode TEXT NOT NULL,
    equity_usdt REAL NOT NULL,
    balance_usdt REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    positions_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reconciliation_events (
    recon_id TEXT PRIMARY KEY,
    execution_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    unknown_orders INTEGER NOT NULL DEFAULT 0,
    position_mismatch INTEGER NOT NULL DEFAULT 0,
    details_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alpha_blueprints (
    blueprint_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    blueprint_hash TEXT NOT NULL UNIQUE,
    family TEXT NOT NULL,
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    falsifiable_claim TEXT NOT NULL,
    blueprint_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    superseded_by TEXT,
    PRIMARY KEY (blueprint_id, version)
);

CREATE TABLE IF NOT EXISTS alpha_robustness_runs (
    run_id TEXT PRIMARY KEY,
    alpha_id TEXT NOT NULL,
    blueprint_id TEXT NOT NULL,
    variant_type TEXT NOT NULL,
    variant_value TEXT NOT NULL,
    simulation_id TEXT,
    metrics_source TEXT NOT NULL,
    sharpe REAL,
    fitness REAL,
    turnover REAL,
    returns REAL,
    drawdown REAL,
    pnl REAL,
    year INTEGER,
    regime TEXT,
    universe TEXT,
    parameters_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(alpha_id, variant_type, variant_value, simulation_id)
);

CREATE TABLE IF NOT EXISTS alpha_robustness_reports (
    alpha_id TEXT PRIMARY KEY,
    blueprint_id TEXT NOT NULL,
    blueprint_hash TEXT NOT NULL,
    baseline_run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL,
    report_json TEXT NOT NULL,
    promotion_ready INTEGER NOT NULL DEFAULT 0,
    evaluated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_lessons (
    lesson_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    data_domain TEXT NOT NULL,
    failure_gate TEXT NOT NULL,
    effective_fix TEXT,
    ineffective_fixes_json TEXT NOT NULL,
    evidence_run_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""



def apply_sqlite_migrations(conn: sqlite3.Connection) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.executescript(SQLITE_DDL)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
            ("20261118_001_batches_a_to_j", now, "Full 20-table production governance schema"),
        )
    tables = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    missing = [t for t in REQUIRED_TABLES if t not in tables]
    apply_alpha_vault_quality_columns(conn, "alphas")
    apply_alpha_vault_quality_columns(conn, "alpha_vault")
    return {"applied_version": "20261119_002_blueprint_robustness", "tables_count": len(tables), "missing_tables": missing}


ALPHA_VAULT_QUALITY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("quality_class", "TEXT NOT NULL DEFAULT 'UNVERIFIED'"),
    ("quality_score", "REAL"),
    ("good_ready", "INTEGER NOT NULL DEFAULT 0"),
    ("submit_ready", "INTEGER NOT NULL DEFAULT 0"),
    ("metrics_source", "TEXT NOT NULL DEFAULT 'UNKNOWN'"),
    ("official_simulation_id", "TEXT"),
    ("self_correlation", "REAL"),
    ("production_correlation", "REAL"),
    ("max_instrument_weight", "REAL"),
    ("sub_universe_passed", "INTEGER"),
    ("weight_test_passed", "INTEGER"),
    ("ranked_sharpe_passed", "INTEGER"),
    ("bias_test_passed", "INTEGER"),
    ("quality_report_json", "TEXT"),
    ("evaluated_at", "TEXT"),
    ("robustness_status", "TEXT NOT NULL DEFAULT 'NOT_EVALUATED'"),
    ("robustness_score", "REAL"),
    ("promotion_ready", "INTEGER NOT NULL DEFAULT 0"),
    ("blueprint_id", "TEXT"),
    ("blueprint_hash", "TEXT"),
)


def apply_alpha_vault_quality_columns(conn: sqlite3.Connection, table_name: str = "alphas") -> list[str]:
    """Idempotently add 004_alpha_quality + robustness columns without blind duplicate ALTER TABLE."""
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if not existing:
        return []
    added: list[str] = []
    with conn:
        for col_name, col_def in ALPHA_VAULT_QUALITY_COLUMNS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}")
                added.append(col_name)
    return added


def rollback_phase_blueprint_robustness(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 9: Safe rollback for blueprint, robustness, and research lesson tables."""
    dropped = []
    with conn:
        for tbl in ("research_lessons", "alpha_robustness_reports", "alpha_robustness_runs", "alpha_blueprints"):
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            dropped.append(tbl)
        conn.execute("DELETE FROM schema_migrations WHERE version='20261119_002_blueprint_robustness'")
    return {"rolled_back_tables": dropped}
