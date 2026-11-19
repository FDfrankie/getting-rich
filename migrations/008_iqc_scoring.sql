PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS iqc_candidate_scores (
    alpha_id TEXT PRIMARY KEY,
    candidate_score REAL NOT NULL,
    timing_score REAL NOT NULL,
    submission_utility REAL NOT NULL,
    grade TEXT NOT NULL,
    decision TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    input_snapshot_hash TEXT NOT NULL,
    calculated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iqc_calculation_audits (
    audit_id TEXT PRIMARY KEY,
    alpha_id TEXT NOT NULL,
    candidate_score REAL NOT NULL,
    timing_score REAL NOT NULL,
    submission_utility REAL NOT NULL,
    grade TEXT NOT NULL,
    decision TEXT NOT NULL,
    breakdown_json TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    input_snapshot_hash TEXT NOT NULL,
    calculated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_iqc_audits_alpha_time
ON iqc_calculation_audits(alpha_id, calculated_at DESC);
