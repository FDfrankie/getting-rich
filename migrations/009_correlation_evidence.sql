PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS alpha_correlation_evidence (
    correlation_evidence_id TEXT PRIMARY KEY,
    candidate_alpha_id TEXT NOT NULL,
    compared_alpha_id TEXT,
    correlation_type TEXT NOT NULL CHECK (
        correlation_type IN (
            'SELF', 'PROD', 'INNER', 'PARENT_CHILD',
            'D0_D1', 'LOCAL_PNL_PROXY', 'STRUCTURAL_PROXY'
        )
    ),
    correlation_source TEXT NOT NULL,
    correlation_value REAL CHECK (
        correlation_value IS NULL OR correlation_value BETWEEN -1.0 AND 1.0
    ),
    candidate_delay INTEGER CHECK (
        candidate_delay IS NULL OR candidate_delay IN (0, 1)
    ),
    compared_delay INTEGER CHECK (
        compared_delay IS NULL OR compared_delay IN (0, 1)
    ),
    pnl_period_start TEXT,
    pnl_period_end TEXT,
    observations INTEGER CHECK (observations IS NULL OR observations >= 2),
    official INTEGER NOT NULL DEFAULT 0 CHECK (official IN (0, 1)),
    expression_similarity REAL,
    field_overlap REAL,
    operator_overlap REAL,
    scope TEXT NOT NULL DEFAULT 'OWN_ALPHA_POOL',
    raw_response_json TEXT NOT NULL DEFAULT '{}',
    checked_at TEXT NOT NULL,
    UNIQUE (
        candidate_alpha_id,
        compared_alpha_id,
        correlation_type,
        correlation_source,
        pnl_period_start,
        pnl_period_end
    )
);

CREATE INDEX IF NOT EXISTS idx_correlation_candidate
ON alpha_correlation_evidence (candidate_alpha_id, correlation_type, checked_at);

CREATE INDEX IF NOT EXISTS idx_correlation_pair
ON alpha_correlation_evidence (candidate_alpha_id, compared_alpha_id, correlation_type);
