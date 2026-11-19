PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. Canonical AST Edit Motif
-- ============================================================

CREATE TABLE IF NOT EXISTS research_edit_motifs (
    motif_id TEXT PRIMARY KEY,

    motif_type TEXT NOT NULL CHECK (
        motif_type IN (
            'WRAP_OPERATOR',
            'UNWRAP_OPERATOR',
            'REPLACE_OPERATOR',
            'REPLACE_FIELD',

            'ADD_CONFIRMATION',
            'REMOVE_CONFIRMATION',

            'ADD_CONDITION',
            'REMOVE_CONDITION',

            'CHANGE_WINDOW',
            'CHANGE_GROUP',
            'CHANGE_NEUTRALIZATION',
            'CHANGE_DECAY',
            'CHANGE_TRUNCATION',

            'ADD_BACKFILL',
            'CHANGE_BACKFILL',
            'REMOVE_BACKFILL',

            'ADD_RANK',
            'REMOVE_RANK',

            'ADD_WINSORIZE',
            'CHANGE_WINSORIZE',

            'ADD_HUMP',
            'CHANGE_HUMP',

            'ADD_TRADE_WHEN',
            'CHANGE_TRADE_WHEN',

            'VECTOR_REDUCER_CHANGE',
            'STRUCTURAL_REWRITE',
            'SETTING_ONLY_CHANGE'
        )
    ),

    motif_hash TEXT NOT NULL UNIQUE,

    target_ast_path TEXT,

    before_subtree_hash TEXT,
    after_subtree_hash TEXT,

    added_operators_json TEXT NOT NULL DEFAULT '[]',
    removed_operators_json TEXT NOT NULL DEFAULT '[]',

    changed_fields_json TEXT NOT NULL DEFAULT '[]',
    changed_literals_json TEXT NOT NULL DEFAULT '[]',

    setting_changes_json TEXT NOT NULL DEFAULT '{}',
    canonical_edit_json TEXT NOT NULL,

    structural_change INTEGER NOT NULL DEFAULT 0 CHECK (
        structural_change IN (0, 1)
    ),

    parameter_only INTEGER NOT NULL DEFAULT 0 CHECK (
        parameter_only IN (0, 1)
    ),

    permanent_invalid INTEGER NOT NULL DEFAULT 0 CHECK (
        permanent_invalid IN (0, 1)
    ),

    permanent_invalid_reason TEXT,

    ast_schema_version TEXT NOT NULL,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    CHECK (
        NOT (
            structural_change = 1
            AND parameter_only = 1
        )
    ),

    CHECK (
        permanent_invalid = 0
        OR permanent_invalid_reason IS NOT NULL
    )
);

CREATE INDEX IF NOT EXISTS idx_edit_motifs_type
ON research_edit_motifs (
    motif_type,
    structural_change,
    parameter_only
);

CREATE INDEX IF NOT EXISTS idx_edit_motifs_subtree
ON research_edit_motifs (
    before_subtree_hash,
    after_subtree_hash
);

CREATE INDEX IF NOT EXISTS idx_edit_motifs_permanent
ON research_edit_motifs (
    permanent_invalid,
    motif_type
);


-- ============================================================
-- 2. Motif Context
--
-- 相同 Edit Motif 在不同 Family、Domain、Universe、
-- Delay、Failure Gate 下不可直接共用失败经验。
-- ============================================================

CREATE TABLE IF NOT EXISTS research_motif_contexts (
    motif_context_id TEXT PRIMARY KEY,

    motif_id TEXT NOT NULL,

    parent_context_hash TEXT NOT NULL,

    hypothesis_family TEXT NOT NULL,
    data_domain TEXT NOT NULL,
    failure_gate TEXT NOT NULL,

    region TEXT NOT NULL,
    universe TEXT NOT NULL,

    delay INTEGER NOT NULL CHECK (
        delay IN (0, 1)
    ),

    objective_lane TEXT NOT NULL CHECK (
        objective_lane IN (
            'RETURN',
            'SHARPE',
            'FITNESS',
            'CAPACITY',
            'BALANCED'
        )
    ),

    parent_turnover_band TEXT CHECK (
        parent_turnover_band IS NULL
        OR parent_turnover_band IN (
            'FAILED',
            'BELOW_FITNESS_FLOOR',
            'PREFERRED',
            'ACCEPTABLE',
            'HIGH_BUT_ELIGIBLE'
        )
    ),

    parent_correlation_band TEXT CHECK (
        parent_correlation_band IS NULL
        OR parent_correlation_band IN (
            'UNKNOWN',
            'LOW',
            'MEDIUM',
            'HIGH',
            'FAILED'
        )
    ),

    parent_coverage_band TEXT CHECK (
        parent_coverage_band IS NULL
        OR parent_coverage_band IN (
            'UNKNOWN',
            'LOW',
            'MEDIUM',
            'HIGH'
        )
    ),

    parent_ast_signature TEXT NOT NULL,

    field_catalog_version TEXT,
    operator_schema_version TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (motif_id)
        REFERENCES research_edit_motifs(motif_id)
        ON DELETE CASCADE,

    UNIQUE (
        motif_id,
        parent_context_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_motif_context_lookup
ON research_motif_contexts (
    hypothesis_family,
    data_domain,
    failure_gate,
    region,
    universe,
    delay
);

CREATE INDEX IF NOT EXISTS idx_motif_context_parent
ON research_motif_contexts (
    parent_context_hash,
    motif_id
);


-- ============================================================
-- 3. Raw Motif Evidence
--
-- 每次 Parent -> Child 实验建立一笔不可覆盖 Evidence。
-- ============================================================

CREATE TABLE IF NOT EXISTS research_motif_evidence (
    motif_evidence_id TEXT PRIMARY KEY,

    motif_id TEXT NOT NULL,
    motif_context_id TEXT NOT NULL,

    parent_artifact_id TEXT NOT NULL,
    child_artifact_id TEXT NOT NULL,

    parent_trial_id TEXT NOT NULL,
    child_trial_id TEXT NOT NULL,

    batch_id TEXT NOT NULL,

    target_failure_gate TEXT NOT NULL,

    target_gate_resolved INTEGER NOT NULL CHECK (
        target_gate_resolved IN (0, 1)
    ),

    success INTEGER NOT NULL CHECK (
        success IN (0, 1)
    ),

    official INTEGER NOT NULL DEFAULT 0 CHECK (
        official IN (0, 1)
    ),

    evidence_source TEXT NOT NULL CHECK (
        evidence_source IN (
            'WORLDQUANT_BRAIN',
            'BRAIN_LABS',
            'LOCAL_REPLICATION',
            'RECORDED_FIXTURE',
            'INFERRED_PROXY'
        )
    ),

    parent_sharpe REAL,
    child_sharpe REAL,
    sharpe_delta REAL,

    parent_fitness REAL,
    child_fitness REAL,
    fitness_delta REAL,

    parent_returns REAL,
    child_returns REAL,
    returns_delta REAL,

    parent_turnover REAL,
    child_turnover REAL,
    turnover_delta REAL,

    parent_margin REAL,
    child_margin REAL,
    margin_delta REAL,

    parent_drawdown REAL,
    child_drawdown REAL,
    drawdown_delta REAL,

    parent_self_correlation REAL CHECK (
        parent_self_correlation IS NULL
        OR parent_self_correlation BETWEEN -1.0 AND 1.0
    ),

    child_self_correlation REAL CHECK (
        child_self_correlation IS NULL
        OR child_self_correlation BETWEEN -1.0 AND 1.0
    ),

    correlation_delta REAL,

    parent_sub_universe_sharpe REAL,
    child_sub_universe_sharpe REAL,
    sub_universe_delta REAL,

    parent_sub_universe_passed INTEGER CHECK (
        parent_sub_universe_passed IS NULL
        OR parent_sub_universe_passed IN (0, 1)
    ),

    child_sub_universe_passed INTEGER CHECK (
        child_sub_universe_passed IS NULL
        OR child_sub_universe_passed IN (0, 1)
    ),

    simulation_minutes REAL CHECK (
        simulation_minutes IS NULL
        OR simulation_minutes >= 0.0
    ),

    raw_metrics_json TEXT NOT NULL DEFAULT '{}',

    evidence_snapshot_hash TEXT NOT NULL,

    created_at TEXT NOT NULL,

    FOREIGN KEY (motif_id)
        REFERENCES research_edit_motifs(motif_id)
        ON DELETE RESTRICT,

    FOREIGN KEY (motif_context_id)
        REFERENCES research_motif_contexts(motif_context_id)
        ON DELETE RESTRICT,

    UNIQUE (
        motif_id,
        motif_context_id,
        child_trial_id,
        target_failure_gate
    ),

    CHECK (
        official = 0
        OR evidence_source = 'WORLDQUANT_BRAIN'
    )
);

CREATE INDEX IF NOT EXISTS idx_motif_evidence_context
ON research_motif_evidence (
    motif_context_id,
    success,
    created_at
);

CREATE INDEX IF NOT EXISTS idx_motif_evidence_child
ON research_motif_evidence (
    child_trial_id,
    motif_id
);

CREATE INDEX IF NOT EXISTS idx_motif_evidence_batch
ON research_motif_evidence (
    batch_id,
    motif_context_id
);


-- ============================================================
-- 4. Exactly-once Motif Evidence Consumption
-- ============================================================

CREATE TABLE IF NOT EXISTS research_motif_consumption_log (
    consumption_id TEXT PRIMARY KEY,

    motif_evidence_id TEXT NOT NULL,
    motif_context_id TEXT NOT NULL,

    child_trial_id TEXT NOT NULL,
    target_failure_gate TEXT NOT NULL,

    consumed_at TEXT NOT NULL,

    FOREIGN KEY (motif_evidence_id)
        REFERENCES research_motif_evidence(motif_evidence_id)
        ON DELETE RESTRICT,

    FOREIGN KEY (motif_context_id)
        REFERENCES research_motif_contexts(motif_context_id)
        ON DELETE RESTRICT,

    UNIQUE (
        motif_evidence_id,
        motif_context_id,
        child_trial_id,
        target_failure_gate
    )
);

CREATE INDEX IF NOT EXISTS idx_motif_consumption_context
ON research_motif_consumption_log (
    motif_context_id,
    consumed_at
);


-- ============================================================
-- 5. Aggregated Motif Policy
-- ============================================================

CREATE TABLE IF NOT EXISTS research_motif_policies (
    motif_policy_id TEXT PRIMARY KEY,

    motif_id TEXT NOT NULL,
    motif_context_id TEXT NOT NULL,

    policy_version TEXT NOT NULL,

    attempts INTEGER NOT NULL DEFAULT 0 CHECK (
        attempts >= 0
    ),

    successes INTEGER NOT NULL DEFAULT 0 CHECK (
        successes >= 0
    ),

    failures INTEGER NOT NULL DEFAULT 0 CHECK (
        failures >= 0
    ),

    distinct_parent_count INTEGER NOT NULL DEFAULT 0 CHECK (
        distinct_parent_count >= 0
    ),

    distinct_batch_count INTEGER NOT NULL DEFAULT 0 CHECK (
        distinct_batch_count >= 0
    ),

    posterior_failure_probability REAL NOT NULL DEFAULT 0.5 CHECK (
        posterior_failure_probability BETWEEN 0.0 AND 1.0
    ),

    wilson_lower_bound REAL NOT NULL DEFAULT 0.0 CHECK (
        wilson_lower_bound BETWEEN 0.0 AND 1.0
    ),

    mean_sharpe_delta REAL NOT NULL DEFAULT 0.0,
    sharpe_delta_m2 REAL NOT NULL DEFAULT 0.0 CHECK (
        sharpe_delta_m2 >= 0.0
    ),

    mean_fitness_delta REAL NOT NULL DEFAULT 0.0,
    fitness_delta_m2 REAL NOT NULL DEFAULT 0.0 CHECK (
        fitness_delta_m2 >= 0.0
    ),

    mean_turnover_delta REAL NOT NULL DEFAULT 0.0,
    mean_correlation_delta REAL NOT NULL DEFAULT 0.0,
    mean_sub_universe_delta REAL NOT NULL DEFAULT 0.0,

    veto_level TEXT NOT NULL DEFAULT 'NONE' CHECK (
        veto_level IN (
            'NONE',
            'COOLDOWN',
            'SOFT',
            'STRONG',
            'PERMANENT'
        )
    ),

    veto_reason TEXT,

    prior_multiplier REAL NOT NULL DEFAULT 1.0 CHECK (
        prior_multiplier BETWEEN 0.0 AND 1.0
    ),

    exploration_probability REAL NOT NULL DEFAULT 0.15 CHECK (
        exploration_probability BETWEEN 0.0 AND 1.0
    ),

    cooldown_until TEXT,

    evidence_version INTEGER NOT NULL DEFAULT 1 CHECK (
        evidence_version >= 1
    ),

    first_evidence_at TEXT,
    last_evidence_at TEXT,
    last_updated_at TEXT NOT NULL,

    FOREIGN KEY (motif_id)
        REFERENCES research_edit_motifs(motif_id)
        ON DELETE CASCADE,

    FOREIGN KEY (motif_context_id)
        REFERENCES research_motif_contexts(motif_context_id)
        ON DELETE CASCADE,

    UNIQUE (
        motif_id,
        motif_context_id,
        policy_version
    ),

    CHECK (
        successes + failures = attempts
    ),

    CHECK (
        veto_level != 'PERMANENT'
        OR prior_multiplier = 0.0
    )
);

CREATE INDEX IF NOT EXISTS idx_motif_policy_lookup
ON research_motif_policies (
    motif_context_id,
    policy_version,
    veto_level
);

CREATE INDEX IF NOT EXISTS idx_motif_policy_failure
ON research_motif_policies (
    posterior_failure_probability DESC,
    wilson_lower_bound DESC,
    attempts DESC
);


-- ============================================================
-- 6. Motif Decision Audit
-- ============================================================

CREATE TABLE IF NOT EXISTS research_motif_decisions (
    motif_decision_id TEXT PRIMARY KEY,

    candidate_id TEXT NOT NULL,
    parent_artifact_id TEXT NOT NULL,

    motif_id TEXT NOT NULL,
    motif_context_id TEXT NOT NULL,

    policy_version TEXT NOT NULL,

    veto_level TEXT NOT NULL CHECK (
        veto_level IN (
            'NONE',
            'COOLDOWN',
            'SOFT',
            'STRONG',
            'PERMANENT'
        )
    ),

    prior_before REAL NOT NULL CHECK (
        prior_before BETWEEN 0.0 AND 1.0
    ),

    prior_after REAL NOT NULL CHECK (
        prior_after BETWEEN 0.0 AND 1.0
    ),

    exploration_probability REAL NOT NULL CHECK (
        exploration_probability BETWEEN 0.0 AND 1.0
    ),

    exploration_override INTEGER NOT NULL DEFAULT 0 CHECK (
        exploration_override IN (0, 1)
    ),

    action TEXT NOT NULL CHECK (
        action IN (
            'ALLOW',
            'LOWER_PRIORITY',
            'COOLDOWN',
            'ALLOW_EXPLORATION',
            'REJECT'
        )
    ),

    reason TEXT NOT NULL,

    decision_context_json TEXT NOT NULL DEFAULT '{}',

    created_at TEXT NOT NULL,

    FOREIGN KEY (motif_id)
        REFERENCES research_edit_motifs(motif_id)
        ON DELETE RESTRICT,

    FOREIGN KEY (motif_context_id)
        REFERENCES research_motif_contexts(motif_context_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_motif_decisions_candidate
ON research_motif_decisions (
    candidate_id,
    created_at
);

CREATE INDEX IF NOT EXISTS idx_motif_decisions_context
ON research_motif_decisions (
    motif_context_id,
    veto_level,
    action
);


-- ============================================================
-- 7. Permanent Veto Reason Allowlist
-- ============================================================

CREATE TABLE IF NOT EXISTS research_permanent_veto_reasons (
    veto_reason_code TEXT PRIMARY KEY,

    description TEXT NOT NULL,

    active INTEGER NOT NULL DEFAULT 1 CHECK (
        active IN (0, 1)
    ),

    created_at TEXT NOT NULL
);

INSERT OR IGNORE INTO research_permanent_veto_reasons (
    veto_reason_code,
    description,
    created_at
)
VALUES
    (
        'LOOK_AHEAD',
        'Confirmed look-ahead or future-data leakage',
        CURRENT_TIMESTAMP
    ),
    (
        'INVALID_OPERATOR',
        'Operator unavailable or invalid in the active operator schema',
        CURRENT_TIMESTAMP
    ),
    (
        'VECTOR_WITHOUT_REDUCER',
        'Vector field used without a valid reducer',
        CURRENT_TIMESTAMP
    ),
    (
        'UNSUPPORTED_SETTING',
        'Officially unsupported simulation setting',
        CURRENT_TIMESTAMP
    ),
    (
        'PARSE_FAILURE',
        'Canonical parser rejected the expression',
        CURRENT_TIMESTAMP
    ),
    (
        'PROHIBITED_STRUCTURE',
        'Structure explicitly prohibited by official platform policy',
        CURRENT_TIMESTAMP
    );
