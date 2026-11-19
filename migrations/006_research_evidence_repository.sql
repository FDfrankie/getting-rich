PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS research_memory_versions (
    policy_version TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alpha_research_experiments (
    experiment_id TEXT PRIMARY KEY,
    alpha_id TEXT,
    candidate_id TEXT NOT NULL,
    parent_experiment_id TEXT,
    blueprint_id TEXT NOT NULL,
    blueprint_hash TEXT NOT NULL,
    hypothesis_family TEXT NOT NULL,
    data_domain TEXT NOT NULL,
    expression TEXT NOT NULL,
    expression_hash TEXT NOT NULL,
    normalized_expression_hash TEXT NOT NULL,
    structural_expression_hash TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    settings_hash TEXT NOT NULL,
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL CHECK (delay IN (0, 1)),
    experiment_type TEXT NOT NULL,
    experiment_status TEXT NOT NULL,
    metrics_source TEXT NOT NULL DEFAULT 'UNKNOWN',
    official_simulation_id TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    error_code TEXT,
    safe_error TEXT,
    FOREIGN KEY(parent_experiment_id)
        REFERENCES alpha_research_experiments(experiment_id),
    UNIQUE(expression_hash, settings_hash, region, universe, delay)
);

CREATE INDEX IF NOT EXISTS idx_research_experiments_context
ON alpha_research_experiments(
    hypothesis_family, data_domain, blueprint_id, experiment_status
);
CREATE INDEX IF NOT EXISTS idx_research_experiments_parent
ON alpha_research_experiments(parent_experiment_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_experiments_simulation
ON alpha_research_experiments(official_simulation_id)
WHERE official_simulation_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS alpha_experiment_metrics (
    experiment_id TEXT PRIMARY KEY,
    sharpe REAL,
    fitness REAL,
    returns REAL,
    turnover REAL CHECK (turnover IS NULL OR turnover >= 0.0),
    margin REAL,
    drawdown REAL,
    self_correlation REAL CHECK (
        self_correlation IS NULL OR self_correlation BETWEEN -1.0 AND 1.0
    ),
    production_correlation REAL CHECK (
        production_correlation IS NULL OR production_correlation BETWEEN -1.0 AND 1.0
    ),
    max_instrument_weight REAL CHECK (
        max_instrument_weight IS NULL OR
        (max_instrument_weight >= 0.0 AND max_instrument_weight <= 1.0)
    ),
    long_count INTEGER CHECK (long_count IS NULL OR long_count >= 0),
    short_count INTEGER CHECK (short_count IS NULL OR short_count >= 0),
    weight_test_passed INTEGER CHECK (weight_test_passed IS NULL OR weight_test_passed IN (0, 1)),
    sub_universe_passed INTEGER CHECK (sub_universe_passed IS NULL OR sub_universe_passed IN (0, 1)),
    ranked_sharpe_passed INTEGER CHECK (ranked_sharpe_passed IS NULL OR ranked_sharpe_passed IN (0, 1)),
    bias_test_passed INTEGER CHECK (bias_test_passed IS NULL OR bias_test_passed IN (0, 1)),
    submit_ready INTEGER NOT NULL DEFAULT 0 CHECK (submit_ready IN (0, 1)),
    good_ready INTEGER NOT NULL DEFAULT 0 CHECK (good_ready IN (0, 1)),
    robust_good INTEGER NOT NULL DEFAULT 0 CHECK (robust_good IN (0, 1)),
    quality_class TEXT NOT NULL DEFAULT 'UNVERIFIED',
    quality_score REAL,
    raw_metrics_json TEXT NOT NULL DEFAULT '{}',
    recorded_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS alpha_experiment_failures (
    failure_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    gate_name TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('WARNING', 'BLOCKING', 'TERMINAL')),
    actual_value REAL,
    required_value REAL,
    failure_code TEXT NOT NULL,
    failure_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE CASCADE,
    UNIQUE(experiment_id, gate_name, failure_code)
);
CREATE INDEX IF NOT EXISTS idx_experiment_failures_context
ON alpha_experiment_failures(gate_name, failure_code, severity);

CREATE TABLE IF NOT EXISTS alpha_experiment_fixes (
    experiment_fix_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    fix_name TEXT NOT NULL,
    fix_category TEXT NOT NULL CHECK (
        fix_category IN (
            'PARAMETER', 'CONDITION', 'NORMALIZATION', 'NEUTRALIZATION',
            'DATA_FIELD', 'DATA_DOMAIN', 'ECONOMIC_HYPOTHESIS',
            'EXECUTION_SHAPING'
        )
    ),
    before_value_json TEXT,
    after_value_json TEXT,
    intended_gate TEXT NOT NULL,
    structural_change INTEGER NOT NULL DEFAULT 0 CHECK (structural_change IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE CASCADE,
    UNIQUE(experiment_id, fix_name, intended_gate)
);
CREATE INDEX IF NOT EXISTS idx_experiment_fixes_context
ON alpha_experiment_fixes(fix_name, intended_gate, structural_change);

CREATE TABLE IF NOT EXISTS research_fix_evidence (
    evidence_id TEXT PRIMARY KEY,
    hypothesis_family TEXT NOT NULL,
    data_domain TEXT NOT NULL,
    failure_gate TEXT NOT NULL,
    fix_name TEXT NOT NULL,
    fix_category TEXT NOT NULL,
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL CHECK (delay IN (0, 1)),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    successful_attempts INTEGER NOT NULL DEFAULT 0 CHECK (successful_attempts >= 0),
    failed_attempts INTEGER NOT NULL DEFAULT 0 CHECK (failed_attempts >= 0),
    mean_sharpe_delta REAL NOT NULL DEFAULT 0.0,
    mean_fitness_delta REAL NOT NULL DEFAULT 0.0,
    mean_turnover_delta REAL NOT NULL DEFAULT 0.0,
    mean_correlation_delta REAL NOT NULL DEFAULT 0.0,
    mean_margin_delta REAL NOT NULL DEFAULT 0.0,
    mean_drawdown_delta REAL NOT NULL DEFAULT 0.0,
    sharpe_delta_m2 REAL NOT NULL DEFAULT 0.0 CHECK (sharpe_delta_m2 >= 0.0),
    fitness_delta_m2 REAL NOT NULL DEFAULT 0.0 CHECK (fitness_delta_m2 >= 0.0),
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    effectiveness_score REAL NOT NULL DEFAULT 0.0,
    first_evidence_at TEXT NOT NULL,
    last_evidence_at TEXT NOT NULL,
    last_experiment_id TEXT,
    evidence_version INTEGER NOT NULL DEFAULT 1 CHECK (evidence_version >= 1),
    UNIQUE(
        hypothesis_family, data_domain, failure_gate, fix_name,
        region, universe, delay
    ),
    CHECK (successful_attempts + failed_attempts = attempts),
    FOREIGN KEY(last_experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_fix_evidence_lookup
ON research_fix_evidence(
    hypothesis_family, data_domain, failure_gate,
    effectiveness_score DESC, confidence DESC
);

CREATE TABLE IF NOT EXISTS evidence_consumption_log (
    consumption_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    experiment_fix_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    source_simulation_id TEXT,
    consumed_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE CASCADE,
    FOREIGN KEY(experiment_fix_id)
        REFERENCES alpha_experiment_fixes(experiment_fix_id)
        ON DELETE CASCADE,
    FOREIGN KEY(evidence_id)
        REFERENCES research_fix_evidence(evidence_id)
        ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
    UNIQUE(experiment_id, experiment_fix_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_consumption_experiment
ON evidence_consumption_log(experiment_id, consumed_at);
CREATE INDEX IF NOT EXISTS idx_consumption_evidence
ON evidence_consumption_log(evidence_id, consumed_at);

CREATE TABLE IF NOT EXISTS research_lessons (
    lesson_id TEXT PRIMARY KEY,
    hypothesis_family TEXT NOT NULL,
    data_domain TEXT NOT NULL,
    failure_gate TEXT NOT NULL,
    lesson_type TEXT NOT NULL CHECK (
        lesson_type IN (
            'PREFER_FIX', 'AVOID_FIX', 'ARCHIVE_FAMILY',
            'EXPLORE_NEW_DOMAIN', 'REQUIRE_MORE_EVIDENCE'
        )
    ),
    recommendation TEXT NOT NULL,
    preferred_fix TEXT,
    blocked_fixes_json TEXT NOT NULL DEFAULT '[]',
    fallback_fixes_json TEXT NOT NULL DEFAULT '[]',
    supporting_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    supporting_experiment_ids_json TEXT NOT NULL DEFAULT '[]',
    sample_size INTEGER NOT NULL DEFAULT 0 CHECK (sample_size >= 0),
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    expected_sharpe_delta REAL,
    expected_fitness_delta REAL,
    expected_turnover_delta REAL,
    expected_correlation_delta REAL,
    applicable_region TEXT,
    applicable_universe TEXT,
    applicable_delay INTEGER CHECK (applicable_delay IS NULL OR applicable_delay IN (0, 1)),
    policy_version TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    lesson_status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (
        lesson_status IN ('ACTIVE', 'SUPERSEDED', 'RETIRED')
    ),
    superseded_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(policy_version)
        REFERENCES research_memory_versions(policy_version),
    FOREIGN KEY(superseded_by)
        REFERENCES research_lessons(lesson_id)
);
CREATE INDEX IF NOT EXISTS idx_research_lessons_active
ON research_lessons(
    hypothesis_family, data_domain, failure_gate,
    lesson_status, confidence DESC
);

CREATE TABLE IF NOT EXISTS research_policy_decisions (
    decision_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    blueprint_id TEXT NOT NULL,
    hypothesis_family TEXT NOT NULL,
    data_domain TEXT NOT NULL,
    detected_failure_gate TEXT,
    considered_lesson_ids_json TEXT NOT NULL DEFAULT '[]',
    considered_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    selected_action TEXT NOT NULL,
    selected_fix TEXT,
    rejected_actions_json TEXT NOT NULL DEFAULT '[]',
    expected_utility REAL,
    exploration_probability REAL NOT NULL DEFAULT 0.0 CHECK (
        exploration_probability BETWEEN 0.0 AND 1.0
    ),
    policy_version TEXT NOT NULL,
    decision_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(policy_version)
        REFERENCES research_memory_versions(policy_version)
);
CREATE INDEX IF NOT EXISTS idx_policy_decisions_candidate
ON research_policy_decisions(candidate_id, created_at);

CREATE TABLE IF NOT EXISTS research_experiment_denylist (
    denylist_id TEXT PRIMARY KEY,
    normalized_expression_hash TEXT,
    structural_expression_hash TEXT,
    blueprint_hash TEXT,
    hypothesis_family TEXT,
    data_domain TEXT,
    deny_reason TEXT NOT NULL,
    evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count >= 1),
    minimum_retry_after TEXT,
    source_type TEXT NOT NULL,
    source_reference TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_denylist_expression_reason
ON research_experiment_denylist(normalized_expression_hash, deny_reason)
WHERE normalized_expression_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS alpha_correlation_checks (
    correlation_check_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    alpha_id TEXT,
    official_simulation_id TEXT,
    check_source TEXT NOT NULL,
    estimated_self_correlation REAL CHECK (
        estimated_self_correlation IS NULL OR
        estimated_self_correlation BETWEEN -1.0 AND 1.0
    ),
    official_self_correlation REAL CHECK (
        official_self_correlation IS NULL OR
        official_self_correlation BETWEEN -1.0 AND 1.0
    ),
    production_correlation REAL CHECK (
        production_correlation IS NULL OR
        production_correlation BETWEEN -1.0 AND 1.0
    ),
    threshold REAL NOT NULL DEFAULT 0.70,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    compared_alpha_count INTEGER,
    most_correlated_alpha_id TEXT,
    raw_response_json TEXT NOT NULL DEFAULT '{}',
    checked_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_correlation_experiment
ON alpha_correlation_checks(experiment_id, checked_at);
