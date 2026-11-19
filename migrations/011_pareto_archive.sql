PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pareto_archive_entries (
    archive_entry_id TEXT PRIMARY KEY,

    trial_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,

    policy_version TEXT NOT NULL,
    objective_schema_version TEXT NOT NULL,
    normalization_version TEXT NOT NULL,

    region TEXT NOT NULL,

    delay INTEGER NOT NULL CHECK (
        delay IN (0, 1)
    ),

    universe TEXT NOT NULL,
    data_domain TEXT NOT NULL,
    blueprint_family TEXT NOT NULL,

    objective_lane TEXT NOT NULL CHECK (
        objective_lane IN (
            'RETURN',
            'SHARPE',
            'FITNESS',
            'CAPACITY',
            'BALANCED'
        )
    ),

    official_quality REAL,
    os_stability REAL,
    perturbation_robustness REAL,
    pnl_uniqueness REAL,
    merged_portfolio_utility REAL,
    cost_efficiency REAL,
    capacity REAL,
    information_gain REAL,

    correlation_risk REAL,
    drawdown_risk REAL,
    overfitting_risk REAL,
    data_quality_risk REAL,
    complexity_cost REAL,

    simulation_minutes REAL CHECK (
        simulation_minutes IS NULL
        OR simulation_minutes >= 0.0
    ),

    objective_known_mask INTEGER NOT NULL,

    pareto_rank INTEGER CHECK (
        pareto_rank IS NULL
        OR pareto_rank >= 1
    ),

    crowding_distance REAL CHECK (
        crowding_distance IS NULL
        OR crowding_distance >= 0.0
    ),

    archive_scope TEXT NOT NULL CHECK (
        archive_scope IN (
            'GLOBAL',
            'OBJECTIVE',
            'COHORT',
            'FAMILY'
        )
    ),

    archive_scope_key TEXT NOT NULL,

    archive_status TEXT NOT NULL CHECK (
        archive_status IN (
            'PROVISIONAL',
            'NON_DOMINATED',
            'DOMINATED',
            'EVICTED',
            'PROMOTED',
            'INVALIDATED'
        )
    ),

    dominated_by_entry_id TEXT,

    admitted_at TEXT NOT NULL,
    last_recomputed_at TEXT NOT NULL,

    input_snapshot_hash TEXT NOT NULL,
    objective_vector_hash TEXT NOT NULL,

    FOREIGN KEY (dominated_by_entry_id)
        REFERENCES pareto_archive_entries(archive_entry_id)
        ON DELETE SET NULL,

    UNIQUE (
        trial_id,
        policy_version,
        archive_scope,
        archive_scope_key
    )
);

CREATE INDEX IF NOT EXISTS idx_pareto_scope_rank
ON pareto_archive_entries (
    archive_scope,
    archive_scope_key,
    archive_status,
    pareto_rank,
    crowding_distance DESC
);

CREATE INDEX IF NOT EXISTS idx_pareto_candidate
ON pareto_archive_entries (
    candidate_id,
    policy_version
);

CREATE INDEX IF NOT EXISTS idx_pareto_cohort
ON pareto_archive_entries (
    region,
    delay,
    universe,
    data_domain,
    blueprint_family
);


CREATE TABLE IF NOT EXISTS pareto_archive_snapshots (
    snapshot_id TEXT PRIMARY KEY,

    policy_version TEXT NOT NULL,

    archive_scope TEXT NOT NULL CHECK (
        archive_scope IN (
            'GLOBAL',
            'OBJECTIVE',
            'COHORT',
            'FAMILY'
        )
    ),

    archive_scope_key TEXT NOT NULL,

    alpha_pool_hash TEXT NOT NULL,

    entry_count INTEGER NOT NULL CHECK (
        entry_count >= 0
    ),

    front_count INTEGER NOT NULL CHECK (
        front_count >= 0
    ),

    non_dominated_count INTEGER NOT NULL CHECK (
        non_dominated_count >= 0
    ),

    snapshot_json TEXT NOT NULL,

    created_at TEXT NOT NULL,

    UNIQUE (
        policy_version,
        archive_scope,
        archive_scope_key,
        alpha_pool_hash
    )
);

CREATE INDEX IF NOT EXISTS idx_pareto_snapshots_scope
ON pareto_archive_snapshots (
    archive_scope,
    archive_scope_key,
    created_at DESC
);


CREATE TABLE IF NOT EXISTS pareto_dominance_edges (
    snapshot_id TEXT NOT NULL,

    dominator_entry_id TEXT NOT NULL,
    dominated_entry_id TEXT NOT NULL,

    epsilon_policy_version TEXT NOT NULL,

    created_at TEXT NOT NULL,

    PRIMARY KEY (
        snapshot_id,
        dominator_entry_id,
        dominated_entry_id
    ),

    FOREIGN KEY (snapshot_id)
        REFERENCES pareto_archive_snapshots(snapshot_id)
        ON DELETE CASCADE,

    FOREIGN KEY (dominator_entry_id)
        REFERENCES pareto_archive_entries(archive_entry_id)
        ON DELETE CASCADE,

    FOREIGN KEY (dominated_entry_id)
        REFERENCES pareto_archive_entries(archive_entry_id)
        ON DELETE CASCADE,

    CHECK (
        dominator_entry_id != dominated_entry_id
    )
);


CREATE TABLE IF NOT EXISTS pareto_normalization_policies (
    normalization_id TEXT PRIMARY KEY,

    normalization_version TEXT NOT NULL,
    objective_name TEXT NOT NULL,

    cohort_key TEXT NOT NULL,

    minimum_value REAL,
    maximum_value REAL,
    median_value REAL,
    mad_value REAL,

    sample_size INTEGER NOT NULL CHECK (
        sample_size >= 0
    ),

    method TEXT NOT NULL CHECK (
        method IN (
            'FIXED_POLICY',
            'GLOBAL_ROBUST',
            'COHORT_ROBUST',
            'EMPIRICAL_PERCENTILE'
        )
    ),

    low_confidence INTEGER NOT NULL DEFAULT 0 CHECK (
        low_confidence IN (0, 1)
    ),

    calculated_at TEXT NOT NULL,

    UNIQUE (
        normalization_version,
        objective_name,
        cohort_key
    )
);
