PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS alpha_optimization_policies (
    policy_id TEXT PRIMARY KEY,
    policy_version TEXT NOT NULL,
    objective TEXT NOT NULL CHECK (
        objective IN ('RETURN', 'SHARPE', 'FITNESS', 'CAPACITY', 'BALANCED')
    ),
    config_json TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL,
    retired_at TEXT,
    UNIQUE(policy_version, objective)
);

CREATE TABLE IF NOT EXISTS alpha_optimization_decisions (
    optimization_decision_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    parent_experiment_id TEXT,
    policy_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    diagnoses_json TEXT NOT NULL DEFAULT '[]',
    selected_action TEXT,
    considered_actions_json TEXT NOT NULL DEFAULT '[]',
    turnover_efficiency REAL,
    before_metrics_json TEXT,
    after_metrics_json TEXT NOT NULL,
    required_validation_json TEXT NOT NULL DEFAULT '[]',
    decision_context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE CASCADE,
    FOREIGN KEY(parent_experiment_id)
        REFERENCES alpha_research_experiments(experiment_id),
    FOREIGN KEY(policy_id)
        REFERENCES alpha_optimization_policies(policy_id),
    UNIQUE(experiment_id, policy_id)
);

CREATE INDEX IF NOT EXISTS idx_optimization_decisions_objective
ON alpha_optimization_decisions(objective, selected_action, created_at);

CREATE TABLE IF NOT EXISTS alpha_objective_lane_runs (
    lane_run_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    experiment_id TEXT,
    objective TEXT NOT NULL CHECK (
        objective IN ('RETURN', 'SHARPE', 'FITNESS', 'CAPACITY', 'BALANCED')
    ),
    expected_yield_per_minute REAL NOT NULL,
    expected_minutes REAL NOT NULL CHECK (expected_minutes > 0),
    expected_net_return REAL,
    expected_sharpe REAL,
    expected_fitness REAL,
    expected_capacity REAL,
    expected_portfolio_utility REAL,
    novelty REAL,
    robustness_probability REAL,
    information_gain REAL,
    correlation_risk REAL,
    complexity_penalty REAL,
    queue_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_lane_runs_queue
ON alpha_objective_lane_runs(
    queue_status, objective, expected_yield_per_minute DESC
);

CREATE TABLE IF NOT EXISTS alpha_execution_shaping_trials (
    shaping_trial_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    parent_experiment_id TEXT,
    shaping_type TEXT NOT NULL CHECK (
        shaping_type IN (
            'DECAY', 'TRADE_WHEN', 'HUMP', 'VOLATILITY_SCALING',
            'NEUTRALIZATION', 'HORIZON', 'EVENT_CONDITION'
        )
    ),
    before_config_json TEXT NOT NULL,
    after_config_json TEXT NOT NULL,
    intended_objective TEXT NOT NULL,
    intended_gate TEXT NOT NULL,
    turnover_efficiency REAL,
    cost_adjusted_return_delta REAL,
    sharpe_delta REAL,
    fitness_delta REAL,
    turnover_delta REAL,
    margin_delta REAL,
    drawdown_delta REAL,
    correlation_delta REAL,
    successful INTEGER CHECK (successful IS NULL OR successful IN (0, 1)),
    created_at TEXT NOT NULL,
    evaluated_at TEXT,
    FOREIGN KEY(experiment_id)
        REFERENCES alpha_research_experiments(experiment_id)
        ON DELETE CASCADE,
    FOREIGN KEY(parent_experiment_id)
        REFERENCES alpha_research_experiments(experiment_id),
    UNIQUE(experiment_id, shaping_type, intended_gate)
);

CREATE INDEX IF NOT EXISTS idx_shaping_trials_learning
ON alpha_execution_shaping_trials(
    shaping_type, intended_objective, intended_gate, successful
);
