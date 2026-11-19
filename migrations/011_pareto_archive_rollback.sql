PRAGMA foreign_keys = OFF;

DROP INDEX IF EXISTS idx_pareto_snapshots_scope;
DROP INDEX IF EXISTS idx_pareto_cohort;
DROP INDEX IF EXISTS idx_pareto_candidate;
DROP INDEX IF EXISTS idx_pareto_scope_rank;

DROP TABLE IF EXISTS pareto_dominance_edges;
DROP TABLE IF EXISTS pareto_archive_snapshots;
DROP TABLE IF EXISTS pareto_archive_entries;
DROP TABLE IF EXISTS pareto_normalization_policies;

PRAGMA foreign_keys = ON;
