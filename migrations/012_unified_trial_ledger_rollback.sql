PRAGMA foreign_keys = OFF;
DROP INDEX IF EXISTS idx_trials_lineage;
DROP INDEX IF EXISTS idx_trials_domain_stage;
DROP TABLE IF EXISTS trial_test_access_log;
DROP TABLE IF EXISTS unified_trial_events;
DROP TABLE IF EXISTS crypto_trial_details;
DROP TABLE IF EXISTS alpha_trial_details;
DROP TABLE IF EXISTS unified_trials;
PRAGMA foreign_keys = ON;
