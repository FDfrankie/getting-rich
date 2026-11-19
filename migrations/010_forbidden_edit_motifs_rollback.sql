PRAGMA foreign_keys = OFF;

DROP INDEX IF EXISTS idx_motif_decisions_context;
DROP INDEX IF EXISTS idx_motif_decisions_candidate;
DROP INDEX IF EXISTS idx_motif_policy_failure;
DROP INDEX IF EXISTS idx_motif_policy_lookup;
DROP INDEX IF EXISTS idx_motif_consumption_context;
DROP INDEX IF EXISTS idx_motif_evidence_batch;
DROP INDEX IF EXISTS idx_motif_evidence_child;
DROP INDEX IF EXISTS idx_motif_evidence_context;
DROP INDEX IF EXISTS idx_motif_context_parent;
DROP INDEX IF EXISTS idx_motif_context_lookup;
DROP INDEX IF EXISTS idx_edit_motifs_permanent;
DROP INDEX IF EXISTS idx_edit_motifs_subtree;
DROP INDEX IF EXISTS idx_edit_motifs_type;

DROP TABLE IF EXISTS research_motif_decisions;
DROP TABLE IF EXISTS research_motif_policies;
DROP TABLE IF EXISTS research_motif_consumption_log;
DROP TABLE IF EXISTS research_motif_evidence;
DROP TABLE IF EXISTS research_motif_contexts;
DROP TABLE IF EXISTS research_edit_motifs;
DROP TABLE IF EXISTS research_permanent_veto_reasons;

PRAGMA foreign_keys = ON;
