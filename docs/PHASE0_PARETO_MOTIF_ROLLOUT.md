# Phase 0 Convergence + Shadow Pareto Archive & Forbidden Edit Motif Rollout

## Included Scope
1. **Phase 0 Convergence**
   - `worldquant_brain/canonical_ast.py`: Single-source-of-truth FastExpr AST parser (`exact_hash`, `normalized_hash`, `structural_hash`, `ast_family_hash`, operator/field multisets, and parent->child `diff_ast_edit`).
   - Unified expression normalization/hashing across `alpha_experiment_store.py`, `alpha_repository.py`, `alpha_novelty.py`, `two_stage_funnel.py`, and `alpha_miner.py`.
   - `worldquant_brain/alpha_miner.py`: Bounded `<=10%` logged exploration (`_selection_audit`), prime windows downgraded to `PARAMETER_VARIANT` only (`structural_novelty=False`), and Sub-universe root-cause diagnosis before mutating.
   - `worldquant_brain/sub_universe_classifier.py`: 8-class Sub-universe Root-cause Classifier preventing blind `rank(...)` wrapping on sentiment/alternative domains.
2. **Shadow-mode Pareto Research Archive & Forbidden Edit Motif Memory**
   - `migrations/010_forbidden_edit_motifs.sql` (+ rollback)
   - `migrations/011_pareto_archive.sql` (+ rollback)
   - `worldquant_brain/pareto_archive.py`: Vector reward, non-dominated sorting, crowding distance, cohort isolation, capacity trimming, and `ParetoArchiveShadowStore`.
   - `worldquant_brain/forbidden_edit_motif.py`: Context-specific AST edit motifs, gate-specific repair outcomes, Beta-Binomial posterior, Wilson lower bound, asymmetric veto policy, `build_edit_motif_from_expressions`, and `ForbiddenEditMotifShadowStore`.

## Rollout Policy (Shadow First)
- **0–20 trials**: Pareto Archive in Shadow ranking only; Forbidden Edit Motif in Record-only mode (no queue override).
- **20–60 trials**: Pareto breaks ties within equal-priority queue candidates.
- **60–100 trials**: Enable Soft Veto (`prior_multiplier = 0.20`, retaining `15%` exploration).
- **100+ trials**: Evaluate Strong Veto; keep Permanent Veto strictly restricted to allowlisted structural/invalidity reasons (`LOOK_AHEAD`, `INVALID_OPERATOR`, `VECTOR_WITHOUT_REDUCER`, `UNSUPPORTED_SETTING`, `PROHIBITED_STRUCTURE`, `PARSE_FAILURE`).
- **Defer full RL / AlphaQCM** until 500–1,000 structured trials are accumulated.
