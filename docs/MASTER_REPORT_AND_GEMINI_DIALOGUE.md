# Antigravity Quant: High-Quality Alpha Research Factory

## Executive conclusion
The current system has a strong official-result parser, submission/GOOD quality gates, and controlled parameter variants. It is still too simple to maximize high-quality alpha generation because it optimizes existing expressions more than it researches economic hypotheses. The next architecture must be hypothesis-driven, multi-domain, robustness-first, provenance-aware, and portfolio-marginal-utility aware.

## Target lifecycle
CANDIDATE -> SIMULATED -> SUBMIT_READY -> BRAIN_GOOD -> ROBUST_GOOD -> PORTFOLIO_GOOD -> PROMOTION_READY

## Integrated workstreams
1. Taiwan canonical PIT universe and lineage.
2. Triple-barrier/meta-labeling/purged OOF foundations.
3. BRAIN official result parser and GOOD grading.
4. Alpha Blueprint hypothesis registry.
5. Robustness Evaluator as the next priority.
6. Later: portfolio marginal utility and research memory.

## Blueprint design
Each blueprint records economic rationale, falsifiable claim, data fields and domains, transforms, peer groups, conditions, expected horizon and turnover, expected failure modes, expression template, parameter space, complexity budget, and robustness plan. A blueprint is not an expression. It is a research contract from which controlled expressions can be generated and falsified.

## Robustness Evaluator priority
Robustness must be evaluated before any candidate is promoted beyond BRAIN_GOOD. Required dimensions:
- Temporal stability: positive-year ratio and no single-year PnL dominance.
- Parameter stability: nearby windows/decay/truncation must degrade smoothly.
- Universe stability: TOP3000/TOP2000/TOP1000 or project-specific universes.
- Neutralization stability: industry/subindustry comparison.
- Cost stress: 1x/1.5x/2x cost assumptions.
- Regime stability: bull/bear/high-vol/low-vol.
- Later extensions: missing-data perturbation, field substitutions, bootstrap confidence, DSR/PBO/CPCV, and portfolio marginal utility.

## Promotion policy
- BRAIN_GOOD: official BRAIN gates and strengthened Sharpe/Fitness buffer.
- ROBUST_GOOD: baseline official run plus all required robustness dimensions robust, score >= 70, and required pass ratios.
- PORTFOLIO_GOOD: positive incremental portfolio utility after correlation, turnover, tail-risk and capacity effects.
- Local metrics can never promote official quality.

## What Gemini must do
Use this bundle as a starting implementation, inspect actual repository classes/schema, and integrate rather than rewrite. Connect Blueprint and Robustness records to the existing Alpha Vault, Brain Job Queue, API and UI. Build idempotent migrations and tests. Do not run BRAIN live or submit alphas in this stage.

## Required repository additions
- `worldquant_brain/alpha_blueprint.py`
- `worldquant_brain/robustness_evaluator.py`
- existing parser/quality bridge from the prior master bundle
- migrations for blueprint registry, robustness runs, reports, and promotion status
- API: blueprint CRUD/read-only plan generation, robustness report endpoint, quality lifecycle filters
- UI: show economic rationale, failure modes, tested variants, dimension scores and promotion blockers

## Suggested DB tables
### alpha_blueprints
blueprint_id, version, blueprint_hash, family, title, rationale, falsifiable_claim, blueprint_json, created_at, superseded_by

### alpha_robustness_runs
run_id, alpha_id, blueprint_id, variant_type, variant_value, simulation_id, metrics_source, sharpe, fitness, turnover, returns, drawdown, pnl, year, regime, universe, parameters_json, created_at

### alpha_robustness_reports
alpha_id, blueprint_id, blueprint_hash, baseline_run_id, status, score, report_json, promotion_ready, evaluated_at

### research_lessons
lesson_id, family, data_domain, failure_gate, effective_fix, ineffective_fixes_json, evidence_run_ids_json, created_at

## Restrictions
No direct GOOD or ROBUST_GOOD DB rewrite. No local proxy promotion. No mock official result as evidence. No test weakening. No push/main merge. No BRAIN live in this integration task.

## Final Gemini instruction
Inspect the current repository, map actual call sites and schemas, integrate supplied modules, add idempotent migrations and targeted tests, run full non-live tests, then output READY_FOR_ANTIGRAVITY/PARTIAL/BLOCKED. Stop before commit/push/live actions.
