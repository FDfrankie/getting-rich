[skip-interview] /swarm /teamwork /boost

# OBJECTIVE-AWARE ALPHA ARCHITECTURE INTEGRATION

Integrate the supplied objective-aware modules into the existing Alpha Blueprint, Research Memory, Evidence Repository, Simulation Queue, Robustness Evaluator and Submission Pipeline.

## Architectural rule
Turnover and Decay are not monotonic quality targets. Optimize cost-adjusted return, Sharpe, Fitness and Capacity according to an explicit objective. Distinguish productive turnover from wasteful turnover.

## Required components
- `optimization_policy.py`: diagnoses and objective-specific actions.
- `objective_aware_scheduler.py`: Return/Sharpe/Capacity/Balanced lanes and ExpectedYieldPerMinute.
- `evidence_objective_bridge.py`: persists diagnosis, objective, turnover efficiency and action context into Evidence/Policy decisions.
- `007_objective_aware_optimization.sql`: policy, decision, lane and shaping-trial storage.

## Integration points
1. After official BRAIN result parsing, create `AlphaPerformance` using cost-adjusted returns.
2. Before generating repair variants, run objective diagnosis.
3. Store the policy snapshot in `research_policy_decisions` and `alpha_optimization_decisions`.
4. Generate only actions mapped to the diagnosed failure. Do not apply higher decay to every high-turnover alpha.
5. Candidate Queue uses lane quotas and ExpectedYieldPerMinute.
6. Evidence updates must include objective and turnover efficiency.
7. Robustness must stress costs, drawdown, correlation and parameter stability.
8. Submission remains governed by official hard gates; objective policy cannot bypass them.

## Lane defaults
RETURN 30%, SHARPE 30%, CAPACITY 15%, BALANCED 25%. These are configurable policy values, not hard-coded truths. Random exploration remains <=10% within the aggregate candidate-generation budget.

## Required behavior
- Productive high turnover: preserve, run 1x/1.5x/2x cost and drawdown stress.
- Wasteful high turnover: prefer hump, trade_when, then higher decay.
- Low-turnover stale signal: test lower decay and shorter horizon.
- Low-return/low-information: survey analyst/news/fundamental fields or change hypothesis.
- High-volatility/low-return: neutralization and volatility scaling.
- Sparse event alpha: relax event condition and verify coverage/long-short counts.
- Decay documentation and code must describe smoothing Alpha output weights, not raw input fields.

## Tests
Add tests for productive turnover not being suppressed, wasteful turnover actions, stale signal lowering decay, objective-specific recommendations, lane budgets, Evidence persistence, migration second apply, disposable rollback/reapply, and full non-live pytest.

No BRAIN live submission, push, or main merge. Output READY_FOR_ANTIGRAVITY/PARTIAL/BLOCKED.
