# WorldQuant BRAIN Webinar, Paper & RL Alpha Playbook

Distilled from WorldQuant BRAIN official webinars, documentation, and 2023–2026 RL alpha-mining papers, integrated directly into `worldquant_brain/two_stage_funnel.py` and `scripts/headless_brain_miner.py`.

## 1. Official Webinar Techniques Implemented
- **Regime-Switching News/Sentiment + Mean Reversion** ([Delay Zero Tutorial](http://www.youtube.com/watch?v=gV4PXY3qZsw), [Beginner Alpha Workshop](http://www.youtube.com/watch?v=BZ-kMA2mnUk)):
  - `rank(ts_sum(ts_backfill(f, 20), 60)) > 0.5 ? (1 + rank(ts_backfill(f, 20))) : rank(-ts_delta(close, 3))`
  - Extending lookback from 30d to 60d improves stability and Fitness past 1.0.
- **Volume-Bucket Custom Neutralization** ([Beginner Alpha Workshop](http://www.youtube.com/watch?v=BZ-kMA2mnUk)):
  - `group_neutralize(rank(ts_backfill(f, 20)), bucket(rank(volume), range="0, 1, 0.2"))`
  - Cuts excessive turnover (e.g. 120% -> ~60%) on high-buzz social/sentiment datasets.
- **Market-Cap Bucket Neutralization & Cap Normalization** ([Alpha Research KB](https://www.scribd.com/document/1058183779/WorldQuant-Brain-Alpha-Research-Knowledge-Base)):
  - `group_neutralize(rank(ts_backfill(f, 20)), bucket(rank(cap), range="0, 1, 0.1"))`
  - Fixes mega-cap bias and Sub-Universe Sharpe failures.
- **`nanHandling="ON"` Stage-2 Sweep for Sparse/Analyst Datasets** ([Delay Zero Tutorial](http://www.youtube.com/watch?v=gV4PXY3qZsw)):
  - Setting `nanHandling="ON"` assigns zero (net short after neutralization) to uncovered stocks, capturing the analyst-coverage premium.
- **Square-Root Compression & `hump()` Smoothing** ([Delay Zero Tutorial](http://www.youtube.com/watch?v=gV4PXY3qZsw), [Sub-Universe Sharpe Guide](https://blog.csdn.net/m0_73177400/article/details/148221442)):
  - `sqrt(rank(ts_backfill(f, 20)))` reduces single-stock weight concentration; `hump(..., hump=0.01)` caps daily weight jumps toward the 12.5% Fitness turnover floor.
- **Forward Valuation Ratio** ([Delay Zero Tutorial](http://www.youtube.com/watch?v=gV4PXY3qZsw)):
  - `group_rank(ts_rank(divide(ts_backfill(f, 20), close), 60), industry)` compares price-scaled estimates against each stock's own 60-day history within its industry.
- **Event / Low-Frequency Revision Surprise & Orthogonal Residual**:
  - `rank(f - last_diff_value(f, 252))` captures the most recent revision on quarterly/report datasets.
  - `rank(ts_regression(f, returns, 60, lag=0, rettype=0))` extracts market-orthogonal residual alpha.

## 2. RL & Memory Architecture (Papers)
- **AlphaPareto Multi-Objective Reward** ([arXiv:2609.34188](https://arxiv.org/html/2609.34188v1)): RL arm rewards combine Sharpe, Fitness, Turnover proximity to 12.5%, and check pass bonus rather than a flat binary score.
- **Structured Search-Process Memory & Forbidden Patterns** ([arXiv:2606.20625](https://arxiv.org/html/2606.20625v2)): `funnel_reference_db` stores all `PASSED`, `NEAR_MISS`, `REJECTED`, `SUBMITTED`, and `SUBMIT_FAILED` outcomes with explicit reasons; `(dataset, probe)` arms with >=5 pulls and 0 hits are auto-deprioritized as forbidden patterns, while submitted skeletons are blocked from parameter-only cloning.
- **Immediate In-Session Near-Miss Refinement**: Only current-session multi-factor `NEAR_MISS` seeds are refined immediately (including `nanHandling="ON"`, `hump()`, `decay`, `truncation`, and `neutralization` sweeps); historical DB records guide RL policy and repulsion without re-mining old seeds.
