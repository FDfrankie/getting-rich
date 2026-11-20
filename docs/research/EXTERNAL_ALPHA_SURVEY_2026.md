# External Alpha Research & Quantitative Optimization Survey (2026)

This document synthesizes verified techniques from recent peer-reviewed papers, quantitative finance practitioner guides, and WorldQuant BRAIN documentation, mapping each directly to modules implemented in this repository.

## 1. Formulaic Alpha Discovery & Self-Evolving Search
- **[AgonAlpha: Autonomous Alpha Discovery via Prompt Economy and Scalable Agentic Search (arXiv:2608.11250)](https://arxiv.org/pdf/2608.11250)**
  - **Key Insight**: Separate candidate generation from evidence verification (fresh-context adversarial reviewer checks evidence integrity & look-ahead; platform scores quality), allocate simulation budget with pending-aware scheduling and successive halving tournaments.
  - **Repo Adoption**: `worldquant_brain/pre_submission_gate.py` enforces official `WORLDQUANT_BRAIN` evidence provenance; `worldquant_brain/two_stage_funnel.py` implements two-stage budget concentration.
- **[AlphaPareto: Formulaic Alpha Discovery with LLM-Guided Multi-Objective Reinforcement Learning (arXiv:2609.34188, NeurIPS 2026)](https://arxiv.org/abs/2609.34188)**
  - **Key Insight**: Scalar rewards miss multi-dimensional alpha quality (predictive power, temporal stability, perturbation robustness, diversity) and suffer from non-stationarity as the alpha pool evolves.
  - **Repo Adoption**: `worldquant_brain/pareto_archive.py` + `migrations/011_pareto_archive.sql` (Shadow-first Pareto frontier with non-dominated sorting and crowding distance).
- **[AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents (arXiv:2606.20625)](https://arxiv.org/abs/2606.20625)**
  - **Key Insight**: Record context-specific AST edit motifs and apply confidence-gated asymmetric process vetoes instead of globally banning useful operators.
  - **Repo Adoption**: `worldquant_brain/forbidden_edit_motif.py` + `migrations/010_forbidden_edit_motifs.sql` + `worldquant_brain/canonical_ast.py`.
- **[AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay (arXiv:2502.16789, KDD 2025)](https://arxiv.org/abs/2502.16789)**
  - **Key Insight**: Enforce structured economic hypotheses (observation, theory, justification, parameter specification) and complexity regularization to resist crowding and alpha decay.
  - **Repo Adoption**: `worldquant_brain/alpha_blueprint.py` + `worldquant_brain/canonical_ast.py` structural deduplication in `worldquant_brain/alpha_miner.py`.
- **[AlphaForge (arXiv:2406.18394)](https://arxiv.org/abs/2406.18394)**, **[QuantFactor REINFORCE (arXiv:2409.05144)](https://arxiv.org/abs/2409.05144)**, **[Trajectory-level Reward Shaping / TLRS (arXiv:2507.20263)](https://arxiv.org/abs/2507.20263)**, and **[AlphaPilot (arXiv:2610.15488)](https://arxiv.org/html/2610.15488)**
  - **Key Insight**: Separate atomic factor discovery from dynamic portfolio combination; use IR/cost-adjusted reward shaping and variance-reducing baselines.
  - **Repo Adoption**: Atomic alpha discovery in `worldquant_brain/alpha_miner.py`; cost-adjusted Belief-State Contextual RL in `crypto_trading/strategies/conditional_prob_rl_strategy.py`.

## 2. Multi-Parameter Optimization & Anti-Overfitting Controls
- **[Cross-Regime Bayesian Optimisation for Equity Signal Generation (arXiv:2608.27076v1)](https://arxiv.org/html/2608.27076v1)** & **[Optuna Optimization + Deflated Sharpe Ratio Guide (QuantJourney)](https://backtester.quantjourney.cloud/research/optuna-optimization)**
  - **Key Insight**: Parameter search and statistical validation are inseparable; every tuning trial spends statistical power and must be bounded by trial caps, Deflated Sharpe Ratio (DSR), and cross-regime stability.
  - **Repo Adoption**: `worldquant_brain/multi_param_optimizer.py` caps per-family polishing (`max_polish_trials_per_family=8`), absorbs dominated passing variants into `PASSED_DO_NOT_SUBMIT`, and promotes only 1 `SUBMITTABLE_BEST` champion per `structural_hash` family.
- **[Trade Selection with Supervised Learning and Optimal Coordinate Ascent (arXiv:1812.04486v1)](https://arxiv.org/html/1812.04486v1)** & **[Numerical Optimization Without a Closed-Form Solution (Risk Hub)](https://riskhub.org/blogs/numerical-optimization-without-a-closed-form-solution)**
  - **Key Insight**: Coordinate ascent with directional step memory plus golden-section / ternary search efficiently locates unimodal optima without exponential grid blow-up.
  - **Repo Adoption**: `propose_multi_param_tuning_candidates()` implements Binary Search (monotonic constraints), Ternary Search (unimodal curves), and Directional Coordinate Ascent (using `observe_parameter_delta`).
- **[Machine Learning for Alpha: What Works in Equities (RupeeCase)](https://www.rupeecase.com/learn/path-5/module-5-3-machine-learning-for-alpha)** & **[Finding Signal: EDA for Alpha Discovery](https://medium.com/@brian-curry-research/finding-signal-a-practitioners-guide-to-eda-methods-for-alpha-discovery-932c0d44e5dc)**
  - **Key Insight**: Financial cross-sections have small effective sample sizes; parsimonious, regularized models with strict walk-forward splits outperform deep over-parameterized networks.
  - **Repo Adoption**: Low-dimensional 11-feature Belief-State Contextual RL with L2 regularization and volatility targeting in `crypto_trading/strategies/conditional_prob_rl_strategy.py`.

## 3. Official Platform & Exchange Documentation (WorldQuant BRAIN, Binance, Bitunix)
- **[WorldQuant BRAIN Research Consultant Program](https://worldquantbrain.com/consultant)** & **[IQC 2026 Official Guidelines](https://www.worldquant.com/brain/iqc/)**
  - **Verified Rule**: Reaching 10,000 points (Gold) unlocks invitation to the BRAIN Research Consultant Program, which enables Multi-Simulation, SuperAlphas, 400,000+ fields across 12 regions, and merit-based quarterly payouts ($2,000+ Master / $8,000+ Grandmaster).
  - **Repo Adoption**: `multi_sim_client.py` HTTP 403 session circuit breaker (for pre-Consultant accounts) + `daily_candidate_selector.py` `reserve_high_value=True` mode (save top-tier alphas until Consultant onboarding).
- **[Binance USD-M Futures Leverage & Maintenance Margin](https://www.binance.com/en/support/faq/detail/360033162192)**, **[Liquidation Price Formula](https://www.binance.com/en/support/faq/detail/b3c689c1f50a44cabb3a84e663b81d93)**, and **[Funding Rate Rules](https://www.binance.com/en/support/faq/detail/360033525031)**
  - **Verified Rule**: Isolated margin keeps other contracts' UPNL isolated (`UPNL=0`), maintenance margin is bracketed (~0.4%–0.5% Tier 1), and funding settles every 8h (accelerating to 1h if hitting ±0.3% cap).
  - **Repo Adoption**: `crypto_trading/risk_manager.py` `calculate_equity_aware_position_size()` enforces `liquidation_safety_factor=3.0` (`(1/leverage) - MMR >= 3 * stop_pct`) so controlled 2x–3x isolated leverage on small ($300) accounts always stops out well before liquidation.
- **[Bitunix Futures Fees & Calculation](https://www.bitunix.com/hub/helpcenter/article/trading-fees-explanation-calculation-bitunix-futures?id=224)** & **[Bitunix Base Tier Fees](https://www.bitunix.com/hub/blog/trading/bitunix-vs-lbank)**
  - **Verified Rule**: Base-tier Futures fees are 0.020% Maker / 0.060% Taker (12 bps round-trip Taker before slippage); leverage increases notional size and fees proportionally.
  - **Repo Adoption**: `ConditionalProbRLStrategy` defaults to `fee_roundtrip=0.0014` (14 bps) on 12h/24h horizons to clear real Taker+slippage drag.

## 4. Applied Takeaways from YouTube Quant Lectures (Transcript-Verified)
- **Tri-State Verification & Future-Tampering Audit**: Implemented directly in `worldquant_brain/leakage_audit.py` (`PASS` / `FAIL` / `UNVERIFIED`) so unverified strategies are never promoted and look-ahead preprocessing leaks are caught automatically.
