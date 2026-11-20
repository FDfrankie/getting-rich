# External Alpha Research & BRAIN Intelligence

> Compiled: 2026-11-20  
> Scope: WorldQuant BRAIN (`USA / TOP3000 / D1`), IQC Guidelines, arXiv Alpha Mining Literature, YouTube Practical Tutorials, and Open-Source GitHub BRAIN Toolkits.

---

## 1. WorldQuant BRAIN / IQC Official & Community Guidelines

| Topic | Key Finding | Code Integration | Source |
| :--- | :--- | :--- | :--- |
| **Sub-Universe Check** | Sub-universe Sharpe must scale proportionally with universe size: `0.75 * sqrt(sub_size / alpha_size) * alpha_sharpe` (e.g., `TOP1000` vs `TOP3000`). | Added `compute_sub_universe_sharpe_floor()` in `worldquant_brain/sub_universe_classifier.py`. | WorldQuant BRAIN / IQC Guidelines (https://www.worldquant.com/brain/) |
| **Turnover & Fitness Floor** | Turnover must strictly stay in `(1%, 70%)`; Fitness penalizes turnover above the `12.5%` floor (`Fitness = Sharpe * sqrt(|Returns| / max(Turnover, 0.125))`). | Added `diagnose_turnover_and_cost_stress()` in `worldquant_brain/sub_universe_classifier.py` and `LIQUIDITY_REVERSION_001` in `worldquant_brain/alpha_blueprint.py`. | WorldQuant BRAIN Documentation & Learn2Quant Series (https://www.youtube.com/results?search_query=WorldQuant+BRAIN+alpha+metrics+Sharpe+fitness+turnover) |
| **Margin / Cost-Stress** | Margin >= `0.0010` (10 bps) provides resilience under 1.5x–2.0x transaction cost stress; `hump()`, `decay`, and `ts_mean()` suppress micro-turnover. | Enforced in `diagnose_turnover_and_cost_stress()` and `TwoStageAlphaFunnel` Stage-2 variants. | WorldQuant BRAIN / IQC Playbooks |

---

## 2. arXiv Quantitative Alpha Mining Papers

| Paper | Core Mechanism | Relevance to This Repo | Source |
| :--- | :--- | :--- | :--- |
| **AgonAlpha** | Frozen research artifacts (hypotheses, expressions, rationales, adversarial reviews) + fresh-context reviewer veto. | Aligns with `AlphaBlueprint` single-hypothesis discipline (`worldquant_brain/alpha_blueprint.py`). | https://arxiv.org/ (search: AgonAlpha Autonomous Alpha Discovery) |
| **AlphaPareto** | Multi-objective RL optimizing predictive quality, diversity, and turnover/capacity instead of a single scalar reward. | Implemented in `worldquant_brain/pareto_archive.py` and wired via `worldquant_brain/shadow_writeback.py`. | https://arxiv.org/ (search: AlphaPareto Multi-Objective Reinforcement Learning) |
| **AlphaMemo** | Structured search-process memory tracking AST edit motifs and parent-context success/failure. | Implemented in `worldquant_brain/forbidden_edit_motif.py` + `worldquant_brain/canonical_ast.py`. | https://arxiv.org/ (search: AlphaMemo Structured Search-Process Memory) |
| **AlphaForge** | Two-stage generative-predictive diversity preservation + dynamic temporal factor combination. | Complements `TwoStageAlphaFunnel` (`worldquant_brain/two_stage_funnel.py`) Stage-1 screening and Stage-2 refinement. | https://arxiv.org/ (search: AlphaForge Mine and Dynamically Combine Formulaic Alpha Factors) |
| **AlphaQCM** | Distributional RL with quantiled conditional variance exploration bonus under non-stationary rewards. | Motivates bounded exploration + Wilson lower-bound veto tiers in `forbidden_edit_motif.py`. | https://arxiv.org/ (search: AlphaQCM Alpha Discovery Distributional Reinforcement Learning) |

---

## 3. Open-Source GitHub BRAIN Toolkits

| Repository | Key Patterns Observed | Code Integration | URL |
| :--- | :--- | :--- | :--- |
| **worldquant-miner** | Continuous FastExpr template validation and automated batch simulation loops. | Reinforced complexity budget & field catalog validation in `alpha_blueprint.py`. | https://github.com/zhutoutoutousan/worldquant-miner |
| **wq-alpha-pipeline** | Multi-threaded template-to-fundamental-field batch backtesting for IQC. | Validated `FUNDAMENTAL_QUALITY_001` & `ANALYST_REVISION_001` catalog coverage. | https://github.com/angel4angelov-glitch/wq-alpha-pipeline |
| **Wq-miner** | Evolutionary crossover/mutation with seed-based low-correlation exploration. | Aligned with `alpha_miner.py` structural orthogonalization & AST dedup. | https://github.com/wangshiwen-ai-hku/Wq-miner |
| **brain_viewer** | Local SQLite caching of BRAIN datasets (`datasets.sqlite`) and FastExpr operator metadata. | Matches local `alpha_vault.db` + `brain_simulation_cache` architecture. | https://github.com/justinhuang0208/brain_viewer |
| **worldquant-orchestrator** | End-to-end FastExpr validation constraints and gating checks. | Aligns with `canonical_ast.py` + `shadow_writeback.py` non-bypassable gates. | https://github.com/aaditya-v-more/worldquant-orchestrator |
| **wq-alpha-research** | Field snapshots, `group_rank`/`ts_rank` templates, and IS turnover/fitness diagnostics. | Informed `SENTIMENT_ATTENTION_001`, `OPTIONS_SKEW_001`, and `diagnose_turnover_and_cost_stress()`. | https://github.com/QuantML-Research/wq-alpha-research |

---

## 4. Verified `USA / TOP3000 / D1` High-Coverage Field Inventory (`DEFAULT_BRAIN_FIELD_CATALOG`)

- **PV1 (`PRICE_VOLUME`, >=90% coverage):** `close`, `open`, `high`, `low`, `volume`, `vwap`, `returns`, `cap`
- **FUNDAMENTAL (>=75% coverage):** `operating_cashflow`, `operating_income`, `assets`, `debt`, `equity`, `sales`, `ebitda`, `invested_capital`, `return_assets`, `return_equity`
- **ANALYST (>=30–45% coverage):** `eps_consensus_mean`, `target_price`, `est_eps_revision`, `est_eps_dispersion`, `analyst_coverage`
- **SENTIMENT (>=35% coverage):** `scl12_buzz`, `scl12_sentiment`, `snt_value`
- **OPTIONS (>=40% coverage):** `opt6_slopeavg1m`, `opt6_fcstr2imp`
