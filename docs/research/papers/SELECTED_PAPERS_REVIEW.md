# 精選量化金融與深度學習學術論文導讀 (Curated Research Papers)

本目錄收錄專案策略與損失函數設計所依據之 5 篇重要論文精華導讀：

---

## 1. arXiv:2607.09426
- **論文題目**: *Deep Reinforcement Learning for Adaptive Market Making in Crypto Perpetual Swaps*
- **核心內容**: 探討加密貨幣永續合約（Perpetual Futures）的做市與套利策略。利用 Actor-Critic 強化學習演算法動態調整掛單價差（Bid-Ask Spread）與資金費率（Funding Rate）對沖比例，在波動加劇時透過偏置庫存（Inventory Skew）抑制最大回撤。
- **專案對應**: `crypto_trading/strategies/rl_strategy.py`

---

## 2. arXiv:2508.02356
- **論文題目**: *Cross-Asset Latent Representation Learning with Variational Autoencoders for Portfolio Allocation*
- **核心內容**: 運用變分自動編碼器 (VAE) 將多幣種高維指標（多週期 RSI、MACD、委託簿深度差）壓縮至連續低維潛在空間 (Latent Space)，消除特徵間的共線性並提升樣本外汎化能力。
- **專案對應**: `crypto_trading/feature_engineer.py`

---

## 3. arXiv:2510.07943
- **論文題目**: *Direct Optimization of Differentiable Sharpe and Sortino Ratios in Gradient-Boosted Trees and Deep Networks*
- **核心內容**: 證明傳統 MSE 損失函數無法有效捕捉金融市場的風險調整後報酬。提出可微夏普損失函數（Differentiable Sharpe Ratio Loss），將投資組合回報的一階矩（均值）與二階矩（方差）梯度直接反向傳播。
- **專案對應**: `crypto_trading/loss_functions.py` (`SoftSharpeLoss`)

---

## 4. arXiv:2604.15973
- **論文題目**: *Deep Order Flow Imbalance and Microstructure Alpha in High-Frequency Crypto Exchanges*
- **核心內容**: 分析加密合約微觀結構中的訂單流失衡（Order Flow Imbalance, OFI）與高頻成交量分佈，建構具備高預測力的高頻短線 Alpha。
- **專案對應**: `crypto_trading/bitunix_client.py` & `crypto_trading/testnet_client.py`

---

## 5. arXiv:2406.02172
- **論文題目**: *Temporal Fusion Transformers for Multi-Horizon Asset Return Forecasting Under High Volatility Regimes*
- **核心內容**: 結合自注意力機制（Self-Attention）與門控殘差網絡（GRN），在極端市場波動狀態（Regime Shifts）下實現多時間尺度（15s ~ 1d）之資產回報預測。
- **專案對應**: `docs/research/ADVANCED_QUANT_RESEARCH.md`
