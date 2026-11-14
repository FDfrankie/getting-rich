# GEMINI.md - 量化操盤與工程開發規範指南 (Antigravity Quant Engine Specifications)

本規範檔案定義了本專案的核心架構原則、交易安全準則、跨市場數據模型以及與 Google Drive / 雲端 OnRender 之持久化協定。

---

## 一、系統核心架構與常駐執行原則

1. **24/7 不間斷背景操盤**：
   - 核心交易引擎（`QuantAppContext` / `CryptoTradingBot`）必須常駐於獨立的後台守護線程（Background Daemon Thread）運行。
   - 不得因使用者切換前端分頁（如從加密貨幣切換至台股）或重新整理瀏覽器（F5）而終止交易迴圈。
2. **Google Drive / 本地持久化數據庫**：
   - 所有訂單歷史、帳戶權益變化（Equity Curve）、持倉狀態與系統日誌一律即時同步至 `data_persistence/system_state.db` (SQLite) 與 JSON 快照。
   - 伺服器冷重啟或斷線重連時，一律由持久化資料庫自動還原歷史日誌與倉位，杜絕狀態遺失。
3. **無縫雲端部署架構 (OnRender)**：
   - 本地沙盒模擬與雲端正式服務架構完全解耦。在雲端 OnRender 部署時，由 Docker 容器環境接管背景任務，提供全球 SSL 訪問。

---

## 二、加密貨幣合約操盤準則 (Binance Testnet & Bitunix)

1. **USDC 零手續費優先與幣種對應 (Pair Mapper)**：
   - 深度學習與特徵工程訓練：利用流動性最高、歷史最豐富的 **USDT** 現貨與合約數據進行特徵萃取。
   - 實時策略執行與撮合：切換為 **USDC** 本位永續合約（享受幣安官方 0 手續費優勢，大幅降低高頻交易磨損）。
2. **全時間週期技術分析 (Multi-Timeframe Support)**：
   - 數據引擎支援 `15s` (極速高頻)、`1m`、`3m`、`5m`、`15m`、`1h`、`1d` 等週期切換與擬真幾何布朗運動（GBM + 厚尾波動）。
3. **雙交易所架構 (Binance -> Bitunix)**：
   - 練習與策略驗證：在幣安官方合約測試網（Binance Demo Testnet）進行模擬下單與風控檢驗。
   - 正式實盤：無縫轉接至 Bitunix OpenAPI（標準 HMAC-SHA256 簽名），支援市價/限價撮合與一鍵全平倉。

---

## 三、精選學術研究論文與工程知識庫 (Research & Literature)

本系統之損失函數與策略演算法基於以下精選學術文獻建立：
1. **arXiv:2607.09426**: *Deep Reinforcement Learning for Adaptive Market Making in Crypto Perpetual Swaps* (動態價差與資金費率對沖)
2. **arXiv:2508.02356**: *Cross-Asset Latent Representation Learning with Variational Autoencoders for Portfolio Allocation* (多幣種 VAE 潛在空間特徵)
3. **arXiv:2510.07943**: *Direct Optimization of Differentiable Sharpe and Sortino Ratios in Gradient-Boosted Trees* (可微夏普比率優化，實現於 `crypto_trading/loss_functions.py`)
4. **arXiv:2604.15973**: *Deep Order Flow Imbalance and Microstructure Alpha in High-Frequency Crypto Exchanges* (L2/L3 委託簿微觀結構特徵)
5. **arXiv:2406.02172**: *Temporal Fusion Transformers for Multi-Horizon Asset Return Forecasting Under High Volatility Regimes* (時間序列 TFT 架構)

### 頂尖量化工程資源與研究博客
- **Flashbots Research**: MEV 搜尋者策略、原子套利與抗三明治攻擊架構
- **Paradigm Research**: 自動做市商 (AMM) 集中流動性數學、永續合約資金費率套利模型
- **Crypto Quant Strategies 2026**: 截面多因子動量排序、統計套利協整配對交易
- **YouTube 數據爬取與字幕提取管線**: `scripts/collect_crypto_videos.py` 與 `scripts/fetch_top_transcripts.py`

---

## 四、WorldQuant BRAIN 量化顧問分潤標準

1. **模擬評估門檻**：
   - Sharpe Ratio $\ge 1.25$
   - Fitness $\ge 1.0$
   - Annual Turnover: $1\% \sim 70\%$
   - Margin $> 10 \text{ bps}$
2. **提交與相關性審查**：
   - 與個人及全平台庫存 Self-Correlation 須 $< 0.70$
3. **顧問季度提成公式**：
   $$\text{Payout} = \text{Quarterly PnL} \times \text{Base Rate (8\%)} \times \text{Sharpe Multiplier} \times \text{Turnover Penalty}$$
