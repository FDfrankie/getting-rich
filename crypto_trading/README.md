# 🪙 虛擬貨幣自動量化操盤、機器學習與強化學習系統

本模組為加密貨幣量化交易引擎，具備**統計學理論**、**技術圖型動量**、**機器學習預測 (LightGBM/XGBoost)**、**深度強化學習自適應倉位控管 (PPO/Actor-Critic)**、**專業回測系統**與**嚴格風控熔斷**機制。

---

## 📌 模組架構

1. **特徵工程器 (`feature_engineer.py`)**：
   - 提取跨週期對數收益率（1h, 3h, 6h, 12h, 24h）、均線乖離率（MA5, 10, 20, 50）。
   - 波動度估計：Parkinson 極端值波動率、Garman-Klass OHLC 波動率、布林帶寬度與 %B。
   - 微觀結構與委託流代理：成交量 Z-Score、VWAP 偏離度、K 線實體位置因子（HL Location）。
   - 目標建構：前瞻多週期超額收益（`target_fwd_ret_3`）與方向性分類標籤。

2. **機器學習訓練核心 (`ml_trainer.py`)**：
   - 實作無未來數據偷看的**滾動式前向交叉驗證（Walk-Forward Cross Validation）**。
   - 支援 LightGBM 與 XGBoost 回歸器與分類器。
   - 計算嚴格指標：Rank IC (資訊係數 Spearman)、年化夏普比率、方向預測勝率。
   - 支援特徵重要性分析與權重序列化導出 (`.joblib`)。

3. **強化學習環境與 Agent (`rl_env.py` & `rl_agent.py`)**：
   - `CryptoTradingEnv`：相容 Gymnasium 規範之交易模擬環境。
   - 獎勵函數：**微分夏普比率（Differential Sharpe Ratio）**、下行波動度二次懲罰、回撤懲罰與摩擦成本調節。
   - `PPOAgent`：純 PyTorch 實作之 Actor-Critic PPO 演算法，支援連續動作空間（持倉權重 $[-1.0, 1.0]$），自動調控換手率與防範大跌回撤。

4. **策略模組 (`strategies/`)**：
   - **`ml_strategy.py`（ML Alpha 預測策略）**：結合 Walk-Forward 訓練之 LightGBM 預測與信心門檻值（Hurdle Filter）過濾噪聲開倉。
   - **`rl_strategy.py`（RL PPO 策略）**：以深度 Actor-Critic 狀態網絡動態調節現貨或合約多空持倉。
   - **`momentum_breakout.py`（動量通道突破）**：布林帶與唐奇安通道單邊趨勢突破。
   - **`statistical_arbitrage.py`（統計套利）**：Z-Score 極端波動均值回歸。

5. **回測引擎 (`backtest_engine.py`)**：
   - 支援 Taker 撮合手續費（預設 0.05%）與滑價（Slippage）模型。
   - 輸出全套專業指標：總報酬率、年化夏普比率（Sharpe Ratio）、最大回撤（Max Drawdown）、勝率與盈虧因子（Profit Factor）。

6. **風控核心 (`risk_manager.py`)**：
   - **ATR 動態部位管理**：單筆風險嚴格鎖定在總資金之 1%~2%。
   - **全域熔斷（Circuit Breaker）**：當帳戶自歷史峰值回撤達 8% 時，立即停用所有新開倉操作。

7. **實盤/模擬盤操盤主程式 (`bot_runner.py`)**：
   - 預設啟用 **Paper Trading（模擬下單）**，以虛擬保證金（10,000 USDT）即時搓合買賣，杜絕實盤風險。

---

## 🚀 執行指南

### 1. 執行策略回測 (經典動量 / ML / RL)
```bash
python crypto_trading/backtest_engine.py
```

### 2. 本機訓練 ML Alpha 模型
```python
from crypto_trading.data_loader import CryptoDataLoader
from crypto_trading.ml_trainer import CryptoMLTrainer

loader = CryptoDataLoader(symbol="BTC/USDT", timeframe="1h")
df = loader.fetch_ohlcv(limit=1000)

trainer = CryptoMLTrainer(model_type="lightgbm", task="regression")
results = trainer.walk_forward_train(df, n_splits=5)
print("Walk-forward Mean Rank IC:", results["mean_rank_ic"])
trainer.save_weights("crypto_lgbm_weights.joblib")
```

### 3. 啟動模擬自動操盤
```bash
python crypto_trading/bot_runner.py
```

### 4. 在 Kaggle 上運用 GPU 加速訓練 RL 模型
本專案提供零配置、全自包含的 Kaggle 筆記本：
- 檔案路徑：[`kaggle_notebooks/kaggle_crypto_rl_trading.ipynb`](../kaggle_notebooks/kaggle_crypto_rl_trading.ipynb)
- 直接將此 `.ipynb` 上傳至 Kaggle Notebook，選取 GPU (T4/P100)，一鍵點擊「Run All」即可完成多千根 K 線的 PPO 訓練與權重導出。
