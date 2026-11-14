# 🚀 全自動化多市場量化交易、機器學習與強化學習研發系統
### Automated Multi-Market Quant Trading & ML/RL Research Platform

歡迎來到本專案！這是一個專為全自動化獲利與量化投資策略開發打造的綜合研發系統，涵蓋了三大核心金融市場與量化平台：
1. **台股大盤分析、多因子 ML 預測排序與每日雙時段 (08:00 盤前 / 16:30 盤後) 決策報告系統 (`taiwan_stock/`)**
2. **加密貨幣自動操盤、TradingView Lightweight Charts 專業看盤與 PyTorch PPO 深度強化學習引擎 (`crypto_trading/`, `web_ui/`)**
3. **WorldQuant BRAIN 量化 Alpha 研發、變現模式與高通過率策略庫 (`worldquant_brain/`)**
4. **雲端訓練 Kaggle Notebooks 庫 (`kaggle_notebooks/`)**

本項目採模組化與高內聚鬆耦合設計，每個子模組均可獨立回測、單獨運行或整合調度。

---

## 📂 系統目錄結構與模組分工

```text
.
├── README.md                      # [當前文件] 專案總覽、架構與 AI 協同開發協議
├── requirements.txt               # 全專案 Python 依賴清單
├── Dockerfile                     # 🐳 輕量雲端容器封裝檔 (支援 24h 免電腦常駐)
├── render.yaml                    # ☁️ Render.com 一鍵自動部署設定
├── GEMINI.md                      # 📜 量化操盤與工程開發規範指南
│
├── web_ui/                        # 📱 一體化響應式 Web APP 控制台 (手機/電腦通用)
│   ├── app.py                     # 服務啟動入口
│   ├── server.py                  # 多線程高效後端 API Server (ThreadingHTTPServer)
│   └── static/
│       └── index.html             # 現代質感 SPA 儀表板 (Lightweight Charts + Tailwind)
│
├── kaggle_notebooks/              # ☁️ 雲端 GPU/CPU 訓練與回測筆記本
│   ├── README.md                  # Kaggle 筆記本使用說明
│   ├── generate_notebooks.py      # 一鍵生成/更新 Kaggle ipynb 腳本
│   ├── kaggle_crypto_rl_trading.ipynb      # 加密貨幣 PPO 強化學習自適應倉位操盤筆記本
│   └── kaggle_taiwan_stock_ml_ranker.ipynb # 台股 LightGBM/XGBoost 多因子報酬排序筆記本
│
├── taiwan_stock/                  # 🇹🇼 1. 台股自動分析、ML 選股與晨報模組
│   ├── README.md                  # 台股模組詳細運作邏輯與說明
│   ├── config.py                  # 台股參數與 API/Webhook 設定
│   ├── market_analyzer.py         # 大盤多空掌握 (均線趨勢、市場溫度評分 0-100)
│   ├── stock_screener.py          # 智慧選股器 (量價突破、均線糾結發散、選股透明度)
│   ├── ml_ranker.py               # 🤖 機器學習 (LightGBM/XGBoost) 多因子超額報酬排序器 (5日隔離期)
│   ├── report_generator.py        # 每日早報排版產出器 (含大盤診斷 + 技術突破 + ML 排行)
│   ├── scheduler.py               # 每日雙時段 (08:00 盤前 / 16:30 盤後) 自動排程執行器
│   └── reports/                   # 歷次自動產生的每日晨報存檔目錄
│
├── crypto_trading/                # 🪙 2. 虛擬貨幣自動操盤與量化/RL 引擎
│   ├── README.md                  # 加密貨幣模組使用指南與風控說明
│   ├── real_exchange_trading_guide.md # 📖 實體交易所出入金、合約機制與 API 安全教學
│   ├── testnet_client.py          # 🌐 幣安官方合約測試網 (Binance Futures Testnet / USDC 0手續費)
│   ├── bitunix_client.py          # 🌐 Bitunix OpenAPI 實盤撮合串接模組
│   ├── regime_filter.py           # 🛡️ 階層式宏觀時框趨勢濾網 (Hierarchical Regime Filter)
│   ├── config.py                  # 交易所 API 設定、風控閥值與多時框設定
│   ├── data_loader.py             # 歷史行情下載、快取與即時 K 線接收 (含離線擬真行情生成)
│   ├── feature_engineer.py        # 🔬 平穩化特徵工程 (動量、Garman-Klass 波動度、委託流)
│   ├── ml_trainer.py              # 📈 滾動式前向交叉驗證 (Walk-Forward CV) 機器學習訓練器
│   ├── rl_env.py                  # 🎮 Gymnasium 相容之加密貨幣微分夏普比率交易環境
│   ├── rl_agent.py                # 🧠 純 PyTorch Actor-Critic PPO 連續動作交易 Agent
│   ├── loss_functions.py          # 📐 可微夏普比率與 Rank IC 損失函數 (Direct Sharpe Optimization)
│   ├── backtest_engine.py         # 向量化與事件驅動回測引擎 (含手續費與滑價)
│   ├── risk_manager.py            # 即時部位管理、動態停損/停利與 8% 最大回撤熔斷保護
│   ├── bot_runner.py              # 24/7 自動操盤實盤/模擬盤主程式 (支援 SQLite 訂單同步)
│   └── strategies/                # 策略庫
│       ├── __init__.py
│       ├── ml_strategy.py         # 🤖 機器學習 Alpha 預測策略
│       ├── rl_strategy.py         # 🧠 強化學習自適應倉位控管策略
│       ├── momentum_breakout.py   # 動量與波動度突破策略 (Bollinger / Donchian Breakout)
│       └── statistical_arbitrage.py# 統計學策略 (均值回歸、Z-Score 波動套利)
│
├── data_persistence/              # 💾 持久化儲存與 Google Drive 雲端同步
│   ├── system_state.db            # SQLite 狀態資料庫 (orders, positions, trade_logs, account_equity)
│   ├── cloud_state_snapshot.json  # 雲端 JSON 狀態快照
│   ├── gdrive_sync.py             # Google Drive API 雙向備份與 Rolling Backup 同步模組
│   └── models/                    # 已訓練完成之量化 ML/RL 模型權重庫
│       ├── crypto_lgbm_weights.joblib
│       ├── crypto_ppo_agent.pt
│       └── taiwan_stock_ranker.joblib
│
├── tests/                         # 🧪 單元測試與嚴格回歸審計套件 (28 Tests)
│   ├── test_architectural_upgrades.py # 交易所持倉同調、熔斷、流動性守護測試
│   ├── test_no_lookahead_audit.py     # 嚴格防未來函數檢驗、5日隔離期審計
│   ├── test_taiwan_stock_scoring.py   # 台股標準化評分 (0-100) 與透明度測試
│   ├── test_ml_rl_pipeline.py         # ML/RL 特徵工程、Walk-forward 訓練測試
│   ├── test_loss_functions.py         # 可微夏普比率、可微 Rank IC 損失函數測試
│   └── test_web_ui_and_vault.py       # Web UI API、Alpha 倉庫讀寫與心跳測試
│
├── docs/                          # 📚 核心講義、學術研究庫與驗收矩陣
│   ├── BATCH_ACCEPTANCE_MATRIX.md # 38 項全維度驗收矩陣 (GOAL_COMPLETE)
│   ├── USER_ACTION_GUIDE.md       # 👤 使用者實務操作指南 (每日例行/Kaggle訓練/實盤轉移)
│   ├── FILE_CLEANUP_AND_MERGE_REPORT.md # 🧹 檔案整理合併與冗餘清理紀錄
│   ├── handbook/
│   │   ├── QUANT_HANDBOOK.md      # 量化金融核心講義 Markdown
│   │   ├── QUANT_HANDBOOK.html    # 排版美化 HTML (支援 KaTeX 公式與 PDF 列印)
│   │   └── build_html.py          # HTML 自動生成腳本
│   └── research/
│       ├── ADVANCED_QUANT_RESEARCH.md # 🔬 前沿量化研究與論文庫
│       └── papers/
│           └── SELECTED_PAPERS_REVIEW.md # 精選學術論文解析 (arXiv 引用)
│
└── worldquant_brain/              # 🧠 3. WorldQuant BRAIN Alpha 研發與變現
    ├── README.md                  # BRAIN 獲利機制與顧問分潤詳解 (如何拿到美金報酬)
    ├── alpha_vault.py             # 🏛️ Alpha 策略倉庫 (驗證通過因子持久化保存與批量匯出)
    ├── alpha_miner.py             # 🧬 ContinuousAlphaMiner 24/7 背景自主演算法挖掘
    ├── alpha_vault.db             # 🗄️ SQLite 策略資料庫
    ├── verified_alphas.json       # 📑 JSON 策略儲備同步備份
    ├── data_fields_guide.md       # BRAIN 數據庫分類與使用指南 (量價、基本面、分析師預期)
    ├── operators_cheatsheet.md    # BRAIN 運算子速查表與數理語意說明
    ├── brain_client.py            # BRAIN API 模擬測試與提交工具腳本
    └── alpha_templates/           # 實測高通過率 Alpha 表達式庫
        ├── price_volume_alpha.py  # 量價動量與微觀結構反轉 Alpha
        ├── fundamental_alpha.py   # 基本面品質、成長性與盈餘動量 Alpha
        └── analyst_sentiment_alpha.py # 分析師目標價調整與情緒 Alpha
```

---

## ⚡ 快速開始 (Quick Start)

### 1. 安裝環境依賴
建議使用 Python 3.9+ 虛擬環境：
```bash
pip install -r requirements.txt
```

### 2. 執行全系統單元測試
```bash
python -m unittest discover -s tests
```

### 3. 本機訓練與回測 ML / RL 加密貨幣策略
```bash
# 執行量化回測 (支援動量突破、ML Alpha 與 RL 策略)
python crypto_trading/backtest_engine.py

# 啟動模擬操盤機器人 (預設為安全之 Paper Trading，無資金風險)
python crypto_trading/bot_runner.py
```

### 4. 台股每日 08:00 晨報與 AI 選股排序
```bash
# 立即測試生成今日晨報 (包含大盤多空結構 + 技術突破 + ML 多因子 5 日報酬排名)
python taiwan_stock/report_generator.py

# 啟動常駐排程器 (每日 08:00 自動運行並輸出)
python taiwan_stock/scheduler.py
```

### 5. 在 Kaggle 上一鍵雲端訓練 (免本地配置 GPU)
本專案在 `kaggle_notebooks/` 提供開箱即用的獨立 Jupyter Notebooks：
1. [`kaggle_crypto_rl_trading.ipynb`](kaggle_notebooks/kaggle_crypto_rl_trading.ipynb)：
   - 內建 CCXT 行情抓取、平穩化特徵工程、Gymnasium 交易環境、PyTorch PPO 訓練、回測淨值曲線視覺化、模型權重匯出 (`crypto_ppo_model.pt`)。
2. [`kaggle_taiwan_stock_ml_ranker.ipynb`](kaggle_notebooks/kaggle_taiwan_stock_ml_ranker.ipynb)：
   - 內建 yfinance 台股權值股下載、多因子構造、LightGBM 滾動驗證、Rank IC 評估與未來 5 日收益排行輸出。

---

## 🔬 模型規格與架構詳解 (Model Architecture & Specs)

### 1. 加密貨幣方向性預測機器學習模型 (Crypto ML Model)
- **架構**：LightGBM Regressor / XGBoost Regressor
- **輸入特徵 (Input Tensor)**：
  - 跨尺度對數收益率：`ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`
  - 均線平穩化乖離率：`bias_ma_5`, `bias_ma_20`, `bias_ma_50`
  - 波動度估計量：Parkinson 極端值波動率、Garman-Klass 波動率、布林帶寬 `%B`
  - 微觀結構與委託流代理：成交量 Z-Score、VWAP 偏離度、K 線實體位置（HL Location）
  - 技術指標：正規化 RSI(14)、MACD 差值比
- **預測目標 (Target)**：前瞻 3 週期超額報酬 $r_{t+3} = \frac{P_{t+3} - P_t}{P_t}$
- **驗證方式**：滾動式前向交叉驗證（Walk-Forward CV），杜絕未來數據偷看偏差 (Lookahead Bias)。
- **權重儲存格式**：Joblib 格式 (`.joblib`)

### 2. 加密貨幣深度強化學習操盤 Agent (Crypto RL Agent)
- **演算法**：Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
- **神經網絡**：Actor-Critic 雙網絡結構
  - Shared Feature Extractor: Linear(dim, 128) -> LayerNorm -> Mish -> Linear(128, 128) -> LayerNorm -> Mish
  - Actor Head: Linear(128, 64) -> Mish -> Linear(64, 1) -> Tanh (輸出連續動作 $[-1.0, 1.0]$)
  - Critic Head: Linear(128, 64) -> Mish -> Linear(64, 1) (狀態價值 $V(s)$ 估計)
- **狀態觀測空間 (State Space)**：
  - 因子特徵向量 + 代理當前持倉權重 + 未實現損益比例
- **動作空間 (Action Space)**：
  - 連續空間 $[-1.0, 1.0]$，分別代表滿倉做空、平倉觀望、滿倉做多
- **獎勵函數 (Reward Function)**：
  $$R_t = 100 \cdot r_t^{\text{net}} - 200 \cdot (r_t^{\text{net}})^2 \cdot \mathbb{I}_{r_t < 0} - 1.5 \cdot (\text{Drawdown}_t)^2 - 0.1 \cdot |\Delta \text{Position}|$$
  兼顧超額回報與夏普比率最大化，並嚴格懲罰回撤與過度頻繁換手。
- **權重儲存格式**：PyTorch Checkpoint (`.pt`)

### 3. 台股多因子機器學習排序模型 (Taiwan Stock ML Ranker)
- **架構**：LightGBM Multi-Factor Regressor
- **因子體系**：5d/10d/20d/60d 動量、均線長短期斜率、20日年化波動率、ATR 比率、爆量乘數、KD 與 RSI
- **預測標的**：各股未來 5 日超額回報
- **排序應用**：自動輸出前五大預期收益標的，直接整合進每日 08:00 晨報。

---

## 🤖 給協同開發 AI 助理的指引 (AI Collaboration Guide)

如果您是正在接手或協助維護本專案的 AI（例如 Claude、GPT、Antigravity、Cursor 或 Copilot）：

1. **模組邊界清晰**：
   - 台股、虛擬貨幣與 WorldQuant BRAIN 三者數據源與執行環境完全隔離。請勿在台股模組中引入 CCXT，亦勿在 BRAIN 模組中引入實盤下單邏輯。
2. **首重風控（Risk First）**：
   - 在任何涉及真金白銀交易的模組中，`crypto_trading/config.py` 中的 `PAPER_TRADING` 預設必須為 `True`。實盤開關必須由使用者手動明確確認。
   - 所有交易策略必須具備 ATR 停損、最大部位上限以及最大帳戶回撤（Drawdown Stop）機制。
3. **特徵與模型反過擬合（Anti-Overfitting）原則**：
   - 特徵必須平穩化（Stationary），禁止直接使用未經差分或比率化的原始價格作為輸入。
   - 評估 ML 模型時務必使用 Walk-Forward CV，不得隨機打亂時間序列進行 K-Fold。
   - 關注指標優先順序：**Rank IC > Out-of-Sample Sharpe > Accuracy**。
4. **代碼規範**：
   - 保持 Python 程式碼具備完整的 Type Hints（型別標註）與繁體中文 Docstring 註解。
   - 新增策略或模型請編寫對應之單元測試至 `tests/` 並通過測試。
