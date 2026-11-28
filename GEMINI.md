# GEMINI.md — Antigravity Quant Engine 全專案架構、研究知識庫與 AI 協作終極手冊

> **最後更新**：2026-11-28 16:23:55+08:00（整合至 **P61 完整版** + **V108 雙軌量化知識庫**）  
> **GitHub 倉庫**：`FDfrankie/getting-rich`（本機同步路徑：`G:\我的雲端硬碟\炒幣`）  
> **適用對象**：所有 AI 模型（Gemini、Claude、GPT、Antigravity Agent）與人類量化工程師。閱讀本檔即可 100% 掌握專案目錄、資料庫結構、交易風控、WorldQuant BRAIN 挖礦與送出鐵律、頂尖交易員結構模仿加密貨幣引擎以及 Windows 排程維運規範。

## 一、專案定位與雙軌核心目標

本專案是一套 **24/7 雙軌全自動量化研究與交易平台**，由兩大完全解耦、各自獨立閉環的子系統構成：

1. **Track A — 加密貨幣 USDT-M 永續合約自動化操盤系統 (`crypto_trading/`, `portfolio/`)**
   - 標的：`BTC/USDT`、`ETH/USDT`、`SOL/USDT`（支援 Binance Testnet / Demo、Bitunix Live、Local Paper）。
   - **已驗證 Walk-Forward 基準**：`CondProb_MTF_KLine_3m_Maker`（BTCUSDT 3m Maker）在同期大盤暴跌 `-19.86%` 下達成 **77.59% 勝率、+1.657% 淨報酬、Profit Factor 1.469、年化 Sharpe 2.965、MaxDD 1.60%**；同策略改用 Taker（5.0 bps）立刻虧損 `-4.159% / Sharpe -8.27`。
   - **P61 旗艦「頂尖交易員結構模仿」引擎 (`crypto_trading/strategies/fib_condprob_kline_strategy.py`)**：
     1. **費波那契 0.618 / 0.382 提前掛單埋伏（Resting Maker Limit Ambush）**：於大時框順勢回撤位提前掛 Maker 限價單，實現 **0 滑價 + 純 Maker 費率（1.8–2.0 bps）** 成交。
     2. **10 種 SPA 檢定驗證 K 線微結構型態 + 經驗貝氏條件機率**：嚴格因果擴張視窗 Beta-Binomial 更新 `P(Win | Pattern, Fib_Zone, HTF_Regime)`，強制 `EV_bps >= 3.5 × 往返手續費`。
     3. **保本鎖後浮盈零風險加碼（Profit-Locked Pyramiding）+ 波動率目標動態槓桿（0.75x–2.75x）**：第一倉浮盈 >= 1.6×ATR 鎖住保本後才允許加碼 +50%，且帳面回撤 >= 4% 自動降槓桿 0.65×。
     4. **跨幣種相對強度輪動（Relative Strength Rotation）**：對比 BTC 48 根 K 線相對強度，強勢領漲幣放大多單、弱勢落後幣放大空單。

2. **Track B — WorldQuant BRAIN / IQC 2026 自動 Alpha 工廠 (`worldquant_brain/`)**
   - 24/7 無頭背景挖礦 (`scripts/headless_brain_miner.py`)，涵蓋自動資料集探索、MCTS/結構變異、官方 API 模擬、自我相關性檢查、逆序送出與顧問級高分 Alpha 保護庫。
   - **嚴格研究範圍**：僅限硬核量化金融（資產定價、市場微結構、衍生品定價、統計套利、量化 ML/AI、另類數據、交易執行與 QuantOps），嚴禁無關微產業雜訊。

## 二、專案目錄地圖（供所有模型快速定位）

```text
G:\我的雲端硬碟\炒幣\
├── GEMINI.md                                      # ★ 本檔：全專案 AI 協作架構手冊（唯一權威入口）
├── README.md                                      # 專案快速上手與常用指令總覽
├── worldquant_brain/                              # WorldQuant BRAIN / IQC 2026 核心模組
│   ├── brain_client.py                            # 官方 API 客戶端、Session 池重置(P60)、逆序送出、顧問庫與克隆清理
│   ├── multi_sim_client.py                        # 全域 Bronze 3 槽 Semaphore + 2s 節流鎖(P60)、非數值欄位黑名單
│   ├── alpha_miner.py                             # 連續挖礦引擎、V108 研究種子、巢狀 ts_decay_linear 封殺(P60)
│   ├── alpha_vault.py / canonical_ast.py          # SQLite Alpha 倉庫、原子 JSON 匯出與 AST 結構雜湊
│   ├── pre_submission_gate.py / iqc_scoring.py    # 官方提交硬門檻與 IQC 2026 三層評分
│   └── sqlite_resilience.py                       # Windows Drive 鎖定隔離與安全重試連線
├── crypto_trading/                                # 加密貨幣自動化交易引擎
│   ├── bot_runner.py                              # CryptoTradingBot（支援 fib_condprob_3m / condprob_3m / institutional_3m）
│   ├── multi_symbol_runner.py                     # MultiSymbolBotManager 多幣種並行與組合風控
│   ├── settings.py / instruments.py               # TradingMode 驗證與幣種精度規格
│   ├── risk_manager.py / regime_filter.py         # ATR 倉位、保本鎖、Chandelier 移動止盈、多時框濾網
│   └── strategies/
│       ├── fib_condprob_kline_strategy.py         # ★ P61 費波那契埋伏 + 經驗貝氏 K 線 + 浮盈加碼 + 強弱輪動引擎
│       ├── conditional_prob_rl_strategy.py        # Walk-Forward 過關 3m 條件機率引擎 (Sharpe 2.965)
│       ├── institutional_alpha_router.py          # 機構多策略路由引擎 (3m MakerBNB Sharpe 1.696)
│       └── momentum_breakout.py                   # 舊版基準策略（實盤停用，僅供介面繼承）
├── portfolio/risk_allocator.py                    # 組合層級槓桿、波動率目標與敞口上限控制
├── data_persistence/                              # 雲端同步遙測與快照目錄（非熱寫入 DB 本體）
│   ├── brain_pending_status.json / qualified_alpha_formulas_vault.json
│   ├── crypto_pnl_status.json / CRYPTO_WALKFORWARD_PROFITABILITY_REPORT.*
│   ├── headless_miner*.log / headless_crypto_bot*.log / watchdog_audit.log
│   └── state_paths.py                             # 自動將熱寫入 SQLite 導向 C:\quant_envs\state\
├── scripts/                                       # 24/7 背景服務、回測與雲端同步腳本
│   ├── headless_brain_miner.py / headless_crypto_bot.py / watchdog_check.ps1
│   ├── export_crypto_pnl_to_drive.py / backtest_crypto_usdt_walkforward.py
│   └── p60/p61_*.zip (+ .sha256)                  # 經 SHA-256 驗證之部署升級包
├── tests/                                         # 離線 pytest / unittest 自動化驗證套件
├── docs/references/                               # 升級清單 Manifest 與文獻索引
└── BRAIN_官方知識庫與研究紀錄/                      # Google Drive ID: 1kTr0HW48m6aQmL7SUECLj4BQ9F4RfmT8
    ├── crypto_quant_kb_v13.json                   # ★ 1,317 篇獨立加密貨幣量化前沿論文 (V108)
    ├── worldquant_brain_official_kb_v103_clean_quant.json.gz # ★ 14,863 篇硬核量化論文 + 624 官方 = 15,487 筆
    ├── WorldQuant BRAIN 官方指南與 Alpha 研究知識庫 (Google Doc, 113 章)
    └── 加密貨幣量化研究與 1m/3m 實戰知識庫 (Google Doc, 12 節)
```

## 三、SQLite 持久化與雲端鎖定隔離架構

- **熱寫入本機路徑**：`C:\quant_envs\state\alpha_vault.db` 與 `C:\quant_envs\state\system_state.db`（由 `data_persistence/state_paths.py` 自動導向）。
- **嚴禁直接於 Google Drive 掛載路徑對 SQLite 進行高頻寫入**：所有雲端鏡像一律透過 `sqlite3.backup()` 或原子暫存檔 `os.replace()` 定期回寫至 `data_persistence/`，杜絕 `database is locked`。
- **核心資料表（16 表架構）**：嚴禁將 `*.db*`、`.env`、`*.pkl` 納入 Git 版本控制。

## 四、WorldQuant BRAIN 送出政策與 P60 關鍵護欄

1. **Bronze 階段逆序送出與顧問保留庫（不可違反）**：
   - 所有達標 Alpha 必須永久歸檔於 `qualified_alpha_archive` 與 `qualified_alpha_formulas_vault.json`。
   - **顧問保留庫保護**：凡 `Sharpe >= 1.50` 或 `Fitness >= 1.20` 標記為 `CONSULTANT_RESERVE`（目前累計保護 120 顆，最高 Sharpe 2.82 / Fitness 2.40），嚴禁在無獎金的 Bronze 階段自動送出。
   - **逆序送出 (`worst_first`)**：Bronze 每日自動送出（上限 `2` 顆，台北時間 22:00 flush）嚴格依 `ORDER BY fitness ASC, sharpe ASC, score ASC` 消耗最低分達標 Alpha。
2. **P60 並發、連線與品質護欄**：
   - **全域 3 槽硬上限**：`multi_sim_client.py` 採用行程級 `BoundedSemaphore(3)` 與 `>= 2.0s` 全域 POST 節流鎖，消除 429 風暴；Bronze 預設跳過 Multi-Sim 403 探測。
   - **TLS Socket 自動重置**：`brain_client.py` 遇 `SSLError / ConnectionError` 立即重建執行緒局部 `requests.Session`（`HTTPAdapter(4,4)`）。
   - **防克隆與中繼欄位封殺**：封殺巢狀 `ts_decay_linear(ts_decay_linear(...))` 克隆體與 `*_currency_code` 等非數值中繼欄位，開機自動執行 `sweep_duplicate_pending_clones()`。

## 五、加密貨幣操盤獲利核心與 P61 策略切換規範

- **實盤晉升硬門檻 (`backtest_crypto_usdt_walkforward.py`)**：`net_return > 0`、`Sharpe >= 1.00`、`Profit Factor >= 1.25`、`MaxDD <= 15.0%`、`Trades >= 15`。
- **環境變數切換 (`CRYPTO_STRATEGY_PROFILE`)**：
  - `fib_condprob_3m` / `fib_condprob_5m` / `elite_fib_3m`：P61 頂尖交易員結構模仿引擎（費波那契埋伏 + 貝氏 K 線 + 保本浮盈加碼 + 2.75x 波動率目標槓桿 + 跨幣強弱輪動）。
  - `condprob_3m`：Walk-Forward 已過關 3m 穩定版（77.59% WR、Sharpe 2.965、MaxDD 1.60%）。
  - `institutional_3m`：多因子機構路由版（Sharpe 1.696、PF 1.261）。

## 六、Windows 排程、測試與安全重啟鐵律

1. **推送前測試（必須 100% 通過）**：
   `C:\quant_envs\antigravity\Scripts\python.exe -m pytest -q tests/test_p60_alpha_quality_and_crypto_upgrade.py tests/test_p61_fib_condprob_kline_and_docs.py`
2. **Git 提交規範**：永遠顯式指定 `git add <files...>`，嚴禁 `git add -A` 或 `git add .`。
3. **乾淨重啟規範（僅在 `git push origin main` 成功後執行）**：
   ```powershell
   Stop-ScheduledTask -TaskName "Quant_BrainMiner" -ErrorAction SilentlyContinue
   Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*headless_brain_miner*" -or $_.CommandLine -like "*headless_crypto_bot*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
   Start-Sleep 2
   Start-ScheduledTask -TaskName "Quant_BrainMiner"
   Enable-ScheduledTask -TaskName "Quant_Watchdog" -ErrorAction SilentlyContinue | Out-Null
   Start-ScheduledTask -TaskName "Quant_Watchdog" -ErrorAction SilentlyContinue
   Start-Sleep 4
   Get-ScheduledTask -TaskName "Quant_*" | Select-Object TaskName, State
   Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*headless_brain_miner*" -or $_.CommandLine -like "*headless_crypto_bot*" } | Select-Object ProcessId, CreationDate, CommandLine
   ```
