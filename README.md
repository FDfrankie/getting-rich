# Antigravity Quant Engine (`FDfrankie/getting-rich`)

24/7 雙軌全自動量化研究與交易平台，整合 **WorldQuant BRAIN / IQC 2026 Alpha 工廠** 與 **加密貨幣 USDT-M P61 頂尖交易員結構模仿引擎（費波那契掛單埋伏 × 經驗貝氏 K 線條件機率 × 浮盈零風險加碼 × 波動率目標動態槓桿 × 跨幣強弱輪動）**。

> **完整架構、資料表、知識庫索引與 AI 協作規範請詳見根目錄 [`GEMINI.md`](./GEMINI.md)**。

## 快速狀態總覽（截至 2026-11-28 16:23:55+08:00）

- **知識庫規模**：`crypto_quant_kb_v13.json`（**1,317 篇**獨立加密貨幣論文）+ `worldquant_brain_official_kb_v103_clean_quant.json.gz`（**14,863 篇**硬核美股量化論文 + **624 篇**官方文章 = **15,487 筆**）。
- **加密貨幣核心策略**：
  - `FibCondProbKLineStrategy`（P61 完整版）：費波那契 0.618/0.382 掛單埋伏（0 滑價 Maker）+ 10 種 SPA 驗證 K 線 + 經驗貝氏條件機率 + 保本鎖後浮盈加碼 + 0.75x–2.75x 波動率目標槓桿 + 跨幣種相對強度輪動。
  - `CondProb_MTF_KLine_3m_Maker`（基準過關版）：BTCUSDT 3m Maker 勝率 **77.59%、+1.657% 淨報酬（同期大盤 -19.86%）、Sharpe 2.965、MaxDD 1.60%**。
- **WorldQuant BRAIN 狀態**：顧問保留庫保護 **120 顆**高分 Alpha（最高 Sharpe 2.82 / Fitness 2.40）；P60 已全面修復 2×3 並發 429 風暴、TLS Socket 斷線與巢狀克隆問題。

## 常用指令速查

```powershell
# 1. 執行真實 Binance USDT-M Walk-Forward OOS 對決回測（含 P61 完整版）
C:\quant_envs\antigravity\Scripts\python.exe scripts/backtest_crypto_usdt_walkforward.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# 2. 執行 P60 + P61 單元測試
C:\quant_envs\antigravity\Scripts\python.exe -m pytest -q tests/test_p60_alpha_quality_and_crypto_upgrade.py tests/test_p61_fib_condprob_kline_and_docs.py
```
