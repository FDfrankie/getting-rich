# P51 加密貨幣 1m/3m K線條件機率策略專題調研與 Walk-Forward 盈利驗證

- **生成時間**：`2026-11-26 16:47:26+08:00`
- **專屬知識庫**：`docs/references/crypto_quant_kb_v1.json`（14 篇核心加密貨幣微結構、1m/3m K線、條件機率與扣費實證文獻）
- **核心目標**：回應實戰交易者普遍使用 `1m / 3m` K線的結構優勢，解決高頻換手被幣安 USDT-M 手續費吃光的痛點，並以 **Walk-Forward 真實歷史回測硬門檻**驗證是否具備扣費後正期望值。

## 一、為什麼多數實戰派看 1m/3m K線，但機器人直接跑 1m/3m 卻常虧錢？

1. **1m/3m 的真正優勢（入場精準度與風報比）**
   - 幣安 USDT-M 永續合約的爆倉插針（Liquidity Sweep）、訂單流反轉（OFI Reversal）、VWAP 回踩與關鍵結構突破都發生在 1–3 分鐘層級。
   - 在 `1m/3m` 上掛結構性影線停損（例如 15–25 bps），搭配 `15m/1h` 大週期順勢目標（45–90 bps），可創造 **2:1 ~ 3:1 的不對稱風報比**，這是純 `1h` 收盤進場做不到的精準度。
2. **1m/3m 機器人爆虧的數學真相（幣安 USDT-M VIP 0 手續費牆）**
   - 幣安 USDT-M VIP 0 費率為 **Maker `0.02%`（2 bps）/ Taker `0.05%`（5 bps）**；啟用 10% BNB 抵扣為 `0.018% / 0.045%`。
   - BTCUSDT 單根 1m K線平均振幅僅約 `8–15 bps`。若每筆進出皆吃市價單（Taker 往返 `10 bps` + 滑價 `2–4 bps` = `12–14 bps`），**手續費直接吃掉一整根 1m K線的全部波動**。
   - 若改採 **Maker-First（`POST_ONLY` 限價掛單）**，往返費率直接從 `10 bps` 降至 `4 bps`（省 60%），但必須搭配文獻 [CKB-008](https://arxiv.org/html/2502.18625v2) 的**逆向選擇防護（Adverse Selection Guard）**——只在插針掃流動性或 OFI 反轉時掛 Maker，避免在單邊急殺/急拉時被動接刀。

## 二、P51 升級架構：四層 1m/3m K線條件機率引擎

升級後的 `crypto_trading/strategies/conditional_prob_rl_strategy.py` 完整落地 6 篇關鍵文獻：

1. **零未來函數多週期對齊（Causal MTF，依據 [MDPI 2026](https://www.mdpi.com/2571-9394/8/3/40)）**
   - 從 `1m` 或 `3m` 原始 K線因果聚合出高週期（`15m` / `1h`）趨勢與波動狀態，並**強制對已完成的高週期 K線做 `.shift(1)`**，杜絕論文揭露的 `+0.20 ROC-AUC` 未來函數膨風。
2. **真實 K線形態解剖 + 訂單流不平衡（依據 [IREF 2026](https://ideas.repec.org/a/eee/reveco/v108y2026ics1059056026002716.html)、[arXiv:2602.00776](https://arxiv.org/html/2602.00776v1)、[arXiv:2608.21888](https://arxiv.org/html/2608.21888v1)）**
   - 計算 `body_ratio`、`upper_wick_ratio`、`lower_wick_ratio`，自動辨識 `BULL_SWEEP`（下影線插針反轉）、`BEAR_SWEEP`（上影線插針）、`BULL_ENGULF`、`BEAR_ENGULF`、`BREAKOUT_UP/DOWN`。
   - 直接讀取幣安 K線的 `TakerBuyBaseVol` 計算真實主動買賣不平衡 `OFI = 2*(TakerBuy/Vol) - 1`（無此欄時以價量位階代理），並計算 VWAP 偏離度。
3. **CUSUM 事件濾網 + 三重屏障條件機率（依據 [Financial Innovation 2025](https://ideas.repec.org/a/spr/fininn/v11y2025i1d10.1186_s40854-025-00866-w.html)）**
   - 不在盤整雜訊 K線亂開倉，僅在 CUSUM 累積波動達標或出現明確 K線形態事件時觸發評估。
   - 條件機率不再只看固定 N 根收盤漲跌，而是估計 **三重屏障（Triple-Barrier）勝率**：在最大持有時間 `H` 內，先觸及停利目標 `k_tp * ATR`（扣雙邊手續費）而非先觸及停損 `k_sl * ATR` 的條件機率 $P(	ext{TP before SL} \mid 	ext{HTF Regime}, 	ext{Candle Pattern}, 	ext{OFI})$，並採**三層階層貝氏收縮（Global -> Coarse -> Fine）**避免樣本稀疏。
4. **Walk-Forward 盈利硬門檻（Profitability Gate，依據 [arXiv:2606.00060](https://arxiv.org/html/2606.00060v1)）**
   - 新增 `scripts/backtest_crypto_usdt_walkforward.py`，直接抓幣安 USDT-M 實盤歷史 `1m / 3m / 1h` K線與資金費率，同步對比 Maker `4 bps` vs Taker `10 bps`，並輸出自動驗收判定（必須達標：`OOS Net Sharpe >= 1.0`、`Profit Factor >= 1.25`、`Max Drawdown <= 15%`、`Net Return > 0`）。
