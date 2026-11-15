# ADR 001: 交易模式集中管理與安全鎖 (Centralized TradingMode and Safety Lock)

## 狀態 (Status)
已接受 (Accepted) - 2026-11-15

## 背景 (Context)
先前系統中存在多處直接根據 `bool(api_key)` 判斷是否為實盤交易的邏輯，容易在配置缺失或測試過程中意外向交易所送出未預期的委託，且各模組各自讀取環境變數缺乏統一校驗。

## 決策 (Decision)
1. 建立 `crypto_trading.settings.Settings` 與 `TradingMode`（`LOCAL_PAPER`、`BINANCE_TESTNET`、`BITUNIX_LIVE`）。
2. 預設模式強制為 `LOCAL_PAPER`。
3. 啟動 `BITUNIX_LIVE` 必須同時滿足多重防護：明確宣告 `ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK`、金鑰非空、風控限額設定大於 0 以及確認操作員版本。
4. 任何測試一律採用 Mock，嚴禁對外發送真實委託。

## 後果 (Consequences)
- 徹底消除因配置或測試錯誤誤觸實盤下單的風險。
- 所有策略與執行引擎統一由 `Settings` 注入交易模式，架構清晰一致。
