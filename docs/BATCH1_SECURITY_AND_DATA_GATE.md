# BATCH 1: 安全閘門與資料架構升級 (Security Gate & Data Integrity)

## 一、核心目標與任務清單
1. **API Key & 金鑰安全全面抽離 (Zero Hardcoded Secrets)**：
   - 徹底移除 crypto_trading/config.py、web_ui/server.py、worldquant_brain/brain_client.py 等模組內的明文金鑰與模擬預設值。
   - 所有敏感憑證（BINANCE_API_KEY, BINANCE_API_SECRET, BITUNIX_API_KEY, BITUNIX_SECRET_KEY, BRAIN_CREDENTIALS）一律強制由 .env 或系統環境變數加載。
2. **USDC 本位永續合約 0 手續費對應 (USDC Zero-Fee Pair Mapper)**：
   - 特徵計算採用流動性最佳之 USDT 交易對行情（如 BTC/USDT, ETH/USDT, SOL/USDT）。
   - 下單撮合引擎支援自動映射至幣安官方 0 手續費之 USDC 永續合約（BTC/USDC:USDC）。
3. **多時間週期支援與動態年化係數 (Multi-Timeframe & Dynamic Annualization)**：
   - 數據引擎支援 15s, 1m, 3m, 5m, 15m, 1h, 1d 等任意時間週期 K 線計算。
   - 年化 Sharpe Ratio 嚴格依據週期動態調整：年化係數  = \sqrt{N_{periods}}$，防止高頻週期被錯誤套用日線 252 因子導致指標失真。
4. **流動性與極端風控閘門 (Liquidity & Volatility Guard)**：
   - 進入下單前實施 24h 交易量、價差 (Bid-Ask Spread) 與盤口深度過濾，杜絕滑點過大之流動性枯竭風險。
