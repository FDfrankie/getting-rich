import os

# 交易所設定 (支援 CCXT 支援的數十家主流交易所，如 binance, bitunix, okx, bybit)
EXCHANGE_NAME = "binance"
API_KEY = os.getenv("CRYPTO_API_KEY", "")
API_SECRET = os.getenv("CRYPTO_API_SECRET", "")

# 幣安合約測試網專屬金鑰 (Binance Futures Testnet: dapi/fapi)
BINANCE_TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
BINANCE_TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")
if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_API_SECRET:
    # 提醒安全規範：未設置環境變數時自動安全降級
    pass

# Bitunix 實盤對接金鑰 (SHA256 HMAC 簽名，實盤零手續費/低手續費撮合首選)
BITUNIX_API_KEY = os.getenv("BITUNIX_API_KEY", "")
BITUNIX_API_SECRET = os.getenv("BITUNIX_API_SECRET", "")

# 預設交易對與時間週期 (Timeframe)
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_SETTLEMENT_CURRENCY = "USDC"  # 實盤優先 USDC 永續合約 (0 費率優勢)，訓練採用 USDT
DEFAULT_TIMEFRAME = "1h"  # 1m, 3m, 5m, 15m, 1h, 1d

# 安全機制開關：預設開啟 Paper Trading (模擬交易模式，絕不使用真金白銀)
PAPER_TRADING = True
INITIAL_BALANCE_USDT = 10000.0  # 模擬初始保證金 10,000 USDT

# 嚴格風控參數
RISK_PARAMS = {
    "max_risk_per_trade_pct": 0.02,   # 單筆交易最大虧損風險：總資金的 2%
    "max_drawdown_stop_pct": 0.08,    # 最大帳戶回撤熔斷：累計回撤 8% 即停止一切新開倉
    "atr_stop_multiplier": 2.0,       # ATR 動態停損距離倍數
    "atr_take_profit_multiplier": 3.0,# ATR 動態停利距離倍數 (提供 1:1.5 風報比)
    "max_open_positions": 3,          # 最大同時持倉數量
    "slippage_bps": 5,                # 回測滑價假設 (5 bps = 0.05%)
    "taker_fee_pct": 0.0005           # 交易所 Taker 手續費率 (0.05%)
}

# 統計套利與動量參數
STRATEGY_PARAMS = {
    "bollinger_window": 20,
    "bollinger_std": 2.0,
    "zscore_entry_threshold": 2.0,    # 統計套利進場偏離度
    "zscore_exit_threshold": 0.5,     # 均值回歸出場偏離度
    "rsi_period": 14,
    "fast_ema": 12,
    "slow_ema": 26,
    "signal_ema": 9
}
