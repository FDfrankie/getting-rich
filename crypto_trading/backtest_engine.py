import pandas as pd
import numpy as np
from typing import Dict, Any
from crypto_trading.config import INITIAL_BALANCE_USDT, RISK_PARAMS
from crypto_trading.data_loader import CryptoDataLoader
from crypto_trading.strategies.momentum_breakout import MomentumBreakoutStrategy

class BacktestEngine:
    """
    量化回測引擎
    支援：
    1. 手續費扣除 (Taker Fee)
    2. 滑價模型 (Slippage)
    3. 多空雙向交易損益統計
    4. 輸出關鍵績效指標：Sharpe Ratio、最大回撤 (Max Drawdown)、獲利因子 (Profit Factor)、勝率 (Win Rate)
    """
    def __init__(self, initial_capital: float = INITIAL_BALANCE_USDT):
        self.initial_capital = initial_capital
        self.fee_rate = RISK_PARAMS["taker_fee_pct"]
        self.slippage = RISK_PARAMS["slippage_bps"] / 10000.0

    def run(self, df_with_signals: pd.DataFrame) -> Dict[str, Any]:
        df = df_with_signals.copy()
        close = df['Close']
        signal = df['signal']

        # 計算資產價格變動百分比
        market_returns = close.pct_change().fillna(0)

        # 策略部位：持倉信號遞延 1 期生效 (避免未來數據偷看偏差 Lookahead Bias)
        strategy_pos = signal.shift(1).fillna(0)

        # 交易行為檢測：部位發生改變時產生手續費與滑價
        trades = (strategy_pos != strategy_pos.shift(1)).astype(int)
        friction_cost = trades * (self.fee_rate + self.slippage)

        # 策略收益率 = (部位 * 標的收益率) - 摩擦成本
        strategy_returns = (strategy_pos * market_returns) - friction_cost

        # 累積淨值曲線 (Equity Curve)
        equity_curve = (1.0 + strategy_returns).cumprod() * self.initial_capital

        # 計算績效指標
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 滾動最大回撤
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # 年化夏普比率 (假設 1h 週期，一年約 8760 小時)
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(8760)

        # 勝率與損益比
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades) + 1e-9)
        profit_factor = winning_trades.sum() / (abs(losing_trades.sum()) + 1e-9)

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(equity_curve.iloc[-1], 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "total_bars_tested": len(df),
            "total_trade_actions": int(trades.sum())
        }

if __name__ == "__main__":
    loader = CryptoDataLoader(symbol="BTC/USDT", timeframe="1h")
    data = loader.fetch_ohlcv(limit=500)
    strat = MomentumBreakoutStrategy()
    df_signals = strat.generate_signals(data)

    engine = BacktestEngine()
    metrics = engine.run(df_signals)

    print("=== 量化回測績效報告 (Backtest Performance) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
