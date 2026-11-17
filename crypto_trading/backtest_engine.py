from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.config import INITIAL_BALANCE_USDT, RISK_PARAMS


@dataclass
class FeeSchedule:
    """
    明確定義交易手續費、滑價與市場微觀結構限制模型
    具備資料來源標記與生效時間
    """

    source: str = "binance_futures_default"
    effective_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    maker_fee_rate: float = 0.0002  # 2 bps (0.02%)
    taker_fee_rate: float = 0.0005  # 5 bps (0.05%)
    spread_bps: float = 1.0  # 1 bps (0.01%)
    slippage_bps: float = 5.0  # 5 bps (0.05%)
    funding_rate_per_bar: float = 0.0001  # 每期資金費率 (預設 0.01% 或每 8 小時結算，可為 0)
    delay_bars: int = 1  # 信號執行延遲期數 (預設 t+1 即 shift(1))
    tick_size: float = 0.1  # 價格最小變動單位
    step_size: float = 0.001  # 數量最小變動單位
    min_notional: float = 5.0  # 最小名義下單價值 (USD/USDT)


class BacktestEngine:
    """
    量化回測引擎 (P5 增強版)
    支援：
    1. 嚴格 t+1 執行 (executed_position = target_signal.shift(delay_bars))
    2. 分開計算並輸出 maker, taker, spread, slippage, funding, delay, tick, step, min notional
    3. 完整績效輸出：gross/net return, Sharpe, Sortino, Calmar, drawdown, turnover, 各項成本,
       trade count, exposure, hit rate, profit factor, signal delay, data source
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_BALANCE_USDT,
        fee_schedule: FeeSchedule | None = None,
    ):
        self.initial_capital = initial_capital
        self.fee_schedule = fee_schedule or FeeSchedule(
            maker_fee_rate=RISK_PARAMS.get("maker_fee_pct", 0.0002),
            taker_fee_rate=RISK_PARAMS.get("taker_fee_pct", 0.0005),
            slippage_bps=float(RISK_PARAMS.get("slippage_bps", 5.0)),
        )
        self.fee_rate = self.fee_schedule.taker_fee_rate
        self.slippage = self.fee_schedule.slippage_bps / 10000.0

    def run(
        self,
        df_with_signals: pd.DataFrame,
        timeframe: str = "1h",
        data_source: str | None = None,
    ) -> dict[str, Any]:
        df = df_with_signals.copy()
        close = df["Close"]
        signal = df["signal"]

        source_name = data_source or getattr(df, "data_source", self.fee_schedule.source)

        # 1. 計算資產市場收益率
        market_returns = close.pct_change(fill_method=None).fillna(0)

        # 2. 策略部位：嚴格延遲 delay_bars 期 (預設 t+1，避免 Lookahead Bias)
        delay = self.fee_schedule.delay_bars
        strategy_pos = signal.shift(delay).fillna(0)

        # 3. 換手率 (Turnover)：部位改變的絕對值
        turnover = (strategy_pos - strategy_pos.shift(1).fillna(0)).abs()

        # 4. 細項交易成本計算
        # 4.1 Taker 手續費 (以 taker 假設計算換手手續費)
        taker_costs = turnover * self.fee_schedule.taker_fee_rate
        # 4.2 Maker 手續費 (若策略有掛單情境，可分開追蹤)
        maker_costs = turnover * self.fee_schedule.maker_fee_rate * 0.0  # 預設市價撮合全部為 taker
        # 4.3 點差成本 (Spread Cost)
        spread_rate = (self.fee_schedule.spread_bps / 10000.0) / 2.0  # 半個 spread
        spread_costs = turnover * spread_rate
        # 4.4 滑價成本 (Slippage Cost)
        slippage_rate = self.fee_schedule.slippage_bps / 10000.0
        slippage_costs = turnover * slippage_rate
        # 4.5 資金費率成本 (Funding Cost)
        funding_costs = strategy_pos.abs() * self.fee_schedule.funding_rate_per_bar

        # 總摩擦成本
        total_friction_cost = taker_costs + maker_costs + spread_costs + slippage_costs + funding_costs

        # 5. 毛收益率 vs 淨收益率
        gross_returns = strategy_pos * market_returns
        net_returns = gross_returns - total_friction_cost

        # 累積淨值曲線 (Gross vs Net)
        gross_equity = (1.0 + gross_returns).cumprod() * self.initial_capital
        net_equity = (1.0 + net_returns).clip(lower=0.0).cumprod() * self.initial_capital

        gross_return_pct = (gross_equity.iloc[-1] - self.initial_capital) / self.initial_capital
        net_return_pct = (net_equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # 6. 回撤 (Drawdown) 計算
        net_rolling_max = net_equity.cummax()
        drawdowns = (net_equity - net_rolling_max) / net_rolling_max
        max_drawdown = float(drawdowns.min())

        # 7. 年化因子與 Sharpe, Sortino, Calmar
        tf_map = {
            "15s": 2102400.0,
            "1m": 525600.0,
            "3m": 175200.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "30m": 17520.0,
            "1h": 8760.0,
            "2h": 4380.0,
            "4h": 2190.0,
            "1d": 365.0,
        }
        annual_factor = tf_map.get(str(timeframe).lower(), 8760.0)

        mean_net = net_returns.mean()
        std_net = net_returns.std()
        sharpe = float((mean_net / (std_net + 1e-9)) * np.sqrt(annual_factor))

        downside_returns = net_returns[net_returns < 0]
        downside_std = downside_returns.std()
        sortino = float((mean_net / (downside_std + 1e-9)) * np.sqrt(annual_factor))

        calmar = float(net_return_pct / (abs(max_drawdown) + 1e-9))

        # 8. Exposure, Hit Rate, Profit Factor, Trade Count
        exposure = float((strategy_pos != 0).mean())

        winning_bars = net_returns[net_returns > 0]
        losing_bars = net_returns[net_returns < 0]
        hit_rate = float(len(winning_bars) / (len(winning_bars) + len(losing_bars) + 1e-9))
        profit_factor = float(winning_bars.sum() / (abs(losing_bars.sum()) + 1e-9))

        # Closed trades 計算
        trades_pnl = []
        in_trade = False
        curr_pos = 0.0
        entry_p = 0.0
        trade_count = 0
        for i in range(len(df)):
            pos_i = strategy_pos.iloc[i]
            p_i = close.iloc[i]
            if pos_i != curr_pos:
                if in_trade and curr_pos != 0.0:
                    trade_pnl = (p_i - entry_p) / entry_p if curr_pos > 0 else (entry_p - p_i) / entry_p
                    fee_roundtrip = (self.fee_schedule.taker_fee_rate + slippage_rate + spread_rate) * 2
                    trades_pnl.append(trade_pnl - fee_roundtrip)
                    trade_count += 1
                if pos_i != 0.0:
                    in_trade = True
                    curr_pos = pos_i
                    entry_p = p_i
                else:
                    in_trade = False
                    curr_pos = 0.0

        if trades_pnl:
            wins = sum(1 for p in trades_pnl if p > 0)
            trade_win_rate = float(wins / len(trades_pnl))
        else:
            trade_win_rate = hit_rate

        total_turnover = float(turnover.sum())

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(float(net_equity.iloc[-1]), 2),
            "gross_return_pct": round(float(gross_return_pct * 100), 2),
            "net_return_pct": round(float(net_return_pct * 100), 2),
            "total_return_pct": round(float(net_return_pct * 100), 2),
            "max_drawdown_pct": round(float(max_drawdown * 100), 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "turnover": round(total_turnover, 4),
            "exposure": round(exposure, 4),
            "hit_rate_pct": round(hit_rate * 100, 2),
            "win_rate_pct": round(trade_win_rate * 100, 2),
            "bar_hit_rate_pct": round(hit_rate * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "trade_count": trade_count,
            "total_trade_actions": int((turnover > 0).sum()),
            "total_bars_tested": len(df),
            "signal_delay": delay,
            "data_source": source_name,
            "fee_schedule": {
                "source": self.fee_schedule.source,
                "effective_time": self.fee_schedule.effective_time,
                "maker_fee_rate": self.fee_schedule.maker_fee_rate,
                "taker_fee_rate": self.fee_schedule.taker_fee_rate,
                "spread_bps": self.fee_schedule.spread_bps,
                "slippage_bps": self.fee_schedule.slippage_bps,
                "funding_rate_per_bar": self.fee_schedule.funding_rate_per_bar,
                "delay_bars": self.fee_schedule.delay_bars,
                "tick_size": self.fee_schedule.tick_size,
                "step_size": self.fee_schedule.step_size,
                "min_notional": self.fee_schedule.min_notional,
            },
            "costs": {
                "total_cost": round(float(total_friction_cost.sum() * self.initial_capital), 4),
                "taker_costs": round(float(taker_costs.sum() * self.initial_capital), 4),
                "maker_costs": round(float(maker_costs.sum() * self.initial_capital), 4),
                "spread_costs": round(float(spread_costs.sum() * self.initial_capital), 4),
                "slippage_costs": round(float(slippage_costs.sum() * self.initial_capital), 4),
                "funding_costs": round(float(funding_costs.sum() * self.initial_capital), 4),
            },
        }
