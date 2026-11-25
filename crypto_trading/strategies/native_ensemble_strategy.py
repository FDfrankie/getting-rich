"""Native Crypto Ensemble Strategy (1h, strictly causal, fee-hurdle gated).

Blends the two natively-crypto signals that showed positive OOS expectancy:
  * ConditionalProbRLStrategy (belief-state contextual RL)         weight 0.65
  * PerpCarryBasisStrategy    (funding / basis carry, vol-regime)   weight 0.35
and layers, all using ONLY data up to and including the current closed bar:
  * Vectorised trend / vol-shock regime filter (no counter-trend entries, flat in vol shock)
  * Fee + slippage + funding hurdle (`evaluate_cost_barrier`) on every NEW entry
  * Self-simulated drawdown cooldown (flat for `cooldown_bars` after `dd_stop` equity drawdown)

BRAIN-alpha features are NOT used for order generation (research context only).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from crypto_trading.execution_cost_filter import evaluate_cost_barrier
from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy
from crypto_trading.strategies.momentum_breakout import BaseStrategy
from crypto_trading.strategies.perp_carry_basis_strategy import PerpCarryBasisStrategy


class NativeCryptoEnsembleStrategy(BaseStrategy):
    preferred_timeframe = "1h"
    cost_gated = True  # bot_runner must not apply the generic edge hurdle a second time

    def __init__(
        self,
        w_rl: float = 0.65,
        w_carry: float = 0.35,
        enter_thr: float = 0.45,
        exit_thr: float = 0.15,
        dd_stop: float = 0.06,
        cooldown_bars: int = 24,
        side_cost: float = 0.0007,
        edge_window: int = 168,
        edge_horizon: int = 6,
    ) -> None:
        self.w_rl = float(w_rl)
        self.w_carry = float(w_carry)
        self.enter_thr = float(enter_thr)
        self.exit_thr = float(exit_thr)
        self.dd_stop = float(dd_stop)
        self.cooldown_bars = int(cooldown_bars)
        self.side_cost = float(side_cost)
        self.edge_window = int(edge_window)
        self.edge_horizon = int(edge_horizon)
        self.last_diagnostics: dict = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        close = pd.to_numeric(out["Close"], errors="coerce")
        rl = ConditionalProbRLStrategy().generate_signals(df)
        carry = PerpCarryBasisStrategy().generate_signals(df)
        rl_sig = pd.to_numeric(rl["signal"], errors="coerce").reindex(out.index).fillna(0.0).to_numpy(float)
        cr_sig = pd.to_numeric(carry["signal"], errors="coerce").reindex(out.index).fillna(0.0).to_numpy(float)
        score = self.w_rl * np.clip(rl_sig, -1, 1) + self.w_carry * np.clip(cr_sig, -1, 1)
        out["ATR"] = pd.to_numeric(carry["ATR"], errors="coerce").reindex(out.index)

        # --- causal regime (rolling windows only) ---
        ma10 = close.rolling(10, min_periods=10).mean()
        ma30 = close.rolling(30, min_periods=30).mean()
        mom20 = (close / close.shift(20) - 1.0) * 100.0
        atr_pct = (out["ATR"] / close.replace(0, np.nan)) * 100.0
        shock = (atr_pct > 4.5).fillna(False).to_numpy()
        bull = ((close > ma10) & (ma10 > ma30) & (mom20 > 0.5)).fillna(False).to_numpy()
        bear = ((close < ma10) & (ma10 < ma30) & (mom20 < -0.5)).fillna(False).to_numpy()
        allow_long = (~shock) & (~bear)
        allow_short = (~shock) & (~bull)

        # --- expected edge = conviction * trailing mean |h-bar move| (past data only) ---
        mv = close.pct_change(self.edge_horizon).abs()
        trail_move = mv.rolling(self.edge_window, min_periods=48).mean()
        edge = (pd.Series(np.abs(score), index=out.index) * trail_move).to_numpy()
        out["expected_edge"] = edge

        ret = close.pct_change().fillna(0.0).to_numpy()
        sig = np.zeros(n, dtype=float)
        pos = 0.0
        prev_turn = 0.0
        equity, peak, cool = 1.0, 1.0, 0
        for i in range(n):
            # realise PnL of the position decided at bar i-1 over bar i
            if i > 0:
                net = sig[i - 1] * ret[i] - prev_turn * self.side_cost
                equity *= 1.0 + net
                peak = max(peak, equity)
                if cool == 0 and (1.0 - equity / peak) >= self.dd_stop:
                    cool = self.cooldown_bars
            s = score[i]
            target = pos
            if cool > 0:
                target = 0.0
                cool -= 1
                if cool == 0:
                    peak = equity
            else:
                e = edge[i]
                edge_ok = (
                    math.isfinite(e)
                    and evaluate_cost_barrier(expected_return=float(e)).trade_allowed
                )
                if pos == 0.0:
                    if s >= self.enter_thr and allow_long[i] and edge_ok:
                        target = 1.0
                    elif s <= -self.enter_thr and allow_short[i] and edge_ok:
                        target = -1.0
                elif pos > 0.0:
                    if shock[i] or s < self.exit_thr:
                        target = -1.0 if (s <= -self.enter_thr and allow_short[i] and edge_ok) else 0.0
                else:
                    if shock[i] or -s < self.exit_thr:
                        target = 1.0 if (s >= self.enter_thr and allow_long[i] and edge_ok) else 0.0
            prev_turn = abs(target - pos)
            pos = target
            sig[i] = pos

        out["signal"] = sig
        out["ensemble_score"] = score
        if n:
            self.last_diagnostics = {
                "score": round(float(score[-1]), 3),
                "edge": None if not math.isfinite(edge[-1]) else round(float(edge[-1]), 5),
                "signal": int(sig[-1]),
            }
        return out
