"""Native Crypto Ensemble Strategy V2 (1h, USDT Fee-Aware, Anti-Churn & Tail-Protected).

Grounded in KB v92 verified quantitative finance literature:
  * He, Manela, Ross & von Wachter (2026), "Fundamentals of Perpetual Futures" (arXiv:2212.06888)
  * Schmeling, Schrimpf & Todorov (2026), "Crypto Carry" (Management Science / BIS WP 1087)
  * Wood, Roberts & Zohren (2022), "Slow Momentum with Fast Reversion" (arXiv:2105.13727)
  * Tsang, Sit & Wong (2022), "Adaptive Robust Online Portfolio Selection" (arXiv:2206.01064)
  * Detzel, Novy-Marx & Velikov (2023), "Model Comparison with Transaction Costs" (JF)
  * Ryan (2026), "Conformal Kelly" (arXiv:2608.01494)

Upgrades over V1:
  1. USDT-M Maker/Taker fee-aware cost barrier (`evaluate_usdt_cost_barrier`), calibrated for
     Binance USDT-M VIP0 (0.02% Maker / 0.05% Taker) instead of zero-fee USDC.
  2. Minimum holding window (`min_holding_bars=4`) no-trade hysteresis to prevent USDT fee
     churning on adjacent 1h bars, while still permitting immediate exits on volatility shock.
  3. Extreme crypto-carry crash gate: blocks crowded long entries when 8h funding & basis
     are simultaneously overheated (Schmeling et al. 2026).
  4. Rolling quantile / conformal tail-dispersion filter to throttle entries during left-tail
     breakdowns.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from crypto_trading.execution_cost_filter import (
    estimate_usdt_roundtrip_fee,
    evaluate_usdt_cost_barrier,
)
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
        enter_thr: float = 0.55,
        exit_thr: float = 0.10,
        dd_stop: float = 0.06,
        cooldown_bars: int = 24,
        min_holding_bars: int = 12,
        order_execution_mode: str = "maker",
        use_bnb_discount: bool = False,
        side_cost: float | None = None,
        edge_window: int = 168,
        edge_horizon: int = 6,
        crowded_funding_block_bps: float = 3.0,
        vol_shock_atr_pct: float = 4.0,
    ) -> None:
        self.w_rl = float(w_rl)
        self.w_carry = float(w_carry)
        self.enter_thr = float(enter_thr)
        self.exit_thr = float(exit_thr)
        self.dd_stop = float(dd_stop)
        self.cooldown_bars = int(cooldown_bars)
        self.min_holding_bars = max(1, int(min_holding_bars))
        self.order_execution_mode = str(order_execution_mode or "maker").lower()
        self.use_bnb_discount = bool(use_bnb_discount)
        if side_cost is not None:
            self.side_cost = float(side_cost)
        else:
            rt_fee = estimate_usdt_roundtrip_fee(
                entry_mode=self.order_execution_mode,
                exit_mode=self.order_execution_mode,
                use_bnb_discount=self.use_bnb_discount,
            )
            slip_one_way = 0.0001 if self.order_execution_mode == "maker" else 0.00025
            self.side_cost = rt_fee / 2.0 + slip_one_way
        self.edge_window = int(edge_window)
        self.edge_horizon = int(edge_horizon)
        self.crowded_funding_block_bps = float(crowded_funding_block_bps)
        self.vol_shock_atr_pct = float(vol_shock_atr_pct)
        self.tp_atr_mult: float = 4.0
        self.sl_atr_mult: float = 2.0
        self.last_diagnostics: dict = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        close = pd.to_numeric(out["Close"], errors="coerce")
        rl = ConditionalProbRLStrategy(
            timeframe="1h",
            fee_roundtrip=self.side_cost * 2.0,
            maker_mode=(self.order_execution_mode == "maker"),
        ).generate_signals(df)
        carry = PerpCarryBasisStrategy().generate_signals(df)
        rl_sig = pd.to_numeric(rl["signal"], errors="coerce").reindex(out.index).fillna(0.0).to_numpy(float)
        cr_sig = pd.to_numeric(carry["signal"], errors="coerce").reindex(out.index).fillna(0.0).to_numpy(float)
        score = self.w_rl * np.clip(rl_sig, -1, 1) + self.w_carry * np.clip(cr_sig, -1, 1)
        out["ATR"] = pd.to_numeric(carry["ATR"], errors="coerce").reindex(out.index)

        # --- causal regime + extreme carry crash gate (rolling windows only) ---
        ma10 = close.rolling(10, min_periods=10).mean()
        ma30 = close.rolling(30, min_periods=30).mean()
        mom20 = (close / close.shift(20) - 1.0) * 100.0
        atr_pct = (out["ATR"] / close.replace(0, np.nan)) * 100.0
        shock = (atr_pct > self.vol_shock_atr_pct).fillna(False).to_numpy()
        bull = ((close > ma10) & (ma10 > ma30) & (mom20 > 0.5)).fillna(False).to_numpy()
        bear = ((close < ma10) & (ma10 < ma30) & (mom20 < -0.5)).fillna(False).to_numpy()

        # Crypto Carry crash-warning gate (Schmeling et al. 2026)
        funding_bps = (
            pd.to_numeric(carry["funding_bps_8h"], errors="coerce").reindex(out.index).fillna(1.0).to_numpy(float)
            if "funding_bps_8h" in carry.columns
            else np.ones(n, dtype=float)
        )
        basis_bps = (
            pd.to_numeric(carry["basis_bps"], errors="coerce").reindex(out.index).fillna(0.0).to_numpy(float)
            if "basis_bps" in carry.columns
            else np.zeros(n, dtype=float)
        )
        crowded_long_crash = (funding_bps > self.crowded_funding_block_bps) & (basis_bps > 25.0)
        crowded_short_squeeze = (funding_bps < -self.crowded_funding_block_bps) & (basis_bps < -25.0)

        allow_long = (~shock) & (~bear) & (~crowded_long_crash)
        allow_short = (~shock) & (~bull) & (~crowded_short_squeeze)

        # --- conformal / quantile volatility-scaled expected edge (strictly past data) ---
        ret_s = close.pct_change().fillna(0.0)
        q_hi = ret_s.rolling(self.edge_window, min_periods=48).quantile(0.875)
        q_lo = ret_s.rolling(self.edge_window, min_periods=48).quantile(0.125)
        iqr = (q_hi - q_lo).replace(0, np.nan)
        iqr_med = iqr.rolling(self.edge_window, min_periods=48).median().replace(0, np.nan)
        tail_scale = np.clip((iqr_med / iqr).fillna(1.0).to_numpy(float), 0.5, 1.25)

        mv = close.pct_change(self.edge_horizon).abs()
        trail_move = mv.rolling(self.edge_window, min_periods=48).mean()
        edge = (pd.Series(np.abs(score) * tail_scale, index=out.index) * trail_move).to_numpy()
        out["expected_edge"] = edge

        ret = ret_s.to_numpy()
        sig = np.zeros(n, dtype=float)
        pos = 0.0
        held_bars = 0
        prev_turn = 0.0
        equity, peak, cool = 1.0, 1.0, 0
        for i in range(n):
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
                held_bars = 0
                if cool == 0:
                    peak = equity
            else:
                e = edge[i]
                edge_ok = (
                    math.isfinite(e)
                    and evaluate_usdt_cost_barrier(
                        expected_return=float(e),
                        entry_mode=self.order_execution_mode,
                        exit_mode=self.order_execution_mode,
                        use_bnb_discount=self.use_bnb_discount,
                    ).trade_allowed
                )
                if pos == 0.0:
                    if s >= self.enter_thr and allow_long[i] and edge_ok:
                        target = 1.0
                        held_bars = 1
                    elif s <= -self.enter_thr and allow_short[i] and edge_ok:
                        target = -1.0
                        held_bars = 1
                elif pos > 0.0:
                    held_bars += 1
                    if shock[i]:
                        target = 0.0
                        held_bars = 0
                    elif held_bars >= self.min_holding_bars and s < self.exit_thr:
                        if s <= -self.enter_thr and allow_short[i] and edge_ok:
                            target = -1.0
                            held_bars = 1
                        else:
                            target = 0.0
                            held_bars = 0
                else:
                    held_bars += 1
                    if shock[i]:
                        target = 0.0
                        held_bars = 0
                    elif held_bars >= self.min_holding_bars and -s < self.exit_thr:
                        if s >= self.enter_thr and allow_long[i] and edge_ok:
                            target = 1.0
                            held_bars = 1
                        else:
                            target = 0.0
                            held_bars = 0
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
                "order_execution_mode": self.order_execution_mode,
                "side_cost_bps": round(self.side_cost * 10000.0, 2),
                "funding_bps_8h": round(float(funding_bps[-1]), 3),
                "crowded_long_blocked": bool(crowded_long_crash[-1]),
            }
        return out
