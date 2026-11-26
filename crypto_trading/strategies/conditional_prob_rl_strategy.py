"""P51 Multi-Timeframe (1m/3m/1h) Candlestick + Triple-Barrier Conditional Probability Strategy.

Grounded in crypto_quant_kb_v1:
  - Causal Multi-Timeframe (MTF) alignment with mandatory 1-bar shift on completed HTF bars
    (MDPI 2026, prevents +0.20 ROC-AUC lookahead inflation).
  - Real candlestick anatomy (Pin Bar / Liquidity Sweep, Engulfing, Breakout) + Binance
    Taker-Buy Order Flow Imbalance (OFI) & VWAP deviation (IREF 2026; arXiv:2602.00776;
    arXiv:2608.21888).
  - Volatility-scaled CUSUM event filtering + Triple-Barrier Conditional Probability
    P(hit TP before SL | HTF Regime, Candle Pattern, OFI) with 3-tier Hierarchical
    Exponential-Decay Bayesian Shrinkage (Financial Innovation 2025).
  - Binance USDT-M VIP 0 Maker-First cost barrier (4 bps Maker RT vs 10 bps Taker RT)
    plus adverse-selection guard against chasing extreme one-sided OFI spikes (arXiv:2502.18625).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from crypto_trading.strategies.momentum_breakout import BaseStrategy


ACTIONS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

# P52 High-Conviction Low-Turnover presets:
# (htf_mult, short_h, long_h, halflife_bars, cooldown, min_hold, tp_atr, sl_atr, cusum_atr_mult, min_move_cost_ratio, min_edge)
TIMEFRAME_PRESETS: dict[str, tuple[int, int, int, float, int, int, float, float, float, float, float]] = {
    "1m": (15, 20, 60, 720.0, 45, 20, 5.0, 2.5, 2.20, 7.0, 0.0018),
    "3m": (20, 16, 48, 480.0, 24, 12, 4.5, 2.2, 1.80, 6.0, 0.0015),
    "5m": (12, 14, 42, 360.0, 16, 8, 4.0, 2.0, 1.50, 5.0, 0.0012),
    "15m": (4, 10, 30, 280.0, 10, 6, 3.8, 1.9, 1.20, 4.5, 0.0010),
    "1h": (4, 12, 24, 240.0, 8, 6, 3.6, 1.8, 1.00, 4.0, 0.0008),
}


@dataclass(frozen=True)
class BeliefSnapshot:
    p_long_short_h: float
    p_short_short_h: float
    p_long_long_h: float
    p_short_long_h: float
    ev_atr_long: float
    ev_atr_short: float
    posterior_confidence: float
    trend_z: float
    vol_regime: float
    current_pos: float
    ofi_score: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Compact, bounded 12-dim feature vector (with bias term) for low-variance RL."""
        return np.array(
            [
                1.0,
                float(np.clip(self.p_long_short_h - 0.5, -0.5, 0.5)),
                float(np.clip(self.p_short_short_h - 0.5, -0.5, 0.5)),
                float(np.clip(self.p_long_long_h - 0.5, -0.5, 0.5)),
                float(np.clip(self.p_short_long_h - 0.5, -0.5, 0.5)),
                float(np.clip(self.ev_atr_long, -3.0, 3.0)) / 3.0,
                float(np.clip(self.ev_atr_short, -3.0, 3.0)) / 3.0,
                float(np.clip(self.posterior_confidence, 0.0, 1.0)),
                float(np.clip(self.trend_z, -3.0, 3.0)) / 3.0,
                float(np.clip(self.vol_regime, -3.0, 3.0)) / 3.0,
                float(np.clip(self.current_pos, -1.0, 1.0)),
                float(np.clip(self.ofi_score, -1.0, 1.0)),
            ],
            dtype=np.float64,
        )


class BeliefStateContextualRLPolicy:
    """Online-finetunable linear-Q contextual RL policy over the Conditional Probability belief state."""

    def __init__(
        self,
        state_dim: int = 12,
        actions: tuple[float, ...] = ACTIONS,
        learning_rate: float = 0.025,
        l2_reg: float = 0.015,
        cost_one_way: float = 0.0002,
        turnover_penalty: float = 0.0002,
        min_edge_threshold: float = 0.0003,
    ) -> None:
        self.state_dim = int(state_dim)
        self.actions = tuple(float(a) for a in actions)
        self.lr = float(learning_rate)
        self.l2_reg = float(l2_reg)
        self.cost_one_way = float(cost_one_way)
        self.turnover_penalty = float(turnover_penalty)
        self.min_edge_threshold = float(min_edge_threshold)
        self.weights = np.zeros((len(self.actions), self.state_dim), dtype=np.float64)
        for idx, act in enumerate(self.actions):
            if act > 0:
                self.weights[idx, 1] = 0.004 * act
                self.weights[idx, 3] = 0.006 * act
                self.weights[idx, 5] = 0.008 * act
                self.weights[idx, 8] = 0.002 * act
                if self.state_dim > 11:
                    self.weights[idx, 11] = 0.0015 * act
            elif act < 0:
                self.weights[idx, 2] = 0.004 * abs(act)
                self.weights[idx, 4] = 0.006 * abs(act)
                self.weights[idx, 6] = 0.008 * abs(act)
                self.weights[idx, 8] = -0.002 * abs(act)
                if self.state_dim > 11:
                    self.weights[idx, 11] = -0.0015 * abs(act)
            else:
                self.weights[idx, 0] = self.min_edge_threshold
        self.updates = 0

    def q_values(self, state_vec: np.ndarray, current_pos: float) -> np.ndarray:
        raw_q = self.weights @ state_vec
        switch_cost = np.array(
            [abs(a - current_pos) * (self.cost_one_way + self.turnover_penalty) for a in self.actions],
            dtype=np.float64,
        )
        return raw_q - switch_cost

    def select_action(self, belief: BeliefSnapshot) -> tuple[float, float]:
        vec = belief.to_vector()
        q = self.q_values(vec, belief.current_pos)
        best_idx = int(np.argmax(q))
        flat_idx = self.actions.index(0.0)
        best_action = self.actions[best_idx]
        if belief.posterior_confidence < 0.20 and best_action != 0.0:
            return 0.0, float(q[flat_idx])
        if belief.current_pos == 0.0 and best_action != 0.0:
            if (float(q[best_idx]) - float(q[flat_idx])) < self.min_edge_threshold:
                return 0.0, float(q[flat_idx])
        return best_action, float(q[best_idx])

    def online_finetune_step(
        self,
        state_vec: np.ndarray,
        realized_fwd_return: float,
        prev_pos: float,
    ) -> None:
        """Strictly causal counterfactual bandit/Q update once forward return is realized."""
        if not math.isfinite(realized_fwd_return):
            return
        clipped_ret = float(np.clip(realized_fwd_return, -0.08, 0.08))
        eff_lr = self.lr / (1.0 + 0.001 * self.updates)
        for idx, act in enumerate(self.actions):
            net_reward = act * clipped_ret - abs(act - prev_pos) * (self.cost_one_way + self.turnover_penalty)
            pred = float(np.dot(self.weights[idx], state_vec))
            err = float(np.clip(net_reward - pred, -0.05, 0.05))
            self.weights[idx] = (1.0 - eff_lr * self.l2_reg) * self.weights[idx] + eff_lr * err * state_vec
        self.updates += 1


class ConditionalProbRLStrategy(BaseStrategy):
    """P51 MTF 1m/3m/1h Candlestick + Triple-Barrier Conditional Probability + Contextual RL Strategy."""

    def __init__(
        self,
        timeframe: str = "3m",
        short_horizon: int | None = None,
        long_horizon: int | None = None,
        halflife_bars: float | None = None,
        fee_roundtrip: float = 0.0004,  # Binance USDT-M VIP 0 Maker RT = 4 bps (Taker RT = 10 bps)
        maker_mode: bool = True,
        prior_strength: float = 18.0,
        min_history_bars: int = 160,
        rebalance_cooldown: int | None = None,
        use_rl_policy: bool = True,
        rl_learning_rate: float = 0.025,
        min_edge_threshold: float | None = None,
        min_holding_bars: int | None = None,
        tp_atr_mult: float | None = None,
        sl_atr_mult: float | None = None,
        cusum_atr_mult: float | None = None,
        min_expected_move_cost_ratio: float | None = None,
        use_cusum_filter: bool = True,
        require_htf_alignment: bool = True,
    ) -> None:
        tf = str(timeframe or "3m").strip().lower()
        preset = TIMEFRAME_PRESETS.get(tf, TIMEFRAME_PRESETS["3m"])
        self.timeframe = tf
        self.preferred_timeframe = tf
        self.htf_mult = int(preset[0])
        self.short_horizon = max(2, int(short_horizon if short_horizon is not None else preset[1]))
        self.long_horizon = max(self.short_horizon + 1, int(long_horizon if long_horizon is not None else preset[2]))
        self.halflife_bars = max(24.0, float(halflife_bars if halflife_bars is not None else preset[3]))
        self.decay_factor = math.pow(0.5, 1.0 / self.halflife_bars)
        self.maker_mode = bool(maker_mode)
        self.fee_roundtrip = float(fee_roundtrip if fee_roundtrip is not None else (0.0004 if maker_mode else 0.0010))
        self.prior_strength = float(prior_strength)
        self.min_history_bars = max(80, int(min_history_bars))
        self.rebalance_cooldown = max(1, int(rebalance_cooldown if rebalance_cooldown is not None else preset[4]))
        self.min_holding_bars = max(1, int(min_holding_bars if min_holding_bars is not None else preset[5]))
        self.use_rl_policy = bool(use_rl_policy)
        self.rl_learning_rate = float(rl_learning_rate)
        self.tp_atr_mult = float(tp_atr_mult if tp_atr_mult is not None else preset[6])
        self.sl_atr_mult = float(sl_atr_mult if sl_atr_mult is not None else preset[7])
        self.cusum_atr_mult = float(cusum_atr_mult if cusum_atr_mult is not None else preset[8])
        self.min_expected_move_cost_ratio = float(min_expected_move_cost_ratio if min_expected_move_cost_ratio is not None else preset[9])
        self.min_edge_threshold = float(min_edge_threshold if min_edge_threshold is not None else preset[10])
        self.use_cusum_filter = bool(use_cusum_filter)
        self.require_htf_alignment = bool(require_htf_alignment)
        self.last_state_summary: dict[str, Any] = {}

    @staticmethod
    def _compute_causal_htf_trend(close: pd.Series, atr: pd.Series, htf_mult: int) -> pd.Series:
        """Causal higher-timeframe trend proxy with strict 1-completed-HTF-bar shift (zero lookahead)."""
        n = len(close)
        if n == 0 or htf_mult <= 1:
            ema_fast = close.ewm(span=24, adjust=False, min_periods=24).mean()
            ema_slow = close.ewm(span=72, adjust=False, min_periods=72).mean()
            return ((ema_fast - ema_slow) / (atr.replace(0, np.nan) * 4.0)).fillna(0.0)

        # Group indices into non-overlapping HTF blocks: block_id = idx // htf_mult
        # For any bar i in block b, only completed blocks <= b - 1 are visible!
        vals = close.to_numpy(dtype=np.float64)
        n_blocks = (n + htf_mult - 1) // htf_mult
        htf_close = np.full(n_blocks, np.nan, dtype=np.float64)
        for b in range(n_blocks):
            end_i = min((b + 1) * htf_mult - 1, n - 1)
            # Only treat block as completed when full htf_mult bars exist
            if (b + 1) * htf_mult <= n:
                htf_close[b] = vals[end_i]

        htf_s = pd.Series(htf_close)
        htf_ema_fast = htf_s.ewm(span=8, adjust=False, min_periods=4).mean()
        htf_ema_slow = htf_s.ewm(span=24, adjust=False, min_periods=8).mean()
        # Strict 1-period shift on HTF series (MDPI 2026 causal MTF rule)
        htf_diff_shifted = (htf_ema_fast - htf_ema_slow).shift(1).to_numpy(dtype=np.float64)

        expanded = np.zeros(n, dtype=np.float64)
        for i in range(n):
            b = i // htf_mult
            v = htf_diff_shifted[b]
            expanded[i] = 0.0 if np.isnan(v) else v

        denom = (atr.replace(0, np.nan) * max(2.0, math.sqrt(htf_mult))).to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            htf_z = np.where(np.isfinite(denom) & (denom > 0), expanded / denom, 0.0)
        return pd.Series(np.nan_to_num(htf_z, nan=0.0, posinf=3.0, neginf=-3.0), index=close.index)

    def compute_regime_states(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        open_p = pd.to_numeric(out["Open"], errors="coerce") if "Open" in out.columns else pd.to_numeric(out["Close"], errors="coerce").shift(1).bfill()
        high = pd.to_numeric(out["High"], errors="coerce")
        low = pd.to_numeric(out["Low"], errors="coerce")
        close = pd.to_numeric(out["Close"], errors="coerce")
        vol = pd.to_numeric(out["Volume"], errors="coerce").fillna(0.0)

        prev_c = close.shift(1)
        prev_o = open_p.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_c).abs(), (low - prev_c).abs()))
        atr = tr.rolling(20, min_periods=10).mean()
        atr_pct = (atr / close.replace(0, np.nan)).fillna(0.002)

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        # Causal HTF trend + local micro trend
        htf_trend_z = self._compute_causal_htf_trend(close, atr, self.htf_mult)
        ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow = close.ewm(span=36, adjust=False, min_periods=36).mean()
        micro_trend_z = ((ema_fast - ema_slow) / (atr.replace(0, np.nan) * 3.0)).fillna(0.0)
        trend_z = (0.65 * htf_trend_z + 0.35 * micro_trend_z).clip(-4.0, 4.0)

        # Volatility regime (price-level agnostic)
        vol_med = atr_pct.rolling(120, min_periods=20).median().replace(0, np.nan)
        vol_regime = ((atr_pct / vol_med) - 1.0).fillna(0.0).clip(-2.0, 4.0)

        # Candlestick anatomy (pure causal)
        bar_range = (high - low).replace(0, np.nan)
        body = (close - open_p).abs()
        body_ratio = (body / bar_range).fillna(0.0).clip(0.0, 1.0)
        upper_wick = (high - np.maximum(open_p, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_p, close) - low).clip(lower=0.0)
        upper_wick_ratio = (upper_wick / bar_range).fillna(0.0).clip(0.0, 1.0)
        lower_wick_ratio = (lower_wick / bar_range).fillna(0.0).clip(0.0, 1.0)

        # Volume Z-score & Rolling VWAP deviation
        vol_mean = vol.rolling(30, min_periods=10).mean()
        vol_std = vol.rolling(30, min_periods=10).std().replace(0, np.nan)
        vol_z = ((vol - vol_mean) / vol_std).fillna(0.0).clip(-3.0, 5.0)
        typical_price = (high + low + close) / 3.0
        pv_roll = (typical_price * vol).rolling(40, min_periods=10).sum()
        v_roll = vol.rolling(40, min_periods=10).sum().replace(0, np.nan)
        vwap = (pv_roll / v_roll).fillna(close)
        vwap_dev_atr = ((close - vwap) / atr.replace(0, np.nan)).fillna(0.0).clip(-4.0, 4.0)

        # Order Flow Imbalance (OFI): real Binance TakerBuyBaseVol when available, else close-location proxy
        taker_col = None
        for cand in ("TakerBuyBaseVol", "taker_buy_base_vol", "taker_buy_base_asset_volume"):
            if cand in out.columns:
                taker_col = cand
                break
        if taker_col is not None:
            t_buy = pd.to_numeric(out[taker_col], errors="coerce").fillna(0.0)
            ofi_raw = (2.0 * (t_buy / vol.replace(0, np.nan)) - 1.0).fillna(0.0).clip(-1.0, 1.0)
        else:
            clv = (2.0 * (close - low) / bar_range - 1.0).fillna(0.0).clip(-1.0, 1.0)
            ofi_raw = clv * np.tanh(vol_z.clip(lower=0.0) + 0.5)
        ofi_ema = ofi_raw.ewm(span=3, adjust=False).mean().clip(-1.0, 1.0)

        # Discrete candlestick patterns (Pin Bar Liquidity Sweep, Engulfing, Momentum Breakout)
        prev_bear = prev_c < prev_o
        prev_bull = prev_c > prev_o
        curr_bull = close > open_p
        curr_bear = close < open_p
        bull_engulf = prev_bear & curr_bull & (close >= prev_o) & (open_p <= prev_c) & (body_ratio >= 0.55)
        bear_engulf = prev_bull & curr_bear & (close <= prev_o) & (open_p >= prev_c) & (body_ratio >= 0.55)
        bull_sweep = (lower_wick_ratio >= 0.52) & (body_ratio <= 0.35) & (close >= open_p - 0.15 * (high - low))
        bear_sweep = (upper_wick_ratio >= 0.52) & (body_ratio <= 0.35) & (close <= open_p + 0.15 * (high - low))
        breakout_up = curr_bull & (body_ratio >= 0.65) & (vol_z >= 1.0) & (close > high.shift(1))
        breakout_dn = curr_bear & (body_ratio >= 0.65) & (vol_z >= 1.0) & (close < low.shift(1))

        candle_pattern = np.where(
            bull_sweep,
            "BULL_SWEEP",
            np.where(
                bear_sweep,
                "BEAR_SWEEP",
                np.where(
                    bull_engulf,
                    "BULL_ENGULF",
                    np.where(
                        bear_engulf,
                        "BEAR_ENGULF",
                        np.where(breakout_up, "BRK_UP", np.where(breakout_dn, "BRK_DN", "NEUTRAL")),
                    ),
                ),
            ),
        )

        # Causal volatility-scaled CUSUM event trigger
        log_ret = np.log((close / prev_c.replace(0, np.nan)).clip(lower=1e-9)).fillna(0.0).to_numpy(dtype=np.float64)
        atr_pct_arr = atr_pct.to_numpy(dtype=np.float64)
        cusum_event = np.zeros(len(out), dtype=bool)
        s_pos = 0.0
        s_neg = 0.0
        for idx_i in range(1, len(out)):
            thresh = max(0.0005, float(atr_pct_arr[idx_i]) * self.cusum_atr_mult)
            r_i = float(log_ret[idx_i])
            s_pos = max(0.0, s_pos + r_i)
            s_neg = min(0.0, s_neg + r_i)
            if s_pos >= thresh:
                cusum_event[idx_i] = True
                s_pos = 0.0
            elif abs(s_neg) >= thresh:
                cusum_event[idx_i] = True
                s_neg = 0.0

        out["ATR"] = atr
        out["ATR_PCT"] = atr_pct
        out["RSI_14"] = rsi
        out["HTF_TREND_Z"] = htf_trend_z
        out["TREND_Z"] = trend_z
        out["VOL_REGIME"] = vol_regime
        out["BODY_RATIO"] = body_ratio
        out["UPPER_WICK_RATIO"] = upper_wick_ratio
        out["LOWER_WICK_RATIO"] = lower_wick_ratio
        out["VOL_Z"] = vol_z
        out["VWAP_DEV_ATR"] = vwap_dev_atr
        out["OFI"] = ofi_ema
        out["CANDLE_PATTERN"] = candle_pattern
        out["CUSUM_EVENT"] = cusum_event

        htf_lbl = np.where(htf_trend_z > 0.30, "UP", np.where(htf_trend_z < -0.30, "DN", "CHOP"))
        ofi_lbl = np.where(ofi_ema > 0.20, "BUYFLOW", np.where(ofi_ema < -0.20, "SELLFLOW", "BAL"))
        vol_lbl = np.where(vol_regime > 0.25, "HIVOL", "NORM")

        # Hierarchical keys: Coarse = HTF|CandlePattern, Fine = HTF|CandlePattern|OFI|Vol
        out["state_coarse"] = [f"{h}|{p}" for h, p in zip(htf_lbl, candle_pattern)]
        out["state_fine"] = [f"{h}|{p}|{o}|{v}" for h, p, o, v in zip(htf_lbl, candle_pattern, ofi_lbl, vol_lbl)]
        return out

    @staticmethod
    def _triple_barrier_outcome(
        entry_idx: int,
        eval_idx: int,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr_pct: np.ndarray,
        tp_mult: float,
        sl_mult: float,
        fee_rt: float,
    ) -> tuple[float, float, float] | None:
        """Strictly causal triple-barrier label resolved at bar eval_idx for entry at entry_idx.

        Returns (win_long, win_short, net_directional_ret) using ONLY bars [entry_idx+1 .. eval_idx].
        """
        if entry_idx < 30 or eval_idx <= entry_idx:
            return None
        c0 = float(close[entry_idx])
        ap = float(atr_pct[entry_idx])
        if not math.isfinite(c0) or c0 <= 0 or not math.isfinite(ap) or ap <= 0:
            return None

        tp_dist = max(fee_rt * 2.0, ap * tp_mult)
        sl_dist = max(fee_rt * 1.2, ap * sl_mult)

        long_tp = c0 * (1.0 + tp_dist)
        long_sl = c0 * (1.0 - sl_dist)
        short_tp = c0 * (1.0 - tp_dist)
        short_sl = c0 * (1.0 + sl_dist)

        win_long = None
        win_short = None
        for k in range(entry_idx + 1, eval_idx + 1):
            hi = float(high[k])
            lo = float(low[k])
            if win_long is None:
                if lo <= long_sl:
                    win_long = 0.0
                elif hi >= long_tp:
                    win_long = 1.0
            if win_short is None:
                if hi >= short_sl:
                    win_short = 0.0
                elif lo <= short_tp:
                    win_short = 1.0
            if win_long is not None and win_short is not None:
                break

        fwd = (float(close[eval_idx]) - c0) / c0
        if win_long is None:
            win_long = 1.0 if fwd > fee_rt else 0.0
        if win_short is None:
            win_short = 1.0 if fwd < -fee_rt else 0.0
        return float(win_long), float(win_short), float(fwd)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.compute_regime_states(df)
        close = pd.to_numeric(out["Close"], errors="coerce").to_numpy(dtype=np.float64)
        high = pd.to_numeric(out["High"], errors="coerce").to_numpy(dtype=np.float64)
        low = pd.to_numeric(out["Low"], errors="coerce").to_numpy(dtype=np.float64)
        atr_pct = out["ATR_PCT"].to_numpy(dtype=np.float64)
        trend_z = out["TREND_Z"].to_numpy(dtype=np.float64)
        vol_reg = out["VOL_REGIME"].to_numpy(dtype=np.float64)
        ofi_arr = out["OFI"].to_numpy(dtype=np.float64)
        patterns = out["CANDLE_PATTERN"].tolist()
        cusum_arr = out["CUSUM_EVENT"].to_numpy(dtype=bool)
        fine_states = out["state_fine"].tolist()
        coarse_states = out["state_coarse"].tolist()

        n = len(out)
        hs = self.short_horizon
        hl = self.long_horizon
        gamma = self.decay_factor

        def _empty_table() -> dict[str, list[float]]:
            return defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        fine_s, coarse_s = _empty_table(), _empty_table()
        fine_l, coarse_l = _empty_table(), _empty_table()
        glob_s = [0.0, 0.0, 0.0, 0.0]
        glob_l = [0.0, 0.0, 0.0, 0.0]

        policy = BeliefStateContextualRLPolicy(
            state_dim=12,
            learning_rate=self.rl_learning_rate,
            cost_one_way=self.fee_roundtrip / 2.0,
            min_edge_threshold=self.min_edge_threshold,
        )

        saved_state_vecs: list[np.ndarray | None] = [None] * n
        saved_prev_pos: list[float] = [0.0] * n

        signals = np.zeros(n, dtype=float)
        prob_long_arr = np.full(n, 0.5, dtype=float)
        prob_short_arr = np.full(n, 0.5, dtype=float)
        ev_long_arr = np.zeros(n, dtype=float)
        ev_short_arr = np.zeros(n, dtype=float)
        conf_arr = np.zeros(n, dtype=float)
        cost_gate_pass = np.zeros(n, dtype=bool)

        htf_trend_z_arr = out["HTF_TREND_Z"].to_numpy(dtype=np.float64)
        pos = 0.0
        held_bars = 0
        bars_since_trade = self.rebalance_cooldown

        def _decay_tables(fine_tbl: dict[str, list[float]], coarse_tbl: dict[str, list[float]], g: list[float]) -> None:
            for tbl in (fine_tbl, coarse_tbl):
                for vals in tbl.values():
                    vals[0] *= gamma
                    vals[1] *= gamma
                    vals[2] *= gamma
                    vals[3] *= gamma
            g[0] *= gamma
            g[1] *= gamma
            g[2] *= gamma
            g[3] *= gamma

        def _update_horizon(j_idx: int, i_idx: int, fine_tbl: dict[str, list[float]], coarse_tbl: dict[str, list[float]], g: list[float]) -> float | None:
            res = self._triple_barrier_outcome(
                j_idx, i_idx, close, high, low, atr_pct, self.tp_atr_mult, self.sl_atr_mult, self.fee_roundtrip
            )
            if res is None:
                return None
            wl, ws, fwd = res
            for tbl, key in ((fine_tbl, fine_states[j_idx]), (coarse_tbl, coarse_states[j_idx])):
                tbl[key][0] += 1.0
                tbl[key][1] += wl
                tbl[key][2] += ws
                tbl[key][3] += fwd
            g[0] += 1.0
            g[1] += wl
            g[2] += ws
            g[3] += fwd
            return fwd

        def _query_posterior(fs_k: str, cs_k: str, fine_tbl: dict[str, list[float]], coarse_tbl: dict[str, list[float]], g: list[float]) -> tuple[float, float, float, float]:
            p0_l = (g[1] / g[0]) if g[0] > 1e-6 else 0.5
            p0_s = (g[2] / g[0]) if g[0] > 1e-6 else 0.5
            ev0 = (g[3] / g[0]) if g[0] > 1e-6 else 0.0

            cn, cwl, cws, cret = coarse_tbl[cs_k]
            ac = self.prior_strength
            pc_l = (cwl + ac * p0_l) / (cn + ac)
            pc_s = (cws + ac * p0_s) / (cn + ac)
            evc = (cret + ac * ev0) / (cn + ac)

            fn, fwl, fws, fret = fine_tbl[fs_k]
            af = self.prior_strength * 0.6
            pl = (fwl + af * pc_l) / (fn + af)
            ps = (fws + af * pc_s) / (fn + af)
            ev = (fret + af * evc) / (fn + af)
            eff_samples = fn + 0.5 * cn
            return pl, ps, ev, eff_samples

        for i in range(n):
            _decay_tables(fine_s, coarse_s, glob_s)
            _decay_tables(fine_l, coarse_l, glob_l)

            _update_horizon(i - hs, i, fine_s, coarse_s, glob_s)
            fwd_l = _update_horizon(i - hl, i, fine_l, coarse_l, glob_l)

            if self.use_rl_policy and fwd_l is not None and (i - hl) >= self.min_history_bars:
                past_vec = saved_state_vecs[i - hl]
                if past_vec is not None:
                    policy.online_finetune_step(past_vec, fwd_l, saved_prev_pos[i - hl])

            if i < self.min_history_bars or np.isnan(close[i]):
                signals[i] = 0.0
                continue

            pl_s, ps_s, ev_s, fn_s = _query_posterior(fine_states[i], coarse_states[i], fine_s, coarse_s, glob_s)
            pl_l, ps_l, ev_l, fn_l = _query_posterior(fine_states[i], coarse_states[i], fine_l, coarse_l, glob_l)

            # Triple-barrier expected value net of USDT-M round-trip fees
            tp_move = max(self.fee_roundtrip * 2.0, float(atr_pct[i]) * self.tp_atr_mult)
            sl_move = max(self.fee_roundtrip * 1.2, float(atr_pct[i]) * self.sl_atr_mult)
            p_long = 0.45 * pl_s + 0.55 * pl_l
            p_short = 0.45 * ps_s + 0.55 * ps_l

            ev_tb_long = p_long * tp_move - (1.0 - p_long) * sl_move - self.fee_roundtrip
            ev_tb_short = p_short * tp_move - (1.0 - p_short) * sl_move - self.fee_roundtrip
            ev_emp_long = 0.4 * ev_s + 0.6 * ev_l - self.fee_roundtrip
            ev_emp_short = -(0.4 * ev_s + 0.6 * ev_l) - self.fee_roundtrip

            ev_net_long = 0.6 * ev_tb_long + 0.4 * ev_emp_long
            ev_net_short = 0.6 * ev_tb_short + 0.4 * ev_emp_short

            denom = max(float(atr_pct[i]) * math.sqrt(max(1.0, hl / 12.0)), 0.002)
            conf = float(min(1.0, (fn_s + fn_l) / (2.0 * self.prior_strength)))

            # Minimum expected move vs fee wall (prevents scalping dead 1m/3m chop)
            expected_move = float(atr_pct[i]) * self.tp_atr_mult
            clears_cost_wall = expected_move >= (self.fee_roundtrip * self.min_expected_move_cost_ratio)
            cost_gate_pass[i] = clears_cost_wall

            belief = BeliefSnapshot(
                p_long_short_h=pl_s,
                p_short_short_h=ps_s,
                p_long_long_h=pl_l,
                p_short_long_h=ps_l,
                ev_atr_long=ev_net_long / denom,
                ev_atr_short=ev_net_short / denom,
                posterior_confidence=conf,
                trend_z=float(trend_z[i]),
                vol_regime=float(vol_reg[i]),
                current_pos=pos,
                ofi_score=float(ofi_arr[i]),
            )
            saved_state_vecs[i] = belief.to_vector()
            saved_prev_pos[i] = pos

            prob_long_arr[i] = p_long
            prob_short_arr[i] = p_short
            ev_long_arr[i] = ev_net_long
            ev_short_arr[i] = ev_net_short
            conf_arr[i] = conf

            if pos != 0.0:
                held_bars += 1
            if (pos == 0.0 and bars_since_trade < self.rebalance_cooldown) or (pos != 0.0 and held_bars < self.min_holding_bars):
                signals[i] = pos
                bars_since_trade += 1
                continue

            # For new entries from FLAT, require CUSUM or explicit candlestick pattern event + cost wall
            is_event_bar = (not self.use_cusum_filter) or bool(cusum_arr[i]) or (patterns[i] != "NEUTRAL")
            if pos == 0.0 and (not clears_cost_wall or not is_event_bar):
                signals[i] = 0.0
                bars_since_trade += 1
                continue

            if self.use_rl_policy:
                target_pos, _ = policy.select_action(belief)
            else:
                if p_long >= 0.55 and ev_net_long >= self.min_edge_threshold and conf >= 0.25:
                    target_pos = 1.0 if conf >= 0.55 else 0.5
                elif p_short >= 0.55 and ev_net_short >= self.min_edge_threshold and conf >= 0.25:
                    target_pos = -1.0 if conf >= 0.55 else -0.5
                elif pos > 0 and p_long < 0.47:
                    target_pos = 0.0
                elif pos < 0 and p_short < 0.47:
                    target_pos = 0.0
                else:
                    target_pos = pos

            # Maker Adverse-Selection Guard (arXiv:2502.18625):
            # Do not enter passive LONG into extreme sell cascade (OFI < -0.75) or SHORT into extreme buy cascade
            if self.maker_mode and pos == 0.0:
                if target_pos > 0 and ofi_arr[i] < -0.75:
                    target_pos = 0.0
                elif target_pos < 0 and ofi_arr[i] > 0.75:
                    target_pos = 0.0

            # P52 HTF Directional Alignment Gate: never catch falling knives against strong 15m/1h trend
            if self.require_htf_alignment and pos == 0.0:
                if target_pos > 0 and htf_trend_z_arr[i] < 0.25:
                    target_pos = 0.0
                elif target_pos < 0 and htf_trend_z_arr[i] > -0.25:
                    target_pos = 0.0

            # Enforce high-conviction positive net expectancy for brand-new entries
            if pos == 0.0 and target_pos > 0 and (ev_net_long < self.min_edge_threshold or p_long < 0.54):
                target_pos = 0.0
            elif pos == 0.0 and target_pos < 0 and (ev_net_short < self.min_edge_threshold or p_short < 0.54):
                target_pos = 0.0

            # Volatility-targeted scaling
            if target_pos != 0.0 and atr_pct[i] > 0.012:
                vol_scale = min(1.0, 0.010 / float(atr_pct[i]))
                target_pos = round(float(target_pos) * vol_scale, 2)

            if abs(target_pos - pos) >= 0.25 or (target_pos == 0.0 and pos != 0.0):
                if (pos == 0.0 and target_pos != 0.0) or (pos * target_pos < 0):
                    held_bars = 0
                elif target_pos == 0.0:
                    held_bars = 0
                pos = target_pos
                bars_since_trade = 1
            else:
                bars_since_trade += 1
            signals[i] = pos

        out["signal"] = signals
        out["cond_prob_long"] = prob_long_arr
        out["cond_prob_short"] = prob_short_arr
        out["cond_ev_long"] = ev_long_arr
        out["cond_ev_short"] = ev_short_arr
        out["belief_confidence"] = conf_arr
        out["cost_gate_pass"] = cost_gate_pass

        if n > 0:
            self.last_state_summary = {
                "timeframe": self.timeframe,
                "maker_mode": self.maker_mode,
                "fee_roundtrip_bps": round(self.fee_roundtrip * 10000.0, 2),
                "state": fine_states[-1],
                "candle_pattern": patterns[-1],
                "ofi": round(float(ofi_arr[-1]), 3),
                "prob_long": round(float(prob_long_arr[-1]), 4),
                "prob_short": round(float(prob_short_arr[-1]), 4),
                "ev_long_bps": round(float(ev_long_arr[-1] * 10000.0), 2),
                "ev_short_bps": round(float(ev_short_arr[-1] * 10000.0), 2),
                "belief_confidence": round(float(conf_arr[-1]), 3),
                "cost_gate_pass": bool(cost_gate_pass[-1]),
                "rl_updates": policy.updates,
                "signal": float(signals[-1]),
            }
        return out
