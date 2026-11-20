"""Regime-Aware Exponential Conditional Probability + Belief-State Contextual RL Strategy.

Architecture:
  Layer 1 (Belief State): Strictly causal, exponential-decay Bayesian conditional
      probability engine across multi-horizon net-of-cost targets (e.g. 12h & 24h)
      calibrated against realistic perpetual futures round-trip costs (default 14 bps).
  Layer 2 (RL / Online Fine-Tune Policy): Compact, low-dimensional Belief-State
      Contextual Q-Learning / Policy Overlay trained strictly causally on realized
      cost-adjusted returns (Differential Return - Turnover Cost - Drawdown Penalty).
      Outputs confidence-scaled positions in {-1.0, -0.5, 0.0, +0.5, +1.0}, staying
      FLAT in noisy / low-edge regimes instead of overtrading.
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

    def to_vector(self) -> np.ndarray:
        """Compact, bounded 11-dim feature vector (with bias term) for low-variance RL."""
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
            ],
            dtype=np.float64,
        )


class BeliefStateContextualRLPolicy:
    """Online-finetunable linear-Q contextual RL policy over the Conditional Probability belief state.

    Uses L2-regularized ridge/SGD updates on realized cost-adjusted rewards so it adapts
    smoothly to regime shifts without the severe variance/overfitting of deep PPO on raw OHLCV.
    """

    def __init__(
        self,
        state_dim: int = 11,
        actions: tuple[float, ...] = ACTIONS,
        learning_rate: float = 0.03,
        l2_reg: float = 0.01,
        cost_one_way: float = 0.0007,
        turnover_penalty: float = 0.0004,
        min_edge_threshold: float = 0.0003,
    ) -> None:
        self.state_dim = int(state_dim)
        self.actions = tuple(float(a) for a in actions)
        self.lr = float(learning_rate)
        self.l2_reg = float(l2_reg)
        self.cost_one_way = float(cost_one_way)
        self.turnover_penalty = float(turnover_penalty)
        self.min_edge_threshold = float(min_edge_threshold)
        # Initialize Q-weights with weak economically sensible priors:
        # positive weight on (p_long - 0.5) & ev_atr_long for long actions, opposite for short,
        # and slight positive bias on FLAT (0.0) to discourage churning in noise.
        self.weights = np.zeros((len(self.actions), self.state_dim), dtype=np.float64)
        for idx, act in enumerate(self.actions):
            if act > 0:
                self.weights[idx, 1] = 0.004 * act
                self.weights[idx, 3] = 0.006 * act
                self.weights[idx, 5] = 0.008 * act
                self.weights[idx, 8] = 0.002 * act
            elif act < 0:
                self.weights[idx, 2] = 0.004 * abs(act)
                self.weights[idx, 4] = 0.006 * abs(act)
                self.weights[idx, 6] = 0.008 * abs(act)
                self.weights[idx, 8] = -0.002 * abs(act)
            else:
                self.weights[idx, 0] = self.min_edge_threshold
        self.updates = 0

    def q_values(self, state_vec: np.ndarray, current_pos: float) -> np.ndarray:
        raw_q = self.weights @ state_vec
        # Subtract explicit switching cost from actions that change current_pos
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
        # Confidence & Q-margin gate: stay FLAT unless expected net edge clears threshold
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
        # Clip extreme bar shocks so single liquidation wicks don't destabilize weights
        clipped_ret = float(np.clip(realized_fwd_return, -0.08, 0.08))
        # Decay learning rate gently with experience while keeping non-zero adaptability
        eff_lr = self.lr / (1.0 + 0.001 * self.updates)
        for idx, act in enumerate(self.actions):
            # Realized net reward for taking action `act` from `prev_pos`
            net_reward = act * clipped_ret - abs(act - prev_pos) * (self.cost_one_way + self.turnover_penalty)
            pred = float(np.dot(self.weights[idx], state_vec))
            err = float(np.clip(net_reward - pred, -0.05, 0.05))
            self.weights[idx] = (1.0 - eff_lr * self.l2_reg) * self.weights[idx] + eff_lr * err * state_vec
        self.updates += 1


class ConditionalProbRLStrategy(BaseStrategy):
    """Regime-aware Exponential-Decay Conditional Probability + Belief-State Contextual RL Strategy."""

    def __init__(
        self,
        short_horizon: int = 12,
        long_horizon: int = 24,
        halflife_bars: float = 240.0,
        fee_roundtrip: float = 0.0014,
        prior_strength: float = 20.0,
        min_history_bars: int = 160,
        rebalance_cooldown: int = 6,
        use_rl_policy: bool = True,
        rl_learning_rate: float = 0.03,
        min_edge_threshold: float = 0.0004,
    ) -> None:
        self.short_horizon = max(2, int(short_horizon))
        self.long_horizon = max(self.short_horizon + 1, int(long_horizon))
        self.halflife_bars = max(24.0, float(halflife_bars))
        self.decay_factor = math.pow(0.5, 1.0 / self.halflife_bars)
        self.fee_roundtrip = float(fee_roundtrip)
        self.prior_strength = float(prior_strength)
        self.min_history_bars = max(60, int(min_history_bars))
        self.rebalance_cooldown = max(1, int(rebalance_cooldown))
        self.use_rl_policy = bool(use_rl_policy)
        self.rl_learning_rate = float(rl_learning_rate)
        self.min_edge_threshold = float(min_edge_threshold)
        self.last_state_summary: dict[str, Any] = {}

    @staticmethod
    def compute_regime_states(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = pd.to_numeric(out["Close"], errors="coerce")
        high = pd.to_numeric(out["High"], errors="coerce")
        low = pd.to_numeric(out["Low"], errors="coerce")
        vol = pd.to_numeric(out["Volume"], errors="coerce")

        ema24 = close.ewm(span=24, adjust=False, min_periods=24).mean()
        ema72 = close.ewm(span=72, adjust=False, min_periods=72).mean()
        prev_c = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_c).abs(), (low - prev_c).abs()))
        atr = tr.rolling(24, min_periods=14).mean()
        atr_pct = (atr / close.replace(0, np.nan)).fillna(0.01)

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        trend_z = ((ema24 - ema72) / (atr.replace(0, np.nan) * 4.0)).fillna(0.0)
        vol_med = atr_pct.rolling(168, min_periods=24).median().replace(0, np.nan)
        vol_regime = ((atr_pct / vol_med) - 1.0).fillna(0.0)

        out["ATR"] = atr
        out["ATR_PCT"] = atr_pct
        out["RSI_14"] = rsi
        out["TREND_Z"] = trend_z
        out["VOL_REGIME"] = vol_regime

        trend_lbl = np.where(trend_z > 0.35, "UP", np.where(trend_z < -0.35, "DOWN", "CHOP"))
        rsi_lbl = np.where(rsi < 35, "OS", np.where(rsi > 65, "OB", "MID"))
        vol_lbl = np.where(vol_regime > 0.25, "HIVOL", "NORMVOL")

        out["state_fine"] = [f"{t}|{r}|{v}" for t, r, v in zip(trend_lbl, rsi_lbl, vol_lbl)]
        out["state_coarse"] = [f"{t}|{r}" for t, r in zip(trend_lbl, rsi_lbl)]
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.compute_regime_states(df)
        close = pd.to_numeric(out["Close"], errors="coerce").to_numpy(dtype=np.float64)
        atr_pct = out["ATR_PCT"].to_numpy(dtype=np.float64)
        trend_z = out["TREND_Z"].to_numpy(dtype=np.float64)
        vol_reg = out["VOL_REGIME"].to_numpy(dtype=np.float64)
        fine_states = out["state_fine"].tolist()
        coarse_states = out["state_coarse"].tolist()

        n = len(out)
        hs = self.short_horizon
        hl = self.long_horizon
        gamma = self.decay_factor

        # Exponentially decayed stats per horizon: [eff_n, eff_wl, eff_ws, eff_sum_ret]
        def _empty_table() -> dict[str, list[float]]:
            return defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        fine_s, coarse_s = _empty_table(), _empty_table()
        fine_l, coarse_l = _empty_table(), _empty_table()
        glob_s = [0.0, 0.0, 0.0, 0.0]
        glob_l = [0.0, 0.0, 0.0, 0.0]

        policy = BeliefStateContextualRLPolicy(
            learning_rate=self.rl_learning_rate,
            cost_one_way=self.fee_roundtrip / 2.0,
            min_edge_threshold=self.min_edge_threshold,
        )

        # Store past belief vectors & positions so RL updates strictly causally when bar i = j + hl arrives
        saved_state_vecs: list[np.ndarray | None] = [None] * n
        saved_prev_pos: list[float] = [0.0] * n

        signals = np.zeros(n, dtype=float)
        prob_long_arr = np.full(n, 0.5, dtype=float)
        prob_short_arr = np.full(n, 0.5, dtype=float)
        ev_long_arr = np.zeros(n, dtype=float)
        ev_short_arr = np.zeros(n, dtype=float)
        conf_arr = np.zeros(n, dtype=float)

        pos = 0.0
        bars_since_trade = self.rebalance_cooldown

        def _update_horizon(j_idx: int, i_idx: int, fine_tbl: dict[str, list[float]], coarse_tbl: dict[str, list[float]], g: list[float]) -> float | None:
            if j_idx < 72 or np.isnan(close[j_idx]) or np.isnan(close[i_idx]) or close[j_idx] <= 0:
                return None
            fwd = (close[i_idx] - close[j_idx]) / close[j_idx]
            wl = 1.0 if fwd > self.fee_roundtrip else 0.0
            ws = 1.0 if fwd < -self.fee_roundtrip else 0.0
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
            return pl, ps, ev, fn

        # Clean loop with separate table/global decay
        fine_s, coarse_s = _empty_table(), _empty_table()
        fine_l, coarse_l = _empty_table(), _empty_table()
        glob_s = [0.0, 0.0, 0.0, 0.0]
        glob_l = [0.0, 0.0, 0.0, 0.0]

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

        for i in range(n):
            _decay_tables(fine_s, coarse_s, glob_s)
            _decay_tables(fine_l, coarse_l, glob_l)

            _update_horizon(i - hs, i, fine_s, coarse_s, glob_s)
            fwd_l = _update_horizon(i - hl, i, fine_l, coarse_l, glob_l)

            # Strictly causal online RL fine-tune when bar (i - hl)'s long-horizon return is now realized
            if self.use_rl_policy and fwd_l is not None and (i - hl) >= self.min_history_bars:
                past_vec = saved_state_vecs[i - hl]
                if past_vec is not None:
                    policy.online_finetune_step(past_vec, fwd_l, saved_prev_pos[i - hl])

            if i < self.min_history_bars or np.isnan(close[i]):
                signals[i] = 0.0
                continue

            pl_s, ps_s, ev_s, fn_s = _query_posterior(fine_states[i], coarse_states[i], fine_s, coarse_s, glob_s)
            pl_l, ps_l, ev_l, fn_l = _query_posterior(fine_states[i], coarse_states[i], fine_l, coarse_l, glob_l)

            ev_net_long = 0.4 * ev_s + 0.6 * ev_l - self.fee_roundtrip
            ev_net_short = -(0.4 * ev_s + 0.6 * ev_l) - self.fee_roundtrip
            denom = max(float(atr_pct[i]) * math.sqrt(hl / 24.0), 0.005)
            conf = float(min(1.0, (fn_s + fn_l) / (2.0 * self.prior_strength)))

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
            )

            saved_state_vecs[i] = belief.to_vector()
            saved_prev_pos[i] = pos

            prob_long_arr[i] = 0.4 * pl_s + 0.6 * pl_l
            prob_short_arr[i] = 0.4 * ps_s + 0.6 * ps_l
            ev_long_arr[i] = ev_net_long
            ev_short_arr[i] = ev_net_short
            conf_arr[i] = conf

            if bars_since_trade < self.rebalance_cooldown:
                signals[i] = pos
                bars_since_trade += 1
                continue

            if self.use_rl_policy:
                target_pos, _ = policy.select_action(belief)
            else:
                # Layer-1-only conservative cost-aware baseline
                if prob_long_arr[i] >= 0.56 and ev_net_long >= self.min_edge_threshold and conf >= 0.25:
                    target_pos = 1.0 if conf >= 0.55 else 0.5
                elif prob_short_arr[i] >= 0.56 and ev_net_short >= self.min_edge_threshold and conf >= 0.25:
                    target_pos = -1.0 if conf >= 0.55 else -0.5
                elif pos > 0 and prob_long_arr[i] < 0.48:
                    target_pos = 0.0
                elif pos < 0 and prob_short_arr[i] < 0.48:
                    target_pos = 0.0
                else:
                    target_pos = pos

            # Volatility-targeted scaling: prevent high-vol assets (e.g. SOL) from taking oversized risk
            if target_pos != 0.0 and atr_pct[i] > 0.015:
                vol_scale = min(1.0, 0.012 / float(atr_pct[i]))
                target_pos = round(float(target_pos) * vol_scale, 2)
            if abs(target_pos - pos) >= 0.15 or (target_pos == 0.0 and pos != 0.0):
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

        if n > 0:
            self.last_state_summary = {
                "state": fine_states[-1],
                "prob_long": round(float(prob_long_arr[-1]), 4),
                "prob_short": round(float(prob_short_arr[-1]), 4),
                "ev_long_bps": round(float(ev_long_arr[-1] * 10000.0), 2),
                "ev_short_bps": round(float(ev_short_arr[-1] * 10000.0), 2),
                "belief_confidence": round(float(conf_arr[-1]), 3),
                "rl_updates": policy.updates,
                "signal": float(signals[-1]),
            }
        return out
