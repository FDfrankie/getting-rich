"""Unit & strict-causality tests for ConditionalProbRLStrategy."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_rl_strategy import (
    BeliefSnapshot,
    BeliefStateContextualRLPolicy,
    ConditionalProbRLStrategy,
)


def _synthetic_ohlcv(n: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.008, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    high = close * (1.0 + rng.uniform(0.001, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.01, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(100.0, 500.0, size=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})


class TestConditionalProbRLStrategy(unittest.TestCase):
    def test_strict_causality_no_lookahead_bias(self) -> None:
        df1 = _synthetic_ohlcv(n=320, seed=11)
        df2 = df1.copy()
        cutoff = 250
        # Drastically perturb all future bars after cutoff
        df2.loc[cutoff:, "Close"] = df2.loc[cutoff:, "Close"] * 2.5
        df2.loc[cutoff:, "High"] = df2.loc[cutoff:, "High"] * 2.5
        df2.loc[cutoff:, "Low"] = df2.loc[cutoff:, "Low"] * 2.5

        s1 = ConditionalProbRLStrategy(use_rl_policy=True)
        s2 = ConditionalProbRLStrategy(use_rl_policy=True)
        out1 = s1.generate_signals(df1)
        out2 = s2.generate_signals(df2)

        np.testing.assert_allclose(out1["signal"].iloc[:cutoff].to_numpy(), out2["signal"].iloc[:cutoff].to_numpy())
        np.testing.assert_allclose(out1["cond_prob_long"].iloc[:cutoff].to_numpy(), out2["cond_prob_long"].iloc[:cutoff].to_numpy())
        np.testing.assert_allclose(out1["belief_confidence"].iloc[:cutoff].to_numpy(), out2["belief_confidence"].iloc[:cutoff].to_numpy())

    def test_online_rl_finetune_updates_weights_and_respects_confidence_gate(self) -> None:
        pol = BeliefStateContextualRLPolicy()
        low_conf = BeliefSnapshot(
            p_long_short_h=0.65, p_short_short_h=0.35,
            p_long_long_h=0.65, p_short_long_h=0.35,
            ev_atr_long=1.5, ev_atr_short=-1.5,
            posterior_confidence=0.10, trend_z=1.0, vol_regime=0.0, current_pos=0.0,
        )
        act, _ = pol.select_action(low_conf)
        self.assertEqual(act, 0.0)

        w_before = pol.weights.copy()
        pol.online_finetune_step(low_conf.to_vector(), realized_fwd_return=0.03, prev_pos=0.0)
        self.assertEqual(pol.updates, 1)
        self.assertFalse(np.allclose(w_before, pol.weights))

    def test_strategy_outputs_required_columns_and_summary(self) -> None:
        df = _synthetic_ohlcv(n=300, seed=42)
        strat = ConditionalProbRLStrategy(use_rl_policy=True)
        out = strat.generate_signals(df)
        for col in ("signal", "ATR", "cond_prob_long", "cond_prob_short", "cond_ev_long", "belief_confidence"):
            self.assertIn(col, out.columns)
        self.assertGreater(strat.last_state_summary.get("rl_updates", 0), 0)
        self.assertTrue(np.all(np.abs(out["signal"].to_numpy()) <= 1.0))


if __name__ == "__main__":
    unittest.main()
