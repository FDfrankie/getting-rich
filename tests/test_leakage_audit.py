"""Unit tests for Tri-State Future-Tampering Leakage Audit."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy
from worldquant_brain.leakage_audit import VerificationState, run_future_tampering_audit


def _make_df(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, size=n))
    return pd.DataFrame({
        "Open": close,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": rng.uniform(100, 500, size=n),
    })


class TestLeakageAudit(unittest.TestCase):
    def test_catches_lookahead_leakage_as_fail(self) -> None:
        df = _make_df(120)
        # Global z-score leaks future mean/std into past bars!
        leaky_fn = lambda d: (d["Close"] - d["Close"].mean()) / (d["Close"].std() + 1e-9)
        rep = run_future_tampering_audit(df, leaky_fn)
        self.assertEqual(rep.state, VerificationState.FAIL)
        self.assertGreater(rep.changed_rows, 0)

    def test_passes_strictly_causal_conditional_prob_rl_strategy(self) -> None:
        df = _make_df(260)
        strat = ConditionalProbRLStrategy(use_rl_policy=True)
        rep = run_future_tampering_audit(df, lambda d: strat.generate_signals(d)["signal"])
        self.assertEqual(rep.state, VerificationState.PASS)
        self.assertEqual(rep.changed_rows, 0)

    def test_insufficient_data_returns_unverified_not_pass(self) -> None:
        df = _make_df(10)
        rep = run_future_tampering_audit(df, lambda d: d["Close"])
        self.assertEqual(rep.state, VerificationState.UNVERIFIED)


if __name__ == "__main__":
    unittest.main()
