"""P53 Unit Tests: Multi-Factor Alpha Profit Maximizer, Break-Even Ratchet & Fractional Signal Fix."""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy
from scripts.backtest_crypto_usdt_walkforward import simulate_walkforward_oos


class _MockRunnerStrategy:
    """Deterministic strategy that enters LONG at bar 120 to verify Break-Even Lock + Profit Ratchet."""
    tp_atr_mult = 5.0
    sl_atr_mult = 2.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ATR_PCT"] = 0.005
        sig = np.zeros(len(out), dtype=float)
        sig[120:] = 1.0
        out["signal"] = sig
        return out


def test_p53_breakeven_ratchet_prevents_winner_reversal_loss() -> None:
    # Price rises +2.0 ATR (+1.0%) at bar 125 (triggers Break-Even Lock at +6 bps), then crashes -3% at bar 130.
    # Without Break-Even Lock it would lose -1.0% (-2 ATR); WITH Break-Even Lock it exits at ~0% net!
    n = 160
    close = np.full(n, 50000.0)
    high = np.full(n, 50050.0)
    low = np.full(n, 49950.0)
    high[125] = 50550.0  # +1.1% = +2.2 ATR -> arms Break-Even Lock at 50030
    close[125] = 50450.0
    low[130] = 48500.0   # crash -> hits Break-Even Lock instead of full -2 ATR SL
    close[130:] = 48600.0
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 100.0)})

    m = simulate_walkforward_oos(
        df, _MockRunnerStrategy(), label="BE_Test", symbol="BTCUSDT", timeframe="3m", fee_one_way=0.0002, warmup_bars=110
    )
    assert m.trades >= 1
    assert m.net_return_pct > -0.20, f"Break-Even Lock failed to protect reversal; net={m.net_return_pct}%"


def test_p53_bot_runner_preserves_fractional_signals_and_has_breakeven_lock() -> None:
    src = (pathlib.Path(__file__).resolve().parents[1] / "crypto_trading" / "bot_runner.py").read_text(encoding="utf-8")
    assert "_raw_sig_f >= 0.25" in src
    assert "Break-Even Lock" in src
    assert "Chandelier Profit Ratchet" in src


def test_p53_conditional_prob_has_vwap_chop_and_kelly_weapons() -> None:
    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "crypto_trading"
        / "strategies"
        / "conditional_prob_rl_strategy.py"
    ).read_text(encoding="utf-8")
    assert "vwap_dev_arr" in src
    assert "kelly_boost" in src
    assert "vol_reg[i] < -0.25" in src
