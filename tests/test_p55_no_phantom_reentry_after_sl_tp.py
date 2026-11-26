"""P55 Unit Test: Verify Zero Phantom Immediate Re-Entry After Intrabar SL/TP Exit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest_crypto_usdt_walkforward import simulate_walkforward_oos


class _PersistentLongStrategy:
    """Stays signal=+1.0 continuously after bar 120 to test whether simulator phantom-reenters after SL/TP."""
    partial_tp_atr_mult = 5.0  # disable partial TP for clean single-stop test
    tp_atr_mult = 6.0
    sl_atr_mult = 2.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ATR_PCT"] = 0.005  # 50 bps ATR -> SL at -1.0%
        sig = np.zeros(len(out), dtype=float)
        sig[120:] = 1.0
        out["signal"] = sig
        return out


def test_p55_no_phantom_reentry_on_stale_signal_after_sl() -> None:
    n = 180
    close = np.full(n, 50000.0)
    high = np.full(n, 50020.0)
    low = np.full(n, 49980.0)
    # Bar 125 drops -2% -> triggers SL once. Bars 126..179 keep dropping -2% every 5 bars.
    # Previously, simulator phantom-reentered on bar 125 and got stopped out 10+ times!
    for k in range(125, n):
        close[k] = 50000.0 * (0.995 ** (k - 124))
        high[k] = close[k] * 1.001
        low[k] = close[k] * 0.990

    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": np.full(n, 100.0)})
    m = simulate_walkforward_oos(
        df, _PersistentLongStrategy(), label="NoPhantomReentry", symbol="BTCUSDT", timeframe="3m", fee_one_way=0.0002, warmup_bars=110
    )
    # MUST take ONLY 1 trade (stopped out at bar 125, then locked out because signal never reset to 0!)
    assert m.trades == 1, f"Expected exactly 1 trade (no phantom re-entry), got {m.trades}"
