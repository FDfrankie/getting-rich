"""Tests for upgraded RiskManager: auto-cooldown recovery, halted stop-loss safety, and $300 isolated-leverage sizing."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from crypto_trading.risk_manager import RiskManager


class TestRiskManagerEquityAndCooldown(unittest.TestCase):
    def test_halted_position_size_still_returns_stop_loss_keys(self) -> None:
        rm = RiskManager(initial_capital=1000.0)
        rm.reanchor_capital(1000.0)
        rm.update_capital(900.0)
        self.assertTrue(rm.is_halted)
        pos = rm.calculate_position_size(65000.0, 1000.0)
        self.assertEqual(pos["size"], 0.0)
        self.assertIn("stop_loss_long", pos)
        self.assertIn("take_profit_long", pos)
        self.assertIn("stop_loss_short", pos)

    def test_circuit_breaker_auto_recovers_after_cooldown_in_defensive_mode(self) -> None:
        rm = RiskManager(initial_capital=1000.0)
        rm.reanchor_capital(1000.0)
        t0 = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(rm.update_capital(900.0, now=t0))
        self.assertTrue(rm.is_halted)
        self.assertFalse(rm.update_capital(900.0, now=t0 + timedelta(hours=12)))
        self.assertTrue(rm.update_capital(900.0, now=t0 + timedelta(hours=25)))
        self.assertFalse(rm.is_halted)
        self.assertTrue(rm.defensive_mode)
        self.assertEqual(rm.peak_capital, 900.0)

    def test_small_300_usd_with_safe_isolated_leverage_can_trade_btc_and_eth(self) -> None:
        rm = RiskManager(initial_capital=300.0)
        rm.reanchor_capital(300.0)

        # $300 with 3x isolated leverage on BTC ($65,000, ATR=$1,300 -> 4% stop -> liq ~33% away > 3x stop!)
        btc_plan = rm.calculate_equity_aware_position_size(
            current_price=65000.0, atr=1300.0, signal_strength=1.0, target_leverage=3.0, min_notional=20.0, step_size=0.001, active_symbols_count=1
        )
        self.assertIsNone(btc_plan["skip_reason"])
        self.assertGreaterEqual(btc_plan["size"], 0.008)
        self.assertGreater(btc_plan["effective_leverage"], 1.5)
        # Liquidation price MUST be strictly farther than stop-loss price
        self.assertLess(btc_plan["est_liquidation_price_long"], btc_plan["stop_loss_long"])
        self.assertGreater(btc_plan["est_liquidation_price_short"], btc_plan["stop_loss_short"])

        # Extreme volatility (10% ATR -> 20% stop) automatically throttles leverage down so liquidation never overtakes stop!
        hi_vol = rm.calculate_equity_aware_position_size(
            current_price=100.0, atr=10.0, signal_strength=1.0, target_leverage=5.0, min_notional=5.0, step_size=0.1, active_symbols_count=1
        )
        self.assertLessEqual(hi_vol["effective_leverage"], 2.0)


if __name__ == "__main__":
    unittest.main()
