"""Behavioral offline tests for PaperBroker, BrainConnectivityService, TWSE calendar, and ReportSnapshot."""

from datetime import date, datetime, timezone
from decimal import Decimal
import unittest

import numpy as np
import pandas as pd


class TestAuditBehaviors(unittest.TestCase):
    def test_single_trading_mode_enum(self):
        from crypto_trading.runtime_context import TradingMode as RtMode
        from crypto_trading.settings import TradingMode as SetMode
        self.assertIs(RtMode, SetMode)

    def test_paper_broker_dynamic_short_pnl_and_reconcile(self):
        from crypto_trading.paper_broker import PaperBroker

        class DummyBot:
            symbol = "BTC/USDT"
            position = -0.02
            position_side = "SHORT"
            entry_price = 65000.0
            balance_usdt = 10000.0
            open_orders = ()

        broker = PaperBroker(DummyBot())
        now = datetime.now(timezone.utc)
        positions = broker.fetch_positions(Decimal("64000"), now=now)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].unrealized_pnl, Decimal("20.00"))
        acc = broker.account_snapshot(positions, now=now)
        self.assertEqual(acc.equity, Decimal("10020.00"))
        self.assertIsNone(acc.consistency_error())
        rec = broker.reconcile(now=now)
        self.assertEqual(rec.status, "OK")
        self.assertEqual(rec.last_completed_at, now)

    def test_brain_connectivity_service_and_queue_lifecycle(self):
        from worldquant_brain.brain_client import BrainClient
        from worldquant_brain.brain_job_queue import BrainJobQueue
        from worldquant_brain.connectivity_service import BrainAuthFailureClassifier, BrainConnectivityService

        self.assertEqual(
            BrainAuthFailureClassifier.classify(401, {"WWW-Authenticate": "persona"}, "Biometric"),
            "BRAIN_BIOMETRIC_REQUIRED",
        )
        self.assertEqual(
            BrainAuthFailureClassifier.classify(403, {"Content-Type": "text/html"}, "Cloudflare WAF"),
            "BRAIN_WAF_BLOCKED",
        )
        self.assertEqual(BrainAuthFailureClassifier.classify(429, {}, ""), "RATE_LIMITED")

        client = BrainClient(email="", password="")
        client.email = None
        client.password = None
        q = BrainJobQueue(client, autostart=False)
        self.assertFalse(q.worker_alive())
        q.start()
        self.assertTrue(q.worker_alive())
        svc = BrainConnectivityService(client, q, lambda: None)
        st = svc.get_status(local_alpha_worker_alive=True)
        self.assertTrue(st["service_available"])
        self.assertFalse(st["configured"])
        self.assertEqual(st["safe_error_code"], "BRAIN_CREDENTIALS_MISSING")
        q.stop()
        q.join(timeout=1.0)
        self.assertFalse(q.worker_alive())

    def test_twse_trading_calendar_holidays_and_bounds(self):
        from taiwan_stock.twse_calendar import is_twse_trading_day

        # 2026-02-17 is Lunar New Year (Tuesday) -> must be False!
        self.assertFalse(is_twse_trading_day(date(2026, 2, 17)))
        # 2026-11-16 is a normal Monday -> True
        self.assertTrue(is_twse_trading_day(date(2026, 11, 16)))
        # Outside verified calendar bounds -> fails closed False
        self.assertFalse(is_twse_trading_day(date(2035, 5, 2)))

    def test_feature_engineer_preserves_nan_correlation(self):
        from crypto_trading.feature_engineer import CryptoFeatureEngineer

        idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
        btc = pd.DataFrame({"Open": np.linspace(100, 110, 10), "High": np.linspace(101, 111, 10), "Low": np.linspace(99, 109, 10), "Close": np.linspace(100, 110, 10), "Volume": [100.0] * 10}, index=idx)
        eth = pd.DataFrame({"Open": [np.nan] * 10, "High": [np.nan] * 10, "Low": [np.nan] * 10, "Close": [np.nan] * 10, "Volume": [np.nan] * 10}, index=idx)
        cross = CryptoFeatureEngineer().compute_multi_asset_cross_features({"BTC/USDT": btc, "ETH/USDT": eth})
        if "ETH_corr_BTC_24" in cross.columns:
            self.assertTrue(cross["ETH_corr_BTC_24"].isna().all())


if __name__ == "__main__":
    unittest.main()
