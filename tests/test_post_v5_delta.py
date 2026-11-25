"""Behavioral tests for post-v5 delta fixes (OrderExecutionService, Decimal risk/idempotency, Testnet fail-closed, UTC TTL, AlphaVault seed, MarketAnalyzer)."""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd


class TestPostV5Delta(unittest.TestCase):
    def test_decimal_idempotency_and_order_execution_service(self):
        from crypto_trading.order_execution_service import OrderExecutionService, OrderIntent
        from crypto_trading.order_manager import generate_idempotency_key

        k1 = generate_idempotency_key("BTC/USDT", "BUY", "MARKET", "0.1000000", None, "tag1")
        k2 = generate_idempotency_key("BTC/USDT", "BUY", "MARKET", 0.1, 0.0, "tag1")
        self.assertEqual(k1, k2)

        svc = OrderExecutionService()
        with self.assertRaises(ValueError):
            svc.submit(OrderIntent("BTC/USDT", "BUY", "MARKET", "0.001"), client=None, mark_price=100.0)

    def test_risk_manager_decimal_step_no_float_modulo_error(self):
        from crypto_trading.risk_manager import RiskManager

        rm = RiskManager(initial_capital=10000.0)
        # 0.03 % 0.001 fails in naive Python float arithmetic; Decimal modulo must approve
        dec = rm.evaluate_order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            amount=0.03,
            price=1000.0,
            context={
                "data_source": "TEST",
                "data_timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "data_quality_passed": True,
                "reconciliation_fresh": True,
                "has_unknown_orders": False,
                "exchange_connected": True,
                "step_size": 0.001,
                "tick_size": 0.01,
            },
        )
        self.assertTrue(dec.allowed, msg=f"Unexpected rejection: {dec.reasons}")

    def test_manual_controller_aware_ttl_and_alpha_vault_seed_status(self):
        from crypto_trading.manual_controller import ManualPositionController
        from worldquant_brain.alpha_vault import AlphaVault

        import gc

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_p = f"{td}/test.db"
            mc = ManualPositionController(db_path=db_p)
            old_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            conn = sqlite3.connect(db_p)
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO manual_overrides (symbol, is_locked, lock_reason, locked_at, ttl_seconds, updated_at) VALUES (?, 1, ?, ?, 60, ?)",
                        ("BTC/USDT", "test", old_iso, old_iso),
                    )
            finally:
                conn.close()
            self.assertFalse(mc.is_symbol_locked("BTC/USDT"))
            del mc
            gc.collect()

            vault = AlphaVault(db_path=":memory:")
            vault.seed_initial_verified_alphas()
            rows = vault.list_alphas()
            self.assertTrue(len(rows) > 0)
            self.assertTrue(all(r["status"] == "TEMPLATE_SEED" for r in rows))

    def test_testnet_client_never_returns_15000_on_remote_error(self):
        from crypto_trading.testnet_client import BinanceTestnetClient

        c = BinanceTestnetClient(api_key="k", api_secret="s")
        c.is_connected = True
        c.mock_mode = False
        c.exchange = MagicMock()
        c.exchange.fapiPrivateV2GetAccount.side_effect = ConnectionError("network down")
        bal = c.fetch_balance()
        self.assertIsNone(bal["total_equity"])
        self.assertEqual(bal["mode"], "OFFICIAL_BINANCE_TESTNET_ERROR")

    def test_market_analyzer_returns_data_unavailable_on_empty_df(self):
        from taiwan_stock.market_analyzer import MarketAnalyzer

        analyzer = MarketAnalyzer()
        with patch.object(analyzer, "fetch_market_data", return_value=pd.DataFrame()):
            res = analyzer.analyze()
            self.assertEqual(res["status"], "DATA_UNAVAILABLE")
            self.assertIsNone(res["score"])


    def test_reconciliation_service_get_latest_result_and_bitunix_no_fake_price(self):
        from unittest.mock import patch
        from crypto_trading.bitunix_client import BitunixClient
        from crypto_trading.reconciliation_service import ReconciliationService

        svc = ReconciliationService()
        self.assertIsNone(svc.get_latest_result())
        svc.reconcile({}, [], [], [], 1000.0, 1000.0)
        latest = svc.get_latest_result()
        self.assertTrue(latest["in_sync"])
        self.assertIn("completed_at", latest)

        bc = BitunixClient(api_key="", api_secret="")
        with patch("requests.get", side_effect=ConnectionError("down")):
            t = bc.fetch_ticker("BTC/USDT")
            self.assertIsNone(t["last"])
            self.assertEqual(t["error"], "TICKER_UNAVAILABLE")



    def test_bitunix_standby_balance_none_and_unverified_model_quarantine(self):
        import json
        import sys
        from pathlib import Path
        try:
            import joblib  # noqa: F401
        except ImportError:
            sys.modules.setdefault("joblib", MagicMock())
        from crypto_trading.bitunix_client import BitunixClient
        from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
        from crypto_trading.strategies.rl_strategy import RLAlphaStrategy

        bc = BitunixClient(api_key="", api_secret="")
        bal = bc.fetch_balance()
        self.assertIsNone(bal["total_equity"])
        self.assertEqual(bal["error"], "BITUNIX_STANDBY_MODE")

        df = pd.DataFrame({"Open": [100.0, 101.0, 102.0], "High": [101.0, 102.0, 103.0], "Low": [99.0, 100.0, 101.0], "Close": [100.0, 101.0, 102.0], "Volume": [10.0, 11.0, 12.0]})
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            m_path = Path(td) / "model.joblib"
            m_path.write_text("dummy", encoding="utf-8")
            ml = MLAlphaStrategy(model_path=str(m_path))
            out_ml = ml.generate_signals(df)
            self.assertEqual(out_ml["model_status"].iloc[0], "UNVERIFIED_MODEL_QUARANTINED")

            r_path = Path(td) / "ppo.pt"
            r_path.write_text("dummy", encoding="utf-8")
            (Path(td) / "ppo_manifest.json").write_text(
                json.dumps({"is_synthetic": True, "deployment_stage": "RESEARCH_ONLY"}),
                encoding="utf-8",
            )
            rl = RLAlphaStrategy(agent_weights_path=str(r_path))
            out_rl = rl.generate_signals(df)
            self.assertIn(out_rl["model_status"].iloc[0], ("UNVERIFIED_MODEL_QUARANTINED", "MODEL_UNAVAILABLE", "SYNTHETIC_MODEL_QUARANTINED"))



    def test_web_ui_fail_closed_no_hardcoded_15000_20000_or_null_crash(self):
        from pathlib import Path
        srv = Path("web_ui/server.py").read_text(encoding="utf-8")
        self.assertNotIn('bal.get("total_equity", 15000.0)', srv)
        self.assertNotIn('bal.get("total_equity", 20000.0)', srv)
        self.assertNotIn('ticker.get("last", 68500.0)', srv)
        html = Path("web_ui/static/index.html").read_text(encoding="utf-8")
        self.assertIn("d.total_equity !== undefined && d.total_equity !== null", html)
        self.assertIn("d.current_price !== undefined && d.current_price !== null", html)



    def test_phase0_inventory_and_phase2_instrument_endpoint(self):
        import json
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        inv_path = repo_root / "reports" / "baseline_inventory.json"
        if inv_path.exists():
            inv = json.loads(inv_path.read_text(encoding="utf-8"))
            for key in (
                "MODE_AUTHORITY", "ACCOUNT_SOURCE", "POSITION_SOURCE", "MARKET_SOURCE",
                "ORDER_ENTRY_POINT", "BRAIN_CONFIG", "MODEL_ARTIFACT", "RUNTIME_ARTIFACT",
                "HARDCODED_UI_VALUE", "WORKER_STARTUP",
            ):
                self.assertIn(key, inv)
        srv = (repo_root / "web_ui" / "server.py").read_text(encoding="utf-8")
        self.assertIn('"/api/trading/instrument"', srv)
        self.assertIn('"canonical_identity"', srv)


if __name__ == "__main__":
    unittest.main()
