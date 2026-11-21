"""Contract & E2E tests for P19: Wave 10 Caller Wiring across BRAIN, Crypto, TW, and US modules."""
from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


class TestP19Wave10CallerWiring(unittest.TestCase):
    def test_brain_client_active_pnl_cache_wiring_blocks_high_corr(self) -> None:
        from worldquant_brain.brain_client import BrainClient

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "alpha_vault.db"
            client = BrainClient(email="", password="")
            client.quota_db_path = str(db_path)

            base_pnl = list(np.cumsum(np.sin(np.linspace(0, 12, 80)) + 0.1))
            ok = client.cache_submitted_alpha_pnl("ACTIVE_001", region="USA", cum_pnl=base_pnl)
            self.assertTrue(ok)

            # Nearly identical candidate PnL should be blocked (>= 0.70)
            clone_pnl = [x * 1.05 + 0.01 for x in base_pnl]
            res_clone = client.check_local_active_self_correlation(
                "CAND_CLONE", region="USA", threshold=0.70, candidate_pnl=clone_pnl
            )
            self.assertTrue(res_clone["checked"])
            self.assertTrue(res_clone["blocked"])
            self.assertEqual(res_clone["matched_alpha_id"], "ACTIVE_001")
            self.assertGreaterEqual(res_clone["max_corr"], 0.99)

            # Orthogonal candidate PnL should pass
            orth_pnl = list(np.cumsum(np.cos(np.linspace(0, 12, 80)) * np.sin(np.linspace(0, 35, 80))))
            res_orth = client.check_local_active_self_correlation(
                "CAND_ORTH", region="USA", threshold=0.70, candidate_pnl=orth_pnl
            )
            self.assertTrue(res_orth["checked"])
            self.assertFalse(res_orth["blocked"])
            gc.collect()

    def test_headless_miner_wires_active_pnl_cache_hooks(self) -> None:
        src = Path("scripts/headless_brain_miner.py").read_text(encoding="utf-8")
        self.assertIn("check_local_active_self_correlation", src)
        self.assertIn("cache_submitted_alpha_pnl", src)
        self.assertIn("LOCAL_SELF_CORR_SKIP", src)

    def test_testnet_client_routes_algo_order_and_hedge_mode(self) -> None:
        from crypto_trading.testnet_client import BinanceTestnetClient

        client = BinanceTestnetClient.__new__(BinanceTestnetClient)
        client.mock_mode = False
        client.is_connected = True
        client.settlement = "USDT"
        client.market_type = "future"
        mock_ex = MagicMock()
        mock_ex.fapiPrivatePostAlgoOrder.return_value = {"algoId": "A999", "status": "NEW"}
        mock_ex.fapiPrivatePostOrder.return_value = {"orderId": "O123", "status": "NEW"}
        client.exchange = mock_ex

        # Conditional STOP_MARKET must route to /fapi/v1/algoOrder
        r_algo = client.create_order(
            "BTC/USDT",
            "sell",
            "STOP_MARKET",
            0.01,
            hedge_mode=True,
            closing_position_side="LONG",
        )
        self.assertTrue(r_algo["success"])
        self.assertEqual(r_algo["endpoint"], "/fapi/v1/algoOrder")
        self.assertEqual(r_algo["exchange_order_id"], "A999")
        algo_params = mock_ex.fapiPrivatePostAlgoOrder.call_args[0][0]
        self.assertEqual(algo_params["positionSide"], "LONG")
        self.assertNotIn("reduceOnly", algo_params)

        # Post-only LIMIT order must use GTX and /fapi/v1/order
        r_lim = client.place_order("BTC/USDT", "buy", 0.02, order_type="LIMIT", price=90000.0, post_only=True)
        self.assertTrue(r_lim["success"])
        self.assertEqual(r_lim["endpoint"], "/fapi/v1/order")
        lim_params = mock_ex.fapiPrivatePostOrder.call_args[0][0]
        self.assertEqual(lim_params["timeInForce"], "GTX")
        self.assertEqual(lim_params["selfTradePreventionMode"], "EXPIRE_MAKER")

    def test_reconciliation_detects_exchange_risk_event_orders(self) -> None:
        from crypto_trading.reconciliation_service import ReconciliationService

        svc = ReconciliationService()
        rep = svc.reconcile(
            local_positions={"BTC/USDT": 0.0},
            remote_positions=[],
            local_orders=[],
            remote_orders=[
                {
                    "id": "999888",
                    "symbol": "BTCUSDT",
                    "clientOrderId": "autoclose-1732200000",
                    "status": "FILLED",
                }
            ],
            local_balance=1000.0,
            remote_balance=1000.0,
        )
        self.assertFalse(rep["is_reconciled"])
        self.assertEqual(rep["allowed_actions"], "REDUCE_ONLY")
        self.assertEqual(rep["exchange_risk_event_count"], 1)
        self.assertEqual(rep["exchange_risk_events"][0]["client_order_id"], "autoclose-1732200000")
        self.assertEqual(svc.get_latest_result()["exchange_risk_event_count"], 1)

    def test_taiwan_ml_ranker_wires_twse_t86_smart_institutional_lots(self) -> None:
        from taiwan_stock.ml_ranker import TaiwanStockMLRanker

        dates = pd.date_range("2025-01-02", periods=95, freq="B")
        base_close = np.linspace(100.0, 120.0, len(dates)) + np.sin(np.arange(len(dates)) * 0.7) * 3.0
        df_a = pd.DataFrame(
            {
                "Open": base_close * 0.99,
                "High": base_close * 1.02,
                "Low": base_close * 0.98,
                "Close": base_close,
                "Volume": np.full(len(dates), 5000.0),
            },
            index=dates,
        )
        df_b = df_a.copy()
        df_b["Close"] = base_close * 0.95 + np.cos(np.arange(len(dates)) * 0.5) * 2.0
        df_b["High"] = df_b["Close"] * 1.02
        df_b["Low"] = df_b["Close"] * 0.98

        fields = [
            "證券代號",
            "外陸資買賣超股數(不含外資自營商)",
            "外資自營商買賣超股數",
            "投信買賣超股數",
            "自營商買賣超股數(自行買賣)",
            "自營商買賣超股數(避險)",
            "三大法人買賣超股數",
        ]
        # Dealer hedge = -500,000 shares must be EXCLUDED from smart_institutional_lots
        t86_a = {d: (fields, ["2330", "200,000", "0", "100,000", "50,000", "-500,000", "-150,000"]) for d in dates}
        t86_b = {d: (fields, ["2317", "10,000", "0", "0", "0", "0", "10,000"]) for d in dates}

        ranker = TaiwanStockMLRanker()
        clean, feature_cols = ranker.prepare_dataset({"2330": df_a, "2317": df_b}, t86_map={"2330": t86_a, "2317": t86_b})
        self.assertIn("smart_institutional_lots", feature_cols)
        self.assertIn("cs_rank_smart_institutional_lots", feature_cols)
        a_rows = clean[clean["symbol"] == "2330"]
        # (200k + 0 + 100k + 50k) / 1000 = 350.0 lots
        self.assertAlmostEqual(float(a_rows["smart_institutional_lots"].iloc[-1]), 350.0, places=4)

    def test_us_ml_ranker_wires_pit_pead_sue_and_form4_cluster_buy(self) -> None:
        from us_stock.ml_ranker import USStockMLRanker

        dates = pd.date_range("2024-01-02", periods=270, freq="B")
        px = np.linspace(150.0, 190.0, len(dates)) + np.sin(np.arange(len(dates)) * 0.5) * 2.0
        df = pd.DataFrame(
            {"Open": px * 0.99, "High": px * 1.01, "Low": px * 0.98, "Close": px, "Volume": np.full(len(dates), 1e6)},
            index=dates,
        )
        eps_recs = [
            {"end": "2022-03-31", "filed": "2022-05-01", "eps": 1.00},
            {"end": "2022-06-30", "filed": "2022-08-01", "eps": 1.05},
            {"end": "2022-09-30", "filed": "2022-11-01", "eps": 1.10},
            {"end": "2022-12-31", "filed": "2023-02-01", "eps": 1.15},
            {"end": "2023-03-31", "filed": "2023-05-01", "eps": 1.20},
            {"end": "2023-06-30", "filed": "2023-08-01", "eps": 1.35},
            {"end": "2023-09-30", "filed": "2023-11-01", "eps": 1.60},
            # Future filing in 2024-11-15 must NOT leak into 2024-06 dates
            {"end": "2024-09-30", "filed": "2024-11-15", "eps": 9.99},
        ]
        form4 = [
            {"filed": "2024-10-10", "transactionCode": "P", "is_10b5_1": False, "insider_name": "CEO", "shares": 5000, "price": 180.0},
            {"filed": "2024-10-12", "transactionCode": "P", "is_10b5_1": False, "insider_name": "CFO", "shares": 3000, "price": 182.0},
        ]
        panel = USStockMLRanker().prepare_panel(
            {"AAPL": df},
            sector_map={"AAPL": "TECH"},
            eps_records_map={"AAPL": eps_recs},
            form4_filings_map={"AAPL": form4},
        )
        self.assertIn("pead_sue_score", panel.columns)
        self.assertIn("sec_rank_pead_sue_score", panel.columns)
        self.assertIn("form4_cluster_buy_score", panel.columns)
        self.assertIn("sec_rank_form4_cluster_buy_score", panel.columns)
        # Before Form 4 filings (2024-06), cluster_buy_score must be 0.0 (strict PIT)
        early = panel.loc[panel.index < pd.Timestamp("2024-10-10"), "form4_cluster_buy_score"]
        self.assertTrue((early == 0.0).all())
        # Within 30d after 2024-10-12, cluster_buy_score must be > 0
        late = panel.loc[
            (panel.index >= pd.Timestamp("2024-10-12")) & (panel.index <= pd.Timestamp("2024-11-05")),
            "form4_cluster_buy_score",
        ]
        self.assertTrue((late > 0.0).all())


if __name__ == "__main__":
    unittest.main()
