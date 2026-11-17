import unittest
from decimal import Decimal
from unittest.mock import MagicMock

from crypto_trading.exchange_normalizer import normalize_binance_balance
from crypto_trading.reconciliation_service import ReconciliationService
from crypto_trading.testnet_client import BinanceTestnetClient
from worldquant_brain.alpha_miner import ContinuousAlphaMiner


class TestPhase2Enhancements(unittest.TestCase):
    def test_binance_testnet_balance_populates_wallet_and_upnl_for_healthy_snapshot(self):
        c = BinanceTestnetClient(api_key="k", api_secret="s")
        c.is_connected = True
        c.mock_mode = False
        c.settlement = "USDC"
        c.exchange = MagicMock()
        c.exchange.fapiPrivateV2GetAccount.return_value = {
            "assets": [
                {
                    "asset": "USDC",
                    "walletBalance": "4981.11",
                    "availableBalance": "4981.11",
                    "unrealizedProfit": "0.00",
                    "marginBalance": "4981.11",
                }
            ]
        }
        bal = c.fetch_balance()
        snap = normalize_binance_balance(bal, default_currency="USDC")
        self.assertEqual(snap.wallet_balance, Decimal("4981.11000000"))
        self.assertEqual(snap.unrealized_pnl, Decimal("0E-8"))
        self.assertEqual(snap.equity, Decimal("4981.11000000"))

    def test_reconciliation_service_marks_in_sync(self):
        svc = ReconciliationService()
        svc.reconcile({}, [], [], [], 4981.11, 4981.11)
        latest = svc.get_latest_result()
        self.assertIsNotNone(latest)
        self.assertTrue(latest["in_sync"])

    def test_alpha_miner_includes_fundamental_and_sentiment_fields(self):
        miner = ContinuousAlphaMiner()
        self.assertTrue(hasattr(miner, "fundamental_fields"))
        self.assertTrue(hasattr(miner, "sentiment_fields"))
        exprs = [miner.generate_candidate_expression()["expression"] for _ in range(80)]
        has_group = any("group_neutralize" in e or "group_rank" in e for e in exprs)
        self.assertTrue(has_group)


if __name__ == "__main__":
    unittest.main()
