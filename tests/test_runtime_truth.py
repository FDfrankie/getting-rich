from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from crypto_trading.paper_broker import calculate_pnl
from crypto_trading.runtime_context import ConnectionState, RuntimeTradingContext, TradingMode
from crypto_trading.trading_snapshot import (
    AccountSnapshot,
    ConnectionSnapshot,
    MarketSnapshot,
    ReconciliationSnapshot,
    RiskSnapshot,
    SnapshotStatus,
    TradingSnapshot,
)


class TestRuntimeTruth(unittest.TestCase):
    def test_remote_degradation_is_explicit_and_cannot_claim_remote_account(self):
        ctx = RuntimeTradingContext.degraded_from_remote(
            TradingMode.BINANCE_TESTNET,
            account_id="testnet-account",
            venue="BINANCE",
            reason="AUTH_FAILED",
            connection_state=ConnectionState.AUTH_FAILED,
        )
        self.assertIs(ctx.requested_mode, TradingMode.BINANCE_TESTNET)
        self.assertIs(ctx.effective_mode, TradingMode.LOCAL_PAPER)
        self.assertTrue(ctx.degraded)
        self.assertFalse(ctx.remote_account_claim_allowed)

    def test_live_mode_requires_authentication(self):
        with self.assertRaises(ValueError):
            RuntimeTradingContext(
                requested_mode=TradingMode.BITUNIX_LIVE,
                effective_mode=TradingMode.BITUNIX_LIVE,
                environment_id="prod",
                account_id="live",
                venue="BITUNIX",
                connection_state=ConnectionState.UNREACHABLE,
            )

    def test_local_paper_short_profit_when_price_falls(self):
        pnl = calculate_pnl("SHORT", Decimal("-0.01"), Decimal("65000"), Decimal("64000"))
        self.assertEqual(pnl, Decimal("10.00"))
        pnl_long = calculate_pnl("LONG", Decimal("0.01"), Decimal("64000"), Decimal("65000"))
        self.assertEqual(pnl_long, Decimal("10.00"))

    def test_equity_mismatch_fails_closed(self):
        now = datetime.now(timezone.utc)
        snapshot = TradingSnapshot(
            context=RuntimeTradingContext.local_paper(),
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot("BTC", "BTC/USDT", "PERPETUAL", "USDT", Decimal("100"), Decimal("100"), "LOCAL", now, 0.1),
            account=AccountSnapshot(Decimal("1000"), Decimal("25"), Decimal("1100"), Decimal("900"), None, None, None, "USDT", source="TEST", timestamp=now),
            positions=(),
            open_orders=(),
            reconciliation=ReconciliationSnapshot("OK", 0, now),
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "REDUCE_ONLY"), ()),
            configured_strategy="PPO",
            effective_strategy="MOMENTUM_FALLBACK",
            model_id=None,
        ).validated()
        self.assertIs(snapshot.status, SnapshotStatus.INCONSISTENT)
        self.assertFalse(snapshot.risk.trading_enabled)
        self.assertNotIn("OPEN_LONG", snapshot.risk.allowed_actions)
        self.assertIn("REDUCE_ONLY", snapshot.risk.allowed_actions)

    def test_stale_account_or_position_or_unknown_orders_block_entries(self):
        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(seconds=20)
        # 1. Stale account (>15s)
        s1 = TradingSnapshot(
            context=RuntimeTradingContext.local_paper(),
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot("BTC", "BTC/USDT", "PERPETUAL", "USDT", Decimal("100"), Decimal("100"), "LOCAL", now, 0.1),
            account=AccountSnapshot(Decimal("1000"), Decimal("0"), Decimal("1000"), Decimal("1000"), None, None, None, "USDT", source="TEST", timestamp=old_ts),
            positions=(),
            open_orders=(),
            reconciliation=ReconciliationSnapshot("OK", 0, now),
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "REDUCE_ONLY"), ()),
            configured_strategy="MOMENTUM",
            effective_strategy="MOMENTUM",
            model_id=None,
            generated_at=now,
        ).validated()
        self.assertIs(s1.status, SnapshotStatus.INCONSISTENT)
        self.assertIn("ACCOUNT_DATA_STALE_OR_MISSING_TIMESTAMP", s1.errors)

        # 2. Unknown orders present
        s2 = TradingSnapshot(
            context=RuntimeTradingContext.local_paper(),
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot("BTC", "BTC/USDT", "PERPETUAL", "USDT", Decimal("100"), Decimal("100"), "LOCAL", now, 0.1),
            account=AccountSnapshot(Decimal("1000"), Decimal("0"), Decimal("1000"), Decimal("1000"), None, None, None, "USDT", source="TEST", timestamp=now),
            positions=(),
            open_orders=({"client_order_id": "ord_1", "status": "UNKNOWN"},),
            reconciliation=ReconciliationSnapshot("OK", 0, now),
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "REDUCE_ONLY"), ()),
            configured_strategy="MOMENTUM",
            effective_strategy="MOMENTUM",
            model_id=None,
            generated_at=now,
        ).validated()
        self.assertIs(s2.status, SnapshotStatus.INCONSISTENT)
        self.assertIn("UNKNOWN_ORDERS_PRESENT", s2.errors)
        self.assertFalse(s2.risk.trading_enabled)
