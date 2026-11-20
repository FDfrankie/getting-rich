"""Tests for v5 upgrades: Unified Trial Ledger (012), Exchange Liquidation Verification, and Bitunix Order Mapping."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from crypto_trading.execution_cost_filter import OrderState, map_bitunix_order_status
from crypto_trading.risk_manager import RiskManager
from worldquant_brain.trial_ledger import UnifiedTrialLedger

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestV5LedgerAndExchangeSafety(unittest.TestCase):
    def test_012_trial_ledger_idempotent_events_and_contamination_guard(self) -> None:
        m012 = (REPO_ROOT / "migrations" / "012_unified_trial_ledger.sql").read_text(encoding="utf-8")
        m012_rb = (REPO_ROOT / "migrations" / "012_unified_trial_ledger_rollback.sql").read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "test.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(m012)
            conn.executescript(m012)  # idempotent 2nd apply
            conn.close()

            ledger = UnifiedTrialLedger(db_path)
            tid1, created1 = ledger.create_or_get_trial(
                domain="WORLDQUANT_ALPHA", stage="OFFICIAL_EVALUATION", policy_version="v1",
                code_commit_sha="abc123", config_hash="cfg1", data_snapshot_hash="data1", input_snapshot_hash="inp1",
                evidence_level="OFFICIAL",
            )
            tid2, created2 = ledger.create_or_get_trial(
                domain="WORLDQUANT_ALPHA", stage="OFFICIAL_EVALUATION", policy_version="v1",
                code_commit_sha="abc123", config_hash="cfg1", data_snapshot_hash="data1", input_snapshot_hash="inp1",
                evidence_level="OFFICIAL",
            )
            self.assertTrue(created1)
            self.assertFalse(created2)
            self.assertEqual(tid1, tid2)

            # Exactly-once event
            self.assertTrue(ledger.append_event(tid1, event_type="SIM_DONE", idempotency_key="k1", payload={"sharpe": 1.5}))
            self.assertFalse(ledger.append_event(tid1, event_type="SIM_DONE", idempotency_key="k1", payload={"sharpe": 1.5}))

            # Test contamination guard
            self.assertFalse(ledger.is_promotion_blocked_by_contamination(tid1))
            ledger.record_test_access(tid1, consumer_name="optimizer", purpose="tuning", consumed_by_generator=True)
            self.assertTrue(ledger.is_promotion_blocked_by_contamination(tid1))

            # Clean rollback & reapply
            conn = sqlite3.connect(db_path)
            conn.executescript(m012_rb)
            conn.executescript(m012)
            conn.close()

    def test_exchange_liquidation_verification_fails_closed(self) -> None:
        # Unknown MMR tier -> BLOCK_ORDER
        r1 = RiskManager.verify_exchange_liquidation_price(
            entry_price=65000.0, stop_loss_price=63000.0, side="LONG", exchange_liq_price=45000.0, mmr_tier_known=False
        )
        self.assertFalse(r1["allowed"])
        self.assertEqual(r1["action"], "BLOCK_ORDER")

        # Mismatch > 5% vs local precheck -> BLOCK_ORDER
        r2 = RiskManager.verify_exchange_liquidation_price(
            entry_price=65000.0, stop_loss_price=63000.0, side="LONG", exchange_liq_price=58000.0, local_precheck_liq_price=45000.0
        )
        self.assertFalse(r2["allowed"])
        self.assertIn("LIQUIDATION_MISMATCH", r2["reason"])

        # Exchange-confirmed safe liquidation distance -> APPROVE
        r3 = RiskManager.verify_exchange_liquidation_price(
            entry_price=65000.0, stop_loss_price=63000.0, side="LONG", exchange_liq_price=45000.0, local_precheck_liq_price=44500.0
        )
        self.assertTrue(r3["allowed"])
        self.assertEqual(r3["action"], "APPROVE")

    def test_bitunix_status_mapping(self) -> None:
        self.assertEqual(map_bitunix_order_status("INIT"), OrderState.CREATED)
        self.assertEqual(map_bitunix_order_status("PART_FILLED_CANCELED"), OrderState.CANCELLED)
        self.assertEqual(map_bitunix_order_status("UNKNOWN_STATUS"), OrderState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
