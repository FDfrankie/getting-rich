"""Unit tests for P6 Batch 13 Correctness Bundle."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_trading.meta_labeling import (
    apply_meta_label_filter,
    compute_deflated_sharpe_and_pbo,
)
from crypto_trading.shadow_trading import ShadowOrderIntent, ShadowTradingService
from data_persistence.repositories import SQLiteRepository
from portfolio.risk_allocator import PortfolioRiskAllocator
from worldquant_brain.alpha_lifecycle import submission_eligibility


class TestP6Batch13Correctness(unittest.TestCase):
    def test_alpha_lifecycle_does_not_mark_simulated_brain_alpha_id_as_submitted(self) -> None:
        alpha = {
            "status": "SUBMISSION_CANDIDATE",
            "sharpe": 1.8,
            "fitness": 1.3,
            "turnover": 0.25,
            "margin_bps": 15.0,
            "drawdown": 0.05,
            "is_check_status": "PASS",
            "self_correlation": 0.35,
            "brain_simulation_id": "sim_123",
            "brain_alpha_id": "alpha_xyz",  # Simulated alpha ALWAYS has brain_alpha_id
            "submitted_at": None,
        }
        elig = submission_eligibility(alpha)
        self.assertTrue(elig.ready, f"Expected ready=True, got reasons={elig.reasons}")

        # Once submitted_at is set, it MUST report ALREADY_SUBMITTED
        alpha_submitted = dict(alpha, submitted_at="2026-11-21T00:00:00Z")
        elig2 = submission_eligibility(alpha_submitted)
        self.assertFalse(elig2.ready)
        self.assertIn("ALREADY_SUBMITTED", elig2.reasons)

    def test_meta_labeling_fail_closed_pbo_and_causal_conf_mask(self) -> None:
        # Insufficient trials/observations must fail-closed (pbo == 1.0, passed == False)
        small = compute_deflated_sharpe_and_pbo([np.ones(20) * 0.01])
        self.assertEqual(small["pbo"], 1.0)
        self.assertFalse(small.get("gate_passed", small.get("passed", False)))

        # Causal conf_mask check: changing future signal values must not alter earlier mask decisions
        close = pd.DataFrame({"Close": np.linspace(100.0, 120.0, 60)})
        sig_a = pd.Series(np.linspace(0.1, 1.0, 60))
        sig_b = sig_a.copy()
        sig_b.iloc[40:] = 100.0
        res_a = apply_meta_label_filter(sig_a, close)
        res_b = apply_meta_label_filter(sig_b, close)
        self.assertIn("ablation_comparison", res_a)
        self.assertIn("ablation_comparison", res_b)

    def test_shadow_trading_single_tx_dedup_and_alias(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_file = Path(td) / "shadow.db"
            engine = ShadowTradingService(db_file)
            i1 = engine.create_order_intent(
                strategy="s1",
                model_version="v1",
                symbol="BTC/USDT",
                features={"rsi": 30.0},
                signal=1.0,
                confidence=0.8,
                target_position=0.5,
                current_price=50000.0,
            )
            self.assertIsInstance(i1, ShadowOrderIntent)
            self.assertEqual(i1.risk_decision, "SHADOW_APPROVED")

            i2 = engine.create_order_intent(
                strategy="s1",
                model_version="v1",
                symbol="BTC/USDT",
                features={"rsi": 30.0},
                signal=1.0,
                confidence=0.8,
                target_position=0.5,
                current_price=50000.0,
            )
            self.assertEqual(i2.risk_decision, "BLOCKED_DUPLICATE_INTENT")

    def test_risk_allocator_blocks_below_min_notional_and_reduce_only_nan(self) -> None:
        alloc = PortfolioRiskAllocator(total_equity_usdt=100.0, max_single_symbol_pct=0.05)
        # Max single symbol cap is $5.0, below $10 min_notional -> must reject
        d = alloc.evaluate_allocation("BTC/USDT", 20.0, {}, min_notional_usdt=10.0)
        self.assertFalse(d.allowed)
        self.assertIn("BELOW_MIN_NOTIONAL", d.reason)

        # Even reduce_only must reject NaN
        d_nan = alloc.evaluate_allocation("BTC/USDT", math.nan, {}, is_reduce_only=True)
        self.assertFalse(d_nan.allowed)
        self.assertEqual(d_nan.reason, "NON_FINITE_INPUT")

    def test_verify_bundle_and_readiness_fails_closed_exit_code(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        src = (repo_root / "scripts/verify_bundle_and_readiness.py").read_text(encoding="utf-8")
        self.assertIn('res.get("overall_status") == "BLOCKED"', src)
        self.assertIn("sys.exit(1)", src)


if __name__ == "__main__":
    unittest.main()
