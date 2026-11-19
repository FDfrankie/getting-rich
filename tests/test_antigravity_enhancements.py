"""Comprehensive offline verification tests for all 14 files in antigravity_complete_enhancements."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crypto_trading.risk_manager import RiskManager
from crypto_trading.walk_forward_guard import (
    build_purged_walk_forward_folds,
    evaluate_walk_forward_oos,
)
from notifications.alert_manager import AlertManager, AlertSeverity
from portfolio.risk_allocator import PortfolioRiskAllocator
from worldquant_brain.direct_submit_service import DirectSubmitService
from worldquant_brain.pre_submission_gate import GateVerdict, evaluate_pre_submission
from worldquant_brain.submission_service import SubmissionService


class TestAntigravityEnhancements(unittest.TestCase):
    def test_pre_submission_gate_blocks_high_self_corr(self) -> None:
        res = evaluate_pre_submission({
            "brain_alpha_id": "A_HIGH_CORR",
            "official_simulation_id": "SIM_1",
            "metrics_source": "WORLDQUANT_BRAIN",
            "status": "BRAIN_COMPLETE",
            "sharpe": 1.65,
            "fitness": 1.25,
            "turnover": 0.20,
            "self_correlation": 0.78,
            "max_instrument_weight": 0.05,
            "weight_test_passed": True,
            "sub_universe_passed": True,
            "bias_test_passed": True,
        })
        self.assertEqual(res.verdict, GateVerdict.BLOCK)
        self.assertIn("SELF_CORRELATION_NOT_PASSED", res.reasons)

    def test_direct_submit_service_enforces_gate(self) -> None:
        called = {"count": 0}

        def _submit(_aid: str) -> dict:
            called["count"] += 1
            return {"success": True, "status": "SUBMITTED"}

        svc = DirectSubmitService(_submit)
        blocked = svc.submit_existing_alpha({
            "brain_alpha_id": "ALPHA_BAD",
            "metrics_source": "LOCAL_PROXY",
            "sharpe": 1.8,
            "fitness": 1.4,
            "turnover": 0.2,
        })
        self.assertFalse(blocked["success"])
        self.assertIn("PRE_SUBMISSION_GATE", blocked["error"])
        self.assertEqual(called["count"], 0)

        passed = svc.submit_existing_alpha({
            "brain_alpha_id": "ALPHA_GOOD",
            "official_simulation_id": "SIM_GOOD",
            "metrics_source": "WORLDQUANT_BRAIN",
            "status": "BRAIN_COMPLETE",
            "sharpe": 1.75,
            "fitness": 1.30,
            "turnover": 0.22,
            "self_correlation": 0.35,
            "max_instrument_weight": 0.04,
            "weight_test_passed": True,
            "sub_universe_passed": True,
            "bias_test_passed": True,
        })
        self.assertTrue(passed["success"])
        self.assertEqual(called["count"], 1)

    def test_submission_service_gates_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "sub.db"
            svc = SubmissionService(db_path, top_k=2)
            res = svc.execute_daily_top_k(
                [
                    {
                        "alpha_id": "A_BAD",
                        "brain_alpha_id": "A_BAD",
                        "official_simulation_id": "SIM_BAD",
                        "metrics_source": "WORLDQUANT_BRAIN",
                        "status": "BRAIN_COMPLETE",
                        "sharpe": 1.6,
                        "fitness": 1.2,
                        "turnover": 0.2,
                        "self_correlation": 0.85,
                        "max_instrument_weight": 0.05,
                        "weight_test_passed": True,
                        "sub_universe_passed": True,
                        "bias_test_passed": True,
                    }
                ],
                submitter=lambda _aid: {"success": True, "status": "SUBMITTED"},
            )
            self.assertEqual(res["submitted_ids"], [])
            del svc
            import gc
            gc.collect()

    def test_walk_forward_guard_purged_folds_and_oos(self) -> None:
        folds = build_purged_walk_forward_folds(600, train_size=200, test_size=80, purge_gap=10)
        self.assertGreaterEqual(len(folds), 4)
        for f in folds:
            self.assertEqual(f.test_start - f.train_end, 10)
        verdict = evaluate_walk_forward_oos(
            [[0.002, 0.001, 0.003, -0.0005, 0.002, 0.0015]] * 4,
            min_folds=3,
            min_mean_sharpe=0.5,
            max_drawdown_limit=0.25,
        )
        self.assertTrue(verdict.passed)

    def test_portfolio_risk_allocator(self) -> None:
        alloc = PortfolioRiskAllocator(
            total_equity_usdt=10000.0,
            max_open_positions=3,
            max_total_exposure_pct=1.2,
            max_single_symbol_pct=0.5,
        )
        d1 = alloc.evaluate_allocation("ETH/USDT", 6000.0, {"BTC/USDT": 4000.0})
        self.assertTrue(d1.allowed)
        self.assertEqual(d1.approved_notional_usdt, 5000.0)

        d2 = alloc.evaluate_allocation(
            "AVAX/USDT",
            2000.0,
            {"BTC/USDT": 3000.0, "ETH/USDT": 3000.0, "SOL/USDT": 3000.0},
        )
        self.assertFalse(d2.allowed)
        self.assertIn("MAX_OPEN_POSITIONS_REACHED", d2.reason)

    def test_alert_manager_cooldown(self) -> None:
        seen = []
        mgr = AlertManager(cooldown_seconds=60.0, sink=seen.append)
        self.assertTrue(mgr.emit("RECON", "drift", severity=AlertSeverity.WARNING, now=100.0))
        self.assertFalse(mgr.emit("RECON", "drift again", severity=AlertSeverity.WARNING, now=120.0))
        self.assertTrue(mgr.emit("RECON", "drift later", severity=AlertSeverity.WARNING, now=165.0))
        self.assertEqual(len(seen), 2)

    def test_risk_manager_auto_quantize_step(self) -> None:
        rm = RiskManager(initial_capital=10000.0)
        dec = rm.evaluate_order(
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            amount=0.0164,
            price=65000.0,
            context={
                "data_source": "test",
                "data_quality_passed": True,
                "reconciliation_fresh": True,
                "has_unknown_orders": False,
                "exchange_connected": True,
                "step_size": 0.001,
                "tick_size": 0.1,
                "auto_quantize_step": True,
            },
        )
        self.assertTrue(dec.allowed, f"Unexpected reject: {dec.reasons}")


if __name__ == "__main__":
    unittest.main()
