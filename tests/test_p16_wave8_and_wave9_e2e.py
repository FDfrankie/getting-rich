"""End-to-end contract tests for P16 (Wave 8 + Wave 9 Alpha E2E Fixes)."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


class TestP16Wave8AndWave9E2E(unittest.TestCase):
    def test_w8_alpha_yield_booster_sparse_nan_on_and_word_boundary_close(self) -> None:
        from worldquant_brain.alpha_yield_booster import generate_smart_settings, precision_repair_near_miss

        cfg_fund = generate_smart_settings(domain="FUNDAMENTAL")
        self.assertEqual(cfg_fund["nanHandling"], "ON")
        cfg_pv = generate_smart_settings(domain="PV")
        self.assertEqual(cfg_pv["nanHandling"], "OFF")

        # Word boundary must NOT corrupt anl4_close_price
        repairs = precision_repair_near_miss(
            "rank(anl4_close_price)",
            {"decay": 4, "truncation": 0.08},
            sharpe=1.3,
            fitness=1.1,
            turnover=0.2,
            failed_checks=["SELF_CORRELATION", "CONCENTRATED_WEIGHT", "LOW_2Y_SHARPE"],
        )
        reasons = {r["reason"] for r in repairs}
        self.assertIn("CONCENTRATED_WEIGHT_RANK_BACKFILL_TRUNC05", reasons)
        self.assertIn("LOW_2Y_SHARPE_JUMP_DECAY_STABILIZE", reasons)
        self.assertNotIn("CORR_FIELD_SUBSTITUTION_VWAP", reasons)

        # Standalone close SHOULD be replaced with vwap
        repairs2 = precision_repair_near_miss(
            "rank(close / open)",
            {"decay": 4},
            sharpe=-1.30,
            fitness=-1.1,
            turnover=0.2,
            failed_checks=["SELF_CORRELATION"],
        )
        reasons2 = {r["reason"] for r in repairs2}
        self.assertIn("NEGATIVE_SHARPE_SIGN_FLIP", reasons2)
        self.assertIn("CORR_FIELD_SUBSTITUTION_VWAP", reasons2)
        vwap_expr = [r["expression"] for r in repairs2 if r["reason"] == "CORR_FIELD_SUBSTITUTION_VWAP"][0]
        self.assertEqual(vwap_expr, "rank(vwap / open)")

    def test_w8_two_stage_funnel_rejects_none_turnover_and_normalizes_pct(self) -> None:
        from worldquant_brain.two_stage_funnel import classify_verdict

        v_none, _ = classify_verdict({"sharpe": 1.5, "fitness": 1.2, "turnover": None, "checks_passed": True})
        self.assertEqual(v_none, "REJECTED")

        v_pct, _ = classify_verdict({"sharpe": 1.5, "fitness": 1.2, "turnover": 25.0, "margin": 0.002, "checks_passed": True})
        self.assertEqual(v_pct, "PASSED")

    def test_w8_multi_sim_client_204_refresh_clears_cookies_and_paginates_datasets(self) -> None:
        from worldquant_brain.multi_sim_client import _ensure_authenticated, fetch_available_datasets

        sess = MagicMock()
        sess.cookies = MagicMock()
        client = MagicMock()
        client.session = sess
        client.is_authenticated = True
        client.offline_mode = False
        client.BASE_URL = "https://api.worldquantbrain.com"
        client.authenticate.return_value = True
        client._multi_sim_last_auth_fail_ts = 0.0
        client._multi_sim_last_auth_ok_ts = 0.0

        ok = _ensure_authenticated(client, force_refresh=True)
        self.assertTrue(ok)
        sess.cookies.clear.assert_called_once()

        p1 = MagicMock(status_code=200, content=b"{}")
        p1.json.return_value = {"results": [{"id": f"ds_{i}", "valueScore": 2, "alphaCount": 100} for i in range(50)]}
        p2 = MagicMock(status_code=200, content=b"{}")
        p2.json.return_value = {"results": [{"id": "ds_50", "valueScore": 3, "alphaCount": 50}, {"id": "ds_51", "valueScore": 3, "alphaCount": 60}]}
        sess.request.side_effect = [p1, p2]
        res = fetch_available_datasets(client, max_datasets=60)
        self.assertEqual(len(res), 52)

    def test_w8_candidate_utility_strict_eligible_and_selector_orthogonal_top_k(self) -> None:
        from worldquant_brain.candidate_utility import evaluate_candidate
        from worldquant_brain.daily_candidate_selector import select_daily_candidates

        bad = evaluate_candidate({"alpha_id": "A_BAD", "sharpe": 1.6, "fitness": 1.2, "turnover": 0.2, "self_correlation": 0.75})
        self.assertFalse(bad.eligible)

        rng = np.random.default_rng(7)
        s1 = rng.normal(0, 1, 50)
        s2 = s1 * 0.98 + rng.normal(0, 0.05, 50)
        s3 = rng.normal(0, 1, 50)

        cands = [
            {"alpha_id": "A1", "expression": "rank(fnd6_ebitda / cap)", "dataset_id": "fundamental6", "sharpe": 1.8, "fitness": 1.4, "turnover": 0.15, "self_correlation": 0.2, "daily_pnl_diff": s1, "is_check_status": "PASS"},
            {"alpha_id": "A2_CORR", "expression": "rank(fnd6_ebitda / assets)", "dataset_id": "fundamental6", "sharpe": 1.75, "fitness": 1.35, "turnover": 0.16, "self_correlation": 0.2, "daily_pnl_diff": s2, "is_check_status": "PASS"},
            {"alpha_id": "A3_ORTH", "expression": "rank(opt8_skew / close)", "dataset_id": "option8", "sharpe": 1.5, "fitness": 1.15, "turnover": 0.20, "self_correlation": 0.2, "daily_pnl_diff": s3, "is_check_status": "PASS"},
        ]
        batch = select_daily_candidates(cands, top_k=2)
        chosen_ids = [c.alpha_id for c in batch.selected]
        self.assertIn("A1", chosen_ids)
        self.assertIn("A3_ORTH", chosen_ids)
        self.assertNotIn("A2_CORR", chosen_ids)

    def test_w9_correlation_evidence_delay0_and_brain_client_ny_quota_day(self) -> None:
        from worldquant_brain.correlation_evidence import evaluate_parent_child_repair

        ok_d0, reason_d0 = evaluate_parent_child_repair(
            parent_child_pnl_corr=0.40,
            official_corr=True,
            target_gate_resolved=True,
            parent_sharpe=1.90,
            child_sharpe=1.80,
            performance_improvement_ratio=0.15,
            delay=0,
        )
        self.assertFalse(ok_d0)
        self.assertEqual(reason_d0, "SHARPE_SEVERELY_DEGRADED")

        bc_src = Path("worldquant_brain/brain_client.py").read_text(encoding="utf-8")
        self.assertIn("America/New_York", bc_src)
        self.assertIn("submitted_skels", bc_src)
        self.assertIn("fnd6_", bc_src)
        self.assertNotIn("find_duplicate(expression, [])", bc_src)


if __name__ == "__main__":
    unittest.main()
