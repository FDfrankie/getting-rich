"""Regression tests for deep Alpha pipeline fixes across funnel, multi-sim, templates, yield booster, and job queue."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from worldquant_brain.alpha_templates.analyst_sentiment_alpha import SENTIMENT_ALPHAS
from worldquant_brain.alpha_templates.fundamental_alpha import FUNDAMENTAL_ALPHAS
from worldquant_brain.alpha_yield_booster import precision_repair_near_miss
from worldquant_brain.brain_job_queue import BrainJobQueue, BrainJobStatus
from worldquant_brain.multi_sim_client import simulate_multi_alpha
from worldquant_brain.two_stage_funnel import classify_verdict


class TestAlphaDeepPipelineFixes(unittest.TestCase):
    def test_negative_sharpe_with_negative_fitness_is_rescued_as_near_miss(self) -> None:
        # Exact historical case: ts_rank(close, 10) -> sharpe=-1.48, fitness=-0.87, margin=-0.000686
        verdict, reason = classify_verdict(
            {
                "sharpe": -1.48,
                "fitness": -0.87,
                "turnover": 0.397,
                "margin": -0.000686,
                "checks_passed": False,
                "failed_checks": ["LOW_SHARPE", "LOW_FITNESS"],
                "settings": {"delay": 1},
            },
            near_miss_sharpe=0.90,
        )
        self.assertEqual(verdict, "NEAR_MISS", f"Expected NEAR_MISS for sign-flip rescue, got {verdict}: {reason}")

    def test_weak_negative_sharpe_still_rejected(self) -> None:
        verdict, _ = classify_verdict(
            {
                "sharpe": -0.40,
                "fitness": -0.20,
                "turnover": 0.30,
                "margin": -0.0002,
                "checks_passed": False,
                "failed_checks": ["LOW_SHARPE"],
                "settings": {"delay": 1},
            },
            near_miss_sharpe=0.90,
        )
        self.assertEqual(verdict, "REJECTED")

    def test_multi_sim_403_sets_session_circuit_breaker(self) -> None:
        client = MagicMock()
        client.offline_mode = False
        client.is_authenticated = True
        client.base_url = "https://api.worldquantbrain.com"
        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.headers = {}
        resp_403.text = "Forbidden"
        client.session.request.return_value = resp_403
        client.simulate_alpha.return_value = {"success": True, "sharpe": 1.3, "fitness": 1.1, "turnover": 0.2}

        items = [
            {"expression": "rank(close - open)", "settings": {"delay": 1}},
            {"expression": "rank(high - low)", "settings": {"delay": 1}},
        ]
        # 1st batch hits 403 and sets _multi_sim_forbidden=True
        out1 = simulate_multi_alpha(client, items, fallback_item_delay=0.0)
        self.assertEqual(len(out1), 2)
        self.assertTrue(getattr(client, "_multi_sim_forbidden", False))
        calls_after_first = client.session.request.call_count

        # 2nd batch skips multi-sim POST entirely (zero additional session.request calls)
        out2 = simulate_multi_alpha(client, items, fallback_item_delay=0.0)
        self.assertEqual(len(out2), 2)
        self.assertEqual(client.session.request.call_count, calls_after_first)

    def test_alpha_templates_have_no_invalid_fields(self) -> None:
        all_exprs = [a["expression"] for a in FUNDAMENTAL_ALPHAS + SENTIMENT_ALPHAS]
        for expr in all_exprs:
            self.assertNotIn("operating_cashflow", expr)
            self.assertNotIn("eps_consensus_mean", expr)

    def test_yield_booster_avoids_blind_rank_on_sentiment_buzz(self) -> None:
        repairs = precision_repair_near_miss(
            "group_neutralize(-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), 7), subindustry)",
            {"universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.05},
            sharpe=1.50,
            fitness=0.80,
            turnover=0.10,
            failed_checks=["LOW_FITNESS"],
        )
        fit_repairs = [r for r in repairs if r["name"] == "Repair_Fitness_BoostReturns"]
        self.assertEqual(len(fit_repairs), 1)
        self.assertNotIn("signed_power(rank(", fit_repairs[0]["expression"])

    def test_brain_job_queue_rejects_invalid_fields_before_queuing(self) -> None:
        client = MagicMock()
        client.offline_mode = True
        q = BrainJobQueue(client, db_path=":memory:")
        job, created = q.create_job(
            "hump(group_neutralize(winsorize(rank(ts_zscore(opt6_slopeavg1m, 60)), std=4), subindustry), hump=0.01)",
            {"universe": "TOP3000", "delay": 1},
        )
        self.assertFalse(created)
        self.assertEqual(job.status, BrainJobStatus.FAILED)
        self.assertIn("PRE_QUEUE_REJECTED:INVALID_FIELD:opt6_slopeavg1m", job.safe_error or "")
        self.assertEqual(q.queue_depth(), 0)


if __name__ == "__main__":
    unittest.main()
