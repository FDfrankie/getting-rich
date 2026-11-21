"""Contract tests for P22: Alpha Yield & Turnover-Aware Stage-2 Refinement."""
from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel, classify_verdict


class TestP22AlphaYieldAndRefinement(unittest.TestCase):
    def test_turnover_aware_near_miss_rescues_high_to_and_keeps_low_to_strict(self) -> None:
        # High-turnover (0.45) signal with Sharpe=1.08, Fitness=0.75 -> rescuable via decay -> NEAR_MISS
        high_to = {
            "sharpe": 1.08,
            "fitness": 0.75,
            "turnover": 0.45,
            "margin": 0.0009,
            "checks_passed": False,
            "failed_checks": ["LOW_SHARPE", "LOW_FITNESS"],
        }
        v_hi, _ = classify_verdict(high_to)
        self.assertEqual(v_hi, "NEAR_MISS")

        # Low-turnover (0.12) signal with same Sharpe=1.08, Fitness=0.75 -> little decay leverage -> REJECTED
        low_to = dict(high_to, turnover=0.12)
        v_lo, _ = classify_verdict(low_to)
        self.assertEqual(v_lo, "REJECTED")

    def test_negative_sharpe_seed_enqueues_pure_sign_flip_as_variant_1(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            f = TwoStageAlphaFunnel(Path(td) / "vault.db")
            res = {
                "expression": "rank(ts_backfill(fnd6_ebitda, 63))",
                "settings": dict(f.base_settings),
                "dataset_id": "fundamental6",
                "probe": "rank_bf",
                "sharpe": -1.45,
                "fitness": -1.10,
                "turnover": 0.22,
                "margin": -0.0016,
                "checks_passed": False,
                "failed_checks": ["LOW_SHARPE", "LOW_FITNESS"],
            }
            added = f._enqueue_immediate_refinement(res, "LOW_SHARPE;LOW_FITNESS")
            self.assertGreaterEqual(added, 1)
            q = f.next_stage2_batch(batch_size=4)
            self.assertEqual(q[0]["expression"], "-(rank(ts_backfill(fnd6_ebitda, 63)))")
            self.assertEqual(q[0]["settings"]["decay"], f.base_settings["decay"])
            gc.collect()

    def test_dataset_aware_backfill_nan_handling_and_cap_norm_skip(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            f = TwoStageAlphaFunnel(Path(td) / "vault.db")
            fnd_batch = f.next_batch({"fundamental6": ["fnd6_ebitda"]}, batch_size=25)
            self.assertTrue(len(fnd_batch) > 0)
            self.assertTrue(all(it["settings"].get("nanHandling") == "ON" for it in fnd_batch))
            self.assertTrue(any("ts_backfill(fnd6_ebitda, 63)" in it["expression"] for it in fnd_batch))

            news_batch = f.next_batch({"news18": ["nws18_sent"]}, batch_size=25)
            self.assertTrue(all(it["probe"] != "cap_norm_gauss" for it in news_batch))
            self.assertTrue(all(it["settings"].get("nanHandling") == "ON" for it in news_batch))
            gc.collect()


if __name__ == "__main__":
    unittest.main()
