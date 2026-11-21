"""Unit tests for P7 Stage 2 Budget & Interleaving Bundle."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel


class TestP7Stage2Budget(unittest.TestCase):
    def test_default_seed_sharpe_threshold_raised_to_1_05(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"))
            self.assertAlmostEqual(funnel.seed_sharpe_threshold, 1.05, places=4)
            self.assertEqual(funnel.max_stage2_variants_per_seed, 5)
            self.assertEqual(funnel.stage1_to_stage2_ratio, 2)

    def test_stage2_caps_variants_per_seed_and_prioritizes_higher_sharpe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"))
            added_low = funnel._enqueue_immediate_refinement(
                {
                    "expression": "rank(open)",
                    "sharpe": 1.06,
                    "fitness": 0.9,
                    "turnover": 0.25,
                    "dataset_id": "model16",
                    "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                },
                reason="NEAR_MISS_SHARPE",
            )
            added_high = funnel._enqueue_immediate_refinement(
                {
                    "expression": "rank(close)",
                    "sharpe": 1.22,
                    "fitness": 1.0,
                    "turnover": 0.25,
                    "dataset_id": "model16",
                    "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                },
                reason="NEAR_MISS_SHARPE",
            )
            self.assertLessEqual(added_low, 5)
            self.assertLessEqual(added_high, 5)
            # Highest-Sharpe seed (1.22) must be sorted to the front of the Stage 2 queue
            self.assertAlmostEqual(funnel._session_refinement_queue[0]["_priority_sharpe"], 1.22, places=4)

    def test_next_batch_interleaves_stage1_and_stage2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"), stage1_to_stage2_ratio=2)
            # Pre-seed Stage 2 queue
            funnel._enqueue_immediate_refinement(
                {
                    "expression": "rank(close)",
                    "sharpe": 1.20,
                    "fitness": 0.95,
                    "turnover": 0.20,
                    "dataset_id": "model16",
                    "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                },
                reason="NEAR_MISS_SHARPE",
            )
            ds_fields = {"model16": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]}

            b1 = funnel.next_batch(ds_fields, batch_size=2)
            b2 = funnel.next_batch(ds_fields, batch_size=2)
            b3 = funnel.next_batch(ds_fields, batch_size=2)
            b4 = funnel.next_batch(ds_fields, batch_size=2)

            self.assertEqual(b1[0]["stage"], "STAGE1")
            self.assertEqual(b2[0]["stage"], "STAGE1")
            self.assertEqual(b3[0]["stage"], "STAGE2")
            self.assertEqual(b4[0]["stage"], "STAGE1")


if __name__ == "__main__":
    unittest.main()
