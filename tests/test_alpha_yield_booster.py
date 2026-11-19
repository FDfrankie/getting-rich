"""Offline unit tests for Alpha Yield Booster (combos, universe rotation, gate-specific repair)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from worldquant_brain.alpha_miner import ContinuousAlphaMiner, extract_expression_skeleton
from worldquant_brain.alpha_yield_booster import (
    generate_orthogonal_combo_candidate,
    generate_smart_settings,
    precision_repair_near_miss,
)


class TestAlphaYieldBooster(unittest.TestCase):
    def test_orthogonal_combo_diversity(self) -> None:
        skels = set()
        for _ in range(40):
            c = generate_orthogonal_combo_candidate()
            self.assertIn("expression", c)
            self.assertIn("COMBO_", c["domain"])
            skels.add(extract_expression_skeleton(c["expression"]))
        self.assertGreaterEqual(len(skels), 10)

    def test_smart_settings_rotates_universes(self) -> None:
        universes = {generate_smart_settings("COMBO_FUND")["universe"] for _ in range(40)}
        self.assertTrue({"TOP3000", "TOP1000", "TOP500"} & universes)
        self.assertGreaterEqual(len(universes), 2)

    def test_smart_settings_iqc_delay0_and_test_period(self) -> None:
        with patch.dict(
            os.environ,
            {
                "IQC_ENABLE_DELAY0": "true",
                "IQC_USE_TEST_PERIOD": "true",
                "IQC_TEST_PERIOD": "P1Y",
            },
            clear=False,
        ):
            delays = {generate_smart_settings("COMBO_FUND")["delay"] for _ in range(30)}
            self.assertTrue(delays.issubset({0, 1}))
            self.assertIn(0, delays)
            cfg = generate_smart_settings("COMBO_FUND")
            self.assertEqual(cfg.get("testPeriod"), "P1Y")

    def test_precision_repair_handles_specific_gates(self) -> None:
        repairs = precision_repair_near_miss(
            "rank(close - open)",
            {"universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 4},
            sharpe=1.05,
            fitness=0.80,
            turnover=0.55,
            margin=0.0007,
            failed_checks=["LOW_MARGIN", "HIGH_TURNOVER", "LOW_SHARPE", "LOW_FITNESS", "SELF_CORRELATION"],
        )
        names = [r["name"] for r in repairs]
        self.assertTrue(any("LowMargin" in n for n in names))
        self.assertTrue(any("HighTurnover" in n or "WastefulTurnover" in n for n in names))
        self.assertTrue(any("Universe" in n for n in names))
        self.assertTrue(any("Fitness" in n for n in names))
        self.assertTrue(any("Corr" in n for n in names))

    def test_miner_integrates_combo_and_smart_settings(self) -> None:
        miner = ContinuousAlphaMiner()
        s = miner.generate_candidate_settings("COMBO_FUND_QUALITY_ANALYST_REV")
        self.assertIn(s["universe"], {"TOP3000", "TOP1000", "TOP500", "TOPSP500"})
        cand = miner.think_and_select_best_candidate(batch_size=6)
        self.assertTrue(bool(cand.get("expression")))


if __name__ == "__main__":
    unittest.main()
