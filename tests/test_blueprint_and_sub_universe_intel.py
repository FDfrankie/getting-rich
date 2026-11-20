from __future__ import annotations

import unittest

from worldquant_brain.alpha_blueprint import (
    DEFAULT_BRAIN_FIELD_CATALOG,
    builtin_blueprints,
    generate_controlled_candidates,
)
from worldquant_brain.sub_universe_classifier import (
    compute_sub_universe_sharpe_floor,
    diagnose_turnover_and_cost_stress,
)


class TestBlueprintAndSubUniverseIntel(unittest.TestCase):
    def test_all_builtin_blueprints_generate_valid_candidates(self) -> None:
        bps = builtin_blueprints()
        self.assertGreaterEqual(len(bps), 5)
        ids = {bp.blueprint_id for bp in bps}
        self.assertIn("ANALYST_REVISION_001", ids)
        self.assertIn("FUNDAMENTAL_QUALITY_001", ids)
        self.assertIn("SENTIMENT_ATTENTION_001", ids)
        self.assertIn("OPTIONS_SKEW_001", ids)
        self.assertIn("LIQUIDITY_REVERSION_001", ids)

        for bp in bps:
            errs = bp.validate()
            self.assertEqual(errs, [], msg=f"{bp.blueprint_id} failed validation: {errs}")
            cands = generate_controlled_candidates(bp, catalog=DEFAULT_BRAIN_FIELD_CATALOG)
            self.assertGreater(len(cands), 0, msg=f"{bp.blueprint_id} generated 0 candidates")
            for c in cands:
                self.assertIn("expression", c)
                self.assertTrue(c["expression"].strip())

    def test_compute_sub_universe_sharpe_floor(self) -> None:
        # 0.75 * sqrt(1000 / 3000) * 1.50 ≈ 0.6495
        floor = compute_sub_universe_sharpe_floor(1.50, alpha_universe_size=3000, sub_universe_size=1000)
        self.assertAlmostEqual(floor, 0.6495, places=4)
        with self.assertRaises(ValueError):
            compute_sub_universe_sharpe_floor(1.50, alpha_universe_size=0, sub_universe_size=1000)

    def test_diagnose_turnover_and_cost_stress(self) -> None:
        healthy = diagnose_turnover_and_cost_stress(sharpe=1.50, fitness=1.20, turnover=0.12, margin=0.0018, delay=1)
        self.assertTrue(healthy["healthy"])
        self.assertTrue(healthy["at_fitness_sweet_spot"])
        self.assertEqual(healthy["issues"], [])

        high_to = diagnose_turnover_and_cost_stress(sharpe=1.60, fitness=0.80, turnover=0.75, margin=8.0, delay=1)
        self.assertFalse(high_to["healthy"])
        self.assertIn("TURNOVER_ABOVE_70PCT_LIMIT", high_to["issues"])
        self.assertIn("LOW_MARGIN_COST_STRESS_RISK", high_to["issues"])
        self.assertIn("ADD_HUMP", high_to["recommended_repairs"])

        low_to = diagnose_turnover_and_cost_stress(sharpe=1.30, fitness=1.05, turnover=0.005, margin=0.002, delay=1)
        self.assertFalse(low_to["healthy"])
        self.assertIn("TURNOVER_TOO_LOW", low_to["issues"])


if __name__ == "__main__":
    unittest.main()
