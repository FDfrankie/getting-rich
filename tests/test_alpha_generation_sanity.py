"""Tests for alpha_miner.py & alpha_blueprint.py hotfix: invalid field quarantine, degenerate guards, and structural dedup."""
from __future__ import annotations

import unittest

from worldquant_brain.alpha_blueprint import builtin_blueprints, fundamental_quality_blueprint
from worldquant_brain.alpha_miner import (
    ContinuousAlphaMiner,
    deduplicate_by_structure,
    validate_candidate_expression,
)


class TestAlphaGenerationSanity(unittest.TestCase):
    def test_rejects_known_invalid_brain_fields(self) -> None:
        for bad in (
            "hump(group_neutralize(winsorize(rank(ts_zscore(opt6_slopeavg1m, 60)) * rank(opt6_fcstr2imp), std=4), subindustry), hump=0.01)",
            "group_neutralize(rank(operating_cashflow / (cap + debt + 0.001)), sector)",
            "ts_decay_linear(rank(eps_consensus_mean - ts_delay(eps_consensus_mean, 20)), 5)",
        ):
            ok, reason = validate_candidate_expression(bad)
            self.assertFalse(ok)
            self.assertTrue(reason.startswith("INVALID_FIELD:"))

    def test_rejects_degenerate_self_correlation_and_window_one(self) -> None:
        ok_self, reason_self = validate_candidate_expression("-1 * ts_std_dev(ts_corr(vwap, vwap, 10), 10)")
        self.assertFalse(ok_self)
        self.assertEqual(reason_self, "DEGENERATE_SELF_CORR:vwap")

        ok_win1, reason_win1 = validate_candidate_expression("-1 * ts_delta(ts_corr(open, returns, 1), 1)")
        self.assertFalse(ok_win1)
        self.assertEqual(reason_win1, "DEGENERATE_CORR_WINDOW:1")

    def test_generated_expressions_always_pass_sanity_validator(self) -> None:
        miner = ContinuousAlphaMiner()
        for _ in range(150):
            cand = miner.generate_candidate_expression()
            ok, reason = validate_candidate_expression(cand["expression"])
            self.assertTrue(ok, f"Bad generated expr {cand['expression']}: {reason}")

    def test_default_blueprints_replaced_invalid_fields(self) -> None:
        fq = fundamental_quality_blueprint()
        self.assertIn("cashflow_op", fq.parameter_space["cf_field"])
        self.assertNotIn("operating_cashflow", fq.parameter_space["cf_field"])
        self.assertGreaterEqual(len(builtin_blueprints()), 5)

    def test_deduplicate_by_structure_collapses_decay_clones(self) -> None:
        clones = [
            "group_neutralize(rank(debt / (abs(equity) + 0.001)), industry)",
            "ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)",
            "ts_decay_linear(ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 3), 8)",
            "group_neutralize(rank(close - open), subindustry)",
        ]
        deduped = deduplicate_by_structure(clones)
        self.assertEqual(len(deduped), 2)


if __name__ == "__main__":
    unittest.main()
