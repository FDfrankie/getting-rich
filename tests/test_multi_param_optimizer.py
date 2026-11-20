"""Unit tests for Multi-Parameter Adaptive Alpha Optimizer, Winner-Only Pool & Historical Rescue Scanner."""
from __future__ import annotations

import unittest
from pathlib import Path

from worldquant_brain.multi_param_optimizer import (
    PoolDisposition,
    SearchMethod,
    WinnerOnlyFamilyPool,
    extract_tunable_parameters,
    observe_parameter_delta,
    propose_multi_param_tuning_candidates,
    scan_rescuable_historical_failures,
    shrink_ternary_bracket,
    ternary_search_points,
)


class TestMultiParamOptimizer(unittest.TestCase):
    def test_extracts_all_formula_and_setting_parameters_skipping_epsilons(self) -> None:
        expr = "hump(group_neutralize(winsorize(signed_power(ts_zscore(close / (vwap + 0.0001), 20), 0.75), std=3.5), subindustry), hump=0.01)"
        params = extract_tunable_parameters(expr, {"decay": 4, "truncation": 0.05})
        roles = {p.role for p in params}
        self.assertEqual(roles, {"HUMP", "STD", "WINDOW", "EXPONENT", "DECAY", "TRUNCATION"})
        # Ensure 0.0001 epsilon was NOT extracted
        self.assertFalse(any(abs(p.current - 0.0001) < 1e-6 for p in params))

    def test_ternary_search_shrinks_bracket_toward_higher_score(self) -> None:
        m1, m2 = ternary_search_points(10.0, 40.0, is_int=True)
        self.assertEqual((m1, m2), (20.0, 30.0))
        lo, hi = shrink_ternary_bracket(10.0, 40.0, m1, m2, score_m1=1.20, score_m2=1.55)
        self.assertEqual((lo, hi), (20.0, 40.0))

    def test_coordinate_ascent_reverses_when_step_hurts_and_continues_when_improved(self) -> None:
        expr = "group_neutralize(ts_zscore(close, 20), industry)"
        params = extract_tunable_parameters(expr, {"decay": 4})
        w_param = next(p for p in params if p.role == "WINDOW")

        obs_good = observe_parameter_delta(
            w_param,
            before_metrics={"sharpe": 1.30, "fitness": 1.05, "turnover": 0.22},
            after_metrics={"sharpe": 1.55, "fitness": 1.25, "turnover": 0.20},
            after_value=30.0,
        )
        self.assertTrue(obs_good.improved_score)
        self.assertEqual(obs_good.recommended_direction, "UPPER")

        obs_bad = observe_parameter_delta(
            w_param,
            before_metrics={"sharpe": 1.30, "fitness": 1.05, "turnover": 0.22},
            after_metrics={"sharpe": 1.10, "fitness": 0.85, "turnover": 0.25},
            after_value=30.0,
        )
        self.assertFalse(obs_bad.improved_score)
        self.assertEqual(obs_bad.recommended_direction, "LOWER")

        cands = propose_multi_param_tuning_candidates(
            expr,
            {"decay": 4},
            base_metrics={"sharpe": 1.30, "fitness": 1.05, "turnover": 0.22},
            prior_observations=[obs_bad],
            max_candidates=4,
        )
        coord = [c for c in cands if c["search_method"] == SearchMethod.COORDINATE_ASCENT.value]
        self.assertEqual(len(coord), 1)
        self.assertLess(coord[0]["after_value"], 20)

    def test_winner_only_family_pool_demotes_older_passing_variants_to_do_not_submit(self) -> None:
        pool = WinnerOnlyFamilyPool(max_polish_trials_per_family=6)
        r1 = pool.register_result(
            alpha_id="A1",
            expression="group_neutralize(ts_zscore(close, 20), industry)",
            settings={"decay": 4},
            metrics={"sharpe": 1.35, "fitness": 1.10, "turnover": 0.22, "margin": 0.0015},
            passed_official=True,
        )
        self.assertEqual(r1.disposition, PoolDisposition.SUBMITTABLE_BEST)

        # Polished variant from SAME family scores higher -> becomes SUBMITTABLE_BEST, A1 demoted!
        r2 = pool.register_result(
            alpha_id="A2",
            expression="group_neutralize(ts_zscore(close, 30), industry)",
            settings={"decay": 6},
            metrics={"sharpe": 1.72, "fitness": 1.48, "turnover": 0.18, "margin": 0.0022},
            passed_official=True,
        )
        self.assertEqual(r2.disposition, PoolDisposition.SUBMITTABLE_BEST)
        self.assertEqual(pool.records["A1"].disposition, PoolDisposition.PASSED_DO_NOT_SUBMIT)

        # Weaker passing variant -> absorbed directly into PASSED_DO_NOT_SUBMIT
        r3 = pool.register_result(
            alpha_id="A3",
            expression="group_neutralize(ts_zscore(close, 15), industry)",
            settings={"decay": 2},
            metrics={"sharpe": 1.28, "fitness": 1.02, "turnover": 0.28, "margin": 0.0012},
            passed_official=True,
        )
        self.assertEqual(r3.disposition, PoolDisposition.PASSED_DO_NOT_SUBMIT)
        self.assertEqual([r.alpha_id for r in pool.submittable_best()], ["A2"])
        self.assertEqual({r.alpha_id for r in pool.passed_do_not_submit()}, {"A1", "A3"})


if __name__ == "__main__":
    unittest.main()
