"""Tests covering the 20 Alpha Improvement Priority Matrix requirements, TrialRegistry/DSR, and FrequentSubtreeAvoidance."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worldquant_brain.alpha_research_engines import FormulaMCTSSearch
from worldquant_brain.improvement_priority import (
    AlphaDisposition,
    PriorityScoreInputs,
    PriorityTier,
    compute_improvement_priority_score,
    evaluate_improvement_priority,
)
from worldquant_brain.optimization_policy import OptimizationObjective
from worldquant_brain.trial_registry_dsr import TrialRegistry, deflated_sharpe_ratio


def _base_record(**overrides) -> dict:
    rec = {
        "brain_alpha_id": "ALPHA_1",
        "official_simulation_id": "SIM_1",
        "metrics_source": "WORLDQUANT_BRAIN",
        "sharpe": 1.40,
        "fitness": 1.15,
        "returns": 0.12,
        "net_returns": 0.11,
        "turnover": 0.22,
        "margin": 0.0015,
        "drawdown": 0.08,
        "self_correlation": 0.30,
        "novelty": 0.75,
        "weight_test_passed": True,
        "sub_universe_passed": True,
        "bias_test_passed": True,
        "submit_ready": True,
    }
    rec.update(overrides)
    return rec


class TestAlphaImprovementPriorityAndDSR(unittest.TestCase):
    def test_01_bias_failure_is_p0_terminal_no_variants(self) -> None:
        p = evaluate_improvement_priority(_base_record(bias_test_passed=False))
        self.assertEqual(p.priority, PriorityTier.P0)
        self.assertFalse(p.allow_simulation_queue)
        self.assertIn("PARAMETER_VARIANT", p.forbidden_actions)

    def test_02_self_corr_boundary_07000_vs_06999(self) -> None:
        p_block = evaluate_improvement_priority(_base_record(self_correlation=0.7000))
        self.assertEqual(p_block.priority, PriorityTier.P0)
        self.assertFalse(p_block.allow_submission)
        self.assertIn("WINDOW_ONLY_TWEAK", p_block.forbidden_actions)

        p_pass = evaluate_improvement_priority(_base_record(self_correlation=0.6999, novelty=0.60))
        self.assertNotEqual(p_pass.priority, PriorityTier.P0)

    def test_03_productive_vs_wasteful_turnover(self) -> None:
        p_prod = evaluate_improvement_priority(
            _base_record(sharpe=1.18, turnover=0.60, net_returns=0.18, previous_net_returns=0.08, previous_turnover=0.20),
            objective=OptimizationObjective.RETURN,
        )
        self.assertEqual(p_prod.recommended_action, "PRESERVE_TURNOVER_AND_STRESS_COST")

        p_waste = evaluate_improvement_priority(
            _base_record(sharpe=1.18, turnover=0.60, net_returns=0.085, previous_net_returns=0.08, previous_turnover=0.20),
            objective=OptimizationObjective.BALANCED,
        )
        self.assertEqual(p_waste.recommended_action, "APPLY_HUMP")

    def test_04_low_return_low_vol_vs_high_return_high_vol(self) -> None:
        p_low = evaluate_improvement_priority(_base_record(sharpe=0.90, fitness=0.70, returns=0.03, net_returns=0.03, volatility=0.05))
        self.assertEqual(p_low.recommended_action, "ADD_ANALYST_NEWS_OR_FUNDAMENTAL")

        p_hv = evaluate_improvement_priority(_base_record(sharpe=0.95, fitness=0.75, returns=0.12, net_returns=0.10, volatility=0.25))
        self.assertEqual(p_hv.recommended_action, "TEST_NEUTRALIZATION")

    def test_05_weight_and_sub_universe_block_before_sharpe_and_robustness(self) -> None:
        pw = evaluate_improvement_priority(_base_record(weight_test_passed=False, sharpe=1.10))
        self.assertEqual(pw.priority, PriorityTier.P0)
        self.assertEqual(pw.primary_failure_gate, "WEIGHT_TEST_FAILED")

        ps = evaluate_improvement_priority(_base_record(sub_universe_passed=False, sharpe=1.60))
        self.assertEqual(ps.priority, PriorityTier.P0)
        self.assertFalse(ps.allow_robustness_promotion)

    def test_06_repair_budget_limits_and_stop_after_two_no_gain(self) -> None:
        p1_exhausted = evaluate_improvement_priority(_base_record(sharpe=1.15, fitness=0.90), repairs_used=3)
        self.assertEqual(p1_exhausted.remaining_budget, 0)
        self.assertFalse(p1_exhausted.allow_simulation_queue)

        p_stop = evaluate_improvement_priority(_base_record(sharpe=1.15), consecutive_no_gain=2)
        self.assertEqual(p_stop.priority, PriorityTier.P6)
        self.assertEqual(p_stop.disposition, AlphaDisposition.ARCHIVE)
        self.assertFalse(p_stop.allow_simulation_queue)

    def test_07_good_low_novelty_vs_high_novelty(self) -> None:
        p_low_nov = evaluate_improvement_priority(_base_record(novelty=0.20, self_correlation=0.55))
        self.assertEqual(p_low_nov.recommended_action, "EVALUATE_PORTFOLIO_MARGINAL_UTILITY")
        self.assertFalse(p_low_nov.allow_submission)

        p_hi_nov = evaluate_improvement_priority(_base_record(novelty=0.85, self_correlation=0.25))
        self.assertEqual(p_hi_nov.priority, PriorityTier.P5)
        self.assertTrue(p_hi_nov.allow_robustness_promotion)

    def test_08_yield_per_minute_accounts_for_3_min_cost(self) -> None:
        s3, y3 = compute_improvement_priority_score(PriorityScoreInputs(expected_minutes=3.0))
        s6, y6 = compute_improvement_priority_score(PriorityScoreInputs(expected_minutes=6.0))
        self.assertEqual(s3, s6)
        self.assertAlmostEqual(y3, 2.0 * y6, places=3)

    def test_09_trial_registry_and_deflated_sharpe_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reg = TrialRegistry(Path(td) / "trials.db")
            for i in range(50):
                reg.record_trial(expression_hash=f"eh_{i}", settings_hash="sh", accepted=(i == 0), metrics={"sharpe": 1.1})
            self.assertEqual(reg.count_trials(), 50)
        dsr_1 = deflated_sharpe_ratio(1.5, trial_count=1, n_observations=252)
        dsr_1000 = deflated_sharpe_ratio(1.5, trial_count=1000, n_observations=252)
        self.assertGreater(dsr_1.dsr_probability, dsr_1000.dsr_probability)

    def test_10_mcts_frequent_subtree_avoidance(self) -> None:
        mcts = FormulaMCTSSearch(subtree_avoidance_threshold=3)
        b_stale = ("FUND", "WINSORIZE_HUMP", "subindustry")
        for _ in range(5):
            mcts.backpropagate(b_stale, 0.20)
        self.assertGreater(mcts.subtree_penalty(b_stale), 0.0)


if __name__ == "__main__":
    unittest.main()
