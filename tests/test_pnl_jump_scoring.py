"""Offline unit & randomized tests for PnL Jump Diagnosis, Delay-aware Gates, and Turnover Bands."""
from __future__ import annotations

import math
import random
import unittest

from worldquant_brain.pnl_jump_scoring import (
    EvidenceSource,
    PnLJumpDiagnosis,
    PnLJumpInput,
    PnLJumpPolicy,
    TurnoverBand,
    classify_turnover,
    evaluate_pnl_jump,
)


def sample(**overrides) -> PnLJumpInput:
    data = dict(
        alpha_id="a",
        delay=1,
        sharpe=1.8,
        fitness=1.5,
        returns=0.12,
        turnover=0.25,
        margin=0.001,
        drawdown=0.07,
        self_correlation=0.3,
        max_instrument_weight=0.07,
        truncation=0.08,
        vector_evidence_source=EvidenceSource.BRAIN_OFFICIAL,
        alpha_vector_observed=True,
        nan_transition_rate=0.01,
        alpha_change_rate=0.10,
        market_beta=0.10,
        largest_year_pnl_share=0.30,
        field_coverage=0.8,
        backfilled_coverage=0.85,
        backfill_window_days=20,
        field_staleness_days=10,
        point_in_time_verified=True,
        cost_stress_completed=True,
        cost_stress_passed=True,
    )
    data.update(overrides)
    return PnLJumpInput(**data)


def actions(report) -> set[str]:
    return {a.action for a in report.actions}


class TestPnLJumpScoring(unittest.TestCase):
    def test_nan_transition_prefers_causal_backfill(self) -> None:
        r = evaluate_pnl_jump(sample(nan_transition_rate=0.051))
        self.assertIn(PnLJumpDiagnosis.NAN_TRANSITION, r.diagnoses)
        self.assertIn("TEST_CAUSAL_BACKFILL", actions(r))

    def test_nan_transition_exact_boundary_not_triggered(self) -> None:
        r = evaluate_pnl_jump(sample(nan_transition_rate=0.05))
        self.assertNotIn(PnLJumpDiagnosis.NAN_TRANSITION, r.diagnoses)

    def test_backfill_staleness_is_blocker_action(self) -> None:
        r = evaluate_pnl_jump(sample(backfill_window_days=90, field_staleness_days=61))
        self.assertIn("REDUCE_OR_REMOVE_BACKFILL", actions(r))
        self.assertIn(PnLJumpDiagnosis.BACKFILL_STALENESS, r.diagnoses)

    def test_rapid_change_recommends_decay_and_average(self) -> None:
        r = evaluate_pnl_jump(sample(alpha_change_rate=0.301))
        self.assertTrue({"TEST_DECAY", "TEST_SIGNAL_AVERAGE"}.issubset(actions(r)))

    def test_weight_exact_10_percent_triggers_and_below_does_not(self) -> None:
        r_hi = evaluate_pnl_jump(sample(max_instrument_weight=0.10))
        self.assertIn("TEST_TRUNCATION_RANK_WINSORIZE", actions(r_hi))
        r_ok = evaluate_pnl_jump(sample(max_instrument_weight=0.099999))
        self.assertNotIn("TEST_TRUNCATION_RANK_WINSORIZE", actions(r_ok))

    def test_market_beta_uses_absolute_exposure(self) -> None:
        self.assertIn("TEST_NEUTRALIZATION", actions(evaluate_pnl_jump(sample(market_beta=-0.31))))
        self.assertNotIn("TEST_NEUTRALIZATION", actions(evaluate_pnl_jump(sample(market_beta=0.30))))

    def test_single_year_40_percent_boundary(self) -> None:
        self.assertNotIn(
            PnLJumpDiagnosis.REGIME_FAILURE,
            evaluate_pnl_jump(sample(largest_year_pnl_share=0.40)).diagnoses,
        )
        self.assertIn(
            "RUN_REGIME_AND_YEAR_ATTRIBUTION",
            actions(evaluate_pnl_jump(sample(largest_year_pnl_share=0.400001))),
        )

    def test_unobserved_vector_is_explicitly_labeled(self) -> None:
        r = evaluate_pnl_jump(
            sample(alpha_vector_observed=False, vector_evidence_source=EvidenceSource.INFERRED_PROXY)
        )
        self.assertIn(PnLJumpDiagnosis.VECTOR_NOT_OBSERVED, r.diagnoses)
        self.assertIn("ALPHA_VECTOR_NOT_OFFICIALLY_OBSERVED", r.warnings)
        self.assertIn("USE_PROXIES_AND_LABEL_EVIDENCE", actions(r))

    def test_multi_signal_overfit_recommends_split(self) -> None:
        r = evaluate_pnl_jump(sample(primary_signal_count=2, confirmation_signal_count=2, condition_count=4))
        self.assertIn("SPLIT_INTO_ATOMIC_ALPHAS", actions(r))

    def test_turnover_bands(self) -> None:
        cases = [
            (-0.1, TurnoverBand.FAILED),
            (0.01, TurnoverBand.FAILED),
            (0.010001, TurnoverBand.BELOW_FITNESS_FLOOR),
            (0.124999, TurnoverBand.BELOW_FITNESS_FLOOR),
            (0.125, TurnoverBand.PREFERRED),
            (0.299999, TurnoverBand.PREFERRED),
            (0.30, TurnoverBand.ACCEPTABLE),
            (0.399999, TurnoverBand.ACCEPTABLE),
            (0.40, TurnoverBand.HIGH_BUT_ELIGIBLE),
            (0.699999, TurnoverBand.HIGH_BUT_ELIGIBLE),
            (0.70, TurnoverBand.FAILED),
        ]
        pol = PnLJumpPolicy()
        for turnover, band in cases:
            self.assertEqual(classify_turnover(turnover, pol), band, f"turnover={turnover}")

    def test_delay_aware_sharpe(self) -> None:
        cases = [(0, 2.0, False), (0, 2.000001, True), (1, 1.25, False), (1, 1.250001, True)]
        for delay, sharpe, passed in cases:
            r = evaluate_pnl_jump(sample(delay=delay, sharpe=sharpe))
            self.assertIs("SHARPE_NOT_PASSED" not in r.hard_blockers, passed, f"delay={delay}, sharpe={sharpe}")

    def test_turnover_above_30_requires_cost_stress(self) -> None:
        r = evaluate_pnl_jump(sample(turnover=0.31, cost_stress_completed=False))
        self.assertIn(PnLJumpDiagnosis.TRANSACTION_COST_RISK, r.diagnoses)
        self.assertIn("RUN_COST_STRESS", actions(r))

        r_ok = evaluate_pnl_jump(sample(turnover=0.299999, cost_stress_completed=False))
        self.assertNotIn(PnLJumpDiagnosis.TRANSACTION_COST_RISK, r_ok.diagnoses)

    def test_nonfinite_inputs_do_not_break_report(self) -> None:
        r = evaluate_pnl_jump(sample(nan_transition_rate=float("nan"), alpha_change_rate=float("inf")))
        self.assertTrue(math.isfinite(r.smoothness_risk_score))
        self.assertTrue(math.isfinite(r.improvement_score))

    def test_randomized_scores_are_finite_and_bounded(self) -> None:
        rng = random.Random(20261119)
        for _ in range(1000):
            r = evaluate_pnl_jump(
                sample(
                    sharpe=rng.uniform(-2, 5),
                    fitness=rng.uniform(-1, 4),
                    returns=rng.uniform(-0.5, 0.8),
                    turnover=rng.uniform(-0.1, 1),
                    margin=rng.uniform(-0.01, 0.01),
                    drawdown=rng.uniform(0, 1),
                    self_correlation=rng.uniform(0, 0.9999),
                    max_instrument_weight=rng.uniform(0, 1),
                    nan_transition_rate=rng.uniform(0, 1),
                    alpha_change_rate=rng.uniform(0, 1),
                    market_beta=rng.uniform(-2, 2),
                    largest_year_pnl_share=rng.uniform(0, 1),
                )
            )
            self.assertTrue(0 <= r.smoothness_risk_score <= 100)
            self.assertTrue(0 <= r.improvement_score <= 100)
            self.assertTrue(math.isfinite(r.smoothness_risk_score))
            self.assertTrue(math.isfinite(r.improvement_score))

    def test_higher_risk_does_not_lower_risk_score(self) -> None:
        base = evaluate_pnl_jump(
            sample(nan_transition_rate=0.01, alpha_change_rate=0.05, max_instrument_weight=0.03, market_beta=0.05, largest_year_pnl_share=0.1)
        )
        high = evaluate_pnl_jump(
            sample(nan_transition_rate=0.4, alpha_change_rate=0.8, max_instrument_weight=0.3, market_beta=1.0, largest_year_pnl_share=0.8)
        )
        self.assertGreaterEqual(high.smoothness_risk_score, base.smoothness_risk_score)


if __name__ == "__main__":
    unittest.main()
