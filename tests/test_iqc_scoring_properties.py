"""Deterministic seeded randomized property-based tests for IQC 2026 scoring (no external Hypothesis dep)."""
from __future__ import annotations

import json
import math
import random
import unittest

from worldquant_brain.iqc_scoring import (
    IQCCandidateDecision,
    IQCCandidateInput,
    evaluate_iqc_candidate,
    plan_daily_submissions,
)

SEED = 20261119
TRIALS = 80


def _base(**overrides) -> IQCCandidateInput:
    d = dict(
        alpha_id="prop-1",
        expression="rank(close)",
        region="USA",
        universe="TOP1000",
        delay=1,
        dataset_category="ANALYST",
        idea_family="FAM_PROP",
        metrics_source="WORLDQUANT_BRAIN",
        official_simulation_id="sim-p",
        brain_alpha_id="alpha-p",
        sharpe=1.80,
        fitness=1.60,
        returns=0.12,
        turnover=0.20,
        margin=0.0012,
        drawdown=0.05,
        self_correlation=0.30,
        max_instrument_weight=0.05,
        bias_test_passed=True,
        weight_test_passed=True,
        sub_universe_passed=True,
        duplicate_or_public=False,
        parameter_only_variant=False,
        test_period_contaminated=False,
        already_submitted=False,
        submit_ready=True,
        temporal_stability=0.80,
        parameter_stability=0.80,
        universe_stability=0.80,
        regime_stability=0.80,
        dsr_probability=0.95,
        pbo=0.10,
        cpcv_positive_ratio=0.75,
        delta_merged_sharpe=0.06,
        delta_merged_return=0.015,
        delta_merged_drawdown=-0.005,
        delta_return_drawdown_ratio=0.15,
        delta_merged_turnover=-0.002,
        official_before_after_improves=True,
        structural_novelty=0.80,
        dataset_gap_value=0.80,
        delay_universe_gap_value=0.80,
        cost_stress_completed=True,
        merged_analysis_completed=True,
        data_quality_verified=True,
    )
    d.update(overrides)
    return IQCCandidateInput(**d)


class TestIQCScoringProperties(unittest.TestCase):
    def test_randomized_invariants_with_reproducible_seed(self) -> None:
        rng = random.Random(SEED)
        for i in range(TRIALS):
            corr = rng.uniform(0.0, 0.99)
            to = rng.uniform(0.0, 0.95)
            wt = rng.uniform(0.0, 0.25)
            sh = rng.uniform(0.5, 3.0)
            fit = rng.uniform(0.5, 3.0)
            fam_cnt = rng.randint(0, 10)
            cond_cnt = rng.randint(0, 8)
            sub_today = rng.randint(0, 5)
            inp = _base(
                alpha_id=f"p_{i}",
                self_correlation=corr,
                turnover=to,
                max_instrument_weight=wt,
                sharpe=sh,
                fitness=fit,
                family_submission_count=fam_cnt,
                condition_count=cond_cnt,
                submitted_today_count=sub_today,
                parameter_cliff_detected=rng.choice([False, True]),
                complexity_delay_detected=rng.choice([False, True]),
            )
            rep = evaluate_iqc_candidate(inp)

            ctx = f"seed={SEED} trial={i} corr={corr:.6f} to={to:.6f} wt={wt:.6f}"
            # 1-6: Hard Gate boundaries
            if corr >= 0.70:
                self.assertFalse(rep.hard_gates.passed, ctx)
                self.assertIn("SELF_CORRELATION_NOT_PASSED", rep.blockers, ctx)
            if to <= 0.01 or to >= 0.70:
                self.assertFalse(rep.hard_gates.passed, ctx)
                self.assertIn("TURNOVER_NOT_PASSED", rep.blockers, ctx)
            if wt >= 0.10:
                self.assertFalse(rep.hard_gates.passed, ctx)
                self.assertIn("MAX_WEIGHT_NOT_PASSED", rep.blockers, ctx)

            # 7-8: Score & timing bounds + finite JSON
            self.assertTrue(0.0 <= rep.candidate_score <= 100.0, ctx)
            self.assertTrue(0.0 <= rep.timing_score <= 20.0, ctx)
            self.assertTrue(math.isfinite(rep.candidate_score), ctx)
            self.assertTrue(math.isfinite(rep.timing_score), ctx)
            self.assertTrue(math.isfinite(rep.submission_utility), ctx)
            serialized = json.dumps(rep.to_dict(), allow_nan=False)
            self.assertIn('"decision":', serialized)

            # 16: Hard gate dominance
            if not rep.hard_gates.passed:
                self.assertEqual(rep.decision(), IQCCandidateDecision.BLOCKED, ctx)

    def test_monotonicity_properties(self) -> None:
        rng = random.Random(SEED + 1)
        for i in range(40):
            # Penalty monotonicity: adding parameter_cliff cannot increase score
            base_inp = _base(alpha_id=f"m_{i}", sharpe=rng.uniform(1.3, 2.2), parameter_cliff_detected=False)
            pen_inp = _base(alpha_id=f"m_{i}", sharpe=base_inp.sharpe, parameter_cliff_detected=True)
            self.assertLessEqual(
                evaluate_iqc_candidate(pen_inp).candidate_score,
                evaluate_iqc_candidate(base_inp).candidate_score,
            )

            # Daily slot monotonicity
            r0 = evaluate_iqc_candidate(_base(submitted_today_count=0))
            r2 = evaluate_iqc_candidate(_base(submitted_today_count=2))
            r3 = evaluate_iqc_candidate(_base(submitted_today_count=3))
            self.assertGreaterEqual(r0.timing.daily_slot_value, r2.timing.daily_slot_value)
            self.assertGreaterEqual(r2.timing.daily_slot_value, r3.timing.daily_slot_value)

            # Family crowding monotonicity
            f0 = evaluate_iqc_candidate(_base(family_submission_count=0))
            f4 = evaluate_iqc_candidate(_base(family_submission_count=4))
            self.assertGreaterEqual(f0.components.dataset_idea_diversity, f4.components.dataset_idea_diversity)

    def test_daily_planner_uniqueness_and_slot_cap(self) -> None:
        rng = random.Random(SEED + 2)
        cands = [
            _base(
                alpha_id=f"dp_{i}",
                idea_family=f"FAM_{rng.randint(1, 3)}",
                dataset_category=rng.choice(["ANALYST", "NEWS", "FUNDAMENTAL", "SENTIMENT"]),
                universe=rng.choice(["TOP500", "TOP1000", "TOP3000"]),
                delay=rng.choice([0, 1]),
            )
            for i in range(15)
        ]
        for sub_today in (0, 1, 2, 3):
            plan = plan_daily_submissions(cands, max_daily=3, max_per_family=1, submitted_today=sub_today)
            self.assertLessEqual(len(plan), max(0, 3 - sub_today))
            ids = [r.alpha_id for r in plan]
            fams = [r.idea_family for r in plan]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(fams), len(set(fams)))


if __name__ == "__main__":
    unittest.main()
