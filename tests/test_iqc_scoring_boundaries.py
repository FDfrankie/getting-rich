"""Boundary & edge-case unit tests for IQC 2026 Score-Aware Alpha Architecture."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from worldquant_brain.iqc_scoring import (
    IQCCandidateDecision,
    IQCCandidateInput,
    IQCGrade,
    IQCScoringPolicy,
    compute_official_fitness,
    evaluate_iqc_candidate,
    plan_daily_submissions,
)


def _cand(**overrides) -> IQCCandidateInput:
    base = dict(
        alpha_id="bnd-1",
        expression="rank(close)",
        region="USA",
        universe="TOP1000",
        delay=1,
        dataset_category="ANALYST",
        idea_family="FAM_A",
        metrics_source="WORLDQUANT_BRAIN",
        official_simulation_id="sim-1",
        brain_alpha_id="alpha-1",
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
        max_single_year_pnl_share=0.25,
        delta_merged_sharpe=0.06,
        delta_merged_return=0.015,
        delta_merged_drawdown=-0.005,
        delta_return_drawdown_ratio=0.15,
        delta_merged_turnover=-0.002,
        official_before_after_improves=True,
        structural_novelty=0.85,
        dataset_gap_value=0.80,
        delay_universe_gap_value=0.80,
        cost_stress_completed=True,
        merged_analysis_completed=True,
        data_quality_verified=True,
    )
    base.update(overrides)
    return IQCCandidateInput(**base)


class TestIQCScoringBoundaries(unittest.TestCase):
    def test_every_hard_gate_blocks_individually_and_collects_all_blockers(self) -> None:
        single_failures = [
            ({"metrics_source": "LOCAL_PROXY"}, "UNVERIFIED_METRICS_SOURCE"),
            ({"official_simulation_id": None}, "MISSING_OFFICIAL_SIMULATION_ID"),
            ({"brain_alpha_id": ""}, "MISSING_BRAIN_ALPHA_ID"),
            ({"bias_test_passed": False}, "BIAS_TEST_NOT_PASSED"),
            ({"weight_test_passed": False}, "WEIGHT_TEST_NOT_PASSED"),
            ({"sub_universe_passed": False}, "SUB_UNIVERSE_TEST_NOT_PASSED"),
            ({"duplicate_or_public": True}, "DUPLICATE_OR_PUBLIC_FORMULA"),
            ({"parameter_only_variant": True}, "PARAMETER_ONLY_VARIANT"),
            ({"test_period_contaminated": True}, "TEST_PERIOD_CONTAMINATED"),
            ({"already_submitted": True}, "ALREADY_SUBMITTED"),
            ({"submit_ready": False}, "SUBMIT_READY_FALSE"),
        ]
        for kw, expected_code in single_failures:
            rep = evaluate_iqc_candidate(_cand(**kw))
            self.assertFalse(rep.hard_gates.passed, f"Expected fail for {kw}")
            self.assertEqual(rep.grade, IQCGrade.BLOCKED)
            self.assertEqual(rep.decision(), IQCCandidateDecision.BLOCKED)
            self.assertIn(expected_code, rep.blockers)

        multi = evaluate_iqc_candidate(
            _cand(metrics_source="LOCAL_PROXY", bias_test_passed=False, submit_ready=False, self_correlation=0.75)
        )
        for code in (
            "UNVERIFIED_METRICS_SOURCE",
            "BIAS_TEST_NOT_PASSED",
            "SUBMIT_READY_FALSE",
            "SELF_CORRELATION_NOT_PASSED",
        ):
            self.assertIn(code, multi.blockers)

    def test_sharpe_and_fitness_exact_strict_boundaries(self) -> None:
        self.assertFalse(evaluate_iqc_candidate(_cand(sharpe=1.249999)).hard_gates.passed)
        self.assertFalse(evaluate_iqc_candidate(_cand(sharpe=1.250000)).hard_gates.passed)
        self.assertTrue(evaluate_iqc_candidate(_cand(sharpe=1.250001)).hard_gates.passed)

        self.assertFalse(evaluate_iqc_candidate(_cand(fitness=0.999999)).hard_gates.passed)
        self.assertFalse(evaluate_iqc_candidate(_cand(fitness=1.000000)).hard_gates.passed)
        self.assertTrue(evaluate_iqc_candidate(_cand(fitness=1.000001)).hard_gates.passed)

    def test_turnover_open_interval(self) -> None:
        self.assertFalse(evaluate_iqc_candidate(_cand(turnover=0.010000)).hard_gates.passed)
        self.assertTrue(evaluate_iqc_candidate(_cand(turnover=0.010001)).hard_gates.passed)
        self.assertTrue(evaluate_iqc_candidate(_cand(turnover=0.699999)).hard_gates.passed)
        self.assertFalse(evaluate_iqc_candidate(_cand(turnover=0.700000)).hard_gates.passed)

    def test_self_correlation_and_max_weight_boundaries(self) -> None:
        self.assertFalse(evaluate_iqc_candidate(_cand(self_correlation=None)).hard_gates.passed)
        self.assertTrue(evaluate_iqc_candidate(_cand(self_correlation=0.699999)).hard_gates.passed)
        self.assertFalse(evaluate_iqc_candidate(_cand(self_correlation=0.700000)).hard_gates.passed)

        self.assertFalse(evaluate_iqc_candidate(_cand(max_instrument_weight=None)).hard_gates.passed)
        self.assertTrue(evaluate_iqc_candidate(_cand(max_instrument_weight=0.099999)).hard_gates.passed)
        self.assertFalse(evaluate_iqc_candidate(_cand(max_instrument_weight=0.100000)).hard_gates.passed)

    def test_fitness_125_floor_and_penalty_boundaries(self) -> None:
        self.assertAlmostEqual(
            compute_official_fitness(2.0, 0.10, 0.00),
            compute_official_fitness(2.0, 0.10, 0.125),
            places=8,
        )
        r_40 = evaluate_iqc_candidate(_cand(max_single_year_pnl_share=0.40))
        r_41 = evaluate_iqc_candidate(_cand(max_single_year_pnl_share=0.4001))
        self.assertEqual(r_40.penalties.single_year_dominance, 0.0)
        self.assertEqual(r_41.penalties.single_year_dominance, 10.0)

        r_pen = evaluate_iqc_candidate(
            _cand(
                family_submission_count=20,
                condition_count=10,
                complexity_delay_detected=True,
                parameter_cliff_detected=True,
                max_single_year_pnl_share=0.55,
            )
        )
        self.assertLessEqual(r_pen.penalties.idea_crowding, 10.0)
        self.assertLessEqual(r_pen.penalties.excessive_conditions, 8.0)
        self.assertEqual(r_pen.penalties.complexity_delay, 3.0)
        self.assertGreaterEqual(r_pen.candidate_score, 0.0)
        self.assertLessEqual(r_pen.candidate_score, 100.0)

    def test_null_robustness_never_produces_nan_or_inf(self) -> None:
        rep = evaluate_iqc_candidate(
            _cand(
                temporal_stability=None,
                parameter_stability=None,
                universe_stability=None,
                regime_stability=None,
                dsr_probability=None,
                pbo=None,
                cpcv_positive_ratio=None,
                merged_analysis_completed=False,
            )
        )
        self.assertTrue(math.isfinite(rep.candidate_score))
        self.assertTrue(math.isfinite(rep.timing_score))
        self.assertTrue(math.isfinite(rep.submission_utility))
        self.assertIn("MISSING_ROBUSTNESS_DATA", rep.warnings)
        self.assertIn("MERGED_ANALYSIS_MISSING", rep.warnings)

    def test_custom_iqc_scoring_policy_overrides_thresholds(self) -> None:
        pol = IQCScoringPolicy(
            policy_version="2026.custom.1",
            minimum_sharpe=1.50,
            minimum_fitness=1.20,
            maximum_self_correlation=0.60,
        )
        rep = evaluate_iqc_candidate(_cand(sharpe=1.40, fitness=1.10, self_correlation=0.65), policy=pol)
        self.assertEqual(rep.policy_version, "2026.custom.1")
        self.assertFalse(rep.hard_gates.passed)
        for code in ("SHARPE_NOT_PASSED", "FITNESS_NOT_PASSED", "SELF_CORRELATION_NOT_PASSED"):
            self.assertIn(code, rep.blockers)

    def test_daily_planner_with_full_slots_returns_empty(self) -> None:
        plan = plan_daily_submissions([_cand()], max_daily=3, max_per_family=1, submitted_today=3)
        self.assertEqual(plan, [])

    def test_boundary_cases_json_fixture(self) -> None:
        fixture_path = Path(__file__).resolve().parent.parent / "examples" / "iqc_boundary_cases.json"
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))
        for item in cases:
            overrides = {
                k: v for k, v in item.items()
                if k not in {"case", "expected_hard_gates_passed", "expected_blocker"}
            }
            rep = evaluate_iqc_candidate(_cand(**overrides))
            self.assertEqual(rep.hard_gates.passed, item["expected_hard_gates_passed"], item["case"])
            if "expected_blocker" in item:
                self.assertIn(item["expected_blocker"], rep.blockers, item["case"])



    def test_delay0_requires_sharpe_above_2_0(self) -> None:
        self.assertFalse(evaluate_iqc_candidate(_cand(delay=0, sharpe=1.999999)).hard_gates.passed)
        self.assertFalse(evaluate_iqc_candidate(_cand(delay=0, sharpe=2.000000)).hard_gates.passed)
        self.assertTrue(evaluate_iqc_candidate(_cand(delay=0, sharpe=2.000001)).hard_gates.passed)

    def test_count_conditional_ops_auto_detects_if_else_bloat(self) -> None:
        from worldquant_brain.iqc_scoring import build_iqc_input_from_trusted_record, count_conditional_ops
        expr = "if_else(a>1, if_else(b>2, if_else(c>3, if_else(d>4, x, y), z), w), v)"
        self.assertEqual(count_conditional_ops(expr), 4)
        inp = build_iqc_input_from_trusted_record({"id": "c1", "expression": expr})
        self.assertEqual(inp.condition_count, 4)



    def test_neutralization_none_warns_unless_group_neutralize_present(self) -> None:
        r_warn = evaluate_iqc_candidate(_cand(neutralization="NONE", expression="rank(-ts_delta(close, 2))"))
        self.assertIn("UNBALANCED_NEUTRALIZATION_NONE", r_warn.warnings)
        r_ok = evaluate_iqc_candidate(_cand(neutralization="NONE", expression="group_neutralize(ts_returns(close, 5), industry)"))
        self.assertNotIn("UNBALANCED_NEUTRALIZATION_NONE", r_ok.warnings)


if __name__ == "__main__":
    unittest.main()
