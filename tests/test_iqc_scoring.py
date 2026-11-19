"""Offline unit tests for IQC 2026 Score-Aware Alpha Architecture & Scoring."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.iqc_scoring import (
    IQC_POLICY_VERSION,
    IQCCandidateDecision,
    IQCCandidateInput,
    IQCGrade,
    IQCScoringRepository,
    build_iqc_input_from_trusted_record,
    compute_official_fitness,
    compute_os_promotion_score,
    compute_submission_utility,
    evaluate_iqc_candidate,
    plan_daily_submissions,
)
from worldquant_brain.pre_submission_gate import GateVerdict, evaluate_pre_submission


def _make_base_candidate(**overrides) -> IQCCandidateInput:
    defaults = dict(
        alpha_id="alpha-A",
        expression="group_rank(ts_zscore(est_eps_revision, 40), subindustry)",
        region="USA",
        universe="TOP1000",
        delay=1,
        dataset_category="ANALYST",
        idea_family="ANALYST_REVISION",
        metrics_source="WORLDQUANT_BRAIN",
        official_simulation_id="sim-A",
        brain_alpha_id="alpha-A",
        sharpe=1.92,
        fitness=1.73,
        returns=0.145,
        turnover=0.22,
        margin=0.00125,
        drawdown=0.055,
        self_correlation=0.31,
        max_instrument_weight=0.065,
        bias_test_passed=True,
        weight_test_passed=True,
        sub_universe_passed=True,
        duplicate_or_public=False,
        parameter_only_variant=False,
        test_period_contaminated=False,
        already_submitted=False,
        submit_ready=True,
        temporal_stability=0.86,
        parameter_stability=0.78,
        universe_stability=0.82,
        regime_stability=0.77,
        dsr_probability=0.97,
        pbo=0.14,
        cpcv_positive_ratio=0.72,
        max_single_year_pnl_share=0.25,
        delta_merged_sharpe=0.09,
        delta_merged_return=0.018,
        delta_merged_drawdown=-0.009,
        delta_return_drawdown_ratio=0.18,
        delta_merged_turnover=-0.004,
        official_before_after_improves=True,
        structural_novelty=0.90,
        dataset_gap_value=0.85,
        delay_universe_gap_value=0.80,
        dataset_value_score=0.75,
        value_factor=0.70,
        pyramid_gap_value=0.80,
        family_submission_count=0,
        condition_count=1,
        cost_stress_completed=True,
        merged_analysis_completed=True,
        complexity_delay_detected=False,
        data_quality_verified=True,
        parameter_cliff_detected=False,
        field_crowding_level=0.1,
        submitted_today_count=0,
        same_family_submitted_today=False,
        same_slice_submitted_today=False,
        relative_competition_estimate=None,
        operational_ready=True,
        sharpe_percentile=0.92,
        fitness_percentile=0.90,
        returns_percentile=0.85,
        margin_percentile=0.85,
        return_drawdown_percentile=0.90,
    )
    defaults.update(overrides)
    return IQCCandidateInput(**defaults)


class TestIQCScoring(unittest.TestCase):
    def test_01_hard_gate_failure_blocks_even_with_high_metrics(self) -> None:
        c = _make_base_candidate(bias_test_passed=False)
        r = evaluate_iqc_candidate(c)
        self.assertFalse(r.hard_gates.passed)
        self.assertEqual(r.grade, IQCGrade.BLOCKED)
        self.assertEqual(r.decision(), IQCCandidateDecision.BLOCKED)
        self.assertIn("BIAS_TEST_NOT_PASSED", r.blockers)

    def test_02_self_correlation_07000_blocked(self) -> None:
        c = _make_base_candidate(self_correlation=0.7000)
        r = evaluate_iqc_candidate(c)
        self.assertFalse(r.hard_gates.passed)
        self.assertEqual(r.decision(), IQCCandidateDecision.BLOCKED)
        self.assertIn("SELF_CORRELATION_NOT_PASSED", r.blockers)

    def test_03_self_correlation_06999_allowed_for_scoring(self) -> None:
        c = _make_base_candidate(self_correlation=0.6999)
        r = evaluate_iqc_candidate(c)
        self.assertTrue(r.hard_gates.passed)
        self.assertNotEqual(r.decision(), IQCCandidateDecision.BLOCKED)

    def test_04_high_sharpe_cannot_override_local_proxy_or_bias_failure(self) -> None:
        proxy = _make_base_candidate(
            alpha_id="alpha-E",
            metrics_source="LOCAL_PROXY",
            official_simulation_id=None,
            brain_alpha_id=None,
            sharpe=2.40,
            fitness=2.10,
            submit_ready=False,
        )
        r = evaluate_iqc_candidate(proxy)
        self.assertEqual(r.decision(), IQCCandidateDecision.BLOCKED)
        self.assertIn("UNVERIFIED_METRICS_SOURCE", r.blockers)

    def test_05_candidate_score_capped_at_100_and_floored_at_0(self) -> None:
        c_hi = _make_base_candidate(
            sharpe_percentile=1.0,
            fitness_percentile=1.0,
            returns_percentile=1.0,
            margin_percentile=1.0,
            temporal_stability=1.0,
            parameter_stability=1.0,
            universe_stability=1.0,
            regime_stability=1.0,
            dsr_probability=1.0,
            pbo=0.0,
            cpcv_positive_ratio=1.0,
        )
        r_hi = evaluate_iqc_candidate(c_hi)
        self.assertLessEqual(r_hi.candidate_score, 100.0)
        self.assertGreaterEqual(r_hi.timing_score, 0.0)
        self.assertLessEqual(r_hi.timing_score, 20.0)

        c_lo = _make_base_candidate(
            sharpe_percentile=0.0,
            fitness_percentile=0.0,
            returns_percentile=0.0,
            margin_percentile=0.0,
            temporal_stability=0.0,
            parameter_stability=0.0,
            universe_stability=0.0,
            regime_stability=0.0,
            test_period_contaminated=True,
            parameter_cliff_detected=True,
            max_single_year_pnl_share=0.65,
            family_submission_count=10,
            condition_count=6,
            cost_stress_completed=False,
            merged_analysis_completed=False,
            data_quality_verified=False,
        )
        r_lo = evaluate_iqc_candidate(c_lo)
        self.assertEqual(r_lo.candidate_score, 0.0)

    def test_06_grade_s_low_timing_holds_for_better_day(self) -> None:
        c = _make_base_candidate(submitted_today_count=3, same_family_submitted_today=True)
        r = evaluate_iqc_candidate(c)
        self.assertEqual(r.grade, IQCGrade.S)
        self.assertLess(r.timing_score, 14.0)
        self.assertEqual(r.decision(), IQCCandidateDecision.HOLD_FOR_BETTER_DAY)

    def test_07_grade_a_high_timing_submits_if_slot_available(self) -> None:
        c = _make_base_candidate(
            sharpe_percentile=0.70,
            fitness_percentile=0.70,
            returns_percentile=0.70,
            margin_percentile=0.60,
            temporal_stability=0.70,
            parameter_stability=0.65,
            universe_stability=0.70,
            regime_stability=0.65,
            delta_merged_sharpe=0.05,
            submitted_today_count=0,
        )
        r = evaluate_iqc_candidate(c)
        self.assertEqual(r.grade, IQCGrade.A)
        self.assertGreaterEqual(r.timing_score, 16.0)
        self.assertEqual(r.decision(), IQCCandidateDecision.SUBMIT_IF_SLOT_AVAILABLE)

    def test_08_grades_b_c_d_map_to_repair_research_archive(self) -> None:
        b = _make_base_candidate(
            sharpe_percentile=0.65,
            fitness_percentile=0.65,
            returns_percentile=0.6,
            margin_percentile=0.6,
            temporal_stability=0.65,
            parameter_stability=0.65,
            universe_stability=0.6,
            regime_stability=0.6,
            delta_merged_sharpe=0.02,
            return_drawdown_percentile=0.65,
            structural_novelty=0.6,
            dataset_gap_value=0.6,
            delay_universe_gap_value=0.6,
            dataset_value_score=0.4,
            pyramid_gap_value=0.4,
            value_factor=0.4,
        )
        rb = evaluate_iqc_candidate(b)
        self.assertEqual(rb.grade, IQCGrade.B)
        self.assertEqual(rb.decision(), IQCCandidateDecision.TARGETED_REPAIR)

        d = _make_base_candidate(
            sharpe_percentile=0.1,
            fitness_percentile=0.1,
            returns_percentile=0.1,
            margin_percentile=0.1,
            temporal_stability=0.1,
            parameter_stability=0.1,
            universe_stability=0.1,
            regime_stability=0.1,
            self_correlation=0.66,
            delta_merged_sharpe=0.0,
            return_drawdown_percentile=0.1,
            structural_novelty=0.1,
            dataset_gap_value=0.1,
            delay_universe_gap_value=0.1,
            cost_stress_completed=False,
            merged_analysis_completed=False,
            parameter_cliff_detected=True,
        )
        rd = evaluate_iqc_candidate(d)
        self.assertEqual(rd.grade, IQCGrade.D)
        self.assertEqual(rd.decision(), IQCCandidateDecision.ARCHIVE)

    def test_09_negative_merged_score_never_submits_now(self) -> None:
        c = _make_base_candidate(
            alpha_id="alpha-B",
            sharpe=2.05,
            fitness=1.82,
            self_correlation=0.66,
            delta_merged_sharpe=-0.025,
            official_before_after_improves=False,
            family_submission_count=8,
            cost_stress_completed=False,
            parameter_cliff_detected=True,
        )
        r = evaluate_iqc_candidate(c)
        self.assertIn("MERGED_SCORE_NEGATIVE", r.blockers)
        self.assertNotIn(
            r.decision(),
            {IQCCandidateDecision.SUBMIT_NOW, IQCCandidateDecision.SUBMIT_IF_SLOT_AVAILABLE},
        )

    def test_10_fitness_uses_turnover_floor_0125(self) -> None:
        f_at_floor = compute_official_fitness(1.8, 0.10, 0.125)
        f_below_floor = compute_official_fitness(1.8, 0.10, 0.05)
        self.assertAlmostEqual(f_at_floor, f_below_floor, places=8)

        r_at_floor = evaluate_iqc_candidate(_make_base_candidate(turnover=0.125))
        r_below_floor = evaluate_iqc_candidate(_make_base_candidate(turnover=0.04))
        self.assertLessEqual(
            r_below_floor.components.turnover_efficiency,
            r_at_floor.components.turnover_efficiency,
        )

    def test_11_idea_crowding_and_window_only_variant(self) -> None:
        fresh = evaluate_iqc_candidate(_make_base_candidate(family_submission_count=0))
        crowded = evaluate_iqc_candidate(_make_base_candidate(family_submission_count=6))
        self.assertGreater(fresh.candidate_score, crowded.candidate_score)

        win_only = evaluate_iqc_candidate(_make_base_candidate(parameter_only_variant=True))
        self.assertEqual(win_only.components.dataset_idea_diversity, 0.0)
        self.assertEqual(win_only.decision(), IQCCandidateDecision.BLOCKED)

    def test_12_clean_test_beats_contaminated_test(self) -> None:
        clean = evaluate_iqc_candidate(_make_base_candidate(test_period_contaminated=False))
        contam = evaluate_iqc_candidate(_make_base_candidate(test_period_contaminated=True))
        self.assertGreater(clean.components.os_robustness, contam.components.os_robustness)
        self.assertGreater(clean.os_promotion_score, contam.os_promotion_score)

    def test_13_excessive_conditions_penalized_and_warned(self) -> None:
        r = evaluate_iqc_candidate(_make_base_candidate(condition_count=5, condition_complexity_exception=False))
        self.assertGreater(r.penalties.excessive_conditions, 0.0)
        self.assertIn("EXCESSIVE_CONDITIONS_OVER_3", r.warnings)
        self.assertEqual(r.components.simplicity, 0.0)

    def test_14_relative_competition_estimate_remains_none_when_unavailable(self) -> None:
        rec = {
            "id": "a1",
            "brain_alpha_id": "a1",
            "official_simulation_id": "s1",
            "metrics_source": "WORLDQUANT_BRAIN",
            "sharpe": 1.9,
            "fitness": 1.6,
            "turnover": 0.2,
            "self_correlation": 0.3,
            "max_instrument_weight": 0.05,
            "bias_test_passed": True,
            "weight_test_passed": True,
            "sub_universe_passed": True,
            "submit_ready": True,
        }
        inp = build_iqc_input_from_trusted_record(rec)
        self.assertIsNone(inp.relative_competition_estimate)
        r = evaluate_iqc_candidate(inp)
        self.assertIsNone(r.relative_competition_estimate)

    def test_15_daily_planner_max_3_and_max_1_per_family(self) -> None:
        cands = [
            _make_base_candidate(alpha_id="a1", idea_family="FAM_1", dataset_category="ANALYST", universe="TOP1000"),
            _make_base_candidate(alpha_id="a2", idea_family="FAM_1", dataset_category="FUNDAMENTAL", universe="TOP500"),
            _make_base_candidate(alpha_id="a3", idea_family="FAM_2", dataset_category="NEWS", universe="TOP500", delay=0),
            _make_base_candidate(alpha_id="a4", idea_family="FAM_3", dataset_category="SENTIMENT", universe="TOP1000"),
            _make_base_candidate(alpha_id="a5", idea_family="FAM_4", dataset_category="OPTIONS", universe="TOP3000"),
        ]
        plan = plan_daily_submissions(cands, max_daily=3, max_per_family=1)
        self.assertLessEqual(len(plan), 3)
        fams = [r.idea_family for r in plan]
        self.assertEqual(len(fams), len(set(fams)))
        self.assertNotIn("a2", [r.alpha_id for r in plan])

    def test_16_ranking_never_bypasses_pre_submission_gate(self) -> None:
        rec = {
            "brain_alpha_id": "a_bad",
            "official_simulation_id": "s_bad",
            "metrics_source": "WORLDQUANT_BRAIN",
            "sharpe": 2.2,
            "fitness": 1.9,
            "turnover": 0.2,
            "self_correlation": 0.72,
            "max_instrument_weight": 0.05,
            "bias_test_passed": True,
            "weight_test_passed": True,
            "sub_universe_passed": True,
            "submit_ready": True,
        }
        gate = evaluate_pre_submission(rec, automatic=True)
        self.assertEqual(gate.verdict, GateVerdict.BLOCK)
        iqc = evaluate_iqc_candidate(build_iqc_input_from_trusted_record(rec))
        self.assertEqual(iqc.decision(), IQCCandidateDecision.BLOCKED)

    def test_17_repository_migration_and_calculation_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "iqc_test.db"
            repo = IQCScoringRepository(db_path)
            repo.migrate()
            repo.migrate()  # idempotent
            r = evaluate_iqc_candidate(_make_base_candidate())
            audit_id = repo.save_report_with_audit(r)
            self.assertTrue(audit_id.startswith("iqc_audit_"))
            stored = repo.get_score("alpha-A")
            self.assertIsNotNone(stored)
            self.assertEqual(stored["policy_version"], IQC_POLICY_VERSION)
            self.assertEqual(stored["input_snapshot_hash"], r.input_snapshot_hash)
            audits = repo.list_audits("alpha-A")
            self.assertEqual(len(audits), 1)
            parsed = json.loads(audits[0]["breakdown_json"])
            self.assertEqual(parsed["alpha_id"], "alpha-A")


if __name__ == "__main__":
    unittest.main()
