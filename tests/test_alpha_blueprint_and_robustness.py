from __future__ import annotations

import dataclasses
import pathlib
import tempfile
import unittest

from crypto_trading.risk_manager import RiskManager
from worldquant_brain.alpha_blueprint import (
    AlphaBlueprint,
    AlphaFamily,
    ComplexityBudget,
    DataFieldSpec,
    RobustnessPlan,
    analyst_revision_blueprint,
    check_expression_complexity_budget,
    fundamental_quality_blueprint,
    generate_controlled_candidates,
    validate_blueprint_fields_against_catalog,
)
from worldquant_brain.alpha_lifecycle import (
    QualityStage,
    require_quality_stage_progression,
)
from worldquant_brain.alpha_quality import AlphaQuality
from worldquant_brain.brain_quality_bridge import parse_and_evaluate_brain_result
from worldquant_brain.research_memory import (
    AlphaResearchFactoryService,
    evaluate_portfolio_marginal_utility,
)
from worldquant_brain.robustness_evaluator import (
    EvaluationRun,
    RobustnessStatus,
    evaluate_robustness,
)


def _baseline() -> EvaluationRun:
    return EvaluationRun(
        run_id="base",
        variant_type="BASELINE",
        variant_value="default",
        sharpe=1.80,
        fitness=1.35,
        turnover=0.22,
        returns=0.14,
        drawdown=0.07,
        pnl=420000.0,
        passed=True,
        metrics_source="WORLDQUANT_BRAIN",
        simulation_id="sim-base",
    )


def _full_valid_runs() -> list[EvaluationRun]:
    return [
        # 1. Temporal stability (4 years, each share <= 0.40)
        EvaluationRun("y22", "YEAR", "2022", 1.5, 1.1, 0.22, pnl=100000.0, simulation_id="sim-y22", year=2022),
        EvaluationRun("y23", "YEAR", "2023", 1.6, 1.2, 0.21, pnl=110000.0, simulation_id="sim-y23", year=2023),
        EvaluationRun("y24", "YEAR", "2024", 1.7, 1.25, 0.23, pnl=105000.0, simulation_id="sim-y24", year=2024),
        EvaluationRun("y25", "YEAR", "2025", 1.4, 1.05, 0.24, pnl=95000.0, simulation_id="sim-y25", year=2025),
        # 2. Parameter stability
        EvaluationRun("p1", "WINDOW", "20", 1.65, 1.22, 0.23, simulation_id="sim-p1"),
        EvaluationRun("p2", "WINDOW", "40", 1.80, 1.35, 0.22, simulation_id="sim-p2"),
        EvaluationRun("p3", "WINDOW", "60", 1.58, 1.18, 0.20, simulation_id="sim-p3"),
        EvaluationRun("p4", "DECAY", "4", 1.62, 1.24, 0.21, simulation_id="sim-p4"),
        EvaluationRun("p5", "TRUNCATION", "0.06", 1.55, 1.19, 0.22, simulation_id="sim-p5"),
        # 3. Universe stability
        EvaluationRun("u1", "UNIVERSE", "TOP3000", 1.80, 1.35, 0.22, simulation_id="sim-u1", universe="TOP3000"),
        EvaluationRun("u2", "UNIVERSE", "TOP2000", 1.62, 1.22, 0.21, simulation_id="sim-u2", universe="TOP2000"),
        EvaluationRun("u3", "UNIVERSE", "TOP1000", 1.44, 1.10, 0.19, simulation_id="sim-u3", universe="TOP1000"),
        # 4. Neutralization stability
        EvaluationRun("n1", "NEUTRALIZATION", "SUBINDUSTRY", 1.80, 1.35, 0.22, simulation_id="sim-n1"),
        EvaluationRun("n2", "NEUTRALIZATION", "INDUSTRY", 1.50, 1.16, 0.22, simulation_id="sim-n2"),
        # 5. Cost stress
        EvaluationRun("c1", "COST", "1.0x", 1.80, 1.35, 0.22, simulation_id="sim-c1"),
        EvaluationRun("c2", "COST", "1.5x", 1.56, 1.16, 0.22, simulation_id="sim-c2"),
        EvaluationRun("c3", "COST", "2.0x", 1.34, 1.04, 0.22, simulation_id="sim-c3"),
        # 6. Regime stability
        EvaluationRun("r1", "REGIME", "HIGH_VOL", 1.48, 1.12, 0.24, simulation_id="sim-r1", regime="HIGH_VOL"),
        EvaluationRun("r2", "REGIME", "LOW_VOL", 1.62, 1.21, 0.20, simulation_id="sim-r2", regime="LOW_VOL"),
        EvaluationRun("r3", "REGIME", "BEAR", 1.36, 1.04, 0.23, simulation_id="sim-r3", regime="BEAR"),
    ]


# 1. Missing economic_rationale -> rejected
def test_01_missing_economic_rationale_rejected():
    bp = dataclasses.replace(analyst_revision_blueprint(), economic_rationale="   ")
    errs = bp.validate()
    assert any("Economic rationale is required" in e for e in errs)


# 2. Complexity budget exceeded -> rejected
def test_02_complexity_budget_exceeded_rejected():
    bp = analyst_revision_blueprint()
    long_expr = "rank(ts_zscore(ts_delta(winsorize(ts_mean(ts_rank(group_rank(anl4_est_eps, industry), 10), 20), 3), 20), 60))"
    errs = check_expression_complexity_budget(long_expr, ComplexityBudget(maximum_tree_depth=4, maximum_operator_count=5))
    assert any("REJECT_COMPLEXITY" in e for e in errs)


# 3. Field missing from catalog -> blocked
def test_03_missing_field_in_catalog_blocked():
    bp = fundamental_quality_blueprint()
    params = {"cf_field": "nonexistent_field_xyz", "debt_field": "debt", "group": "industry", "hump_val": 0.01}
    errs = validate_blueprint_fields_against_catalog(bp, params)
    assert any("FIELD_NOT_IN_CATALOG:nonexistent_field_xyz" in e for e in errs)


# 4. Local proxy baseline -> rejected
def test_04_local_proxy_baseline_rejected():
    bp = analyst_revision_blueprint()
    local_base = dataclasses.replace(_baseline(), metrics_source="LOCAL_PROXY")
    raised = False
    try:
        evaluate_robustness(blueprint=bp, baseline=local_base, runs=_full_valid_runs())
    except ValueError as e:
        assert "Baseline must be an official, passing BRAIN run" in str(e)
        raised = True
    assert raised


# 5. Missing yearly runs -> blocks promotion
def test_05_missing_yearly_runs_blocks_promotion():
    bp = analyst_revision_blueprint()
    runs = [r for r in _full_valid_runs() if r.year is None]
    report = evaluate_robustness(blueprint=bp, baseline=_baseline(), runs=runs)
    assert not report.promotion_ready
    assert any("No yearly runs" in b for b in report.blockers)


# 6. Missing universe runs -> blocks promotion
def test_06_missing_universe_runs_blocks_promotion():
    bp = analyst_revision_blueprint()
    runs = [r for r in _full_valid_runs() if r.variant_type != "UNIVERSE"]
    report = evaluate_robustness(blueprint=bp, baseline=_baseline(), runs=runs)
    assert not report.promotion_ready
    assert any("UNIVERSE_STABILITY has no runs" in b for b in report.blockers)


# 7. Missing cost stress runs -> blocks promotion
def test_07_missing_cost_stress_blocks_promotion():
    bp = analyst_revision_blueprint()
    runs = [r for r in _full_valid_runs() if r.variant_type != "COST"]
    report = evaluate_robustness(blueprint=bp, baseline=_baseline(), runs=runs)
    assert not report.promotion_ready
    assert any("COST_STRESS has no runs" in b for b in report.blockers)


# 8. Missing regime runs -> blocks promotion
def test_08_missing_regime_runs_blocks_promotion():
    bp = analyst_revision_blueprint()
    runs = [r for r in _full_valid_runs() if r.variant_type != "REGIME"]
    report = evaluate_robustness(blueprint=bp, baseline=_baseline(), runs=runs)
    assert not report.promotion_ready
    assert any("REGIME_STABILITY has no runs" in b for b in report.blockers)


# 9. Single year PnL > 40% -> blocked
def test_09_single_year_pnl_concentration_over_40_percent_blocked():
    bp = analyst_revision_blueprint()
    runs = [r for r in _full_valid_runs() if r.year is None]
    runs.extend([
        EvaluationRun("y22", "YEAR", "2022", 2.1, 1.5, 0.22, pnl=300000.0, simulation_id="sim-y22", year=2022),
        EvaluationRun("y23", "YEAR", "2023", 1.2, 0.9, 0.22, pnl=40000.0, simulation_id="sim-y23", year=2023),
        EvaluationRun("y24", "YEAR", "2024", 1.1, 0.9, 0.22, pnl=40000.0, simulation_id="sim-y24", year=2024),
        EvaluationRun("y25", "YEAR", "2025", 1.1, 0.9, 0.22, pnl=40000.0, simulation_id="sim-y25", year=2025),
    ])
    report = evaluate_robustness(blueprint=bp, baseline=_baseline(), runs=runs)
    assert not report.promotion_ready
    assert any("Single-year PnL share" in b for b in report.blockers)


# 10. Cliff parameter drop -> FRAGILE
def test_10_cliff_parameter_drop_marked_fragile():
    bp = analyst_revision_blueprint()
    runs = [r for r in _full_valid_runs() if r.variant_type not in {"WINDOW", "DECAY", "TRUNCATION"}]
    runs.extend([
        EvaluationRun("p1", "WINDOW", "20", 0.55, 0.40, 0.42, simulation_id="sim-p1"),
        EvaluationRun("p2", "WINDOW", "40", 1.80, 1.35, 0.22, simulation_id="sim-p2"),
        EvaluationRun("p3", "WINDOW", "60", 0.40, 0.31, 0.48, simulation_id="sim-p3"),
        EvaluationRun("p4", "DECAY", "4", 0.50, 0.35, 0.55, simulation_id="sim-p4"),
        EvaluationRun("p5", "TRUNCATION", "0.06", 0.45, 0.28, 0.50, simulation_id="sim-p5"),
    ])
    report = evaluate_robustness(blueprint=bp, baseline=_baseline(), runs=runs)
    assert not report.promotion_ready
    assert report.status == RobustnessStatus.FRAGILE


# 11. Only passing TOP3000 and failing TOP2000/TOP1000 -> FRAGILE
def test_11_universe_only_passing_top3000_marked_fragile():
    bp = analyst_revision_blueprint()
    runs = [r for r in _full_valid_runs() if r.variant_type != "UNIVERSE"]
    runs.extend([
        EvaluationRun("u1", "UNIVERSE", "TOP3000", 1.80, 1.35, 0.22, simulation_id="sim-u1", universe="TOP3000"),
        EvaluationRun("u2", "UNIVERSE", "TOP2000", 0.30, 0.20, 0.22, simulation_id="sim-u2", universe="TOP2000"),
        EvaluationRun("u3", "UNIVERSE", "TOP1000", 0.20, 0.15, 0.22, simulation_id="sim-u3", universe="TOP1000"),
    ])
    report = evaluate_robustness(blueprint=bp, baseline=_baseline(), runs=runs)
    assert not report.promotion_ready
    assert report.status == RobustnessStatus.FRAGILE


# 12. BRAIN offline / local proxy never produces GOOD / ROBUST_GOOD
def test_12_brain_offline_never_produces_good():
    from worldquant_brain.alpha_quality import AlphaMetrics, evaluate_alpha
    m = AlphaMetrics(
        alpha_id="local_001",
        expression="rank(close)",
        delay=1,
        sharpe=2.5,
        fitness=1.8,
        returns=0.20,
        turnover=0.15,
        metrics_source="LOCAL_PROXY",
        official_simulation_id=None,
    )
    rep = evaluate_alpha(m)
    assert rep.quality == AlphaQuality.UNVERIFIED
    assert rep.submit_ready is False


# 13. Direct stage-skipping or direct DB write to ROBUST_GOOD is blocked
def test_13_stage_skipping_and_direct_db_write_to_robust_good_is_blocked():
    jumped = False
    try:
        require_quality_stage_progression(QualityStage.CANDIDATE, QualityStage.ROBUST_GOOD)
    except ValueError as e:
        assert "Cannot skip" in str(e)
        jumped = True
    assert jumped

    db_file = str(pathlib.Path(tempfile.mkdtemp()) / "guard.db")
    svc = AlphaResearchFactoryService(db_path=db_file)
    blocked = False
    try:
        svc.set_quality_class_guarded("unverified_alpha_99", "ROBUST_GOOD")
    except PermissionError as e:
        assert "Direct write of ROBUST_GOOD" in str(e)
        blocked = True
    assert blocked


# 14. API / service returns only verified ROBUST_GOOD alphas
def test_14_api_returns_only_verified_robust_good():
    db_file = str(pathlib.Path(tempfile.mkdtemp()) / "api_test.db")
    svc = AlphaResearchFactoryService(db_path=db_file)
    bp = fundamental_quality_blueprint()
    rep = svc.evaluate_and_persist_robustness(
        alpha_id="alpha_verified_01",
        blueprint=bp,
        baseline=_baseline(),
        runs=_full_valid_runs(),
    )
    assert rep.promotion_ready
    rg_items = svc.list_robust_good_alphas()
    assert len(rg_items) == 1
    assert rg_items[0]["alpha_id"] == "alpha_verified_01"
    assert rg_items[0]["quality_class"] == "ROBUST_GOOD"


# 15. Same simulation replay is idempotent + generates research lessons
def test_15_idempotent_persistence_and_research_memory():
    db_file = str(pathlib.Path(tempfile.mkdtemp()) / "factory.db")
    svc = AlphaResearchFactoryService(db_path=db_file)
    bp = fundamental_quality_blueprint()
    svc.register_blueprint(bp)
    svc.register_blueprint(bp)
    assert len(svc.list_blueprints()) >= 2

    r1 = svc.record_robustness_run("alpha_01", bp.blueprint_id, _baseline())
    r2 = svc.record_robustness_run("alpha_01", bp.blueprint_id, _baseline())
    assert r1["idempotent_replay"] is False
    assert r2["idempotent_replay"] is True

    lesson1 = svc.record_research_lesson(
        family=bp.family.value,
        data_domain="FUNDAMENTAL",
        failure_gate="COST_STRESS",
        effective_fix="Add hump(0.01) to suppress micro-turnover",
        ineffective_fixes=["Decrease decay below 2"],
        evidence_run_ids=["sim-c2", "sim-c3"],
    )
    lesson2 = svc.record_research_lesson(
        family=bp.family.value,
        data_domain="FUNDAMENTAL",
        failure_gate="COST_STRESS",
        effective_fix="Add hump(0.01) to suppress micro-turnover",
        ineffective_fixes=["Decrease decay below 2"],
        evidence_run_ids=["sim-c2", "sim-c3"],
    )
    assert lesson1["lesson_id"] == lesson2["lesson_id"]
    assert len(svc.list_research_lessons()) == 1


# 16. Verify 50.19% false drawdown bug is resolved
def test_16_risk_manager_reanchor_prevents_false_50_percent_drawdown():
    rm = RiskManager(initial_capital=10000.0)
    rm.reanchor_capital(4981.0)
    assert rm.initial_capital == 4981.0
    assert rm.peak_capital == 4981.0
    assert rm.check_drawdown(4981.0) is True
    assert rm.is_trading_halted is False


class TestAlphaBlueprintAndRobustnessSuite(unittest.TestCase):
    def test_all_16_gates(self):
        test_01_missing_economic_rationale_rejected()
        test_02_complexity_budget_exceeded_rejected()
        test_03_missing_field_in_catalog_blocked()
        test_04_local_proxy_baseline_rejected()
        test_05_missing_yearly_runs_blocks_promotion()
        test_06_missing_universe_runs_blocks_promotion()
        test_07_missing_cost_stress_blocks_promotion()
        test_08_missing_regime_runs_blocks_promotion()
        test_09_single_year_pnl_concentration_over_40_percent_blocked()
        test_10_cliff_parameter_drop_marked_fragile()
        test_11_universe_only_passing_top3000_marked_fragile()
        test_12_brain_offline_never_produces_good()
        test_13_stage_skipping_and_direct_db_write_to_robust_good_is_blocked()
        test_14_api_returns_only_verified_robust_good()
        test_15_idempotent_persistence_and_research_memory()
        test_16_risk_manager_reanchor_prevents_false_50_percent_drawdown()


if __name__ == "__main__":
    unittest.main()
