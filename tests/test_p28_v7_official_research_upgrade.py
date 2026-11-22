"""Regression tests for P28 (v7 Official BRAIN Research Alpha Engine Upgrade)."""
from __future__ import annotations


def test_all_stage1_probes_and_v7_additions_are_bronze_unlocked(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.multi_sim_client import find_locked_bronze_operators
    from worldquant_brain.two_stage_funnel import STAGE1_PROBES

    names = {n for n, _ in STAGE1_PROBES}
    assert {"liq_bucket_neut", "kth_sparse90", "is_ladder_boost"}.issubset(names)

    for name, tmpl in STAGE1_PROBES:
        expr = tmpl.format(f="close")
        locked = find_locked_bronze_operators(expr)
        assert not locked, f"Stage-1 probe {name} uses locked Bronze operators: {locked}"


def test_2y_sharpe_repair_uses_bronze_safe_is_ladder_not_jump_decay(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.alpha_yield_booster import precision_repair_near_miss
    from worldquant_brain.multi_sim_client import find_locked_bronze_operators

    repairs = precision_repair_near_miss(
        "rank(ts_backfill(est_eps, 20))",
        {"neutralization": "INDUSTRY", "decay": 2, "universe": "TOP3000"},
        sharpe=1.45,
        fitness=1.10,
        turnover=0.18,
        failed_checks=["LOW_2Y_SHARPE"],
    )
    assert repairs
    for rep in repairs:
        assert "jump_decay" not in rep["expression"]
        locked = find_locked_bronze_operators(rep["expression"])
        assert not locked, f"Repair {rep['name']} uses locked Bronze operators: {locked}"
    assert any("1.5 - rank(" in r["expression"] for r in repairs)


def test_overfitting_robustness_redlines():
    from worldquant_brain.alpha_yield_booster import (
        build_overfitting_robustness_probes,
        evaluate_overfitting_robustness,
    )

    probes = build_overfitting_robustness_probes("ts_zscore(close, 20)")
    assert probes["rank_test"] == "rank(ts_zscore(close, 20))"
    assert probes["binary_test"] == "sign(ts_zscore(close, 20))"

    ok = evaluate_overfitting_robustness(
        is_sharpe=2.0, test_period_sharpe=1.4, rank_test_sharpe=1.2, binary_test_sharpe=0.9
    )
    assert ok["robust"] is True

    bad = evaluate_overfitting_robustness(
        is_sharpe=2.0, test_period_sharpe=0.8, rank_test_sharpe=0.4, binary_test_sharpe=0.3
    )
    assert bad["robust"] is False
    assert set(bad["overfitting_flags"]) == {
        "TEST_PERIOD_DROP_OVER_50PCT",
        "RANK_TEST_DROP_OVER_70PCT",
        "BINARY_TEST_DROP_OVER_70PCT",
    }
