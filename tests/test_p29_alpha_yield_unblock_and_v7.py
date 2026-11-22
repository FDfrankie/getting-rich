"""Regression tests for P29 (Alpha Yield Unblock + v7 Official Research Upgrade + Bronze Guard)."""
from __future__ import annotations


def test_5bps_margin_passes_and_high_sharpe_high_to_rescue_bypasses_max_fc(monkeypatch):
    monkeypatch.delenv("BRAIN_MIN_MARGIN", raising=False)
    from worldquant_brain.two_stage_funnel import classify_verdict

    # 6 bps margin (0.0006) + all official BRAIN checks passed -> MUST PASS!
    verdict, reason = classify_verdict({
        "sharpe": 1.52,
        "fitness": 1.18,
        "turnover": 0.22,
        "margin": 0.0006,
        "checks_passed": True,
        "failed_checks": [],
    })
    assert verdict == "PASSED", f"Expected PASSED at 6bps margin, got {verdict}: {reason}"

    # P05M7vpq-class gem with 3 failed checks (LOW_FITNESS, HIGH_TURNOVER, LOW_2Y_SHARPE) -> MUST rescue as NEAR_MISS!
    v2, r2 = classify_verdict({
        "sharpe": 2.06,
        "fitness": 0.58,
        "turnover": 1.01,
        "margin": 0.00016,
        "checks_passed": False,
        "failed_checks": ["LOW_FITNESS", "HIGH_TURNOVER", "LOW_2Y_SHARPE"],
    })
    assert v2 == "NEAR_MISS", f"Expected NEAR_MISS rescue, got {v2}: {r2}"

    # 75% turnover high-Sharpe gem (0.70 < to <= 0.78) -> MUST rescue as NEAR_MISS!
    v3, r3 = classify_verdict({
        "sharpe": 1.75,
        "fitness": 0.55,
        "turnover": 0.75,
        "margin": 0.0003,
        "checks_passed": False,
        "failed_checks": ["LOW_FITNESS", "HIGH_TURNOVER"],
    })
    assert v3 == "NEAR_MISS", f"Expected NEAR_MISS for 75% TO gem, got {v3}: {r3}"


def test_all_stage1_probes_and_repairs_are_bronze_unlocked(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.alpha_yield_booster import precision_repair_near_miss
    from worldquant_brain.multi_sim_client import find_locked_bronze_operators
    from worldquant_brain.two_stage_funnel import STAGE1_PROBES

    names = {n for n, _ in STAGE1_PROBES}
    assert {"liq_bucket_neut", "kth_sparse90", "is_ladder_boost"}.issubset(names)
    for name, tmpl in STAGE1_PROBES:
        locked = find_locked_bronze_operators(tmpl.format(f="close"))
        assert not locked, f"Stage1 probe {name} uses locked ops: {locked}"

    repairs = precision_repair_near_miss(
        "rank(ts_backfill(est_eps, 20))",
        {"neutralization": "INDUSTRY", "decay": 2, "universe": "TOP3000"},
        sharpe=1.45,
        fitness=1.10,
        turnover=0.18,
        failed_checks=["LOW_2Y_SHARPE"],
    )
    assert repairs and all("jump_decay" not in r["expression"] for r in repairs)
    for r in repairs:
        assert not find_locked_bronze_operators(r["expression"])


def test_alpha_miner_validator_blocks_deprecated_opt6_and_locked_ops(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.alpha_miner import validate_candidate_expression

    ok1, reason1 = validate_candidate_expression("rank(opt6_nexterntod)")
    assert not ok1 and "opt6_nexterntod" in reason1

    ok2, reason2 = validate_candidate_expression("jump_decay(rank(close), 5)")
    assert not ok2 and "LOCKED_BRONZE_OPERATOR:jump_decay" in reason2

    ok3, reason3 = validate_candidate_expression("hump(group_neutralize(rank(close), subindustry), hump=0.01)")
    assert ok3 and reason3 == "OK"


def test_overfitting_robustness_redlines():
    from worldquant_brain.alpha_yield_booster import (
        build_overfitting_robustness_probes,
        evaluate_overfitting_robustness,
    )

    probes = build_overfitting_robustness_probes("ts_zscore(close, 20)")
    assert probes == {"rank_test": "rank(ts_zscore(close, 20))", "binary_test": "sign(ts_zscore(close, 20))"}
    assert evaluate_overfitting_robustness(is_sharpe=2.0, test_period_sharpe=1.3, rank_test_sharpe=1.1, binary_test_sharpe=0.8)["robust"] is True
    bad = evaluate_overfitting_robustness(is_sharpe=2.0, test_period_sharpe=0.8, rank_test_sharpe=0.4, binary_test_sharpe=0.3)
    assert not bad["robust"] and len(bad["overfitting_flags"]) == 3
