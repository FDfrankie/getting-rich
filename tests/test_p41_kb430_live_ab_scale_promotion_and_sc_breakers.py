from __future__ import annotations

import tempfile
from pathlib import Path

from worldquant_brain.multi_sim_client import find_locked_bronze_operators
from worldquant_brain.two_stage_funnel import (
    STAGE1_PROBES,
    V8_ECONOMIC_HYPOTHESIS_SEEDS,
    TwoStageAlphaFunnel,
    generate_kb304_corr_breaker_variants,
    is_obscure_footnote_field,
)


def test_p41_stage1_probes_unit_clean_and_bronze_safe() -> None:
    names = [n for n, _ in STAGE1_PROBES]
    assert names[0] == "cap_ts120_scale_ind", f"Expected cap_ts120_scale_ind at #1, got {names[0]}"
    assert "cap_ts252_quantile" not in names, "Un-neutralized cap_ts252_quantile must be purged"
    for expected in (
        "cap_ts120_scale_sub",
        "cap_av_diff60_grp_rank",
        "cap_multi_win_prod",
        "cap_delta2_zscore20_sub",
        "cap_spow05_scale120",
    ):
        assert expected in names, f"Missing P41 probe: {expected}"
    for name, tmpl in STAGE1_PROBES:
        assert "(cap + 0.001)" not in tmpl, f"Unit-warning denominator still in {name}: {tmpl}"
        assert "(close + 0.001)" not in tmpl, f"Unit-warning denominator still in {name}: {tmpl}"
        sample = tmpl.format(f="sales")
        locked = find_locked_bronze_operators(sample)
        assert locked == [], f"Probe {name} uses locked Bronze operators: {locked}"


def test_p41_curated_seeds_all_bronze_safe() -> None:
    p41_seeds = [s for s in V8_ECONOMIC_HYPOTHESIS_SEEDS if str(s.get("name", "")).startswith("p41_")]
    assert len(p41_seeds) >= 7
    for s in p41_seeds:
        locked = find_locked_bronze_operators(s["expression"])
        assert locked == [], f"Seed {s['name']} uses locked Bronze operators: {locked}"


def test_p41_footnote_filter_unblocks_winners_and_wired_into_stage1() -> None:
    for winner in (
        "accumulated_amortization_finite_intangibles",
        "allocated_sbp_expense_total",
        "accumulated_depreciation_depletion_amortization_ppne",
    ):
        assert not is_obscure_footnote_field(winner), f"Winner field wrongly blocked: {winner}"

    for loser in (
        "federal_statutory_tax_rate",
        "equipment_maximum_useful_life",
        "exercisable_options_avg_exercise_price_2",
        "deferred_tax_assets_valuation_allowance_value",
        "fn_avg_diluted_sharesout_adj_q",
        "diluted_shares_outstanding_adjustment",
    ):
        assert is_obscure_footnote_field(loser), f"Loser footnote field not blocked: {loser}"

    with tempfile.TemporaryDirectory() as td:
        funnel = TwoStageAlphaFunnel(Path(td) / "test.db")
        batch = funnel.next_batch(
            {"fundamental6": ["federal_statutory_tax_rate", "equipment_maximum_useful_life", "accumulated_amortization_finite_intangibles"]},
            batch_size=10,
        )
        fields_used = [item["field"] for item in batch]
        assert "federal_statutory_tax_rate" not in fields_used
        assert "equipment_maximum_useful_life" not in fields_used
        assert "accumulated_amortization_finite_intangibles" in fields_used


def test_p41_corr_breaker_includes_sc_breakers_and_bronze_safe() -> None:
    variants = generate_kb304_corr_breaker_variants(
        "group_rank(ts_rank(ts_backfill(sales, 60) / cap, 126), subindustry)",
        {"neutralization": "SUBINDUSTRY", "decay": 4},
        max_variants=6,
    )
    assert len(variants) == 6
    reasons = [v["reason"] for v in variants]
    assert any("ts_scale" in r for r in reasons)
    for v in variants:
        assert find_locked_bronze_operators(v["expression"]) == []

    variants2 = generate_kb304_corr_breaker_variants(
        "group_rank(ts_scale(ts_backfill(sales, 60) / cap, 120), industry)",
        {"neutralization": "INDUSTRY", "decay": 4},
        max_variants=6,
    )
    reasons2 = [v["reason"] for v in variants2]
    assert "p41_corr_signed_power_045" in reasons2
    assert "p41_corr_multi_window_prod" in reasons2
    for v in variants2:
        assert find_locked_bronze_operators(v["expression"]) == []


def test_p41_stage2_fitness_and_drawdown_rescue_bronze_safe() -> None:
    with tempfile.TemporaryDirectory() as td:
        funnel = TwoStageAlphaFunnel(Path(td) / "test.db", max_stage2_variants_per_seed=16)
        funnel._enqueue_immediate_refinement(
            {
                "expression": "group_rank(ts_scale(ts_backfill(accumulated_depreciation_depletion_amortization_ppne, 60) / cap, 120), industry)",
                "settings": {"neutralization": "INDUSTRY", "decay": 4, "truncation": 0.08, "universe": "TOP3000"},
                "dataset_id": "fundamental6",
                "probe": "cap_ts120_scale_ind",
                "sharpe": 1.33,
                "fitness": 0.96,
                "turnover": 0.1811,
                "margin": 0.00105,
            },
            "LOW_FITNESS_HIGH_DRAWDOWN",
        )
        exprs = [item["expression"] for item in funnel._session_refinement_queue]
        assert any(e.startswith("ts_mean(") for e in exprs), "Missing P41 ts_mean(..., 2) Fitness rescue"
        assert any("volume > adv20" in e for e in exprs), "Missing P41 volume > adv20 Fitness rescue"
        assert any(e.startswith("winsorize(") for e in exprs), "Missing P41 winsorize(..., std=3) Drawdown rescue"
        for e in exprs:
            assert find_locked_bronze_operators(e) == [], f"Locked op in Stage-2 rescue: {e}"
