from __future__ import annotations

import os
from unittest.mock import patch

from worldquant_brain.multi_sim_client import (
    apply_consultant_quick_mode,
    correlation_poll_interval_seconds,
    find_locked_bronze_operators,
    is_empty_correlation_computing,
)
from worldquant_brain.two_stage_funnel import (
    STAGE1_PROBES,
    TwoStageAlphaFunnel,
    generate_kb304_corr_breaker_variants,
)


def test_p40_purges_neg_sharpe_legacy_probes_and_promotes_kb304_champions() -> None:
    names = [name for name, _ in STAGE1_PROBES]
    assert "last_revision" not in names
    assert "ts_av_diff60" not in names
    assert "cap_norm_gauss" not in names
    assert "corr_ret20" not in names
    assert names[0] in ("cap_ts120_scale_ind", "cap_ts126_grp_rank")
    for required in (
        "cap_ts126_grp_rank",
        "cap_ts120_scale_ind",
        "cap_ts63_grp_zscore",
        "cap_ts252_zscore_rank",
        "small_and_stable_hump",
        "delta4_sector_neut",
        "double_excess_ts_resid",
        "group_backfill_ts252",
        "group_scale_delta_diff",
    ):
        assert required in names, f"Missing P40 probe: {required}"


def test_p40_all_stage1_probes_are_100_percent_bronze_unlocked_and_clean() -> None:
    for name, tmpl in STAGE1_PROBES:
        rendered = tmpl.format(f="fnd6_sales")
        assert "winsorize(rank(" not in rendered, f"No-op winsorize(rank in {name}"
        locked = find_locked_bronze_operators(rendered)
        assert locked == [], f"Probe {name} uses locked Bronze operators: {locked}"


def test_p40_corr_breaker_swaps_are_distinct_and_bronze_safe() -> None:
    base = "group_rank(ts_rank(ts_backfill(vec_avg(fnd6_x), 20) / (cap + 0.001), 126), subindustry)"
    variants = generate_kb304_corr_breaker_variants(base, {"neutralization": "SUBINDUSTRY"}, max_variants=6)
    assert len(variants) >= 4
    exprs = [v["expression"] for v in variants]
    assert any("ts_scale(" in e for e in exprs)
    assert any("ts_zscore(" in e for e in exprs)
    assert any("vec_sum(" in e for e in exprs)
    for v in variants:
        assert find_locked_bronze_operators(v["expression"]) == [], f"Locked op in {v}"


def test_p40_stage2_near_miss_rescue_emits_kb304_rules_bronze_safe(tmp_path) -> None:
    funnel = TwoStageAlphaFunnel(tmp_path / "vault.db", max_stage2_variants_per_seed=12)
    base_expr = "group_rank(ts_rank(ts_backfill(sales, 20) / (cap + 0.001), 126), subindustry)"

    # Fitness near-miss -> volatility regime gate + outer quantile
    funnel._enqueue_immediate_refinement(
        {"expression": base_expr, "sharpe": 1.65, "fitness": 0.91, "turnover": 0.18, "settings": {"neutralization": "SUBINDUSTRY"}},
        "LOW_FITNESS",
    )
    q_exprs = [item["expression"] for item in funnel._session_refinement_queue]
    assert any("trade_when(ts_rank(ts_std_dev(returns, 20), 63) > 0.5" in e for e in q_exprs)
    assert any(e.startswith("ts_mean(") or e.startswith("quantile(") for e in q_exprs)
    for e in q_exprs:
        assert find_locked_bronze_operators(e) == []

    # Sub-universe near-miss -> liquidity-tiered decay + hump(group_neutralize)
    funnel._session_refinement_queue.clear()
    funnel._enqueue_immediate_refinement(
        {"expression": base_expr, "sharpe": 1.70, "fitness": 1.10, "turnover": 0.15, "settings": {"neutralization": "MARKET"}},
        "LOW_SUB_UNIVERSE_SHARPE_AND_WEIGHT",
    )
    q2 = funnel._session_refinement_queue
    assert any("rank(volume * close)" in item["expression"] for item in q2)
    assert any(item["settings"].get("neutralization") == "INDUSTRY" for item in q2)
    for item in q2:
        assert find_locked_bronze_operators(item["expression"]) == []


def test_p40_multi_sim_corr_polling_and_quick_mode_guardrails() -> None:
    assert correlation_poll_interval_seconds() >= 30.0
    assert is_empty_correlation_computing(200, "") is True
    assert is_empty_correlation_computing(200, b"  ") is True
    assert is_empty_correlation_computing(200, {"max": 0.5}) is False
    assert is_empty_correlation_computing(429, "") is False

    base_sett = {"region": "USA", "universe": "TOP3000"}
    with patch.dict(os.environ, {"BRAIN_CONSULTANT_MODE": "false"}):
        s_bronze = apply_consultant_quick_mode(base_sett, quick_mode=True)
        assert "simulationMode" not in s_bronze
    with patch.dict(os.environ, {"BRAIN_CONSULTANT_MODE": "true"}):
        s_cons = apply_consultant_quick_mode(base_sett, quick_mode=True)
        assert s_cons["simulationMode"] == "QUICK"
        assert s_cons["visualization"] is False
