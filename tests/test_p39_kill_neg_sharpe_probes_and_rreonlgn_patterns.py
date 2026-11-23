"""P39 unit tests: Kill negative-Sharpe corr_ret20 & no-op winsorize(rank), promote RReoNlgn probes & Bronze 3-sim cap."""
from __future__ import annotations

import pytest

from worldquant_brain.two_stage_funnel import STAGE1_PROBES, is_obscure_footnote_field
from worldquant_brain.multi_sim_client import (
    api_min_spacing_seconds,
    bronze_max_concurrent_sims,
    find_locked_bronze_operators,
)


def test_p39_no_corr_ret20_and_no_noop_winsorize_rank() -> None:
    probe_dict = dict(STAGE1_PROBES)
    assert "corr_ret20" not in probe_dict
    for name, tmpl in STAGE1_PROBES:
        expr = tmpl.format(f="operating_income")
        compact = expr.replace(" ", "").lower()
        assert "winsorize(rank(" not in compact, f"No-op winsorize(rank()) found in {name}: {expr}"
        assert "ts_corr(ts_backfill(operating_income,20),returns,20)" not in compact, f"Negative-Sharpe corr_ret20 in {name}"
        if "bucket(" in compact:
            assert "densify(bucket(" in compact, f"bucket() must be wrapped in densify() in {name}: {expr}"
        locked = find_locked_bronze_operators(expr)
        assert locked == [], f"Locked Bronze op {locked} in {name}: {expr}"


def test_p39_rreonlgn_winning_probes_present() -> None:
    names = {n for n, _ in STAGE1_PROBES}
    for required in ("ts126_grp_rank", "cap_ts126_grp_rank", "rreonlgn_combo_eps", "last_diff_surprise"):
        assert required in names, f"Missing P39 winning probe: {required}"


def test_p39_obscure_footnote_filter() -> None:
    assert is_obscure_footnote_field("antidilutive_securities_eps_exclusion_count") is True
    assert is_obscure_footnote_field("federal_statutory_tax_rate") is True
    assert is_obscure_footnote_field("equipment_maximum_useful_life") is True
    assert is_obscure_footnote_field("operating_income") is False
    assert is_obscure_footnote_field("cashflow_op") is False
    assert is_obscure_footnote_field("est_eps") is False


def test_p39_bronze_concurrent_limit_and_api_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_MAX_CONCURRENT_SIMS", "8")
    assert bronze_max_concurrent_sims() == 3
    assert api_min_spacing_seconds() >= 2.0

    monkeypatch.setenv("BRAIN_CONSULTANT_MODE", "true")
    assert bronze_max_concurrent_sims() == 8
