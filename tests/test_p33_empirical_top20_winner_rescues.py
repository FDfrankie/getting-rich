"""Regression tests for P33 (Empirical Top-20 Winner Family Rescues + P32 hump-freeze fix)."""
from __future__ import annotations

import re


def test_all_48_seeds_bronze_unlocked_and_zero_freezing_hump(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.multi_sim_client import (
        BRONZE_ALLOWED_NEUTRALIZATIONS,
        find_locked_bronze_operators,
    )
    from worldquant_brain.two_stage_funnel import V8_ECONOMIC_HYPOTHESIS_SEEDS

    assert len(V8_ECONOMIC_HYPOTHESIS_SEEDS) == 48
    hump_re = re.compile(r"hump\s*=\s*([0-9.]+)")
    for seed in V8_ECONOMIC_HYPOTHESIS_SEEDS:
        assert not find_locked_bronze_operators(seed["expression"]), f"Locked op in {seed['name']}"
        assert seed["neutralization"] in BRONZE_ALLOWED_NEUTRALIZATIONS
        for m in hump_re.finditer(seed["expression"]):
            assert float(m.group(1)) <= 0.002


def test_p33_top20_winner_rescues_dispatch_first(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel

    funnel = TwoStageAlphaFunnel(tmp_path / "f.db")
    batch = funnel.next_batch(
        {"socialmedia12": ["scl12_buzz"], "model51": ["unsystematic_risk_last_360_days"], "pv1": ["close"]},
        batch_size=5,
    )
    p33 = [b for b in batch if str(b.get("probe", "")).startswith("p33_")]
    assert len(p33) == 3
    assert p33[0]["probe"] == "p33_58JQGZ7n_subuniv_top1000"
    assert p33[0]["settings"]["universe"] == "TOP1000"
