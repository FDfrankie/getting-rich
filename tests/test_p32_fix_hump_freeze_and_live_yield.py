"""Regression tests for P32 (Fix TOP3000 hump>=0.01 Freeze + Live Rescue Seeds + P31 Superset)."""
from __future__ import annotations

import re


def test_zero_freezing_hump_parameters_in_seeds_probes_or_booster(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.alpha_yield_booster import generate_orthogonal_combo_candidate
    from worldquant_brain.multi_sim_client import find_locked_bronze_operators
    from worldquant_brain.two_stage_funnel import STAGE1_PROBES, V8_ECONOMIC_HYPOTHESIS_SEEDS

    hump_re = re.compile(r"hump\s*=\s*([0-9.]+)")

    # 38 total seeds (6 live rescue + 20 v8 + 12 v9), NONE with freezing hump >= 0.005
    assert len(V8_ECONOMIC_HYPOTHESIS_SEEDS) == 38
    for seed in V8_ECONOMIC_HYPOTHESIS_SEEDS:
        assert not find_locked_bronze_operators(seed["expression"])
        for m in hump_re.finditer(seed["expression"]):
            assert float(m.group(1)) <= 0.002, f"Freezing hump in {seed['name']}: {seed['expression']}"

    for name, tmpl in STAGE1_PROBES:
        expr = tmpl.format(f="close")
        assert not find_locked_bronze_operators(expr)
        for m in hump_re.finditer(expr):
            assert float(m.group(1)) <= 0.002, f"Freezing hump in probe {name}: {expr}"

    for _ in range(15):
        cand = generate_orthogonal_combo_candidate()
        for m in hump_re.finditer(cand["expression"]):
            assert float(m.group(1)) <= 0.002, f"Freezing hump in combo: {cand['expression']}"


def test_frozen_low_to_high_sharpe_9ql2kwdx_enters_stage2_rescue():
    from worldquant_brain.two_stage_funnel import classify_verdict

    verdict, _ = classify_verdict({
        "sharpe": 0.98,
        "fitness": 0.43,
        "turnover": 0.0047,
        "margin": 0.010319,
        "checks_passed": False,
        "failed_checks": ["LOW_SHARPE", "LOW_FITNESS", "LOW_TURNOVER"],
    })
    assert verdict == "NEAR_MISS"


def test_p32_live_rescue_seeds_emit_with_top1000_and_decay0(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel

    funnel = TwoStageAlphaFunnel(tmp_path / "f.db")
    batch = funnel.next_batch({"option9": ["pcr_oi_30"], "model51": ["beta_last_30_days_spy"]}, batch_size=5)
    p32_items = [b for b in batch if str(b.get("probe", "")).startswith("p32_rescue_")]
    assert len(p32_items) == 3
    assert any(b["settings"]["universe"] == "TOP1000" and b["settings"]["decay"] == 0 for b in p32_items)
