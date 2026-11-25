"""Regression tests for P30 (v8 Multi-Field Economic Seeds + Dataset-Calibrated Neutralization)."""
from __future__ import annotations


def test_all_v8_economic_seeds_are_bronze_safe_and_valid_neutralization(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.multi_sim_client import (
        BRONZE_ALLOWED_NEUTRALIZATIONS,
        find_locked_bronze_operators,
    )
    from worldquant_brain.two_stage_funnel import V8_ECONOMIC_HYPOTHESIS_SEEDS

    assert len(V8_ECONOMIC_HYPOTHESIS_SEEDS) >= 20
    for seed in V8_ECONOMIC_HYPOTHESIS_SEEDS:
        locked = find_locked_bronze_operators(seed["expression"])
        assert not locked, f"Seed {seed['name']} uses locked Bronze operators: {locked}"
        assert seed["neutralization"] in BRONZE_ALLOWED_NEUTRALIZATIONS


def test_funnel_injects_v8_economic_seeds_and_uses_dataset_neutralization(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    import worldquant_brain.two_stage_funnel as tsf
    from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel

    v8_only = tuple(s for s in tsf.V8_ECONOMIC_HYPOTHESIS_SEEDS if str(s.get("name", "")).startswith("v8_"))
    monkeypatch.setattr(tsf, "V8_ECONOMIC_HYPOTHESIS_SEEDS", v8_only)

    funnel = TwoStageAlphaFunnel(tmp_path / "funnel.db")
    batch = funnel.next_batch({"option9": ["pcr_oi_30", "pcr_oi_60"], "model51": ["beta_last_30_days_spy"]}, batch_size=5)
    assert len(batch) == 5
    econ_items = [b for b in batch if str(b.get("probe", "")).startswith("v8_")]
    assert 2 <= len(econ_items) <= 3
    # Single-field probes should use dataset-calibrated neutralization (option9 -> INDUSTRY, model51 -> INDUSTRY)
    for item in batch:
        if not str(item.get("probe", "")).startswith("v8_"):
            if item["dataset_id"] == "option9":
                assert item["settings"]["neutralization"] in {"INDUSTRY", "SECTOR"}
            elif item["dataset_id"] == "model51":
                assert item["settings"]["neutralization"] == "INDUSTRY"


def test_yield_booster_v8_atoms_are_bronze_unlocked(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.alpha_yield_booster import _HIGH_YIELD_ATOMS
    from worldquant_brain.multi_sim_client import find_locked_bronze_operators

    for dom, atoms in _HIGH_YIELD_ATOMS.items():
        for expr in atoms:
            locked = find_locked_bronze_operators(expr)
            assert not locked, f"Atom in {dom} uses locked ops {locked}: {expr}"
