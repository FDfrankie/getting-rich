from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from worldquant_brain.alpha_miner import ContinuousAlphaMiner


def test_near_miss_skips_weak_alphas(tmp_path: Path) -> None:
    miner = ContinuousAlphaMiner()
    miner.vault.db_path = str(tmp_path / "vault.db")
    variants = miner.generate_near_miss_refinements(
        "rank(close - open)",
        {"decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08},
        sharpe=0.25,
        fitness=0.20,
        turnover=0.30,
        margin=0.0002,
    )
    assert variants == []


def test_near_miss_generates_sign_flip_and_decay_for_high_turnover_or_negative() -> None:
    miner = ContinuousAlphaMiner()
    miner.has_tried_expression = lambda _expr: False  # type: ignore[method-assign]
    miner.mark_expression_tried = lambda _expr: None  # type: ignore[method-assign]

    variants = miner.generate_near_miss_refinements(
        "ts_delta(close, 3)",
        {"decay": 2, "neutralization": "SUBINDUSTRY", "truncation": 0.08},
        sharpe=-1.35,
        fitness=0.90,
        turnover=0.62,
        margin=0.0007,
        failed_checks=["HIGH_TURNOVER"],
        max_variants=6,
    )
    assert len(variants) >= 3
    names = [v["name"] for v in variants]
    assert "NearMiss_SignFlip" in names
    assert "NearMiss_DecayTurnover" in names
    decay_var = next(v for v in variants if v["name"] == "NearMiss_DecayTurnover")
    assert decay_var["settings"]["decay"] >= 6
    assert decay_var["settings"]["truncation"] <= 0.05
    assert decay_var["expression"].startswith("ts_decay_linear(")


def test_near_miss_rotates_neutralization_for_borderline_sharpe() -> None:
    miner = ContinuousAlphaMiner()
    miner.has_tried_expression = lambda _expr: False  # type: ignore[method-assign]
    miner.mark_expression_tried = lambda _expr: None  # type: ignore[method-assign]

    variants = miner.generate_near_miss_refinements(
        "rank(vwap - close)",
        {"decay": 4, "neutralization": "SUBINDUSTRY", "truncation": 0.05},
        sharpe=1.12,
        fitness=0.88,
        turnover=0.25,
        margin=0.0012,
        max_variants=3,
    )
    assert len(variants) >= 2
    assert any(v["settings"]["neutralization"] == "INDUSTRY" for v in variants)
