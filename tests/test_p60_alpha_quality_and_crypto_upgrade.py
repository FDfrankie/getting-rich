"""P60 Regression & Systemic Audit Verification Tests."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_non_numeric_metadata_field_blacklist():
    from worldquant_brain.multi_sim_client import is_non_numeric_metadata_field
    from worldquant_brain.alpha_miner import validate_candidate_expression

    assert is_non_numeric_metadata_field("actuals_value_currency_code") is True
    assert is_non_numeric_metadata_field("VEC:company_id") is True
    assert is_non_numeric_metadata_field("operating_income") is False
    assert is_non_numeric_metadata_field("pv13_com_page_rank") is False

    ok, reason = validate_candidate_expression("signed_power(rank(ts_backfill(actuals_value_currency_code, 5)) - 0.5, 2)")
    assert ok is False
    assert "METADATA_FIELD" in reason


def test_nested_ts_decay_linear_rejected_and_refinement_safe():
    from worldquant_brain.alpha_miner import validate_candidate_expression, ContinuousAlphaMiner

    nested = "ts_decay_linear(ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5), 8)"
    ok, reason = validate_candidate_expression(nested)
    assert ok is False
    assert reason == "NESTED_DECAY_CLONE"

    clean = "ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 8)"
    ok2, reason2 = validate_candidate_expression(clean)
    assert ok2 is True, reason2

    miner = ContinuousAlphaMiner()
    refs = miner.generate_near_miss_refinements(
        clean,
        {"decay": 4, "neutralization": "INDUSTRY", "truncation": 0.08},
        sharpe=1.30,
        fitness=0.95,
        turnover=0.55,
        margin=0.0008,
        failed_checks=["HIGH_TURNOVER"],
    )
    for r in refs:
        assert "ts_decay_linear(ts_decay_linear(" not in r["expression"].lower()


def test_v108_research_seeds_all_valid():
    from worldquant_brain.alpha_miner import V108_RESEARCH_HIGH_FITNESS_SEEDS, validate_candidate_expression

    assert len(V108_RESEARCH_HIGH_FITNESS_SEEDS) >= 7
    for seed in V108_RESEARCH_HIGH_FITNESS_SEEDS:
        ok, reason = validate_candidate_expression(seed["expression"])
        assert ok is True, f"{seed['name']} failed: {reason}"


def test_global_bronze_concurrency_semaphore_caps_nested_pools(monkeypatch):
    from worldquant_brain import multi_sim_client as msc

    monkeypatch.setenv("BRAIN_CONSULTANT_MODE", "false")
    monkeypatch.setenv("BRAIN_MAX_CONCURRENT_SIMS", "3")
    msc._GLOBAL_SIM_SEM = None

    active = {"n": 0, "peak": 0}
    lock = threading.Lock()

    class DummyClient:
        offline_mode = False
        is_authenticated = True
        BASE_URL = "https://api.worldquantbrain.com"

        def simulate_alpha(self, expr, settings=None, max_wait_seconds=60, poll_interval=1):
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            return {"success": True, "status": "COMPLETE", "expression": expr, "settings": settings or {}}

    client = DummyClient()
    items = [{"expression": f"rank(close) + {i}", "dataset_id": "pv1"} for i in range(6)]
    t1 = threading.Thread(target=msc.simulate_multi_alpha, args=(client, items[:3]), kwargs={"fallback_item_delay": 0.0})
    t2 = threading.Thread(target=msc.simulate_multi_alpha, args=(client, items[3:]), kwargs={"fallback_item_delay": 0.0})
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert active["peak"] <= 3, f"Exceeded Bronze 3-slot cap: peak={active['peak']}"


def test_sweep_duplicate_pending_clones_sqlite(tmp_path):
    from worldquant_brain.brain_client import BrainClient

    db = tmp_path / "vault.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE brain_candidate_pool ("
        "alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, "
        "fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, "
        "score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
    )
    rows = [
        ("A1", "group_neutralize(rank(debt / (abs(equity) + 0.001)), industry)", 1.30, 1.07, 0.025, 0.006, 33.0, "2026-11-25", "PENDING", "2026-11-25T00:00:00Z"),
        ("A2", "ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)", 1.30, 1.07, 0.025, 0.006, 33.1, "2026-11-25", "PENDING", "2026-11-25T00:00:01Z"),
        ("A3", "ts_decay_linear(ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5), 8)", 1.30, 1.08, 0.025, 0.006, 33.2, "2026-11-25", "PENDING", "2026-11-25T00:00:02Z"),
        ("A4", "signed_power(rank(ts_backfill(actuals_value_currency_code, 5)) - 0.5, 2)", 1.38, 1.24, 0.018, 0.011, 40.0, "2026-11-25", "PENDING", "2026-11-25T00:00:03Z"),
        ("A5", "group_rank(ts_scale(ts_backfill(pv13_com_page_rank, 2) / cap, 120), subindustry)", 1.29, 1.08, 0.11, 0.0015, 36.6, "2026-11-26", "PENDING", "2026-11-26T00:00:00Z"),
    ]
    conn.executemany("INSERT INTO brain_candidate_pool VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    client = BrainClient(email="", password="")
    client._quota_db = lambda: str(db)
    res = client.sweep_duplicate_pending_clones()
    assert res["dedup_skipped"] == 2
    assert res["invalid_removed"] == 1

    conn = sqlite3.connect(db)
    statuses = dict(conn.execute("SELECT alpha_id, status FROM brain_candidate_pool").fetchall())
    conn.close()
    assert statuses["A1"] == "PENDING"
    assert statuses["A2"] == "DEDUP_SKIPPED"
    assert statuses["A3"] == "DEDUP_SKIPPED"
    assert statuses["A4"] == "FAILED_CHECKS"
    assert statuses["A5"] == "PENDING"


def test_crypto_microstructure_entry_guard():
    from scripts.headless_crypto_bot import evaluate_microstructure_entry_guard

    ok, _ = evaluate_microstructure_entry_guard(expected_edge_bps=18.0, round_trip_fee_bps=4.0, funding_rate_8h=0.0001, proposed_side="LONG", seconds_since_last_close=300.0)
    assert ok is True

    ok_fee, reason_fee = evaluate_microstructure_entry_guard(expected_edge_bps=8.0, round_trip_fee_bps=4.0)
    assert ok_fee is False and "EDGE_BELOW_FEE_FLOOR" in reason_fee

    ok_fund, reason_fund = evaluate_microstructure_entry_guard(expected_edge_bps=25.0, funding_rate_8h=0.0009, proposed_side="LONG")
    assert ok_fund is False and "ADVERSE_LONG_FUNDING" in reason_fund

    ok_cd, reason_cd = evaluate_microstructure_entry_guard(expected_edge_bps=25.0, seconds_since_last_close=60.0)
    assert ok_cd is False and "COOLDOWN_ACTIVE" in reason_cd
