"""Unit tests for P36 Mining Efficiency, Dataset Profiles, Concurrency & Backlog Flush."""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from worldquant_brain.alpha_template_expander import _hump, _wrap_hygiene, generate_diverse_domain_candidate
from worldquant_brain.multi_sim_client import (
    default_sim_profile_for_dataset,
    fetch_available_datasets,
    find_locked_bronze_operators,
    sanitize_sim_settings,
    simulate_multi_alpha,
)
from worldquant_brain.two_stage_funnel import V8_ECONOMIC_HYPOTHESIS_SEEDS


def test_p36_dataset_sim_profiles_and_sanitize() -> None:
    fund = default_sim_profile_for_dataset("fundamental6")
    assert fund["nanHandling"] == "ON"
    assert fund["decay"] == 0
    assert fund["neutralization"] == "SUBINDUSTRY"

    anl = default_sim_profile_for_dataset("analyst4")
    assert anl["nanHandling"] == "ON"
    assert anl["decay"] == 2

    pv = default_sim_profile_for_dataset("pv1")
    assert pv["nanHandling"] == "OFF"
    assert pv["decay"] == 12

    # Caller explicit override must be respected while still enforcing Bronze neutralization
    sanitized = sanitize_sim_settings({"neutralization": "CROWDING", "decay": 6}, dataset_id="fundamental6")
    assert sanitized["decay"] == 6
    assert sanitized["nanHandling"] == "ON"
    assert sanitized["neutralization"] in {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"}


def test_p36_dataset_discovery_prioritizes_high_pass_rate_tiers() -> None:
    client = MagicMock()
    client.offline_mode = False
    client.is_authenticated = True
    client.BASE_URL = "https://api.worldquantbrain.com"
    client.timeout = 10.0

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.content = b"ok"
    fake_resp.json.return_value = {
        "results": [
            {"id": "socialmedia8", "valueScore": 1, "alphaCount": 17914, "coverage": 0.9, "category": {"id": "socialmedia"}},
            {"id": "pv1", "valueScore": 1, "alphaCount": 2000000, "coverage": 1.0, "category": {"id": "pv"}},
            {"id": "fundamental6", "valueScore": 1, "alphaCount": 1500000, "coverage": 0.95, "category": {"id": "fundamental"}},
            {"id": "analyst4", "valueScore": 1, "alphaCount": 795268, "coverage": 0.9, "category": {"id": "analyst"}},
            {"id": "option8", "valueScore": 1, "alphaCount": 236121, "coverage": 0.85, "category": {"id": "option"}},
        ]
    }
    client.session.request.return_value = fake_resp

    ranked = fetch_available_datasets(client)
    ids = [d["id"] for d in ranked]
    assert ids[0] == "fundamental6"
    assert set(ids[1:3]) == {"analyst4", "option8"}
    assert ids.index("fundamental6") < ids.index("socialmedia8")


def test_p36_bronze_403_fallback_runs_concurrently_with_300s_default_timeout() -> None:
    client = MagicMock()
    client.offline_mode = False
    client.is_authenticated = True
    client._multi_sim_forbidden = True
    client._unavailable_neutralizations = set()

    active = 0
    peak_active = 0
    seen_timeouts: list[float] = []
    lock = threading.Lock()

    def fake_sim(expr, settings=None, max_wait_seconds=None, poll_interval=None):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            seen_timeouts.append(float(max_wait_seconds or 0.0))
        time.sleep(0.08)
        with lock:
            active -= 1
        return {"success": True, "status": "COMPLETED", "alpha_id": f"ID_{expr}", "sharpe": 1.6, "fitness": 1.2}

    client.simulate_alpha.side_effect = fake_sim
    items = [
        {"expression": f"rank(f{i})", "dataset_id": "fundamental6", "stage": "STAGE1"}
        for i in range(6)
    ]

    with patch.dict(os.environ, {"BRAIN_BRONZE_CONCURRENCY": "3", "BRAIN_SIM_TIMEOUT_SEC": "300"}):
        res = simulate_multi_alpha(client, items, fallback_item_delay=0.0)

    assert len(res) == 6
    assert [r["alpha_id"] for r in res] == [f"ID_rank(f{i})" for i in range(6)]
    assert peak_active >= 2, f"Expected concurrent Bronze execution, got peak_active={peak_active}"
    assert all(t == 300.0 for t in seen_timeouts)


def test_p36_template_expander_removes_turnover_freezing_hump_and_noop_winsorize() -> None:
    for _ in range(30):
        assert _hump() <= 0.0015
        expr = _wrap_hygiene("operating_income / (equity + 0.001)", grp="subindustry")
        assert "winsorize(rank(" not in expr
        assert "hump(" not in expr
        cand = generate_diverse_domain_candidate("FUNDAMENTAL_CF_EV")
        cand_expr = cand[1] if isinstance(cand, tuple) and len(cand) >= 2 else str(cand)
        assert "winsorize(rank(" not in cand_expr
        m = re.search(r"hump=([0-9.]+)", cand_expr)
        if m:
            assert float(m.group(1)) <= 0.0015


def test_p36_gold_and_agonalpha_seeds_are_bronze_compliant() -> None:
    p36_seeds = [s for s in V8_ECONOMIC_HYPOTHESIS_SEEDS if str(s.get("name", "")).startswith("p36_")]
    assert len(p36_seeds) >= 6
    for s in p36_seeds:
        locked = find_locked_bronze_operators(str(s["expression"]))
        assert not locked, f"Seed {s['name']} uses locked Bronze operators: {locked}"


def test_p36_flush_daily_candidates_fast_skips_corr_clones_to_reach_diverse_alpha() -> None:
    from worldquant_brain.brain_client import BrainClient

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "vault.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE brain_candidate_pool ("
            "alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, "
            "fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, "
            "score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', "
            "created_at TEXT NOT NULL)"
        )
        # 3 clones of the same skeleton (high score, will fail SELF_CORRELATION) + 1 diverse alpha (will pass)
        rows = [
            ("CLONE1", "rank(ts_delta(close, 5))", 1.8, 1.4, 0.10, 0.0020, 95.0, "2026-11-22", "PENDING", "2026-11-22T00:00:00Z"),
            ("CLONE2", "rank(ts_delta(close, 10))", 1.7, 1.3, 0.10, 0.0020, 94.0, "2026-11-22", "PENDING", "2026-11-22T00:00:01Z"),
            ("CLONE3", "rank(ts_delta(close, 20))", 1.6, 1.2, 0.10, 0.0020, 93.0, "2026-11-22", "PENDING", "2026-11-22T00:00:02Z"),
            ("DIVERSE1", "group_rank(ts_rank(operating_income / equity, 126), subindustry)", 1.65, 1.35, 0.05, 0.0080, 90.0, "2026-11-22", "PENDING", "2026-11-22T00:00:03Z"),
        ]
        conn.executemany("INSERT INTO brain_candidate_pool VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()

        client = BrainClient(email="test@example.com", password="pw")
        client._quota_db = lambda: str(db_path)  # type: ignore[method-assign]
        client.get_daily_submission_count = lambda day=None: 0  # type: ignore[method-assign]

        submitted_calls: list[str] = []

        def fake_submit(aid, **kwargs):
            submitted_calls.append(aid)
            if aid.startswith("CLONE"):
                return {"success": False, "alpha_id": aid, "status": "SUBMISSION_CHECK_FAILED", "corr_rejected": True, "error": "SELF_CORRELATION=FAIL (value=0.95)"}
            return {"success": True, "alpha_id": aid, "status": "SUBMITTED"}

        client.submit_alpha = fake_submit  # type: ignore[method-assign]
        with patch.dict(os.environ, {"BRAIN_SUBMIT_ORDER": "best_first", "BRAIN_RESERVE_MIN_SHARPE": "99", "BRAIN_RESERVE_MIN_FITNESS": "99"}):
            res = client.flush_daily_candidates(day_str="2026-11-22", max_daily_submissions=2)

        # CLONE1 fails SELF_CORRELATION -> CLONE2 & CLONE3 fast-skipped without API calls -> DIVERSE1 submitted!
        assert submitted_calls == ["CLONE1", "DIVERSE1"]
        assert any(r["alpha_id"] == "DIVERSE1" and r["success"] for r in res)

        conn = sqlite3.connect(str(db_path))
        statuses = dict(conn.execute("SELECT alpha_id, status FROM brain_candidate_pool").fetchall())
        conn.close()
        assert statuses["CLONE1"] == "FAILED_CORR"
        assert statuses["CLONE2"] == "FAILED_CORR"
        assert statuses["CLONE3"] == "FAILED_CORR"
        assert statuses["DIVERSE1"] == "SUBMITTED"
