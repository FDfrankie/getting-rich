"""P38 unit tests: Startup Consultant Reserve Sweep + Bronze 429 Worker Throttle + Official KB 116 Templates."""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from worldquant_brain.brain_client import WorldQuantBrainClient
from worldquant_brain.alpha_template_expander import _TEMPLATE_FAMILIES, generate_diverse_domain_candidate
from worldquant_brain.multi_sim_client import find_locked_bronze_operators
from scripts.headless_brain_miner import run_headless_miner


def test_p38_startup_sweep_protects_legacy_pending_gems_before_flush_hour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_RESERVE_MIN_SHARPE", "1.50")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_FITNESS", "1.20")

    db_path = tmp_path / "alpha_vault.db"
    client = WorldQuantBrainClient()
    client._quota_db = lambda: str(db_path)  # type: ignore[method-assign]

    # Seed legacy PENDING rows matching live 01:33 state (RReoNlgn 2.82, A1x5nL5W 2.01, rK7oWXmJ 1.51) + 1 low Bronze alpha
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, "
        "fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, "
        "candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
    )
    rows = [
        ("RReoNlgn", "rank(a)", 2.82, 2.40, 0.1414, 0.001446, 90.45, "2026-11-22", "PENDING", "2026-11-22T15:26:01Z"),
        ("A1x5nL5W", "rank(b)", 2.01, 1.32, 0.0630, 0.001721, 56.79, "2026-11-22", "PENDING", "2026-11-22T15:26:11Z"),
        ("rK7oWXmJ", "rank(c)", 1.51, 1.26, 0.0147, 0.011866, 41.88, "2026-11-22", "PENDING", "2026-11-22T15:26:11Z"),
        ("LOW_130", "rank(d)", 1.30, 1.05, 0.1500, 0.001000, 28.00, "2026-11-22", "PENDING", "2026-11-22T15:27:00Z"),
    ]
    conn.executemany("INSERT INTO brain_candidate_pool VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    moved = client.sweep_consultant_reserve(day_str="2026-11-23")
    assert moved == 3

    exp = client.export_qualified_formula_vault()
    assert exp["summary"]["consultant_reserve_count"] == 3
    assert {r["alpha_id"] for r in exp["consultant_reserve"]} == {"RReoNlgn", "A1x5nL5W", "rK7oWXmJ"}

    conn = sqlite3.connect(db_path)
    pending_left = [r[0] for r in conn.execute("SELECT alpha_id FROM brain_candidate_pool WHERE status='PENDING'").fetchall()]
    conn.close()
    assert pending_left == ["LOW_130"]


def test_p38_all_official_kb_templates_are_bronze_unlocked_and_hygienic() -> None:
    assert "OFFICIAL_KB_LOW_CORR" in _TEMPLATE_FAMILIES
    assert len(_TEMPLATE_FAMILIES["OFFICIAL_KB_LOW_CORR"]) >= 7

    for dom, builders in _TEMPLATE_FAMILIES.items():
        for builder in builders:
            for tr in (None, "WINSORIZE_BEFORE_RANK", "LIQUIDITY_TIERED_DECAY"):
                name, expr = builder(grp="industry", transform=tr)
                locked = find_locked_bronze_operators(expr)
                assert locked == [], f"Locked Bronze op {locked} in {dom}/{name}: {expr}"
                assert "winsorize(rank(" not in expr.replace(" ", "").lower(), f"No-op winsorize(rank()) in {expr}"

    cand = generate_diverse_domain_candidate("OFFICIAL_KB_LOW_CORR")
    assert cand["domain"] == "OFFICIAL_KB_LOW_CORR"
    assert find_locked_bronze_operators(cand["expression"]) == []


def test_p38_default_workers_is_one_for_bronze_rate_limit() -> None:
    sig = inspect.signature(run_headless_miner)
    assert sig.parameters["num_workers"].default == 1
