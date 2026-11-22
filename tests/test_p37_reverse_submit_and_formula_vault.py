"""P37 unit tests: Permanent Formula Archive + Bronze Reverse-Order Submit + Consultant Reserve Protection."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from worldquant_brain.brain_client import WorldQuantBrainClient
from scripts.headless_brain_miner import sync_vault_snapshot_to_drive


def _make_client(tmp_path: Path) -> WorldQuantBrainClient:
    db_path = tmp_path / "alpha_vault.db"
    client = WorldQuantBrainClient()
    client._quota_db = lambda: str(db_path)  # type: ignore[method-assign]
    return client


def test_p37_bronze_reverse_submit_and_consultant_reserve_protection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_SUBMIT_ORDER", "worst_first")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_SHARPE", "1.50")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_FITNESS", "1.20")

    client = _make_client(tmp_path)

    # 1. Stage 3 alphas: worst qualifying, medium qualifying, and high-value Consultant gem
    st_gem = client.stage_candidate_for_daily_selection(
        "GEM_168", "rank(ts_delta(fnd6_ebitda, 60))", sharpe=1.68, fitness=1.35, turnover=0.12, margin=0.0025,
        day_str="2026-11-22", settings={"region": "USA", "universe": "TOP3000", "decay": 6},
    )
    st_mid = client.stage_candidate_for_daily_selection(
        "MID_142", "rank(ts_rank(anl4_eps, 20))", sharpe=1.42, fitness=1.15, turnover=0.18, margin=0.0012,
        day_str="2026-11-22", settings={"region": "USA", "universe": "TOP3000", "decay": 4},
    )
    st_worst = client.stage_candidate_for_daily_selection(
        "WORST_128", "rank(-goodwill / sales)", sharpe=1.28, fitness=1.03, turnover=0.15, margin=0.0009,
        day_str="2026-11-22", settings={"region": "USA", "universe": "TOP3000", "decay": 4},
    )

    assert st_gem == "CONSULTANT_RESERVE"
    assert st_mid == "PENDING"
    assert st_worst == "PENDING"

    # 2. Also inject a legacy pre-P37 high-Sharpe row directly as PENDING to test defense-in-depth guard
    conn = sqlite3.connect(client._quota_db())
    conn.execute(
        "INSERT INTO brain_candidate_pool VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', '2026-11-22T00:00:00Z')",
        ("LEGACY_GEM_155", "rank(cashflow / enterprise_value)", 1.55, 1.18, 0.10, 0.0020, 99.0, "2026-11-22"),
    )
    conn.commit()
    conn.close()

    submitted_order: list[str] = []

    def _fake_submit(aid: str, **kwargs):
        submitted_order.append(aid)
        return {"success": True, "status": "SUBMITTED", "alpha_id": aid}

    client.submit_alpha = _fake_submit  # type: ignore[method-assign]
    res = client.flush_daily_candidates(day_str="2026-11-22", max_daily_submissions=2)

    # Worst qualifying MUST be submitted first, then medium; BOTH gems (>=1.50) MUST be protected!
    assert submitted_order == ["WORST_128", "MID_142"]
    assert len(res) == 2

    # Verify exported formula vault preserves ALL 4 formulas permanently
    vault_path = tmp_path / "qualified_alpha_formulas_vault.json"
    exp = client.export_qualified_formula_vault(output_path=vault_path)
    assert vault_path.exists()
    data = json.loads(vault_path.read_text(encoding="utf-8"))
    assert data["summary"]["total_archived"] == 4
    assert data["summary"]["consultant_reserve_count"] == 2
    assert data["summary"]["submitted_count"] == 2
    reserve_ids = {r["alpha_id"] for r in data["consultant_reserve"]}
    assert reserve_ids == {"GEM_168", "LEGACY_GEM_155"}
    gem_rec = next(r for r in data["consultant_reserve"] if r["alpha_id"] == "GEM_168")
    assert gem_rec["expression"] == "rank(ts_delta(fnd6_ebitda, 60))"
    assert gem_rec["settings"]["universe"] == "TOP3000"


def test_p37_consultant_mode_unlocks_best_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_CONSULTANT_MODE", "true")
    monkeypatch.delenv("BRAIN_SUBMIT_ORDER", raising=False)

    client = _make_client(tmp_path)
    assert client._submit_order_mode() == "best_first"
    assert client._is_consultant_reserve(1.80, 1.50) is False

    st_hi = client.stage_candidate_for_daily_selection(
        "HI_180", "rank(ts_delta(fnd6_ebitda, 60))", sharpe=1.80, fitness=1.50, turnover=0.14, margin=0.0025,
        day_str="2026-11-22",
    )
    st_lo = client.stage_candidate_for_daily_selection(
        "LO_130", "rank(-goodwill / sales)", sharpe=1.30, fitness=1.05, turnover=0.14, margin=0.0010,
        day_str="2026-11-22",
    )
    assert st_hi == "PENDING"
    assert st_lo == "PENDING"

    submitted_order: list[str] = []
    client.submit_alpha = lambda aid, **kw: (submitted_order.append(aid) or {"success": True, "status": "SUBMITTED", "alpha_id": aid})  # type: ignore[method-assign]
    client.flush_daily_candidates(day_str="2026-11-22", max_daily_submissions=1)
    assert submitted_order == ["HI_180"]


def test_p37_harvest_unsubmitted_uses_asc_order_and_archives_top_gems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_SUBMIT_ORDER", "worst_first")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_SHARPE", "1.50")

    client = _make_client(tmp_path)
    client.email = "test@example.com"
    client.password = "secret"
    client.authenticated = True

    calls: list[dict] = []

    def _fake_get(url, params=None, timeout=20.0):
        calls.append(dict(params or {}))
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"ok"
        if (params or {}).get("order") == "is.fitness":
            resp.json.return_value = {
                "results": [{
                    "id": "LOW_Q1",
                    "regular": {"code": "rank(close / ts_delay(close, 5))"},
                    "settings": {"delay": 1, "universe": "TOP3000"},
                    "is": {"sharpe": 1.31, "fitness": 1.04, "turnover": 0.16, "margin": 0.0011, "checks": []},
                }]
            }
        else:
            resp.json.return_value = {
                "results": [{
                    "id": "TOP_GEM1",
                    "regular": {"code": "rank(ts_zscore(fnd6_optgr, 60))"},
                    "settings": {"delay": 1, "universe": "TOP3000"},
                    "is": {"sharpe": 1.92, "fitness": 1.58, "turnover": 0.11, "margin": 0.0031, "checks": []},
                }]
            }
        return resp

    client.session = MagicMock()
    client.session.get.side_effect = _fake_get

    staged = client.harvest_unsubmitted_candidates(limit=20, day_str="2026-11-22")
    assert staged == 1
    assert [c["order"] for c in calls] == ["is.fitness", "-is.fitness"]

    exp = client.export_qualified_formula_vault()
    assert exp["summary"]["total_archived"] == 2
    assert exp["summary"]["consultant_reserve_count"] == 1
    assert exp["consultant_reserve"][0]["alpha_id"] == "TOP_GEM1"
    assert exp["bronze_submittable_worst_first"][0]["alpha_id"] == "LOW_Q1"


def test_p37_sync_vault_snapshot_exports_formula_vault_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_SUBMIT_ORDER", "worst_first")

    client = _make_client(tmp_path)
    client.stage_candidate_for_daily_selection("A_GEM", "rank(a)", 1.72, 1.40, 0.12, 0.0020, day_str="2026-11-22")
    client.stage_candidate_for_daily_selection("A_LOW", "rank(b)", 1.29, 1.02, 0.15, 0.0010, day_str="2026-11-22")

    status_path = tmp_path / "out" / "brain_pending_status.json"
    out = sync_vault_snapshot_to_drive(
        client._quota_db(),
        mirror_dir=tmp_path / "mirror",
        status_json_path=status_path,
    )
    assert out["status_written"] is True
    assert out.get("formula_vault_written") is True
    st_json = json.loads(status_path.read_text(encoding="utf-8"))
    assert st_json["submit_order"] == "worst_first"
    assert st_json["pending_count"] == 1
    assert st_json["consultant_reserve_count"] == 1
    assert st_json["archived_formulas_total"] == 2
    vault_json = status_path.parent / "qualified_alpha_formulas_vault.json"
    assert vault_json.exists()
