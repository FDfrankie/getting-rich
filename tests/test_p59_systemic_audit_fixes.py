"""Unit tests for P59 Whole-Project Systemic Audit & Bugfix Bundle."""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest


def test_watchdog_ps1_single_sync_and_unified_python():
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "watchdog_check.ps1"
    text = p.read_text(encoding="utf-8")
    assert "function Resolve-QuantPython" in text
    assert text.count("export_crypto_pnl_to_drive.py") == 1
    assert "headless_crypto_bot_err.log" in text
    assert "$cryptoProcs.Count -gt 1" in text


def test_headless_crypto_bot_normalizes_shared_equity_and_heartbeat(tmp_path):
    from scripts.headless_crypto_bot import build_pnl_snapshot, _atomic_write_json

    mgr = MagicMock()
    mgr.status_summary.return_value = [
        {"symbol": "BTC/USDT", "position_side": "FLAT", "position": 0.0, "entry_price": 0.0, "balance_usdt": 5000.0, "equity_usdt": 5000.0},
        {"symbol": "ETH/USDT", "position_side": "FLAT", "position": 0.0, "entry_price": 0.0, "balance_usdt": 5000.0, "equity_usdt": 5000.0},
        {"symbol": "SOL/USDT", "position_side": "FLAT", "position": 0.0, "entry_price": 0.0, "balance_usdt": 5000.0, "equity_usdt": 5000.0},
    ]
    snap = build_pnl_snapshot(mgr, initial_total_equity=5000.0, cycle=1, trading_mode="LOCAL_PAPER")
    assert snap["total_equity_usdt"] == 5000.0
    assert snap["pnl_usdt"] == 0.0
    assert len(snap["symbols"]) == 3

    out = tmp_path / "crypto_bot_heartbeat.json"
    _atomic_write_json(out, snap)
    assert json.loads(out.read_text(encoding="utf-8"))["total_equity_usdt"] == 5000.0


def test_alpha_vault_atomic_json_export(tmp_path):
    from worldquant_brain.alpha_vault import AlphaVault

    db_p = tmp_path / "vault.db"
    json_p = tmp_path / "verified_alphas.json"
    v = AlphaVault(str(db_p))
    res = v.save_alpha(
        name="TEST_ALPHA_1",
        expression="rank(close / ts_delay(close, 5))",
        sharpe=1.65,
        fitness=1.35,
        turnover=0.18,
    )
    assert res["success"] is True
    v.export_to_json(str(json_p))
    loaded = json.loads(json_p.read_text(encoding="utf-8"))
    assert len(loaded) == 1
    assert loaded[0]["name"] == "TEST_ALPHA_1"
    # Ensure no leftover .tmp files
    assert not list(tmp_path.glob("*.tmp"))


def test_brain_client_vault_summary_balances_total_archived(tmp_path):
    from worldquant_brain.brain_client import WorldQuantBrainClient

    db_p = tmp_path / "alpha_vault.db"
    conn = sqlite3.connect(str(db_p))
    conn.execute(
        """CREATE TABLE qualified_alpha_archive (
            alpha_id TEXT PRIMARY KEY, expression TEXT, sharpe REAL, fitness REAL,
            turnover REAL, margin REAL, returns REAL, drawdown REAL,
            sub_universe_sharpe REAL, tier TEXT, submit_status TEXT, source TEXT,
            archived_at TEXT, updated_at TEXT, settings_json TEXT
        )"""
    )
    rows = [
        ("A1", "rank(close)", 2.1, 1.8, 0.12, 0.002, 0.1, 0.05, 1.5, "CONSULTANT_RESERVE", "CONSULTANT_RESERVE", "M", "t", "t", "{}"),
        ("A2", "rank(open)", 1.3, 1.05, 0.12, 0.002, 0.1, 0.05, 1.1, "BRONZE_SUBMITTABLE", "PENDING", "M", "t", "t", "{}"),
        ("A3", "rank(high)", 1.4, 1.1, 0.12, 0.002, 0.1, 0.05, 1.1, "BRONZE_SUBMITTABLE", "SUBMITTED", "M", "t", "t", "{}"),
        ("A4", "rank(low)", 1.35, 1.08, 0.12, 0.002, 0.1, 0.05, 1.1, "BRONZE_SUBMITTABLE", "FAILED_CORR", "M", "t", "t", "{}"),
    ]
    conn.executemany("INSERT INTO qualified_alpha_archive VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    client = WorldQuantBrainClient()
    client._quota_db = lambda: str(db_p)
    out_json = tmp_path / "qualified_alpha_formulas_vault.json"
    payload = client.export_qualified_formula_vault(output_path=out_json)
    s = payload["summary"]
    assert s["total_archived"] == 4
    assert s["consultant_reserve_count"] == 1
    assert s["bronze_submittable_count"] == 1
    assert s["submitted_count"] == 1
    assert s["rejected_or_excluded_count"] == 1
    assert (
        s["consultant_reserve_count"]
        + s["bronze_submittable_count"]
        + s["submitted_count"]
        + s["rejected_or_excluded_count"]
        == s["total_archived"]
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_export_crypto_pnl_pid_atomic_write(tmp_path):
    from scripts.export_crypto_pnl_to_drive import _atomic_write_text

    target = tmp_path / "CRYPTO_PNL_DASHBOARD.md"
    _atomic_write_text(target, "# ok")
    assert target.read_text(encoding="utf-8") == "# ok"
    assert not list(tmp_path.glob("*.tmp"))


def test_bot_runner_reversal_exit_present():
    p = pathlib.Path(__file__).resolve().parents[1] / "crypto_trading" / "bot_runner.py"
    text = p.read_text(encoding="utf-8")
    assert "CRYPTO_STRATEGY_PROFILE" in text
    assert "策略反向空頭信號確認平多" in text
    assert "策略反向多頭信號確認平空" in text


def test_headless_brain_miner_boot_banner_and_single_export():
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "headless_brain_miner.py"
    text = p.read_text(encoding="utf-8")
    assert "[MINER_BOOT]" in text
    assert "os.getpid()" in text
