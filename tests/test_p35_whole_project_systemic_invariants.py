"""Cross-module systemic invariant tests for P35 (Threshold Alignment + Backlog Harvester + OrderManager WAL)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def test_cross_module_5bps_margin_and_turnover_alignment():
    from worldquant_brain.alpha_lifecycle import AlphaState, submission_eligibility
    from worldquant_brain.pre_submission_gate import evaluate_pre_submission

    # 7.52 bps margin passes lifecycle gate (5 bps floor)
    elig = submission_eligibility({
        "status": AlphaState.SUBMISSION_CANDIDATE.value,
        "sharpe": 1.67,
        "fitness": 1.02,
        "turnover": 0.2218,
        "margin_bps": 7.52,
        "drawdown": 0.0811,
        "is_check_status": "PASS",
        "self_correlation": 0.15,
        "brain_simulation_id": "sim_123",
    })
    assert elig.ready is True, f"Expected 7.52 bps eligible, got {elig.reasons}"

    # Pre-submission gate rejects 101% turnover (1.01) and enforces Delay-0 Fitness >= 1.30
    bad_to = evaluate_pre_submission({
        "alpha_id": "A1",
        "simulation_id": "S1",
        "metrics_source": "WORLDQUANT_BRAIN",
        "delay": 1,
        "sharpe": 2.0,
        "fitness": 1.2,
        "turnover": 1.01,
        "self_correlation": 0.1,
        "is_checks_passed": True,
    })
    assert "TURNOVER_NOT_PASSED" in bad_to.reasons

    bad_d0 = evaluate_pre_submission({
        "alpha_id": "A2",
        "simulation_id": "S2",
        "metrics_source": "WORLDQUANT_BRAIN",
        "delay": 0,
        "sharpe": 2.2,
        "fitness": 1.15,
        "turnover": 0.20,
        "self_correlation": 0.1,
        "is_checks_passed": True,
    })
    assert "FITNESS_NOT_PASSED" in bad_d0.reasons


def test_order_manager_enables_wal_and_idempotent_claim(tmp_path):
    from crypto_trading.order_manager import OrderManager

    db = tmp_path / "orders.db"
    om = OrderManager(db_path=str(db))
    c1 = om.claim_order("BTCUSDT", "BUY", "MARKET", 0.01, 50000.0, client_tag="t1")
    c2 = om.claim_order("BTCUSDT", "BUY", "MARKET", 0.01, 50000.0, client_tag="t1")
    assert c1["claimed"] is True
    assert c2["claimed"] is False and c2["error"] == "DUPLICATE_ORDER_REJECTED"

    conn = sqlite3.connect(db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert str(mode).lower() == "wal"


def test_watchdog_ps1_deduplicates_orphan_miners():
    ps1 = Path("scripts/watchdog_check.ps1")
    if not ps1.exists():
        ps1 = Path(__file__).resolve().parent.parent / "scripts/watchdog_check.ps1"
    txt = ps1.read_text(encoding="utf-8-sig")
    assert "$minerProcs.Count -gt 1" in txt
    assert "Stop-Process" in txt
