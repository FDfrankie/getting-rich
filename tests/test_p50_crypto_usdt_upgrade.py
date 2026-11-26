"""Unit tests for P50 Crypto USDT Settlement, Fee-Aware Barrier, Ratcheting Stop & Cloud PnL Fixes."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_trading.execution_cost_filter import (
    USDT_M_MAKER_FEE_ONE_WAY,
    USDT_M_TAKER_FEE_ONE_WAY,
    estimate_usdt_roundtrip_fee,
    evaluate_usdt_cost_barrier,
)
from crypto_trading.settings import DEFAULT_SETTLEMENT_CURRENCY, RISK_PARAMS, TradingMode


def test_default_settlement_is_usdt_and_fee_schedule_calibrated():
    assert DEFAULT_SETTLEMENT_CURRENCY == "USDT"
    assert RISK_PARAMS["maker_fee_pct"] == 0.0002
    assert RISK_PARAMS["taker_fee_pct"] == 0.0005
    assert RISK_PARAMS["max_stop_loss_pct"] <= 0.05


def test_usdt_roundtrip_fee_and_cost_barrier_maker_vs_taker():
    rt_maker = estimate_usdt_roundtrip_fee(entry_mode="maker", exit_mode="maker")
    rt_taker = estimate_usdt_roundtrip_fee(entry_mode="taker", exit_mode="taker")
    rt_maker_bnb = estimate_usdt_roundtrip_fee(entry_mode="maker", exit_mode="maker", use_bnb_discount=True)

    assert abs(rt_maker - 0.0004) < 1e-9
    assert abs(rt_taker - 0.0010) < 1e-9
    assert abs(rt_maker_bnb - 0.00036) < 1e-9
    assert USDT_M_MAKER_FEE_ONE_WAY == 0.0002
    assert USDT_M_TAKER_FEE_ONE_WAY == 0.0005

    # A 20 bps expected edge should clear Maker barrier (cost ~6 bps -> req ~7.5 bps)
    # but be REJECTED by Taker barrier (cost ~14.5 bps -> req ~18.1 bps)
    dec_maker = evaluate_usdt_cost_barrier(expected_return=0.0020, entry_mode="maker", exit_mode="maker")
    dec_taker = evaluate_usdt_cost_barrier(expected_return=0.0020, entry_mode="taker", exit_mode="taker")
    assert dec_maker.trade_allowed is True
    assert dec_taker.trade_allowed is False


def test_export_crypto_pnl_unwraps_trading_mode_enum_and_sums_fees(tmp_path: Path):
    from scripts.export_crypto_pnl_to_drive import fetch_exchange_account_ro, read_sqlite_pnl_summary

    # Enum passed directly must unwrap cleanly (never "TRADINGMODE.LOCAL_PAPER")
    res = fetch_exchange_account_ro(TradingMode.LOCAL_PAPER)
    assert res["trading_mode"] == "LOCAL_PAPER"
    assert res["connected"] is False

    # SQLite summary must aggregate fees from filled non-test orders
    db = tmp_path / "test_state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE trade_logs (id INTEGER PRIMARY KEY, symbol TEXT, action TEXT, price REAL, amount REAL, pnl REAL, timestamp TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, exchange_order_id TEXT, price REAL, fee REAL, status TEXT)")
    conn.execute("CREATE TABLE bot_state (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO trade_logs VALUES (1, 'BTC/USDT', 'CLOSE', 65000.0, 0.1, 120.5, '2026-11-26 12:00:00')")
    conn.execute("INSERT INTO orders VALUES (1, 'EX_1', 65000.0, 1.30, 'FILLED')")
    conn.execute("INSERT INTO orders VALUES (2, 'PAPER_999', 50000.0, 25.0, 'FILLED')")  # test fixture excluded
    conn.execute("INSERT INTO bot_state VALUES ('execution_mode', 'BINANCE_TESTNET')")
    conn.commit()
    conn.close()

    sq = read_sqlite_pnl_summary(db_path=db)
    assert sq["total_closes"] == 1
    assert sq["realized_pnl_usdt"] == 120.5
    assert sq["total_fees_usdt"] == 1.3
