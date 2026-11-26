import json
import sqlite3
from pathlib import Path

from scripts.export_crypto_pnl_to_drive import (
    export_crypto_pnl_to_drive,
    fetch_exchange_account_ro,
    read_sqlite_pnl_summary,
)


def _create_sample_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            price REAL,
            amount REAL,
            pnl REAL,
            exchange_order_id TEXT
        )
        """
    )
    conn.execute("CREATE TABLE bot_state (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO bot_state (key, value) VALUES ('trading_mode', 'BINANCE_TESTNET')")
    # Synthetic unit test rows (must be filtered out)
    conn.execute(
        "INSERT INTO trade_logs (timestamp, symbol, action, price, amount, pnl, exchange_order_id) "
        "VALUES ('2026-11-20T00:00:00Z', 'BTC/USDT', 'CLOSE_LONG', 55000.0, 1.0, 5000.0, 'PAPER_123')"
    )
    conn.execute(
        "INSERT INTO trade_logs (timestamp, symbol, action, price, amount, pnl, exchange_order_id) "
        "VALUES ('2026-11-20T00:01:00Z', 'BTC/USDT', 'CLOSE_SHORT', 45000.0, 1.0, 5000.0, 'PAPER_124')"
    )
    # Real closes
    conn.execute(
        "INSERT INTO trade_logs (timestamp, symbol, action, price, amount, pnl, exchange_order_id) "
        "VALUES ('2026-11-21T01:00:00Z', 'BTC/USDT', 'CLOSE_LONG', 91234.5, 0.05, 42.5, 'BINANCE_9001')"
    )
    conn.execute(
        "INSERT INTO trade_logs (timestamp, symbol, action, price, amount, pnl, exchange_order_id) "
        "VALUES ('2026-11-21T02:00:00Z', 'ETH/USDT', 'CLOSE_SHORT', 3100.0, 0.5, -12.5, 'BINANCE_9002')"
    )
    conn.commit()
    conn.close()


def test_sqlite_filters_synthetic_rows_and_computes_pnl(tmp_path: Path) -> None:
    db_path = tmp_path / "system_state.db"
    _create_sample_db(db_path)

    summary = read_sqlite_pnl_summary(db_path=db_path)
    assert summary["db_exists"] is True
    assert summary["filtered_test_rows_count"] == 2
    assert summary["total_closes"] == 2
    assert summary["win_count"] == 1
    assert summary["win_rate_pct"] == 50.0
    assert summary["realized_pnl_usdt"] == 30.0
    assert summary["bot_state"]["trading_mode"] == "BINANCE_TESTNET"


def test_exchange_missing_keys_graceful_degradation(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET", raising=False)
    res = fetch_exchange_account_ro("BINANCE_TESTNET")
    assert res["connected"] is False
    assert "Missing API key/secret" in (res["exchange_error"] or "")


def test_atomic_export_writes_json_and_markdown(tmp_path: Path) -> None:
    db_path = tmp_path / "system_state.db"
    _create_sample_db(db_path)
    json_out = tmp_path / "crypto_pnl_status.json"
    md_out = tmp_path / "CRYPTO_PNL_DASHBOARD.md"

    payload = export_crypto_pnl_to_drive(
        json_path=json_out,
        md_path=md_out,
        db_path=db_path,
        exchange_override={
            "trading_mode": "BINANCE_TESTNET",
            "connected": True,
            "total_equity_usdt": 8450.25,
            "free_balance_usdt": 8000.0,
            "unrealized_pnl_usdt": 15.5,
            "open_positions": [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "LONG",
                    "size": 0.01,
                    "entry_price": 90000.0,
                    "mark_price": 91550.0,
                    "unrealized_pnl": 15.5,
                    "leverage": 3,
                }
            ],
            "exchange_error": None,
        },
    )
    assert payload["sqlite"]["realized_pnl_usdt"] == 30.0
    assert json_out.exists()
    assert md_out.exists()
    loaded = json.loads(json_out.read_text(encoding="utf-8"))
    assert loaded["exchange"]["total_equity_usdt"] == 8450.25
    md_text = md_out.read_text(encoding="utf-8")
    assert "Crypto PnL Cloud Dashboard" in md_text
    assert "8450.25" in md_text
    assert "BTC/USDT:USDT" in md_text
