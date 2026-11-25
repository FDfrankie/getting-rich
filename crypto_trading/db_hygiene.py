"""Safe cleanup helper for synthetic unit-test rows that polluted the live state DB."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

SYNTHETIC_REASONS = ("Open Long", "Close Long", "Open Short", "Close Short", "unit_test")
SYNTHETIC_EXCHANGE_IDS = ("987654321",)


def purge_synthetic_test_rows(db_path: str | None = None, *, dry_run: bool = True, backup_dir: str | None = None) -> dict:
    """Delete synthetic test rows from `trade_logs` and `orders` (never touches real BOT/TESTNET rows).

    dry_run=True only counts. Otherwise takes a sqlite3.backup() snapshot first and deletes in one IMMEDIATE txn.
    """
    if db_path is None:
        from data_persistence.gdrive_sync import STATE_DB_PATH

        db_path = STATE_DB_PATH
    ph = ",".join("?" * len(SYNTHETIC_REASONS))
    ph_ids = ",".join("?" * len(SYNTHETIC_EXCHANGE_IDS))
    where_tl = f"reason IN ({ph})"
    where_ord = f"(reason IN ({ph}) OR exchange_order_id IN ({ph_ids}))"
    params_tl = SYNTHETIC_REASONS
    params_ord = SYNTHETIC_REASONS + SYNTHETIC_EXCHANGE_IDS
    result: dict = {"db_path": db_path, "dry_run": dry_run}
    with closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        result["trade_logs_matched"] = conn.execute(f"SELECT COUNT(*) FROM trade_logs WHERE {where_tl}", params_tl).fetchone()[0]
        result["orders_matched"] = conn.execute(f"SELECT COUNT(*) FROM orders WHERE {where_ord}", params_ord).fetchone()[0]
        if dry_run:
            return result
        bdir = backup_dir or os.path.join(os.path.dirname(os.path.abspath(db_path)), "backups")
        os.makedirs(bdir, exist_ok=True)
        bfile = os.path.join(bdir, f"pre_purge_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.db")
        with closing(sqlite3.connect(bfile)) as dst:
            conn.backup(dst)
        result["backup"] = bfile
        conn.execute("BEGIN IMMEDIATE")
        try:
            result["trade_logs_deleted"] = conn.execute(f"DELETE FROM trade_logs WHERE {where_tl}", params_tl).rowcount
            result["orders_deleted"] = conn.execute(f"DELETE FROM orders WHERE {where_ord}", params_ord).rowcount
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return result
