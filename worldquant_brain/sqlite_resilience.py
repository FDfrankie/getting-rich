"""SQLite resilience helpers for cloud-synced folders (e.g. Google Drive) to prevent disk I/O crashes."""
from __future__ import annotations

import os
import random
import sqlite3
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_TRANSIENT_SQLITE_TOKENS = (
    "disk i/o error",
    "database is locked",
    "database table is locked",
    "unable to open database file",
    "attempt to write a readonly database",
)


def is_transient_sqlite_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return any(tok in msg for tok in _TRANSIENT_SQLITE_TOKENS)


def execute_with_sqlite_retry(
    db_path: str,
    fn: Callable[[sqlite3.Connection], T],
    *,
    max_retries: int = 5,
    base_delay: float = 0.08,
    timeout: float = 15.0,
) -> T:
    """Run a SQLite unit-of-work with exponential backoff on transient Google Drive / lock I/O errors."""
    if db_path and db_path != ":memory:":
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    last_exc: Exception | None = None
    for attempt in range(max(1, int(max_retries))):
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            try:
                conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
                conn.execute("PRAGMA synchronous = NORMAL")
            except sqlite3.OperationalError:
                pass
            with conn:
                return fn(conn)
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if not is_transient_sqlite_error(exc) or attempt >= max_retries - 1:
                raise
            sleep_s = base_delay * (2 ** attempt) + random.uniform(0.0, 0.03)
            time.sleep(sleep_s)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("execute_with_sqlite_retry exhausted without result")
