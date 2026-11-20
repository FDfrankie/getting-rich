"""SQLite resilience helpers for cloud-synced folders (e.g. Google Drive) to prevent disk I/O crashes."""
from __future__ import annotations

import logging
import os
import re
import random
import sqlite3
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

_logger = logging.getLogger(__name__)


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
                except Exception as _exc:
                    _logger.debug("sqlite_resilience suppressed exception: %s", _exc)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("execute_with_sqlite_retry exhausted without result")


@contextmanager
def managed_connection(
    db_path: str,
    *,
    timeout: float = 15.0,
    journal_mode: str | None = None,
) -> Iterator[sqlite3.Connection]:
    """Safe SQLite connection contextmanager that always closes the FD and configures busy_timeout."""
    if db_path and db_path != ":memory:":
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=timeout)
    try:
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
            conn.execute("PRAGMA synchronous = NORMAL")
            mode = (journal_mode or os.getenv("SQLITE_JOURNAL_MODE", "")).strip().upper()
            if not mode:
                lower_path = str(db_path or "").lower()
                is_cloud = any(tok in lower_path for tok in ("google", "雲端硬碟", "onedrive", "dropbox"))
                mode = "DELETE" if is_cloud else "WAL"
            if db_path != ":memory:" and mode in {"DELETE", "WAL", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}:
                conn.execute(f"PRAGMA journal_mode = {mode}")
        except sqlite3.OperationalError:
            pass
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception as _exc:
            _logger.debug("sqlite_resilience suppressed exception: %s", _exc)
        raise
    finally:
        try:
            conn.close()
        except Exception as _exc:
            _logger.debug("sqlite_resilience suppressed exception: %s", _exc)


_DDL_TABLE_RE = re.compile(
    r'CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+["`\[]?([A-Za-z_][A-Za-z0-9_]*)["`\]]?\s*\(',
    re.IGNORECASE,
)

_TABLE_CONSTRAINT_STARTS = (
    "PRIMARY",
    "FOREIGN",
    "UNIQUE",
    "CHECK",
    "CONSTRAINT",
)


def _split_top_level_commas(body: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    for ch in body:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
                continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _sanitize_add_column_def(col_def: str) -> str:
    """Make a column definition safe for SQLite ALTER TABLE ADD COLUMN."""
    cleaned = re.sub(r"\bPRIMARY\s+KEY\b", "", col_def, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAUTOINCREMENT\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bUNIQUE\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "TEXT"
    # SQLite requires a DEFAULT when adding a NOT NULL column to a non-empty table
    if re.search(r"\bNOT\s+NULL\b", cleaned, flags=re.IGNORECASE) and not re.search(
        r"\bDEFAULT\b", cleaned, flags=re.IGNORECASE
    ):
        upper = cleaned.upper()
        if any(tok in upper for tok in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE")):
            cleaned = f"{cleaned} DEFAULT 0"
        else:
            cleaned = f"{cleaned} DEFAULT ''"
    return cleaned


def _extract_tables_from_ddl(ddl_sql: str) -> dict[str, dict[str, str]]:
    """Parse CREATE TABLE IF NOT EXISTS blocks into {table: {col: col_def}}."""
    tables: dict[str, dict[str, str]] = {}
    for match in _DDL_TABLE_RE.finditer(ddl_sql):
        table_name = match.group(1)
        start = match.end()
        depth = 1
        in_single = False
        in_double = False
        end = start
        while end < len(ddl_sql) and depth > 0:
            ch = ddl_sql[end]
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif not in_single and not in_double:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            end += 1
        if depth != 0:
            continue
        body = ddl_sql[start : end - 1]
        # Strip SQL -- line comments
        body_no_comments = re.sub(r"--[^\n]*", "", body)
        cols: dict[str, str] = {}
        for item in _split_top_level_commas(body_no_comments):
            if not item:
                continue
            first_tok = item.split(None, 1)[0].strip('"`[]').upper()
            if first_tok in _TABLE_CONSTRAINT_STARTS:
                continue
            parts = item.split(None, 1)
            col_name = parts[0].strip('"`[]')
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col_name):
                continue
            col_def = parts[1].strip() if len(parts) > 1 else "TEXT"
            cols[col_name] = _sanitize_add_column_def(col_def)
        if cols:
            tables.setdefault(table_name, {}).update(cols)
    return tables


def ensure_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
    expected_columns: dict[str, str],
) -> list[str]:
    """Auto-migrate missing columns onto an existing SQLite table without failing on legacy DBs."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError(f"Invalid SQLite table name: {table_name!r}")
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    existing = {str(row[1]) for row in cur.fetchall()}
    if not existing:
        return []
    added: list[str] = []
    for col_name, col_def in expected_columns.items():
        if col_name in existing:
            continue
        safe_def = _sanitize_add_column_def(col_def)
        try:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {safe_def}")
            existing.add(col_name)
            added.append(col_name)
            _logger.warning(
                "Auto-migrated legacy SQLite table %s: added missing column %s (%s)",
                table_name,
                col_name,
                safe_def,
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                _logger.warning(
                    "Could not auto-migrate column %s.%s (%s): %s",
                    table_name,
                    col_name,
                    safe_def,
                    exc,
                )
    return added


def ensure_schema_from_ddl(
    conn: sqlite3.Connection,
    ddl_sql: str,
) -> dict[str, list[str]]:
    """Verify every CREATE TABLE IF NOT EXISTS in ddl_sql has all declared columns on legacy DBs."""
    migrated: dict[str, list[str]] = {}
    for table_name, cols in _extract_tables_from_ddl(ddl_sql).items():
        added = ensure_table_columns(conn, table_name, cols)
        if added:
            migrated[table_name] = added
    return migrated
