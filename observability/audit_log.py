"""Append-only structured audit log with basic secret scrubbing."""
from __future__ import annotations

import json
import re
import sqlite3
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_SECRET = re.compile(r"(?i)(password|secret|token|authorization|api[_-]?key)")


def scrub(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): ("***" if _SECRET.search(str(k)) else scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


class AuditLog:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        _audit_ddl = """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL DEFAULT '',
                occurred_at TEXT NOT NULL DEFAULT '',
                component TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL DEFAULT '',
                actor TEXT NOT NULL DEFAULT '',
                correlation_id TEXT,
                entity_type TEXT,
                entity_id TEXT,
                outcome TEXT NOT NULL DEFAULT '',
                previous_state TEXT,
                next_state TEXT,
                reason TEXT,
                payload_json TEXT,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT ''
            )
        """
        from worldquant_brain.sqlite_resilience import managed_connection
        with managed_connection(self.db_path, timeout=30.0) as conn:
            conn.execute(_audit_ddl)
            ensure_schema_from_ddl(conn, _audit_ddl)

    def write(self, *, component: str, action: str, outcome: str, details: Mapping[str, Any] | None = None,
              actor: str | None = None, correlation_id: str | None = None,
              entity_type: str | None = None, entity_id: str | None = None) -> str:
        event_id = uuid.uuid4().hex
        now_iso = datetime.now(timezone.utc).isoformat()
        details_str = json.dumps(scrub(details or {}), sort_keys=True, ensure_ascii=False)
        from worldquant_brain.sqlite_resilience import managed_connection
        with managed_connection(self.db_path, timeout=30.0) as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    event_id, event_type, occurred_at, component, action, actor,
                    correlation_id, entity_type, entity_id, outcome,
                    payload_json, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    f"{component}.{action}",
                    now_iso,
                    component,
                    action,
                    actor or "",
                    correlation_id,
                    entity_type,
                    entity_id,
                    outcome,
                    details_str,
                    details_str,
                    now_iso,
                ),
            )
        return event_id
