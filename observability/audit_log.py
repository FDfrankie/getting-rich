"""Append-only structured audit log with basic secret scrubbing."""
from __future__ import annotations

import json
import re
import sqlite3
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    component TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT,
                    correlation_id TEXT,
                    entity_type TEXT,
                    entity_id TEXT,
                    outcome TEXT NOT NULL,
                    details_json TEXT NOT NULL
                )
            """)

    def write(self, *, component: str, action: str, outcome: str, details: Mapping[str, Any] | None = None,
              actor: str | None = None, correlation_id: str | None = None,
              entity_type: str | None = None, entity_id: str | None = None) -> str:
        event_id = uuid.uuid4().hex
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_id, datetime.now(timezone.utc).isoformat(), component, action, actor,
                  correlation_id, entity_type, entity_id, outcome,
                  json.dumps(scrub(details or {}), sort_keys=True, ensure_ascii=False)))
        return event_id
