"""WorldQuant BRAIN Challenge score tracking with evidence classification.

The public/help documentation describes 2,000 as a daily score cap, not a
lifetime cap. Because authenticated API schemas can change, this tracker stores
raw snapshots and only marks the cap as API-confirmed when the authenticated
payload itself exposes a matching daily cap field.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl

TAIPEI = ZoneInfo("Asia/Taipei")
DEFAULT_DOCUMENTED_DAILY_CAP = 2000.0


@dataclass(frozen=True)
class ScoreSnapshot:
    snapshot_id: str
    observed_at: str
    taipei_date: str
    total_score: float | None
    daily_score: float | None
    daily_cap: float | None
    rank_value: int | None
    submission_days_60d: int | None
    uniqueness: float | None
    is_score: float | None
    source: str
    cap_evidence: str
    raw_payload: dict[str, Any]


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _find(payload: Mapping[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        current: Any = payload
        for part in path.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                current = None
                break
        if current is not None:
            return current
    return None


def parse_score_payload(payload: Mapping[str, Any], *, source: str = "AUTHENTICATED_API") -> ScoreSnapshot:
    """Parse several known/likely schema names without inventing missing values."""
    total = _number(_find(payload, (
        "leaderboard.score", "totalScore", "total_score", "score", "data.totalScore", "data.total_score", "leader.totalScore",
    )))
    daily = _number(_find(payload, (
        "dailyScore", "daily_score", "scoreToday", "todayScore", "data.dailyScore", "data.daily_score",
    )))
    cap = _number(_find(payload, (
        "dailyScoreCap", "daily_score_cap", "maxDailyScore", "data.dailyScoreCap", "data.daily_score_cap",
    )))
    rank = _integer(_find(payload, ("leaderboard.rank", "rank", "rankValue", "data.rank", "leader.rank")))
    days = _integer(_find(payload, (
        "daysOfSubmission", "submissionDays60d", "submission_days_60d", "data.daysOfSubmission",
    )))
    uniqueness = _number(_find(payload, ("uniqueness", "data.uniqueness", "scores.uniqueness")))
    is_score = _number(_find(payload, ("isScore", "is_score", "data.isScore", "scores.isScore")))
    cap_evidence = "AUTHENTICATED_API_CONFIRMED" if cap is not None else "DOCUMENTED_UNCONFIRMED_BY_API"
    if cap is None:
        cap = DEFAULT_DOCUMENTED_DAILY_CAP
    now = datetime.now(timezone.utc)
    return ScoreSnapshot(
        snapshot_id=uuid.uuid4().hex,
        observed_at=now.isoformat(),
        taipei_date=now.astimezone(TAIPEI).date().isoformat(),
        total_score=total,
        daily_score=daily,
        daily_cap=cap,
        rank_value=rank,
        submission_days_60d=days,
        uniqueness=uniqueness,
        is_score=is_score,
        source=source,
        cap_evidence=cap_evidence,
        raw_payload=dict(payload),
    )


class ScoreTracker:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS brain_score_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                observed_at TEXT NOT NULL,
                taipei_date TEXT NOT NULL,
                total_score REAL,
                daily_score REAL,
                daily_cap REAL,
                rank_value INTEGER,
                submission_days_60d INTEGER,
                uniqueness REAL,
                is_score REAL,
                source TEXT NOT NULL,
                cap_evidence TEXT NOT NULL,
                raw_payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_brain_score_date
                ON brain_score_snapshots(taipei_date, observed_at);
            CREATE TABLE IF NOT EXISTS brain_submission_score_attribution (
                attribution_id TEXT PRIMARY KEY,
                brain_alpha_id TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                score_before REAL,
                score_after REAL,
                score_delta REAL,
                score_observed_at TEXT,
                attribution_status TEXT NOT NULL,
                batch_date TEXT NOT NULL
            );
            """)
            ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS brain_score_snapshots (\n                snapshot_id TEXT PRIMARY KEY,\n                observed_at TEXT NOT NULL,\n                taipei_date TEXT NOT NULL,\n                total_score REAL,\n                daily_score REAL,\n                daily_cap REAL,\n                rank_value INTEGER,\n                submission_days_60d INTEGER,\n                uniqueness REAL,\n                is_score REAL,\n                source TEXT NOT NULL,\n                cap_evidence TEXT NOT NULL,\n                raw_payload_json TEXT NOT NULL\n            );\n            CREATE INDEX IF NOT EXISTS ix_brain_score_date\n                ON brain_score_snapshots(taipei_date, observed_at);\n            CREATE TABLE IF NOT EXISTS brain_submission_score_attribution (\n                attribution_id TEXT PRIMARY KEY,\n                brain_alpha_id TEXT NOT NULL,\n                submitted_at TEXT NOT NULL,\n                score_before REAL,\n                score_after REAL,\n                score_delta REAL,\n                score_observed_at TEXT,\n                attribution_status TEXT NOT NULL,\n                batch_date TEXT NOT NULL\n            );\n            ')

    def record(self, snapshot: ScoreSnapshot) -> ScoreSnapshot:
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO brain_score_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id, snapshot.observed_at, snapshot.taipei_date,
                snapshot.total_score, snapshot.daily_score, snapshot.daily_cap,
                snapshot.rank_value, snapshot.submission_days_60d,
                snapshot.uniqueness, snapshot.is_score, snapshot.source,
                snapshot.cap_evidence,
                json.dumps(snapshot.raw_payload, ensure_ascii=False, sort_keys=True),
            ))
        return snapshot

    def fetch_and_record(self, fetcher: Callable[[], Mapping[str, Any]], *, source: str) -> ScoreSnapshot:
        payload = fetcher()
        if not isinstance(payload, Mapping):
            raise TypeError("score fetcher must return a mapping")
        return self.record(parse_score_payload(payload, source=source))

    def latest(self) -> ScoreSnapshot | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM brain_score_snapshots ORDER BY observed_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return ScoreSnapshot(
            snapshot_id=row["snapshot_id"], observed_at=row["observed_at"],
            taipei_date=row["taipei_date"], total_score=row["total_score"],
            daily_score=row["daily_score"], daily_cap=row["daily_cap"],
            rank_value=row["rank_value"], submission_days_60d=row["submission_days_60d"],
            uniqueness=row["uniqueness"], is_score=row["is_score"], source=row["source"],
            cap_evidence=row["cap_evidence"], raw_payload=json.loads(row["raw_payload_json"]),
        )

    def cap_status(self) -> dict[str, Any]:
        latest = self.latest()
        if latest is None:
            return {
                "daily_cap": DEFAULT_DOCUMENTED_DAILY_CAP,
                "cap_evidence": "DOCUMENTED_UNCONFIRMED_BY_API",
                "daily_score": None,
                "reached": False,
                "stop_allowed": False,
            }
        confirmed = latest.cap_evidence == "AUTHENTICATED_API_CONFIRMED"
        reached = (
            confirmed and latest.daily_score is not None and latest.daily_cap is not None
            and latest.daily_score >= latest.daily_cap
        )
        return {
            "daily_cap": latest.daily_cap,
            "cap_evidence": latest.cap_evidence,
            "daily_score": latest.daily_score,
            "total_score": latest.total_score,
            "rank": latest.rank_value,
            "submission_days_60d": latest.submission_days_60d,
            "reached": reached,
            "stop_allowed": confirmed,
            "observed_at": latest.observed_at,
        }

    def history(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM brain_score_snapshots ORDER BY observed_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for r in reversed(rows):
            out.append({
                "snapshot_id": r["snapshot_id"],
                "observed_at": r["observed_at"],
                "taipei_date": r["taipei_date"],
                "total_score": r["total_score"],
                "daily_score": r["daily_score"],
                "daily_cap": r["daily_cap"],
                "rank_value": r["rank_value"],
                "submission_days_60d": r["submission_days_60d"],
                "uniqueness": r["uniqueness"],
                "is_score": r["is_score"],
                "source": r["source"],
                "cap_evidence": r["cap_evidence"],
            })
        return out

    def summary(self) -> dict[str, Any]:
        latest = self.latest()
        return {"latest": asdict(latest) if latest else None, "cap": self.cap_status()}
