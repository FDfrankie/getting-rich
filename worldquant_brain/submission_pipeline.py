"""Unified WorldQuant BRAIN Submission Pipeline.

Consolidates:
- DirectSubmitService (from direct_submit_service.py)
- SubmissionService (from submission_service.py)
- SubmitAllAvailablePolicy / PolicyDecision (from adaptive_submission.py)
- BrainAutoSubmitGate / AlphaStatus / migrate_alpha_status (from brain_auto_submit.py)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from worldquant_brain.alpha_repository import AlphaRepository
from worldquant_brain.daily_candidate_selector import SelectionBatch, select_daily_candidates
from worldquant_brain.pre_submission_gate import GateVerdict, evaluate_pre_submission
from worldquant_brain.score_tracker import ScoreTracker
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl

logger = logging.getLogger("BrainAutoSubmit")

SUBMIT_ACK_ENV = "BRAIN_AUTO_SUBMIT_ACK"
REQUIRED_SUBMIT_ACK = "I_UNDERSTAND_SUBMISSION_IS_IRREVERSIBLE"
KILL_SWITCH_ENV = "BRAIN_SUBMISSION_KILL_SWITCH"


class AlphaStatus(str, Enum):
    GENERATED = "GENERATED"
    LOCAL_SCREENED = "LOCAL_SCREENED"
    DEMO_SAMPLE = "DEMO_SAMPLE"
    LEGACY_UNVERIFIED = "LEGACY_UNVERIFIED"
    SIMULATION_QUEUED = "SIMULATION_QUEUED"
    BRAIN_RUNNING = "BRAIN_RUNNING"
    BRAIN_COMPLETE = "BRAIN_COMPLETE"
    BRAIN_REJECTED = "BRAIN_REJECTED"
    SUBMISSION_CHECK_PENDING = "SUBMISSION_CHECK_PENDING"
    READY_FOR_SUBMIT = "READY_FOR_SUBMIT"
    SUBMISSION_CLAIMED = "SUBMISSION_CLAIMED"
    SUBMITTED = "SUBMITTED"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"


def migrate_alpha_status(alpha_record: Dict[str, Any]) -> str:
    current_status = alpha_record.get("status", "")
    is_seed = alpha_record.get("is_seed", False) or alpha_record.get("source") == "seed"
    if is_seed:
        return AlphaStatus.DEMO_SAMPLE.value

    has_location = bool(alpha_record.get("simulation_location") or alpha_record.get("location"))
    has_alpha_id = bool(alpha_record.get("alpha_id") or alpha_record.get("simulation_id"))
    is_local_proxy = alpha_record.get("is_local_proxy", False) or alpha_record.get("proxy_type") == "BTC"

    if is_local_proxy:
        return AlphaStatus.LOCAL_SCREENED.value

    if current_status in ("VERIFIED", "PASS", "PASSED"):
        if not (has_location or has_alpha_id):
            return AlphaStatus.LEGACY_UNVERIFIED.value
        return AlphaStatus.BRAIN_COMPLETE.value

    return current_status or AlphaStatus.GENERATED.value


class BrainAutoSubmitGate:
    def __init__(
        self,
        min_sharpe: float = 1.25,
        min_fitness: float = 1.0,
        min_margin: float = 0.0010,
        min_turnover: float = 0.01,
        max_turnover: float = 0.70,
        max_self_correlation: float = 0.70,
        max_sub_universe_sharpe_decay: float = 0.50,
        daily_quota: int = 10,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.min_sharpe = min_sharpe
        self.min_fitness = min_fitness
        self.min_margin = min_margin
        self.min_turnover = min_turnover
        self.max_turnover = max_turnover
        self.max_self_correlation = max_self_correlation
        self.max_sub_universe_sharpe_decay = max_sub_universe_sharpe_decay
        self.daily_quota = daily_quota
        self.cooldown_seconds = cooldown_seconds

        self._submitted_today = 0
        self._last_submitted_at: float = 0.0
        self._submitted_keys: set[str] = set()
        self._lock = threading.Lock()
        self.audit_log: List[Dict[str, Any]] = []

    def is_kill_switch_active(self) -> bool:
        val = os.getenv(KILL_SWITCH_ENV, "0").strip().lower()
        return val in ("1", "true", "yes", "on")

    def is_ack_valid(self) -> bool:
        return os.getenv(SUBMIT_ACK_ENV, "").strip() == REQUIRED_SUBMIT_ACK

    def evaluate_alpha(self, metrics: Dict[str, Any], correlation: float = 0.0) -> tuple[bool, str]:
        sharpe = metrics.get("sharpe")
        fitness = metrics.get("fitness")
        turnover = metrics.get("turnover")
        margin = metrics.get("margin")

        if sharpe is None or fitness is None or turnover is None or margin is None:
            return False, "Missing core metrics"

        if sharpe < self.min_sharpe:
            return False, f"Sharpe {sharpe:.2f} < threshold {self.min_sharpe:.2f}"
        if fitness < self.min_fitness:
            return False, f"Fitness {fitness:.2f} < threshold {self.min_fitness:.2f}"
        if turnover < self.min_turnover or turnover > self.max_turnover:
            return False, f"Turnover {turnover:.2%} not within [{self.min_turnover:.0%}, {self.max_turnover:.0%}]"
        if margin < self.min_margin:
            return False, f"Margin {margin:.4f} < threshold {self.min_margin:.4f}"
        if correlation >= self.max_self_correlation:
            return False, f"Correlation {correlation:.2f} >= max {self.max_self_correlation:.2f}"

        return True, "PASSED_HARD_GATE"

    def can_submit(
        self,
        submission_key: str,
        metrics: Dict[str, Any],
        correlation: float = 0.0,
        alpha_record: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        if self.is_kill_switch_active():
            return False, "KILL_SWITCH_ACTIVE"

        if not self.is_ack_valid():
            return False, f"MISSING_OR_INVALID_ACK (Requires {SUBMIT_ACK_ENV}={REQUIRED_SUBMIT_ACK})"

        if alpha_record is not None:
            gate = evaluate_pre_submission(alpha_record, automatic=True)
            if gate.verdict != GateVerdict.PASS:
                return False, f"PRE_SUBMISSION_GATE:{','.join(gate.reasons)}"

        passed, reason = self.evaluate_alpha(metrics, correlation)
        if not passed:
            return False, reason

        with self._lock:
            if submission_key in self._submitted_keys:
                return False, "ALREADY_SUBMITTED_IDEMPOTENT"

            if self._submitted_today >= self.daily_quota:
                return False, f"DAILY_QUOTA_EXCEEDED (Max {self.daily_quota}/day)"

            now = time.time()
            elapsed = now - self._last_submitted_at
            if elapsed < self.cooldown_seconds:
                return False, f"COOLDOWN_ACTIVE (Wait {self.cooldown_seconds - elapsed:.1f}s)"

        return True, "READY"

    def record_submission(self, alpha_id: str, submission_key: str, status: str = "SUBMITTED", error: Optional[str] = None):
        with self._lock:
            self._submitted_keys.add(submission_key)
            self._submitted_today += 1
            self._last_submitted_at = time.time()
            entry = {
                "alpha_id": alpha_id,
                "submission_key": submission_key,
                "status": status,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.audit_log.append(entry)
            return entry


class DirectSubmitService:
    """Submits an already-simulated official BRAIN alpha ID directly after passing pre-submission gate."""

    def __init__(self, submit_fn: Callable[[str], Mapping[str, Any]]) -> None:
        self.submit_fn = submit_fn

    def submit_existing_alpha(
        self,
        alpha_record: Mapping[str, Any],
        *,
        enforce_gate: bool = True,
    ) -> dict[str, Any]:
        brain_alpha_id = str(alpha_record.get("brain_alpha_id") or "").strip()
        if not brain_alpha_id:
            return {"success": False, "error": "MISSING_BRAIN_ALPHA_ID"}
        if alpha_record.get("submitted_at") or alpha_record.get("status") == "SUBMITTED":
            return {"success": False, "error": "ALREADY_SUBMITTED_IDEMPOTENT", "brain_alpha_id": brain_alpha_id}

        if enforce_gate:
            gate = evaluate_pre_submission(alpha_record)
            if gate.verdict != GateVerdict.PASS:
                return {
                    "success": False,
                    "error": f"PRE_SUBMISSION_GATE_{gate.verdict.value}",
                    "brain_alpha_id": brain_alpha_id,
                    "gate_result": gate.to_dict(),
                }

        resp = dict(self.submit_fn(brain_alpha_id))
        ok = bool(resp.get("success")) or resp.get("status") in {"SUBMITTED", "SUCCESS"}
        now = datetime.now(timezone.utc).isoformat()
        resp_hash = hashlib.sha256(json.dumps(resp, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {
            "success": ok,
            "brain_alpha_id": brain_alpha_id,
            "status": "SUBMITTED" if ok else "SUBMISSION_CANDIDATE",
            "submitted_at": now if ok else None,
            "submit_response_hash": resp_hash,
            "retained_in_vault": True,
            "raw_response": resp,
        }


class SubmissionService:
    """Daily Top-K Utility submission service with pre-submission gate, audit logging, and non-destructive state transitions."""

    def __init__(self, db_path: str | Path, top_k: int = 2) -> None:
        self.repo = AlphaRepository(db_path)
        self.top_k = int(top_k)

    def execute_daily_top_k(
        self,
        candidates: Iterable[Mapping[str, Any]],
        submitter: Callable[[str], Mapping[str, Any]],
        *,
        formal_submission_count: int = 0,
        enforce_gate: bool = True,
    ) -> dict[str, Any]:
        cand_list = list(candidates)
        cand_by_id = {
            str(c.get("alpha_id") or c.get("brain_alpha_id") or c.get("id") or ""): dict(c)
            for c in cand_list
        }
        batch: SelectionBatch = select_daily_candidates(
            cand_list, top_k=self.top_k, formal_submission_count=formal_submission_count
        )
        submitted_ids: list[str] = []
        failed_ids: list[str] = []
        gate_blocked_ids: list[str] = []

        for sel in batch.selected:
            alpha_id = sel.alpha_id
            record = cand_by_id.get(alpha_id, {})
            if enforce_gate and record:
                gate = evaluate_pre_submission(record)
                if gate.verdict != GateVerdict.PASS:
                    gate_blocked_ids.append(alpha_id)
                    continue

            resp = dict(submitter(alpha_id))
            ok = bool(resp.get("success")) or resp.get("status") in {"SUBMITTED", "SUCCESS"}
            now = datetime.now(timezone.utc).isoformat()
            resp_hash = hashlib.sha256(json.dumps(resp, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            with self.repo.store.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO brain_submissions (
                        submission_id, alpha_id, brain_alpha_id, brain_simulation_id,
                        status, submitted_at, submit_acknowledgement, submit_response_hash,
                        official_status, attribution_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        alpha_id,
                        str(resp.get("brain_alpha_id") or alpha_id),
                        resp.get("simulation_id"),
                        "SUBMITTED" if ok else "FAILED",
                        now,
                        "ACK",
                        resp_hash,
                        str(resp.get("status") or ("SUBMITTED" if ok else "FAILED")),
                        "PENDING_NEXT_REFRESH",
                    ),
                )
            if ok:
                submitted_ids.append(alpha_id)
            else:
                failed_ids.append(alpha_id)

        return {
            "selected_count": len(batch.selected),
            "submitted_ids": submitted_ids,
            "failed_ids": failed_ids,
            "gate_blocked_ids": gate_blocked_ids,
            "retained_pending_ids": list(batch.retained_pending_ids),
            "rejected_ids": list(batch.rejected_ids),
        }


@dataclass(frozen=True)
class PolicyDecision:
    submit: bool
    reason: str


class SubmitAllAvailablePolicy:
    """Adaptive submission policy: submit every pending candidate, without local quality ranking."""

    def __init__(
        self,
        db_path: str | Path,
        score_tracker: ScoreTracker,
        *,
        enabled: bool,
        kill_switch: Callable[[], bool] = lambda: False,
        acknowledgement_valid: Callable[[], bool] = lambda: False,
        cooldown_seconds: float = 2.0,
        order_by: str = "fifo",
    ) -> None:
        self.db_path = str(db_path)
        self.score_tracker = score_tracker
        self.enabled = bool(enabled)
        self.kill_switch = kill_switch
        self.acknowledgement_valid = acknowledgement_valid
        self.cooldown_seconds = float(cooldown_seconds)
        self.order_by = str(order_by or "fifo").lower()
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
            CREATE TABLE IF NOT EXISTS brain_adaptive_submission_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                candidates_seen INTEGER NOT NULL DEFAULT 0,
                attempted INTEGER NOT NULL DEFAULT 0,
                submitted INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stopped_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS brain_adaptive_submission_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                alpha_id TEXT NOT NULL,
                attempted_at TEXT NOT NULL,
                outcome TEXT NOT NULL,
                safe_error TEXT,
                UNIQUE(alpha_id, outcome) ON CONFLICT IGNORE
            );
            """)
            ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS brain_adaptive_submission_runs (\n                run_id TEXT PRIMARY KEY,\n                started_at TEXT NOT NULL,\n                completed_at TEXT,\n                candidates_seen INTEGER NOT NULL DEFAULT 0,\n                attempted INTEGER NOT NULL DEFAULT 0,\n                submitted INTEGER NOT NULL DEFAULT 0,\n                failed INTEGER NOT NULL DEFAULT 0,\n                stopped_reason TEXT\n            );\n            CREATE TABLE IF NOT EXISTS brain_adaptive_submission_events (\n                event_id INTEGER PRIMARY KEY AUTOINCREMENT,\n                run_id TEXT NOT NULL,\n                alpha_id TEXT NOT NULL,\n                attempted_at TEXT NOT NULL,\n                outcome TEXT NOT NULL,\n                safe_error TEXT,\n                UNIQUE(alpha_id, outcome) ON CONFLICT IGNORE\n            );\n            ')

    def decide(self, alpha_id: str) -> PolicyDecision:
        if not self.enabled:
            return PolicyDecision(False, "AUTO_SUBMIT_DISABLED")
        if self.kill_switch():
            return PolicyDecision(False, "KILL_SWITCH_ACTIVE")
        if not self.acknowledgement_valid():
            return PolicyDecision(False, "SUBMIT_ACK_INVALID")
        cap = self.score_tracker.cap_status()
        if cap["stop_allowed"] and cap["reached"]:
            return PolicyDecision(False, "AUTHENTICATED_DAILY_SCORE_CAP_REACHED")
        with self.connect() as conn:
            duplicate = conn.execute(
                "SELECT 1 FROM brain_adaptive_submission_events WHERE alpha_id=? AND outcome='SUBMITTED'",
                (alpha_id,),
            ).fetchone()
        if duplicate:
            return PolicyDecision(False, "ALREADY_SUBMITTED_IDEMPOTENT")
        return PolicyDecision(True, "SUBMIT_ANY_AVAILABLE")

    def pending_candidates(self) -> list[dict[str, Any]]:
        order_clause = "score DESC, created_at ASC, alpha_id ASC" if self.order_by == "score_desc" else "created_at ASC, alpha_id ASC"
        with self.connect() as conn:
            try:
                rows = conn.execute(f"""
                    SELECT alpha_id, expression, candidate_date, created_at, status
                    FROM brain_candidate_pool
                    WHERE status='PENDING'
                    ORDER BY {order_clause}
                """).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def run(self, submitter: Callable[[str], Mapping[str, Any]]) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        started = datetime.now(timezone.utc).isoformat()
        candidates = self.pending_candidates()
        stats = {"run_id": run_id, "seen": len(candidates), "attempted": 0, "submitted": 0, "failed": 0}
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO brain_adaptive_submission_runs(run_id,started_at,candidates_seen) VALUES(?,?,?)",
                (run_id, started, len(candidates)),
            )
        stop_reason = None
        for candidate in candidates:
            alpha_id = str(candidate["alpha_id"])
            decision = self.decide(alpha_id)
            if not decision.submit:
                if decision.reason == "AUTHENTICATED_DAILY_SCORE_CAP_REACHED":
                    stop_reason = decision.reason
                    break
                continue
            stats["attempted"] += 1
            try:
                response = dict(submitter(alpha_id))
                success = bool(response.get("success")) or response.get("status") in {"SUBMITTED", "SUCCESS"}
                outcome = "SUBMITTED" if success else "FAILED"
                safe_error = None if success else str(response.get("error") or response.get("status") or "UNKNOWN")[:500]
            except Exception as exc:
                outcome = "FAILED"
                safe_error = type(exc).__name__
            stats["submitted" if outcome == "SUBMITTED" else "failed"] += 1
            with self.connect() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO brain_adaptive_submission_events
                    (run_id,alpha_id,attempted_at,outcome,safe_error) VALUES(?,?,?,?,?)
                """, (run_id, alpha_id, datetime.now(timezone.utc).isoformat(), outcome, safe_error))
                if outcome == "SUBMITTED":
                    conn.execute(
                        "UPDATE brain_candidate_pool SET status='SUBMITTED' WHERE alpha_id=?", (alpha_id,)
                    )
            if self.cooldown_seconds > 0:
                time.sleep(self.cooldown_seconds)
        completed = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute("""
                UPDATE brain_adaptive_submission_runs SET completed_at=?, attempted=?, submitted=?,
                    failed=?, stopped_reason=? WHERE run_id=?
            """, (completed, stats["attempted"], stats["submitted"], stats["failed"], stop_reason, run_id))
        stats["stopped_reason"] = stop_reason
        return stats


__all__ = [
    "AlphaStatus",
    "BrainAutoSubmitGate",
    "DirectSubmitService",
    "KILL_SWITCH_ENV",
    "PolicyDecision",
    "REQUIRED_SUBMIT_ACK",
    "SUBMIT_ACK_ENV",
    "SubmissionService",
    "SubmitAllAvailablePolicy",
    "logger",
    "migrate_alpha_status",
]
