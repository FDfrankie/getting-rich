import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from observability.audit_log import AuditLog
from worldquant_brain.alpha_experiment_store import AlphaExperimentStore
from worldquant_brain.alpha_lifecycle import AlphaState, is_discovered, require_transition, submission_eligibility
from worldquant_brain.brain_scheduler import BrainDailyScheduler
from worldquant_brain.manual_priority_queue import ManualPriorityQueue

TAIPEI = ZoneInfo("Asia/Taipei")


def test_lifecycle_and_discovered_semantics():
    require_transition(AlphaState.BRAIN_COMPLETE, AlphaState.IS_CHECK_PASSED)
    assert is_discovered({"status": "BRAIN_COMPLETE"})
    assert not is_discovered({"status": "VERIFIED"})


def test_submission_gate_has_no_daily_quota():
    ready = submission_eligibility({
        "status": "SUBMISSION_CANDIDATE", "sharpe": 1.5, "fitness": 1.1,
        "turnover": .2, "margin_bps": 11, "drawdown": .05,
        "is_check_status": "PASS", "self_correlation": .4,
        "brain_simulation_id": "sim-1",
    })
    assert ready.ready


def test_daily_claim_is_restart_safe(tmp_path):
    db = tmp_path / "a.db"
    s1 = BrainDailyScheduler(db, 22)
    s2 = BrainDailyScheduler(db, 22)
    now = datetime(2026, 11, 18, 22, 5, tzinfo=TAIPEI)
    assert s1.claim_due_run(worker_id="a", now=now).acquired
    second = s2.claim_due_run(worker_id="b", now=now)
    assert not second.acquired and second.reason == "ALREADY_CLAIMED"


def test_manual_priority_and_dedup_survive_restart(tmp_path):
    db = tmp_path / "a.db"
    q = ManualPriorityQueue(db)
    auto, _ = q.enqueue("rank(close)", priority=20, source="AUTO_MINING")
    manual, created = q.enqueue("rank(volume)", priority=100, source="MANUAL")
    assert created and q.queue_position(manual["job_id"]) == 1
    claimed = ManualPriorityQueue(db).claim_next("worker")
    assert claimed["job_id"] == manual["job_id"]
    same, created = q.enqueue(" rank( volume ) ", priority=100, source="MANUAL")
    assert not created and same["job_id"] == manual["job_id"]


def test_experiment_feedback_keeps_failure(tmp_path):
    db = tmp_path / "a.db"
    store = AlphaExperimentStore(db)
    store.start_experiment("e1", "rank(close)")
    store.complete_experiment("e1", {"status": "FAILED", "failure_code": "SYNTAX"})
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT failure_count,last_failure_code FROM alpha_experience_memory").fetchone()
    assert row == (1, "SYNTAX")


def test_audit_log_scrubs_secrets(tmp_path):
    db = tmp_path / "a.db"
    AuditLog(db).write(component="brain", action="test", outcome="OK", details={"api_key": "secret", "safe": 1})
    with sqlite3.connect(db) as conn:
        details = conn.execute("SELECT details_json FROM audit_events").fetchone()[0]
    assert "secret" not in details and '"safe": 1' in details
