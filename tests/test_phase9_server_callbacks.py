import sqlite3
import time
from types import SimpleNamespace

from worldquant_brain.alpha_experiment_store import AlphaExperimentStore
from worldquant_brain.brain_job_queue import BrainJobQueue


class OfflineClient:
    offline_mode = True
    def simulate_alpha(self, expression, settings=None):
        return {"status": "OFFLINE", "sharpe": None, "fitness": None,
                "turnover": None, "margin": None, "location": None, "simulation_id": None}


def test_brain_job_queue_manual_priority_and_metadata(tmp_path):
    db = tmp_path / "queue.db"
    q = BrainJobQueue(OfflineClient(), autostart=False, db_path=str(db), num_workers=1)
    auto, _ = q.create_job("rank(close)", priority=20, source="AUTO_MINING", simulation_only=True)
    manual, _ = q.create_job("rank(volume)", priority=100, source="MANUAL",
                             requested_by="ui", manual_note="urgent", simulation_only=True)
    q.start()
    limit = time.time() + 2
    while manual.claimed_at is None and time.time() < limit:
        time.sleep(.01)
    q.stop(); q.join()
    assert manual.claimed_at is not None
    assert manual.source == "MANUAL" and manual.priority == 100
    assert manual.simulation_only and not manual.auto_submit
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT priority,source,requested_by,simulation_only FROM brain_jobs WHERE job_id=?",
                           (manual.job_id,)).fetchone()
    assert row == (100, "MANUAL", "ui", 1)


def test_completion_and_failure_feedback_contract(tmp_path):
    store = AlphaExperimentStore(tmp_path / "events.db")
    store.start_experiment("ok", "rank(close)")
    store.complete_experiment("ok", {"status": "BRAIN_COMPLETE", "sharpe": 1.4})
    store.start_experiment("bad", "rank(open)")
    store.complete_experiment("bad", {"status": "FAILED", "failure_code": "SYNTAX"})
    with sqlite3.connect(tmp_path / "events.db") as conn:
        complete = conn.execute("SELECT simulation_status FROM alpha_experiments WHERE experiment_id='ok'").fetchone()[0]
        failure = conn.execute("SELECT failure_code FROM alpha_experiments WHERE experiment_id='bad'").fetchone()[0]
    assert complete == "BRAIN_COMPLETE"
    assert failure == "SYNTAX"
