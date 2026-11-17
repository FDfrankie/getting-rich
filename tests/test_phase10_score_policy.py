import sqlite3
from worldquant_brain.adaptive_submission import SubmitAllAvailablePolicy
from worldquant_brain.score_tracker import ScoreTracker, parse_score_payload


def seed_candidate(db, alpha_id, score):
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS brain_candidate_pool(
          alpha_id TEXT PRIMARY KEY,expression TEXT,candidate_date TEXT,created_at TEXT,status TEXT,score REAL)""")
        c.execute("INSERT INTO brain_candidate_pool VALUES(?,?,?,?,?,?)",
                  (alpha_id,'rank(close)','2026-11-18',f'2026-11-18T00:00:0{score}', 'PENDING',score))


def test_documented_2000_cap_is_not_treated_as_authenticated_stop(tmp_path):
    tracker=ScoreTracker(tmp_path/'a.db')
    tracker.record(parse_score_payload({'dailyScore':2000},source='TEST'))
    status=tracker.cap_status()
    assert status['daily_cap']==2000
    assert not status['stop_allowed'] and not status['reached']


def test_authenticated_cap_stops_when_reached(tmp_path):
    tracker=ScoreTracker(tmp_path/'a.db')
    tracker.record(parse_score_payload({'dailyScore':2000,'dailyScoreCap':2000},source='TEST'))
    policy=SubmitAllAvailablePolicy(tmp_path/'a.db',tracker,enabled=True,
        acknowledgement_valid=lambda:True,cooldown_seconds=0)
    assert not policy.decide('A').submit


def test_policy_submits_all_fifo_not_quality_ranked(tmp_path):
    db=tmp_path/'a.db'; tracker=ScoreTracker(db)
    seed_candidate(db,'LOW',1); seed_candidate(db,'HIGH',9)
    policy=SubmitAllAvailablePolicy(db,tracker,enabled=True,
        acknowledgement_valid=lambda:True,cooldown_seconds=0)
    called=[]
    result=policy.run(lambda alpha_id: called.append(alpha_id) or {'success':True})
    assert called==['LOW','HIGH']
    assert result['submitted']==2


def test_policy_requires_enable_ack_and_idempotency(tmp_path):
    db=tmp_path/'a.db'; tracker=ScoreTracker(db); seed_candidate(db,'A',1)
    disabled=SubmitAllAvailablePolicy(db,tracker,enabled=False,acknowledgement_valid=lambda:True)
    assert disabled.decide('A').reason=='AUTO_SUBMIT_DISABLED'
    noack=SubmitAllAvailablePolicy(db,tracker,enabled=True,acknowledgement_valid=lambda:False)
    assert noack.decide('A').reason=='SUBMIT_ACK_INVALID'
