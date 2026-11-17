import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.brain_job_queue import BrainJobQueue, BrainJobStatus


class TestBrainSubmitPatch(unittest.TestCase):
    def test_submit_alpha_offline_returns_offline(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            with patch.dict(os.environ, {"BRAIN_EMAIL": "", "BRAIN_PASSWORD": ""}):
                c = BrainClient(email=None, password=None)
                c.quota_db_path = os.path.join(td, "q.db")
                res = c.submit_alpha("alpha123")
                self.assertFalse(res["success"])
                self.assertEqual(res["status"], "OFFLINE")

    def test_submit_alpha_success_200_and_daily_quota_cap_2(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            sess = MagicMock()
            auth_resp = MagicMock(status_code=201)
            check_ok_resp = MagicMock(
                status_code=200,
                content=b'{"is":{"sharpe":1.65,"fitness":1.3,"turnover":0.25,"margin":0.0018,"checks":[{"name":"SELF_CORRELATION","result":"PASS"}]}}',
            )
            check_ok_resp.json.return_value = {
                "is": {
                    "sharpe": 1.65,
                    "fitness": 1.30,
                    "turnover": 0.25,
                    "margin": 0.0018,
                    "checks": [{"name": "SELF_CORRELATION", "result": "PASS"}],
                }
            }
            sess.get.return_value = check_ok_resp
            sub_resp = MagicMock(status_code=200, headers={}, content=b'{"status":"COMPLETE"}')
            sub_resp.json.return_value = {"status": "COMPLETE"}
            sess.post.side_effect = [auth_resp, sub_resp, sub_resp]
            c = BrainClient(email="u@example.com", password="pw", session=sess)
            c.quota_db_path = os.path.join(td, "q.db")

            r1 = c.submit_alpha("alpha_1")
            self.assertTrue(r1["success"])
            self.assertEqual(r1["status"], "SUBMITTED")

            r2 = c.submit_alpha("alpha_2")
            self.assertTrue(r2["success"])
            self.assertEqual(r2["status"], "SUBMITTED")

            r3 = c.submit_alpha("alpha_3")
            self.assertFalse(r3["success"])
            self.assertEqual(r3["status"], "DAILY_QUOTA_REACHED")
            self.assertEqual(r3["daily_limit"], 2)
            self.assertEqual(r3["submitted_today"], 2)
            self.assertEqual(sess.post.call_count, 3)

    def test_submit_alpha_blocked_when_is_check_fails_or_low_margin(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            sess = MagicMock()
            sess.post.return_value = MagicMock(status_code=201)
            check_fail_resp = MagicMock(
                status_code=200,
                content=b'{"is":{"sharpe":1.5,"fitness":1.2,"turnover":0.2,"margin":0.0004,"checks":[{"name":"SUB_UNIVERSE_SHARPE","result":"FAIL"}]}}',
            )
            check_fail_resp.json.return_value = {
                "is": {
                    "sharpe": 1.5,
                    "fitness": 1.2,
                    "turnover": 0.2,
                    "margin": 0.0004,
                    "checks": [{"name": "SUB_UNIVERSE_SHARPE", "result": "FAIL"}],
                }
            }
            sess.get.return_value = check_fail_resp
            c = BrainClient(email="u@example.com", password="pw", session=sess)
            c.quota_db_path = os.path.join(td, "q.db")

            res = c.submit_alpha("alpha_bad")
            self.assertFalse(res["success"])
            self.assertEqual(res["status"], "SUBMISSION_CHECK_FAILED")
            self.assertEqual(sess.post.call_count, 1)
            self.assertEqual(c.get_daily_submission_count(), 0)

    def test_daily_candidate_pool_selects_highest_scoring_2(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            sess = MagicMock()
            sess.post.side_effect = [
                MagicMock(status_code=201),
                MagicMock(status_code=200, headers={}, content=b'{"status":"COMPLETE"}'),
                MagicMock(status_code=200, headers={}, content=b'{"status":"COMPLETE"}'),
            ]
            ok_resp = MagicMock(status_code=200, content=b'{"is":{"sharpe":1.7,"fitness":1.4,"turnover":0.2,"margin":0.002}}')
            ok_resp.json.return_value = {"is": {"sharpe": 1.7, "fitness": 1.4, "turnover": 0.2, "margin": 0.002}}
            sess.get.return_value = ok_resp

            c = BrainClient(email="u@example.com", password="pw", session=sess)
            c.quota_db_path = os.path.join(td, "q.db")
            c.stage_candidate_for_daily_selection("a_low", "rank(close)", 1.30, 1.05, 0.2, 0.0015, day_str="2026-11-17")
            c.stage_candidate_for_daily_selection("a_top", "rank(vwap)", 1.75, 1.40, 0.2, 0.0020, day_str="2026-11-17")
            c.stage_candidate_for_daily_selection("a_mid", "rank(open)", 1.50, 1.20, 0.2, 0.0018, day_str="2026-11-17")

            flushed = c.flush_daily_candidates(day_str="2026-11-17", max_daily_submissions=2)
            self.assertEqual([r["alpha_id"] for r in flushed], ["a_top", "a_mid"])
            self.assertEqual(c.get_daily_submission_count("2026-11-17"), 2)

    def test_brain_job_queue_sqlite_persistence_across_restart(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_p = os.path.join(td, "jobs.db")
            c = BrainClient(email=None, password=None)
            q1 = BrainJobQueue(c, autostart=False, db_path=db_p)
            job, is_new = q1.create_job("rank(close - open)")
            self.assertTrue(is_new)

            q2 = BrainJobQueue(c, autostart=False, db_path=db_p)
            restored = q2.get_job(job.job_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.expression, "rank(close - open)")
            self.assertEqual(restored.status, BrainJobStatus.QUEUED)

    def test_miner_deduplication_and_disabled_local_sim(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            miner = ContinuousAlphaMiner()
            miner.vault = AlphaVault(db_path=os.path.join(td, "v.db"))
            self.assertIsNone(miner.mine_cycle())
            c1 = miner.generate_unique_candidate()
            self.assertTrue(miner.has_tried_expression(c1["expression"]))


if __name__ == "__main__":
    unittest.main()
