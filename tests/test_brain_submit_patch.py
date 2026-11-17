import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.brain_client import BrainClient


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

    def test_local_proxy_simulation_disabled_by_default(self):
        miner = ContinuousAlphaMiner()
        self.assertIsNone(miner.mine_cycle())


if __name__ == "__main__":
    unittest.main()
