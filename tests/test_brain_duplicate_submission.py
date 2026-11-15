"""
Tests for WorldQuant BRAIN duplicate submission idempotency.
Complies with docs/BRAIN_WEB_API_ASYNC_FIX_SPEC.md.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from worldquant_brain.brain_job_queue import BrainJobQueue, compute_simulation_key


class TestBrainDuplicateSubmission(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.offline_mode = True
        self.mock_client.simulate_alpha.return_value = {
            "status": "OFFLINE",
            "sharpe": None,
            "fitness": None,
            "turnover": None,
            "margin": None
        }
        self.queue = BrainJobQueue(self.mock_client, default_poll_interval=0.1)

    def test_duplicate_click_returns_same_job(self):
        """連續點擊相同 expression 回傳同一個既有 job_id，不建立重複 job"""
        expr = "rank(ts_argmax(high, 10))"
        job1, is_new1 = self.queue.create_job(expr)
        self.assertTrue(is_new1)

        # Duplicate click
        job2, is_new2 = self.queue.create_job(expr)
        self.assertFalse(is_new2)
        self.assertEqual(job1.job_id, job2.job_id)
        self.assertEqual(job1.simulation_key, job2.simulation_key)

    def test_whitespace_variation_deduplication(self):
        """僅有前後空格差異的表達式視為同一作業進行去重"""
        expr1 = "ts_rank(close, 20)"
        expr2 = "  ts_rank(close, 20)  "

        job1, is_new1 = self.queue.create_job(expr1)
        job2, is_new2 = self.queue.create_job(expr2)

        self.assertTrue(is_new1)
        self.assertFalse(is_new2)
        self.assertEqual(job1.job_id, job2.job_id)

    def test_different_settings_creates_different_jobs(self):
        """相同 expression 但 settings 不同時，視為不同模擬作業"""
        expr = "rank(close - open)"
        settings_a = {"region": "USA", "universe": "TOP3000"}
        settings_b = {"region": "EUR", "universe": "TOP2000"}

        job1, is_new1 = self.queue.create_job(expr, settings_a)
        job2, is_new2 = self.queue.create_job(expr, settings_b)

        self.assertTrue(is_new1)
        self.assertTrue(is_new2)
        self.assertNotEqual(job1.job_id, job2.job_id)
        self.assertNotEqual(job1.simulation_key, job2.simulation_key)

    def test_simulation_key_consistency(self):
        """SHA-256 simulation key 的雜湊計算穩定性驗證"""
        key1 = compute_simulation_key("close", {"a": 1, "b": 2})
        key2 = compute_simulation_key("close", {"b": 2, "a": 1})
        self.assertEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
