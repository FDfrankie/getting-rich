"""
Tests for Brain Job Queue behavior.
Complies with docs/BRAIN_WEB_API_ASYNC_FIX_SPEC.md.
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock
from worldquant_brain.brain_job_queue import BrainJobQueue, BrainJobStatus, BrainSimulationJob


class MockBrainClient:
    def __init__(self, offline_mode: bool = False, authenticated: bool = True):
        self.offline_mode = offline_mode
        self.authenticated = authenticated
        self.timeout = 5.0
        self.BASE_URL = "https://api.worldquantbrain.com"
        self.session = MagicMock()

    def authenticate(self):
        self.authenticated = True
        return True

    def _number(self, obj, *names):
        return 1.85


class TestBrainJobQueue(unittest.TestCase):
    def test_immediate_202_non_blocking(self):
        """HTTP handler 不等待遠端輪詢（0 秒阻塞），非同步 202 立即返回"""
        mock_client = MockBrainClient(offline_mode=False)
        # Mock session.post and session.get
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 201
        mock_post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_test_123"}
        mock_client.session.post.return_value = mock_post_resp

        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.content = b'{"status": "RUNNING"}'
        mock_get_resp.json.return_value = {"status": "RUNNING"}
        mock_client.session.get.return_value = mock_get_resp

        queue = BrainJobQueue(mock_client, default_poll_interval=0.1)

        t0 = time.monotonic()
        job, is_new = queue.create_job("rank(close / delay(close, 5))")
        t1 = time.monotonic()

        # Job creation should be nearly instantaneous (< 50ms)
        self.assertLess(t1 - t0, 0.1)
        self.assertTrue(is_new)
        self.assertIn(job.status, [BrainJobStatus.QUEUED, BrainJobStatus.RUNNING])

    def test_worker_polls_and_completes_job(self):
        """背景 Worker 自動輪詢更新 Job 狀態與指標"""
        mock_client = MockBrainClient(offline_mode=False)
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 201
        mock_post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_poll_abc"}
        mock_client.session.post.return_value = mock_post_resp

        # First poll RUNNING, second poll COMPLETE
        poll_running = MagicMock()
        poll_running.status_code = 200
        poll_running.content = b'{"status": "RUNNING"}'
        poll_running.headers = {}
        poll_running.json.return_value = {"status": "RUNNING"}

        poll_complete = MagicMock()
        poll_complete.status_code = 200
        poll_complete.content = b'{"status": "COMPLETE", "sharpe": 1.9, "fitness": 1.25}'
        poll_complete.headers = {}
        poll_complete.json.return_value = {"status": "COMPLETE", "sharpe": 1.9, "fitness": 1.25}

        mock_client.session.get.side_effect = [poll_running, poll_complete]

        queue = BrainJobQueue(mock_client, default_poll_interval=0.05)
        job, _ = queue.create_job("rank(volume * close)")

        # Wait up to 2 seconds for worker to complete
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and job.status != BrainJobStatus.COMPLETE:
            time.sleep(0.05)

        self.assertEqual(job.status, BrainJobStatus.COMPLETE)
        self.assertEqual(job.alpha_id, "sim_poll_abc")
        self.assertIsNotNone(job.metrics["sharpe"])
        self.assertIsNone(job.safe_error)

    def test_offline_mode_fallback(self):
        """離線模式自動回退為 OFFLINE 驗證，不發送外網請求"""
        mock_client = MockBrainClient(offline_mode=True)
        mock_client.simulate_alpha = MagicMock(return_value={
            "status": "OFFLINE",
            "sharpe": None,
            "fitness": None,
            "turnover": None,
            "margin": None,
            "location": None,
            "simulation_id": None
        })

        queue = BrainJobQueue(mock_client, default_poll_interval=0.05)
        job, _ = queue.create_job("ts_max(high, 20)")

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and job.status != BrainJobStatus.OFFLINE:
            time.sleep(0.05)

        self.assertEqual(job.status, BrainJobStatus.OFFLINE)
        mock_client.session.post.assert_not_called()

    def test_worker_catches_exception_safely(self):
        """背景例外安全捕獲，不中斷常駐程序，並過濾敏感資訊"""
        mock_client = MockBrainClient(offline_mode=False)
        mock_client.session.post.side_effect = ConnectionResetError("Connection reset by peer: secret_token=abc12345")

        queue = BrainJobQueue(mock_client, default_poll_interval=0.05)
        job, _ = queue.create_job("rank(open / vwap)")

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and job.status != BrainJobStatus.FAILED:
            time.sleep(0.05)

        self.assertEqual(job.status, BrainJobStatus.FAILED)
        self.assertIsNotNone(job.safe_error)
        self.assertNotIn("abc12345", job.safe_error)


if __name__ == "__main__":
    unittest.main()
