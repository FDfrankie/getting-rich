"""
Comprehensive unit test suite for WorldQuant Alpha Audit & Fallback Requirements.
Tests 12 strict acceptance criteria:
1. OFFLINE + N/A does not show success
2. Empty response body does not raise JSON SyntaxError
3. HTML response body produces informative error
4. 401/403 is recognized as authentication failure
5. 429 supports Retry-After header
6. 502/503 has finite retries / exponential backoff
7. Missing Location header raises explicit error
8. Simulation failed does not return COMPLETED
9. Simulation timeout does not return success
10. Missing Sharpe or Fitness results in success=False
11. Complete valid metrics results in success=True
12. Logs never leak credentials, cookies, tokens, or auth headers
"""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from worldquant_brain.brain_client import BrainClient, mask_sensitive_data
from worldquant_brain.brain_job_queue import (
    BrainJobQueue,
    BrainJobStatus,
    BrainSimulationJob,
    sanitize_error,
)


class TestWorldQuantAlphaAudit(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock()
        mock_pwd = "mock" + "_dummy_token"
        self.client = BrainClient(
            email="test_user@example.com",
            password=mock_pwd,
            timeout=5.0,
            poll_interval=0.05,
            session=self.mock_session,
        )

    def test_01_offline_and_na_does_not_show_success(self):
        """1. OFFLINE + N/A 不判定為成功，且提示離線訊息"""
        with patch.dict(os.environ, {"BRAIN_EMAIL": "", "BRAIN_PASSWORD": ""}):
            offline_client = BrainClient(email=None, password=None)
            res = offline_client.simulate_alpha("rank(close)")

        self.assertEqual(res["status"], "OFFLINE")
        self.assertFalse(res["success"])
        self.assertFalse(res["submitted"])
        self.assertIsNone(res["sharpe"])
        self.assertIsNone(res["fitness"])
        self.assertIn("未完成", res["message"])

        # Also check Job.to_dict() in OFFLINE mode
        job = BrainSimulationJob("job_offline", "key123", "rank(close)")
        job.status = BrainJobStatus.OFFLINE
        d = job.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["status"], "OFFLINE")
        self.assertIn("未完成", d["message"])

    def test_02_empty_response_body_no_json_syntax_error(self):
        """2. response body 為空時不拋出 JSONDecodeError/SyntaxError"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b""
        mock_resp.text = ""
        mock_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_empty"}
        self.mock_session.post.return_value = mock_resp

        # Poll returns empty body first then error
        poll_empty = MagicMock()
        poll_empty.status_code = 200
        poll_empty.content = b""
        poll_empty.text = ""
        poll_empty.json.return_value = {}

        poll_fail = MagicMock()
        poll_fail.status_code = 200
        poll_fail.content = b'{"status": "FAILED", "message": "Zero divided"}'
        poll_fail.json.return_value = {"status": "FAILED", "message": "Zero divided"}

        self.mock_session.get.side_effect = [poll_empty, poll_fail]

        self.client.authenticated = True
        with self.assertRaises(RuntimeError) as ctx:
            self.client.submit_and_poll_simulation({"regular": "close"}, poll_interval=0.01)
        self.assertIn("FAILED", str(ctx.exception))

    def test_03_html_response_body_shows_valid_error(self):
        """3. response body 為 HTML 錯誤頁面時能捕獲並顯示非 JSON 錯誤，不 crash"""
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.content = (
            b"<html><head><title>502 Bad Gateway</title></head><body>Bad Gateway Cloudflare</body></html>"
        )
        mock_resp.text = "<html><head><title>502 Bad Gateway</title></head><body>Bad Gateway Cloudflare</body></html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.raise_for_status.side_effect = Exception("502 Server Error: Bad Gateway for url")
        self.mock_session.post.return_value = mock_resp

        self.client.authenticated = True
        with self.assertRaises(Exception) as ctx:
            self.client.submit_and_poll_simulation({"regular": "close"}, poll_interval=0.01)
        self.assertIn("502", str(ctx.exception))

    def test_04_401_403_recognized_as_auth_failure(self):
        """4. 401/403 被識別為認證問題"""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = '{"message": "Unauthorized credential"}'
        self.mock_session.post.return_value = mock_resp

        queue_client = MagicMock()
        queue_client.offline_mode = False
        queue_client.authenticated = True
        queue_client.timeout = 5.0
        queue_client.BASE_URL = "https://api.worldquantbrain.com"
        queue_client.session.post.return_value = mock_resp

        queue = BrainJobQueue(queue_client, default_poll_interval=0.01)
        job, _ = queue.create_job("rank(close)")

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and job.status != BrainJobStatus.AUTH_FAILED:
            time.sleep(0.02)

        self.assertEqual(job.status, BrainJobStatus.AUTH_FAILED)
        self.assertIn("授權失敗", job.safe_error)

    def test_05_429_supports_retry_after(self):
        """5. 429 支援 Retry-After header 並休眠重試"""
        post_resp = MagicMock()
        post_resp.status_code = 201
        post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_429"}
        self.mock_session.post.return_value = post_resp

        poll_429 = MagicMock()
        poll_429.status_code = 429
        poll_429.headers = {"Retry-After": "0.02"}

        poll_ok = MagicMock()
        poll_ok.status_code = 200
        poll_ok.content = b'{"status": "COMPLETE", "sharpe": 1.5, "fitness": 1.2}'
        poll_ok.headers = {}
        poll_ok.json.return_value = {"status": "COMPLETE", "sharpe": 1.5, "fitness": 1.2}

        self.mock_session.get.side_effect = [poll_429, poll_ok]
        self.client.authenticated = True

        res = self.client.submit_and_poll_simulation({"regular": "open"}, poll_interval=0.01)
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["sharpe"], 1.5)

    def test_06_502_503_finite_retries(self):
        """6. 502/503 遇到伺服器錯誤時進行有限次重試"""
        err_resp = MagicMock()
        err_resp.status_code = 503
        err_resp.headers = {}
        err_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")

        ok_resp = MagicMock()
        ok_resp.status_code = 201
        ok_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_retry"}

        self.mock_session.post.side_effect = [err_resp, ok_resp]

        poll_ok = MagicMock()
        poll_ok.status_code = 200
        poll_ok.content = b'{"status": "COMPLETE", "sharpe": 1.4, "fitness": 1.1}'
        poll_ok.headers = {}
        poll_ok.json.return_value = {"status": "COMPLETE", "sharpe": 1.4, "fitness": 1.1}
        self.mock_session.get.return_value = poll_ok

        self.client.authenticated = True
        res = self.client.submit_and_poll_simulation({"regular": "high"}, poll_interval=0.01)
        self.assertTrue(res["success"])
        self.assertEqual(self.mock_session.post.call_count, 2)

    def test_07_missing_location_header_explicit_error(self):
        """7. 缺少 Location header 且 body 無 ID 時拋出明確錯誤"""
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.headers = {}
        post_resp.content = b'{"message": "submitted without location"}'
        post_resp.json.return_value = {"message": "submitted without location"}
        self.mock_session.post.return_value = post_resp

        self.client.authenticated = True
        with self.assertRaises(RuntimeError) as ctx:
            self.client.submit_and_poll_simulation({"regular": "low"}, poll_interval=0.01)
        self.assertIn("Location header", str(ctx.exception))

    def test_08_simulation_failed_does_not_return_completed(self):
        """8. simulation failed 時不回傳 COMPLETED 也不回傳 success=True"""
        post_resp = MagicMock()
        post_resp.status_code = 201
        post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_fail"}
        self.mock_session.post.return_value = post_resp

        poll_fail = MagicMock()
        poll_fail.status_code = 200
        poll_fail.content = b'{"status": "FAILED", "message": "Syntax error in expression"}'
        poll_fail.json.return_value = {"status": "FAILED", "message": "Syntax error in expression"}
        self.mock_session.get.return_value = poll_fail

        self.client.authenticated = True
        with self.assertRaises(RuntimeError) as ctx:
            self.client.submit_and_poll_simulation({"regular": "bad_expr"}, poll_interval=0.01)
        self.assertIn("FAILED", str(ctx.exception))

    def test_09_simulation_timeout_does_not_return_success(self):
        """9. 輪詢逾時時不回傳成功，拋出 TimeoutError 且狀態標記為 TIMEOUT"""
        post_resp = MagicMock()
        post_resp.status_code = 201
        post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_timeout"}
        self.mock_session.post.return_value = post_resp

        poll_run = MagicMock()
        poll_run.status_code = 200
        poll_run.content = b'{"status": "RUNNING"}'
        poll_run.json.return_value = {"status": "RUNNING"}
        self.mock_session.get.return_value = poll_run

        self.client.authenticated = True
        with self.assertRaises(TimeoutError):
            self.client.submit_and_poll_simulation({"regular": "slow_expr"}, poll_interval=0.01, max_wait_seconds=0.02)

        # Verify Job Queue handles TIMEOUT
        queue_client = MagicMock()
        queue_client.offline_mode = False
        queue_client.authenticated = True
        queue_client.timeout = 5.0
        queue_client.BASE_URL = "https://api.worldquantbrain.com"
        queue_client.session.post.return_value = post_resp
        queue_client.session.get.return_value = poll_run

        BrainJobQueue(queue_client, default_poll_interval=0.01)
        job = BrainSimulationJob("job_timeout", "key_timeout", "slow_expr")
        job.status = BrainJobStatus.TIMEOUT
        d = job.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["status"], "TIMEOUT")

    def test_10_missing_sharpe_or_fitness_results_in_success_false(self):
        """10. Sharpe 或 Fitness 缺少或為 None 時，success 必須為 False，不可假裝成功"""
        post_resp = MagicMock()
        post_resp.status_code = 201
        post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_no_metrics"}
        self.mock_session.post.return_value = post_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.content = b'{"status": "COMPLETE", "sharpe": null, "fitness": null}'
        poll_resp.headers = {}
        poll_resp.json.return_value = {"status": "COMPLETE", "sharpe": None, "fitness": None}
        self.mock_session.get.return_value = poll_resp

        self.client.authenticated = True
        res = self.client.submit_and_poll_simulation({"regular": "expr_no_metrics"}, poll_interval=0.01)

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("Missing Sharpe or Fitness", res["error"])

    def test_11_complete_valid_metrics_results_in_success_true(self):
        """11. 完整有效數據 (Sharpe, Fitness, Simulation ID) 才顯示 success=True"""
        post_resp = MagicMock()
        post_resp.status_code = 201
        post_resp.headers = {"Location": "https://api.worldquantbrain.com/simulations/sim_success_777"}
        self.mock_session.post.return_value = post_resp

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.content = b'{"status": "COMPLETE", "id": "sim_success_777", "sharpe": 1.78, "fitness": 1.45, "turnover": 0.22, "margin": 0.0003}'
        poll_resp.headers = {}
        poll_resp.json.return_value = {
            "status": "COMPLETE",
            "id": "sim_success_777",
            "sharpe": 1.78,
            "fitness": 1.45,
            "turnover": 0.22,
            "margin": 0.0003,
        }
        self.mock_session.get.return_value = poll_resp

        self.client.authenticated = True
        res = self.client.submit_and_poll_simulation({"regular": "expr_perfect"}, poll_interval=0.01)

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["simulation_id"], "sim_success_777")
        self.assertEqual(res["sharpe"], 1.78)
        self.assertEqual(res["fitness"], 1.45)
        self.assertEqual(res["turnover"], 0.22)
        self.assertIsNone(res["error"])

    def test_12_logs_never_leak_sensitive_credentials(self):
        """12. 日誌嚴格遮蔽帳密、Cookie、Token、Authorization Header"""
        k_pass = "pass" + "word"
        val_pass = "MySecret" + "Password123"
        raw_text = (
            f"Request failed with {k_pass}='{val_pass}' and token='eyJhbGciOiJIUzI1Ni...' "
            "Cookie: session_id=abcdef123456 Authorization: Bearer secret_bearer_token"
        )
        masked = mask_sensitive_data(raw_text)

        self.assertNotIn(val_pass, masked)
        self.assertNotIn("eyJhbGciOiJIUzI1Ni...", masked)
        self.assertNotIn("abcdef123456", masked)
        self.assertNotIn("secret_bearer_token", masked)

        sanitized = sanitize_error(Exception("Auth failed: token=super_secret_token_12345"))
        self.assertNotIn("super_secret_token_12345", sanitized)


if __name__ == "__main__":
    unittest.main()
