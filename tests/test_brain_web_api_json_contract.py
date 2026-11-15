"""
Tests for WorldQuant BRAIN Web API JSON response contract.
Complies with docs/BRAIN_WEB_API_ASYNC_FIX_SPEC.md.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

# Provide lightweight mock modules for optional heavy dependencies if running in minimal Python env
for mod in [
    "yfinance",
    "pandas",
    "numpy",
    "lightgbm",
    "torch",
    "joblib",
    "sklearn",
    "matplotlib",
    "mplfinance",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from web_ui.server import APP_STATE, AppAPIHandler


class DummyRequest:
    def __init__(
        self, command: str, path: str, body: bytes = b"", headers: dict[str, str] = None
    ):
        self.command = command
        self.path = path
        self.body = body
        self.headers = headers or {
            "Content-Length": str(len(body)),
            "Content-Type": "application/json",
        }

    def makefile(self, mode: str, bufsize: int = 0):
        if "b" in mode:
            if "r" in mode:
                return io.BytesIO(self.body)
            return io.BytesIO()
        return io.StringIO()


class TestBrainWebAPIJsonContract(unittest.TestCase):
    def setUp(self):
        self.client_address = ("127.0.0.1", 12345)
        self.server = MagicMock()

    def _execute_request(
        self, command: str, path: str, payload: dict | bytes | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        if isinstance(payload, dict):
            body = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, bytes):
            body = payload
        else:
            body = b""

        req_io = io.BytesIO(body)
        resp_io = io.BytesIO()

        # Build mock handler
        handler = AppAPIHandler.__new__(AppAPIHandler)
        handler.client_address = self.client_address
        handler.server = self.server
        handler.command = command
        handler.path = path
        handler.request_version = "HTTP/1.1"
        handler.headers = {
            "Content-Length": str(len(body)),
            "Content-Type": "application/json",
        }
        handler.rfile = req_io
        handler.wfile = resp_io

        status_capture = []
        headers_capture = {}

        def send_response(code, message=None):
            status_capture.append(code)

        def send_header(keyword, value):
            headers_capture[keyword.lower()] = value

        def end_headers():
            pass

        handler.send_response = send_response
        handler.send_header = send_header
        handler.end_headers = end_headers

        if command == "POST":
            handler.do_POST()
        elif command == "GET":
            handler.do_GET()

        status_code = status_capture[0] if status_capture else 200
        response_bytes = resp_io.getvalue()
        return status_code, headers_capture, response_bytes

    def test_simulations_post_returns_202_with_valid_json(self):
        """1. 202 回應有合法 JSON 且包含 ok, status, job_id, simulation_key, next_poll_seconds"""
        payload = {"expression": "ts_rank(close, 10)"}
        status, headers, body = self._execute_request(
            "POST", "/api/brain/simulations", payload
        )

        self.assertEqual(status, 202)
        self.assertIn("application/json", headers.get("content-type", ""))
        self.assertEqual(headers.get("cache-control"), "no-store")

        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("ok"))
        self.assertIn("job_id", data)
        self.assertIn("simulation_key", data)
        self.assertIn("status", data)
        self.assertIn("next_poll_seconds", data)

    def test_auth_failure_returns_valid_json_error(self):
        """2. Auth failure 有合法 JSON 且絕不洩漏 Secret"""
        with patch.object(
            APP_STATE.brain_job_queue,
            "create_job",
            side_effect=PermissionError("secret_password_123 invalid"),
        ):
            status, headers, body = self._execute_request(
                "POST", "/api/brain/simulations", {"expression": "open / close"}
            )
            self.assertEqual(status, 500)
            self.assertIn("application/json", headers.get("content-type", ""))
            data = json.loads(body.decode("utf-8"))
            self.assertFalse(data.get("ok"))
            self.assertIn("error", data)
            self.assertNotIn("secret_password_123", json.dumps(data))

    def test_timeout_returns_valid_json_error(self):
        """3. Timeout 有合法 JSON"""
        with patch.object(
            APP_STATE.brain_job_queue,
            "create_job",
            side_effect=TimeoutError("Connection timed out"),
        ):
            status, headers, body = self._execute_request(
                "POST", "/api/brain/simulations", {"expression": "close"}
            )
            self.assertEqual(status, 500)
            data = json.loads(body.decode("utf-8"))
            self.assertFalse(data.get("ok"))
            self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")

    def test_client_exception_returns_valid_json_error(self):
        """4. Client exception 有合法 JSON"""
        status, headers, body = self._execute_request(
            "POST", "/api/brain/simulations", {"expression": ""}
        )
        self.assertEqual(status, 400)
        data = json.loads(body.decode("utf-8"))
        self.assertFalse(data.get("ok"))
        self.assertEqual(data["error"]["code"], "INVALID_EXPRESSION")

    def test_get_job_status_returns_valid_json(self):
        """查詢 GET /api/brain/jobs/{job_id} 回傳標準 JSON"""
        # Create a job first
        job, _ = APP_STATE.brain_job_queue.create_job("rank(volume)")
        status, headers, body = self._execute_request(
            "GET", f"/api/brain/jobs/{job.job_id}"
        )

        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("content-type", ""))
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("ok"))
        self.assertIn("job", data)
        self.assertEqual(data["job"]["job_id"], job.job_id)

    def test_not_found_endpoint_returns_json(self):
        """404 endpoint 必須回傳合法 JSON 錯誤"""
        status, headers, body = self._execute_request(
            "POST", "/api/brain/unknown_route", {}
        )
        self.assertEqual(status, 404)
        data = json.loads(body.decode("utf-8"))
        self.assertFalse(data.get("ok"))
        self.assertEqual(data["error"]["code"], "NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
