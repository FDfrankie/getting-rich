"""Contract tests for P20: HTTP 429 rate limit must trigger cooldown (RATE_LIMITED), never kill miner as AUTH_FAILED."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from worldquant_brain.multi_sim_client import (
    _is_auth_rate_or_captcha_error,
    _is_rate_limit_error,
    _simulate_single_fallback,
    simulate_multi_alpha,
)


class TestP20RateLimitVsAuthSeparation(unittest.TestCase):
    def test_429_is_rate_limit_not_auth_block(self) -> None:
        err = "429 Client Error: Too Many Requests for url: https://api.worldquantbrain.com/simulations"
        self.assertTrue(_is_rate_limit_error(err))
        self.assertFalse(_is_auth_rate_or_captcha_error(err))
        self.assertTrue(_is_auth_rate_or_captcha_error("CAPTCHA verification required"))
        self.assertTrue(_is_auth_rate_or_captcha_error("Authentication failed (401)"))

    def test_single_fallback_429_returns_rate_limited_not_auth_failed(self) -> None:
        client = MagicMock()
        client.simulate_alpha.side_effect = RuntimeError(
            "429 Client Error: Too Many Requests for url: https://api.worldquantbrain.com/simulations"
        )
        items = [
            {"expression": "rank(close)", "stage": "STAGE1", "dataset_id": "model16"},
            {"expression": "rank(open)", "stage": "STAGE1", "dataset_id": "model16"},
        ]
        res = _simulate_single_fallback(client, items, max_wait_seconds=10.0, poll_interval=1.0, item_delay_seconds=0.0)
        self.assertEqual(len(res), 2)
        for r in res:
            self.assertFalse(r["success"])
            self.assertEqual(r["status"], "RATE_LIMITED")
            self.assertNotEqual(r["status"], "AUTH_FAILED")

    def test_multi_sim_post_429_returns_rate_limited(self) -> None:
        client = MagicMock()
        client.offline_mode = False
        client.is_authenticated = True
        client._multi_sim_forbidden = False
        client._multi_sim_last_auth_fail_ts = 0.0
        client._multi_sim_last_auth_ok_ts = 0.0
        client.BASE_URL = "https://api.worldquantbrain.com"
        resp429 = MagicMock()
        resp429.status_code = 429
        resp429.headers = {"Retry-After": "0"}
        resp429.text = "429 Too Many Requests"
        client.session.request.return_value = resp429
        items = [
            {"expression": "rank(close)", "stage": "STAGE1", "dataset_id": "model16"},
            {"expression": "rank(open)", "stage": "STAGE1", "dataset_id": "model16"},
        ]
        res = simulate_multi_alpha(client, items, fallback_item_delay=0.0)
        self.assertEqual(len(res), 2)
        self.assertTrue(all(r["status"] == "RATE_LIMITED" for r in res))


if __name__ == "__main__":
    unittest.main()
