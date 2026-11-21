"""True vertical E2E tests for P21: BrainClient -> multi_sim_client -> TwoStageAlphaFunnel -> AlphaVault -> HeadlessMiner."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.multi_sim_client import _request_with_retry, simulate_multi_alpha
from worldquant_brain.two_stage_funnel import classify_verdict


def _make_resp(status_code: int, json_body=None, headers=None, text: str = ""):
    r = MagicMock()
    r.status_code = status_code
    r.headers = dict(headers or {})
    r.text = text
    r.content = b"{}" if json_body is not None else (text.encode("utf-8") if text else b"")
    r.json.return_value = json_body if json_body is not None else {}
    if status_code >= 400:
        r.raise_for_status.side_effect = RuntimeError(f"HTTP {status_code}: {text}")
    else:
        r.raise_for_status.return_value = None
    return r


class TestP21VerticalAlphaE2E(unittest.TestCase):
    def test_single_sim_fallback_populates_checks_passed_probe_and_reaches_passed(self) -> None:
        client = BrainClient(email="u@test.com", password="pw")
        client.authenticated = True

        # Multi-sim array POST returns 403 -> triggers single-sim fallback
        resp_403 = _make_resp(403, text="Multi-sim not entitled")
        post_ok_1 = _make_resp(201, headers={"Location": "https://api.worldquantbrain.com/simulations/sim1"})
        post_ok_2 = _make_resp(201, headers={"Location": "https://api.worldquantbrain.com/simulations/sim2"})

        poll_1 = _make_resp(200, json_body={"status": "COMPLETE", "id": "sim1", "alpha": "A_PASS"})
        alpha_1 = _make_resp(
            200,
            json_body={
                "id": "A_PASS",
                "is": {
                    "sharpe": 1.85,
                    "fitness": 1.32,
                    "turnover": 0.22,
                    "margin": 0.0018,
                    "returns": 0.14,
                    "drawdown": 0.06,
                    "checks": [
                        {"name": "LOW_SHARPE", "result": "PASS"},
                        {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"},
                    ],
                },
            },
        )
        poll_2 = _make_resp(200, json_body={"status": "COMPLETE", "id": "sim2", "alpha": "A_FAILCHK"})
        alpha_2 = _make_resp(
            200,
            json_body={
                "id": "A_FAILCHK",
                "is": {
                    "sharpe": 1.50,
                    "fitness": 1.10,
                    "turnover": 0.25,
                    "margin": 0.0015,
                    "checks": [
                        {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"},
                    ],
                },
            },
        )

        sess = MagicMock()
        sess.request.return_value = resp_403
        sess.post.side_effect = [post_ok_1, post_ok_2]
        sess.get.side_effect = [poll_1, alpha_1, poll_2, alpha_2]
        client.session = sess

        items = [
            {"expression": "rank(fnd6_ebitda)", "stage": "STAGE1", "dataset_id": "fundamental6", "probe": "rank_raw"},
            {"expression": "rank(fnd6_eps)", "stage": "STAGE1", "dataset_id": "fundamental6", "probe": "rank_ts_delta"},
        ]
        results = simulate_multi_alpha(client, items, max_wait_seconds=30.0, poll_interval=0.01, fallback_item_delay=0.0)
        self.assertEqual(len(results), 2)

        # Item 1: checks_passed=True -> classify_verdict MUST return PASSED & preserve probe!
        r1 = results[0]
        self.assertTrue(r1["success"])
        self.assertTrue(r1["checks_passed"])
        self.assertEqual(r1["failed_checks"], [])
        self.assertEqual(r1["probe"], "rank_raw")
        verdict1, reason1 = classify_verdict(r1)
        self.assertEqual(verdict1, "PASSED", f"Expected PASSED, got {verdict1} ({reason1})")
        self.assertTrue(AlphaVault.passes_qualification(r1["sharpe"], r1["fitness"], r1["turnover"]))

        # Item 2: failed check must propagate accurately
        r2 = results[1]
        self.assertFalse(r2["checks_passed"])
        self.assertEqual(r2["failed_checks"], ["LOW_SUB_UNIVERSE_SHARPE"])
        self.assertEqual(r2["probe"], "rank_ts_delta")
        verdict2, _ = classify_verdict(r2)
        self.assertEqual(verdict2, "NEAR_MISS")

    def test_submit_and_poll_recovers_from_401_expiry_and_poll_disconnect(self) -> None:
        client = BrainClient(email="u@test.com", password="pw")
        client.authenticated = True
        reauth_calls = {"n": 0}

        def _fake_auth() -> bool:
            reauth_calls["n"] += 1
            client.authenticated = True
            return True

        client.authenticate = _fake_auth  # type: ignore[method-assign]

        sess = MagicMock()
        sess.post.side_effect = [
            _make_resp(401, text="Unauthorized"),
            _make_resp(201, headers={"Location": "https://api.worldquantbrain.com/simulations/sim_rec"}),
        ]
        sess.get.side_effect = [
            ConnectionError("RemoteDisconnected"),
            _make_resp(200, json_body={"status": "COMPLETE", "id": "sim_rec", "alpha": "A_REC"}),
            _make_resp(
                200,
                json_body={
                    "id": "A_REC",
                    "is": {
                        "sharpe": 1.60,
                        "fitness": 1.20,
                        "turnover": 0.20,
                        "margin": 0.0014,
                        "checks": [{"name": "LOW_SHARPE", "result": "PASS"}],
                    },
                },
            ),
        ]
        client.session = sess

        res = client.simulate_alpha("rank(close)", max_wait_seconds=30.0, poll_interval=0.01)
        self.assertEqual(reauth_calls["n"], 1)
        self.assertTrue(res["success"])
        self.assertTrue(res["checks_passed"])
        self.assertEqual(res["alpha_id"], "A_REC")

    def test_request_with_retry_recovers_from_ssl_eof(self) -> None:
        client = MagicMock()
        client.timeout = 5.0
        client.session.request.side_effect = [
            OSError("SSL: UNEXPECTED_EOF_WHILE_READING"),
            _make_resp(200, json_body={"results": []}),
        ]
        resp = _request_with_retry(client, "GET", "https://api.worldquantbrain.com/data-fields", max_retries=2)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(client.session.request.call_count, 2)

    def test_turnover_1_5_rejected_consistently_across_funnel_and_vault(self) -> None:
        bad = {
            "sharpe": 1.80,
            "fitness": 1.30,
            "turnover": 1.50,  # 150% decimal turnover -> MUST NOT be divided by 100!
            "margin": 0.0020,
            "checks_passed": True,
            "failed_checks": [],
        }
        v_bad, _ = classify_verdict(bad)
        self.assertEqual(v_bad, "REJECTED")
        self.assertFalse(AlphaVault.passes_qualification(1.80, 1.30, 1.50))

        good_pct = dict(bad, turnover=22.0)  # 22% percentage
        v_pct, _ = classify_verdict(good_pct)
        self.assertEqual(v_pct, "PASSED")
        self.assertTrue(AlphaVault.passes_qualification(1.80, 1.30, 22.0))

    def test_headless_miner_rate_limit_does_not_trip_consec_zero(self) -> None:
        src = Path("scripts/headless_brain_miner.py").read_text(encoding="utf-8")
        self.assertIn("all_rate_limited", src)
        self.assertIn("do NOT clear cookies or trip auth circuit breaker", src)


if __name__ == "__main__":
    unittest.main()
