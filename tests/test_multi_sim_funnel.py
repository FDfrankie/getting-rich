from __future__ import annotations

import contextlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from scripts.headless_brain_miner import run_headless_miner
from worldquant_brain.multi_sim_client import _request_with_retry, simulate_multi_alpha
from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel


class _MockResp:
    def __init__(self, status_code: int, payload: Any = None, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}" if payload is not None else b""
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.post_batches: list[list[dict[str, Any]]] = []
        self._children: dict[str, dict[str, Any]] = {}
        self._alphas: dict[str, dict[str, Any]] = {}
        self._seq = 0

    def request(self, method: str, url: str, json=None, params=None, timeout=30.0):
        if method == "GET" and url.endswith("/data-fields"):
            return _MockResp(200, {"results": [{"id": "anl4_eps_mean", "type": "MATRIX", "coverage": 0.92}, {"id": "anl4_sales_mean", "type": "MATRIX", "coverage": 0.88}]})
        if method == "POST" and url.endswith("/simulations"):
            assert isinstance(json, list), "Expected Multi-Simulation list payload"
            self.post_batches.append(json)
            child_ids = []
            for item in json:
                self._seq += 1
                cid = f"child_{self._seq}"
                aid = f"alpha_{self._seq}"
                expr = item["regular"]
                # Make first expression a negative-Sharpe seed, refined expressions high-Sharpe winners
                if expr.startswith("-("):
                    sharpe, fitness = 1.95, 1.45
                elif "anl4_eps_mean" in expr:
                    sharpe, fitness = -1.15, 0.90
                else:
                    sharpe, fitness = 0.45, 0.30
                self._children[cid] = {"status": "COMPLETE", "alpha": aid}
                self._alphas[aid] = {
                    "id": aid,
                    "is": {
                        "sharpe": sharpe,
                        "fitness": fitness,
                        "turnover": 0.22,
                        "margin": 0.0018,
                        "checks": [{"name": "LOW_SHARPE", "result": "PASS" if sharpe >= 1.25 else "FAIL"}],
                    },
                }
                child_ids.append(cid)
            self._last_children = child_ids
            return _MockResp(201, None, headers={"Location": "/simulations/parent_batch_1"})
        if method == "GET" and "/simulations/parent_batch_1" in url:
            return _MockResp(200, {"status": "COMPLETE", "children": self._last_children})
        if method == "GET" and "/simulations/child_" in url:
            cid = url.rsplit("/", 1)[-1]
            return _MockResp(200, self._children[cid])
        if method == "GET" and "/alphas/" in url:
            aid = url.rsplit("/", 1)[-1]
            return _MockResp(200, self._alphas[aid])
        return _MockResp(404, {})


class _FakeBrainClient:
    BASE_URL = "https://api.worldquantbrain.com"
    offline_mode = False
    is_authenticated = True
    authenticated = True
    timeout = 5.0

    def __init__(self) -> None:
        self.session = _FakeSession()
        self.staged: list[str] = []
        self.submitted: list[str] = []

    def authenticate(self) -> bool:
        return True

    def fetch_self_correlation(self, alpha_id: str) -> dict[str, Any]:
        return {"passed": True, "self_correlation": 0.25}

    def get_daily_submission_count(self, day_str=None) -> int:
        return len(self.submitted)

    def stage_candidate_for_daily_selection(self, alpha_id: str, *args, **kwargs) -> None:
        self.staged.append(alpha_id)

    def submit_alpha(self, alpha_id: str, **kwargs) -> dict[str, Any]:
        self.submitted.append(alpha_id)
        return {"success": True, "status": "SUBMITTED", "alpha_id": alpha_id}

    def flush_daily_candidates(self, **kwargs) -> list[dict[str, Any]]:
        return []


class TestMultiSimAndTwoStageFunnel(unittest.TestCase):
    def test_multi_sim_batch_and_sign_flip_promotion(self) -> None:
        client = _FakeBrainClient()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "funnel.db"
            funnel = TwoStageAlphaFunnel(db, seed_sharpe_threshold=0.90)
            batch1 = funnel.next_stage1_batch("analyst4", ["anl4_eps_mean", "anl4_sales_mean"], batch_size=4)
            self.assertEqual(len(batch1), 4)
            res = simulate_multi_alpha(client, batch1, poll_interval=0.01)
            self.assertEqual(len(res), 4)
            self.assertEqual(len(client.session.post_batches), 1)

            ing = funnel.ingest_results(res)
            self.assertGreaterEqual(ing["promoted"], 1)
            # Negative Sharpe seed must be auto-negated in Stage 2
            batch2 = funnel.next_stage2_batch(batch_size=4)
            self.assertEqual(len(batch2), 4)
            self.assertTrue(all(item["expression"].startswith("-(") for item in batch2))
            self.assertEqual(batch2[0]["stage"], "STAGE2")

            # Restart resilience: new funnel instance on same DB does not repeat tried expressions
            funnel_restarted = TwoStageAlphaFunnel(db, seed_sharpe_threshold=0.90)
            self.assertGreater(funnel_restarted.summary()["tried_total"], 0)

    def test_headless_miner_multi_sim_end_to_end(self) -> None:
        client = _FakeBrainClient()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "vault.db"
            stats = run_headless_miner(
                duration_minutes=0.2,
                num_workers=2,
                batch_size=4,
                datasets=["analyst4"],
                brain_client=client,  # type: ignore[arg-type]
                vault_db_path=db,
            )
            self.assertGreaterEqual(stats["batches"], 2)
            self.assertGreaterEqual(stats["seeds_promoted"], 1)
            self.assertGreaterEqual(stats["accepted"], 1)

    def test_403_single_sim_fallback_no_reauth(self) -> None:
        class _403Session:
            def request(self, method: str, url: str, json=None, params=None, timeout=30.0):
                return _MockResp(403, {"detail": "Multi-simulation disabled for tier"})

        class _FallbackClient(_FakeBrainClient):
            def __init__(self) -> None:
                super().__init__()
                self.session = _403Session()  # type: ignore[assignment]
                self.auth_calls = 0
                self.single_sim_calls: list[str] = []

            def authenticate(self) -> bool:
                self.auth_calls += 1
                return True

            def simulate_alpha(self, expr: str, *, settings=None, max_wait_seconds=1800.0, poll_interval=5.0):
                self.single_sim_calls.append(expr)
                return {
                    "success": True,
                    "status": "COMPLETED",
                    "alpha_id": f"single_{len(self.single_sim_calls)}",
                    "sharpe": 1.4,
                    "fitness": 1.1,
                    "turnover": 0.2,
                    "margin": 0.002,
                    "checks_passed": True,
                    "failed_checks": [],
                }

        client = _FallbackClient()
        batch = [
            {"expression": "rank(anl4_eps_mean)", "stage": "STAGE1", "dataset_id": "analyst4"},
            {"expression": "rank(anl4_sales_mean)", "stage": "STAGE1", "dataset_id": "analyst4"},
        ]
        res = simulate_multi_alpha(client, batch, fallback_item_delay=0.0)
        self.assertEqual(client.auth_calls, 0)
        self.assertEqual(client.single_sim_calls, ["rank(anl4_eps_mean)", "rank(anl4_sales_mean)"])
        self.assertEqual(len(res), 2)
        self.assertTrue(all(r["success"] and r["dataset_id"] == "analyst4" for r in res))

    def test_401_reauth_serialized_across_threads_and_auth_cooldown_stop(self) -> None:
        barrier = threading.Barrier(2)

        class _401Session:
            def __init__(self, client_ref: Any) -> None:
                self._client = client_ref

            def request(self, method: str, url: str, json=None, params=None, timeout=30.0):
                if not self._client.refreshed:
                    try:
                        barrier.wait(timeout=2.0)
                    except threading.BrokenBarrierError:
                        pass
                    return _MockResp(401, {"detail": "expired"})
                return _MockResp(200, {"ok": True})

        class _LockedAuthClient(_FakeBrainClient):
            def __init__(self) -> None:
                super().__init__()
                self.refreshed = False
                self.auth_calls = 0
                self.session = _401Session(self)  # type: ignore[assignment]

            def authenticate(self) -> bool:
                self.auth_calls += 1
                self.refreshed = True
                self.is_authenticated = True
                self.authenticated = True
                return True

        locked_client = _LockedAuthClient()

        def _worker() -> int:
            r = _request_with_retry(locked_client, "GET", "https://api.worldquantbrain.com/test")
            return r.status_code

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        self.assertEqual(locked_client.auth_calls, 1)

        # Headless miner must abort cleanly when batches return CAPTCHA/429 AUTH_FAILED
        class _CaptchaSession(_FakeSession):
            def request(self, method: str, url: str, json=None, params=None, timeout=30.0):
                if method == "POST" and url.endswith("/simulations"):
                    r = _MockResp(429, {"captcha": ["This field is required."]})
                    r.text = '{"captcha":["This field is required."]}'
                    return r
                return super().request(method, url, json=json, params=params, timeout=timeout)

        captcha_client = _FakeBrainClient()
        captcha_client.session = _CaptchaSession()
        buf = io.StringIO()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "vault_captcha.db"
            with contextlib.redirect_stdout(buf):
                stats = run_headless_miner(
                    duration_minutes=0.2,
                    num_workers=2,
                    batch_size=4,
                    datasets=["analyst4"],
                    brain_client=captcha_client,  # type: ignore[arg-type]
                    vault_db_path=db,
                )
        self.assertGreaterEqual(stats["failed"], 1)
        self.assertIn("[AUTH_COOLDOWN]", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
