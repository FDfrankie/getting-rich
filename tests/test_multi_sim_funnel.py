from __future__ import annotations

import contextlib
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from scripts.headless_brain_miner import run_headless_miner
from worldquant_brain.alpha_vault import AlphaVault
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
        if method == "GET" and url.endswith("/data-sets"):
            return _MockResp(
                200,
                {
                    "results": [
                        {"id": "pv1", "valueScore": 1, "alphaCount": 2319487, "fieldCount": 24, "coverage": 1.0, "category": {"id": "pv"}},
                        {"id": "analyst4", "valueScore": 1, "alphaCount": 191793, "fieldCount": 875, "coverage": 0.82, "category": {"id": "analyst"}},
                        {"id": "rare_ds", "valueScore": 3, "alphaCount": 127, "fieldCount": 6, "coverage": 1.0, "category": {"id": "model"}},
                    ]
                },
            )
        if method == "GET" and url.endswith("/data-fields"):
            return _MockResp(
                200,
                {
                    "results": [
                        {"id": "anl4_eps_mean", "type": "MATRIX", "coverage": 0.92},
                        {"id": "anl4_sales_mean", "type": "MATRIX", "coverage": 0.88},
                    ]
                },
            )
        if method == "POST" and url.endswith("/simulations"):
            assert isinstance(json, list), "Expected Multi-Simulation list payload"
            self.post_batches.append(json)
            child_ids = []
            for item in json:
                self._seq += 1
                cid = f"child_{self._seq}"
                aid = f"alpha_{self._seq}"
                expr = item["regular"]
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

    def __init__(self, *, fail_corr_on_flush: bool = False) -> None:
        self.session = _FakeSession()
        self.staged: list[str] = []
        self.submitted: list[str] = []
        self.fail_corr_on_flush = fail_corr_on_flush

    def authenticate(self) -> bool:
        return True

    def get_daily_submission_count(self, day_str=None) -> int:
        return len(self.submitted)

    def stage_candidate_for_daily_selection(self, alpha_id: str, *args, **kwargs) -> None:
        if alpha_id not in self.staged:
            self.staged.append(alpha_id)

    def flush_daily_candidates(self, **kwargs) -> list[dict[str, Any]]:
        if not self.staged:
            return []
        aid = self.staged.pop(0)
        if self.fail_corr_on_flush:
            return [
                {
                    "success": False,
                    "status": "SUBMIT_REJECTED",
                    "alpha_id": aid,
                    "corr_rejected": True,
                    "error": "SELF_CORRELATION > 0.70",
                }
            ]
        self.submitted.append(aid)
        return [{"success": True, "status": "SUBMITTED", "alpha_id": aid}]


class TestMultiSimAndTwoStageFunnel(unittest.TestCase):
    def test_rl_policy_repulsion_and_immediate_session_refinement(self) -> None:
        client = _FakeBrainClient()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = Path(td) / "funnel.db"
            funnel = TwoStageAlphaFunnel(db, seed_sharpe_threshold=0.90)
            batch1 = funnel.next_batch({"analyst4": ["anl4_eps_mean", "anl4_sales_mean"]}, batch_size=4)
            self.assertEqual(len(batch1), 4)
            res = simulate_multi_alpha(client, batch1, poll_interval=0.01)
            ing = funnel.ingest_results(res)
            self.assertGreaterEqual(ing["promoted"], 1)
            self.assertGreaterEqual(ing["rejected"], 1)

            # Immediate in-session refinement works right away
            batch2 = funnel.next_batch({"analyst4": ["anl4_eps_mean", "anl4_sales_mean"]}, batch_size=4)
            self.assertEqual(len(batch2), 4)
            self.assertTrue(all("-(" in item["expression"] for item in batch2))

            # New session on same DB does NOT re-mine old near-miss seeds, but keeps RL policy & reference DB!
            funnel_new_session = TwoStageAlphaFunnel(db, seed_sharpe_threshold=0.90)
            self.assertEqual(funnel_new_session.pending_refinement_count(), 0)
            summ = funnel_new_session.summary()
            self.assertIn("REJECTED", summ["reference_outcomes"])
            self.assertGreater(len(summ["top_rl_arms"]), 0)

            # Parameter-tweaked clone of an already-submitted expression is blocked by skeleton guard
            funnel_new_session.mark_submitted(
                alpha_id="sub_1",
                expression="rank(ts_backfill(anl4_new_field, 20))",
                dataset_id="analyst4",
                probe="rank_bf",
            )
            with funnel_new_session._connect() as conn:
                skels, ftoks = funnel_new_session._load_reference_guards(conn)
                self.assertTrue(
                    funnel_new_session._is_repelled_by_reference(
                        "rank(ts_backfill(anl4_new_field, 10))", skels, ftoks
                    )
                )

    def test_headless_miner_stages_and_removes_failed_submit(self) -> None:
        client = _FakeBrainClient(fail_corr_on_flush=True)
        prev_flush = os.environ.get("BRAIN_DAILY_FLUSH_HOUR")
        os.environ["BRAIN_DAILY_FLUSH_HOUR"] = "0"
        try:
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
                self.assertGreaterEqual(stats["accepted"], 1)
                self.assertGreaterEqual(stats["submit_failed_deleted"], 1)
                vault = AlphaVault(str(db))
                remaining = [a for a in vault.list_alphas() if a.get("brain_alpha_id") == "alpha_9"]
                self.assertEqual(remaining, [])
                funnel = TwoStageAlphaFunnel(db)
                self.assertIn("SUBMIT_FAILED", funnel.summary()["reference_outcomes"])
        finally:
            if prev_flush is None:
                os.environ.pop("BRAIN_DAILY_FLUSH_HOUR", None)
            else:
                os.environ["BRAIN_DAILY_FLUSH_HOUR"] = prev_flush



    def test_multifactor_near_miss_boundaries(self) -> None:
        from worldquant_brain.two_stage_funnel import classify_verdict
        # High Sharpe but dead fitness -> REJECTED
        v1, _ = classify_verdict({"sharpe": 1.15, "fitness": 0.35, "turnover": 0.20, "checks_passed": False, "failed_checks": ["LOW_FITNESS"]})
        self.assertEqual(v1, "REJECTED")
        # Delay 0 requires higher near-miss Sharpe (>= 1.45)
        v2, _ = classify_verdict({"sharpe": 1.15, "fitness": 0.85, "turnover": 0.20, "checks_passed": False, "settings": {"delay": 0}, "failed_checks": ["LOW_SHARPE"]})
        self.assertEqual(v2, "REJECTED")
        # Rescuable single/double gap with solid fitness & turnover -> NEAR_MISS
        v3, _ = classify_verdict({"sharpe": 1.12, "fitness": 0.78, "turnover": 0.25, "margin": 0.0012, "checks_passed": False, "failed_checks": ["LOW_SHARPE"]})
        self.assertEqual(v3, "NEAR_MISS")
        # Non-rescuable LOW_UNIVERSE -> REJECTED
        v4, _ = classify_verdict({"sharpe": 1.30, "fitness": 1.05, "turnover": 0.25, "checks_passed": False, "failed_checks": ["LOW_UNIVERSE"]})
        self.assertEqual(v4, "REJECTED")
        # Negative Sharpe + negative Fitness (BRAIN convention) must be NEAR_MISS for sign-flip, NOT REJECTED
        v5, _ = classify_verdict({"sharpe": -1.18, "fitness": -0.82, "turnover": 0.22, "checks_passed": False, "failed_checks": ["LOW_SHARPE", "LOW_FITNESS"]})
        self.assertEqual(v5, "NEAR_MISS")



    def test_auto_dataset_discovery_prioritizes_high_value_uncrowded(self) -> None:
        from worldquant_brain.multi_sim_client import fetch_available_datasets
        client = _FakeBrainClient()
        ds = fetch_available_datasets(client)
        self.assertEqual([d["id"] for d in ds], ["rare_ds", "analyst4", "pv1"])



    def test_vector_field_wrapping_and_hump_stage2(self) -> None:
        from worldquant_brain.multi_sim_client import fetch_dataset_fields
        class _VecSession(_FakeSession):
            def request(self, method: str, url: str, json=None, params=None, timeout=30.0):
                if method == "GET" and url.endswith("/data-fields"):
                    return _MockResp(200, {"results": [{"id": "nws12_range", "type": "VECTOR", "coverage": 0.85}]})
                return super().request(method, url, json=json, params=params, timeout=timeout)
        client = _FakeBrainClient()
        client.session = _VecSession()
        fields = fetch_dataset_fields(client, "news12")
        self.assertEqual(fields, ["VEC:nws12_range"])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            funnel = TwoStageAlphaFunnel(Path(td) / "vec.db")
            b1 = funnel.next_batch({"news12": fields}, batch_size=3)
            self.assertTrue(all("vec_avg(nws12_range)" in item["expression"] for item in b1))
            # Check hump() is generated when NEAR_MISS has SUB_UNIVERSE or turnover > 0.15
            funnel._enqueue_immediate_refinement(
                {"expression": "rank(vec_avg(nws12_range))", "sharpe": 1.10, "fitness": 0.75, "turnover": 0.25, "dataset_id": "news12"},
                "SUB_UNIVERSE",
            )
            b2 = funnel.next_stage2_batch(batch_size=5)
            self.assertTrue(any("hump(" in item["expression"] for item in b2))



    def test_pareto_reward_forbidden_patterns_and_nan_handling(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            funnel = TwoStageAlphaFunnel(Path(td) / "pareto.db")
            # Record 5 straight REJECTED results on (model16, rank_bf) -> becomes forbidden pattern
            bad_batch = [
                {
                    "expression": f"rank(ts_backfill(f_{i}, 20))",
                    "dataset_id": "model16",
                    "probe": "rank_bf",
                    "stage": "STAGE1",
                    "sharpe": 0.2,
                    "fitness": 0.1,
                    "turnover": 0.80,
                    "checks_passed": False,
                    "failed_checks": ["LOW_SHARPE"],
                }
                for i in range(5)
            ]
            funnel.ingest_results(bad_batch)
            with funnel._connect() as conn:
                ranked = funnel._ucb_rank_arms(conn, ["model16"], [("rank_bf", "rank({f})"), ("vol_bucket_neut", "rank({f})")])
                self.assertEqual(ranked[0][1], "vol_bucket_neut")
                self.assertEqual(ranked[-1][1], "rank_bf")
            # Stage-2 must include nanHandling="ON" variant
            funnel._enqueue_immediate_refinement(
                {"expression": "rank(f_good)", "sharpe": 1.10, "fitness": 0.80, "turnover": 0.20, "dataset_id": "analyst4"},
                "LOW_SHARPE",
            )
            s2_all = funnel.next_stage2_batch(batch_size=30)
            self.assertTrue(any(item["settings"].get("nanHandling") == "ON" for item in s2_all))

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
