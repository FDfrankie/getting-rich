"""API contract tests for IQC 2026 endpoints in web_ui/server.py (using local loopback ephemeral server)."""
from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import importlib.util
import sys
import types

if "yfinance" not in sys.modules and importlib.util.find_spec("yfinance") is None:
    _yf = types.ModuleType("yfinance")
    _yf.download = lambda *a, **kw: None
    _yf.Ticker = lambda *a, **kw: None
    sys.modules["yfinance"] = _yf

if "joblib" not in sys.modules and importlib.util.find_spec("joblib") is None:
    _jl = types.ModuleType("joblib")
    _jl.load = lambda *a, **kw: None
    _jl.dump = lambda *a, **kw: None
    sys.modules["joblib"] = _jl

from web_ui.server import APP_STATE, AppAPIHandler
from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.iqc_scoring import IQCScoringRepository


class TestIQCAPIContract(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._vault_db = Path(self._tmp.name) / "vault.db"
        self._vault = AlphaVault(db_path=self._vault_db)
        # Seed 3 trusted backend records: 1 strong S/A, 1 blocked (0.7000 corr), 1 negative merged

        # Enrich vault rows directly with official check columns via extra dict or monkeypatched list_alphas
        self._trusted_rows = [
            {
                "id": "alpha_good",
                "brain_alpha_id": "alpha_good",
                "official_simulation_id": "sim_good",
                "metrics_source": "WORLDQUANT_BRAIN",
                "expression": "group_rank(ts_zscore(est_eps_revision, 40), subindustry)",
                "region": "USA",
                "universe": "TOP1000",
                "delay": 1,
                "dataset_category": "ANALYST",
                "idea_family": "ANALYST_REV",
                "sharpe": 1.92,
                "fitness": 1.73,
                "returns": 0.145,
                "turnover": 0.22,
                "margin": 0.00125,
                "drawdown": 0.055,
                "self_correlation": 0.31,
                "max_instrument_weight": 0.065,
                "bias_test_passed": True,
                "weight_test_passed": True,
                "sub_universe_passed": True,
                "submit_ready": True,
                "temporal_stability": 0.86,
                "parameter_stability": 0.78,
                "universe_stability": 0.82,
                "regime_stability": 0.77,
                "dsr_probability": 0.97,
                "pbo": 0.14,
                "cpcv_positive_ratio": 0.72,
                "delta_merged_sharpe": 0.09,
                "delta_merged_return": 0.018,
                "delta_merged_drawdown": -0.009,
                "delta_return_drawdown_ratio": 0.18,
                "delta_merged_turnover": -0.004,
                "official_before_after_improves": True,
                "structural_novelty": 0.90,
                "dataset_gap_value": 0.85,
                "delay_universe_gap_value": 0.80,
                "cost_stress_completed": True,
                "merged_analysis_completed": True,
            },
            {
                "id": "alpha_corr_070",
                "brain_alpha_id": "alpha_corr_070",
                "official_simulation_id": "sim_corr",
                "metrics_source": "WORLDQUANT_BRAIN",
                "expression": "rank(close)",
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "dataset_category": "PRICE_VOLUME",
                "idea_family": "PV_1",
                "sharpe": 2.10,
                "fitness": 1.80,
                "returns": 0.16,
                "turnover": 0.25,
                "self_correlation": 0.7000,
                "max_instrument_weight": 0.05,
                "bias_test_passed": True,
                "weight_test_passed": True,
                "sub_universe_passed": True,
                "submit_ready": True,
            },
            {
                "id": "alpha_neg_merged",
                "brain_alpha_id": "alpha_neg_merged",
                "official_simulation_id": "sim_neg",
                "metrics_source": "WORLDQUANT_BRAIN",
                "expression": "rank(open)",
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "dataset_category": "PRICE_VOLUME",
                "idea_family": "PV_2",
                "sharpe": 1.95,
                "fitness": 1.65,
                "returns": 0.14,
                "turnover": 0.22,
                "self_correlation": 0.45,
                "max_instrument_weight": 0.06,
                "bias_test_passed": True,
                "weight_test_passed": True,
                "sub_universe_passed": True,
                "submit_ready": True,
                "delta_merged_sharpe": -0.03,
                "official_before_after_improves": False,
            },
        ]

        self._orig_vault = APP_STATE.alpha_vault
        self._vault.list_alphas = lambda *a, **kw: list(self._trusted_rows)  # type: ignore[method-assign]
        APP_STATE.alpha_vault = self._vault

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), AppAPIHandler)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def tearDown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        APP_STATE.alpha_vault = self._orig_vault
        self._tmp.cleanup()

    def _req(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict, str]:
        conn = http.client.HTTPConnection("127.0.0.1", self._port, timeout=5)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        payload = json.dumps(body) if body is not None else None
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        ctype = resp.getheader("Content-Type", "")
        conn.close()
        return resp.status, json.loads(raw), ctype

    def test_iqc_api_contracts_and_recalculate_idempotency(self) -> None:
        # 1. GET /api/brain/iqc/candidates
        status, data, ctype = self._req("GET", "/api/brain/iqc/candidates")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        self.assertEqual(data["count"], 3)
        by_id = {c["alpha_id"]: c for c in data["candidates"]}
        self.assertEqual(by_id["alpha_corr_070"]["decision"], "BLOCKED")
        self.assertEqual(by_id["alpha_corr_070"]["grade"], "BLOCKED")
        self.assertIn("SELF_CORRELATION_NOT_PASSED", by_id["alpha_corr_070"]["blockers"])
        self.assertTrue(0.0 <= by_id["alpha_good"]["candidate_score"] <= 100.0)
        self.assertTrue(0.0 <= by_id["alpha_good"]["timing_score"] <= 20.0)
        self.assertIsNone(by_id["alpha_good"]["relative_competition_estimate"])

        # 2. GET /api/brain/iqc/candidates/{alpha_id} + 404
        st_ok, d_ok, _ = self._req("GET", "/api/brain/iqc/candidates/alpha_good")
        self.assertEqual(st_ok, 200)
        self.assertEqual(d_ok["alpha_id"], "alpha_good")
        st_404, d_404, _ = self._req("GET", "/api/brain/iqc/candidates/nonexistent_alpha_999")
        self.assertEqual(st_404, 404)
        self.assertNotIn("Traceback", json.dumps(d_404))
        self.assertNotIn(str(self._vault_db), json.dumps(d_404))

        # 3. GET /api/brain/iqc/daily-plan (excludes blocked & negative merged)
        st_dp, d_dp, _ = self._req("GET", "/api/brain/iqc/daily-plan")
        self.assertEqual(st_dp, 200)
        self.assertLessEqual(len(d_dp["selected"]), 3)
        selected_ids = [s["alpha_id"] for s in d_dp["selected"]]
        self.assertIn("alpha_good", selected_ids)
        self.assertNotIn("alpha_corr_070", selected_ids)
        self.assertNotIn("alpha_neg_merged", selected_ids)

        # 4. POST /api/brain/iqc/recalculate ignores client-supplied metrics & is idempotent
        st_r1, d_r1, _ = self._req(
            "POST",
            "/api/brain/iqc/recalculate",
            body={
                "alpha_id": "alpha_corr_070",
                "self_correlation": 0.05,  # Attempted client spoofing MUST be ignored
                "sharpe": 9.99,
                "submit_ready": True,
            },
        )
        self.assertEqual(st_r1, 200)
        self.assertEqual(d_r1["decision"], "BLOCKED")
        self.assertIn("SELF_CORRELATION_NOT_PASSED", d_r1["blockers"])

        repo = IQCScoringRepository(self._vault_db)
        audits_before = len(repo.list_audits("alpha_corr_070"))
        st_r2, d_r2, _ = self._req("POST", "/api/brain/iqc/recalculate", body={"alpha_id": "alpha_corr_070"})
        self.assertEqual(st_r2, 200)
        self.assertFalse(d_r2["audit_created"])
        self.assertEqual(len(repo.list_audits("alpha_corr_070")), audits_before)



        # 5. GET /api/brain/alphas/{alpha_id}/pnl-diagnosis & /improvement-checklist
        st_pd, d_pd, _ = self._req("GET", "/api/brain/alphas/alpha_good/pnl-diagnosis")
        self.assertEqual(st_pd, 200)
        self.assertEqual(d_pd["alpha_id"], "alpha_good")
        self.assertIn("VECTOR_NOT_OBSERVED", d_pd["diagnoses"])
        self.assertIn("ALPHA_VECTOR_NOT_OFFICIALLY_OBSERVED", d_pd["warnings"])
        st_pd404, _, _ = self._req("GET", "/api/brain/alphas/missing_999/pnl-diagnosis")
        self.assertEqual(st_pd404, 404)

        st_cl, d_cl, _ = self._req("GET", "/api/brain/alphas/alpha_good/improvement-checklist")
        self.assertEqual(st_cl, 200)
        self.assertEqual(d_cl["checklist_doc"], "docs/ALPHA_IMPROVEMENT_CHECKLIST.md")


if __name__ == "__main__":
    unittest.main()
