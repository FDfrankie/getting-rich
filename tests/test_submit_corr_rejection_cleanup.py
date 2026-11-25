"""Tests verifying submit_alpha catches failed IS/correlation checks on COMPLETE polls and deletes high-corr alphas from AlphaVault."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.content = b"1" if payload is not None else b""
        self.text = text

    def json(self) -> dict:
        return self._payload


class TestSubmitCorrRejectionCleanup(unittest.TestCase):
    def _make_client(self, tmpdir: str) -> BrainClient:
        with patch.dict(os.environ, {"BRAIN_EMAIL": "u@example.com", "BRAIN_PASSWORD": "pw"}):
            c = BrainClient(poll_interval=0.01)
        c.authenticated = True
        c.get_daily_submission_count = MagicMock(return_value=0)
        return c

    def test_complete_poll_with_failed_self_corr_returns_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = self._make_client(td)
            c.session = MagicMock()
            c.session.post.return_value = _FakeResp(
                202, headers={"Location": "https://api.worldquantbrain.com/alphas/A1/submit"}
            )
            c.session.get.return_value = _FakeResp(
                200,
                payload={
                    "status": "COMPLETE",
                    "is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL"}]},
                },
            )
            with patch.object(c, "_record_daily_submission") as mock_rec:
                res = c.submit_alpha("A1", enforce_checks=False, day_str="2099-01-01")
                self.assertFalse(res["success"])
                self.assertEqual(res["status"], "SUBMIT_REJECTED")
                self.assertTrue(res.get("corr_rejected"))
                mock_rec.assert_not_called()

    def test_complete_poll_with_unsubmitted_remote_status_returns_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            c = self._make_client(td)
            c.session = MagicMock()
            c.session.post.return_value = _FakeResp(
                202, headers={"Location": "https://api.worldquantbrain.com/alphas/A2/submit"}
            )
            c.session.get.side_effect = [
                _FakeResp(200, payload={"status": "COMPLETE"}),
                _FakeResp(
                    200,
                    payload={
                        "id": "A2",
                        "status": "UNSUBMITTED",
                        "is": {"checks": [{"name": "PROD_CORRELATION", "result": "FAIL"}]},
                    },
                ),
            ]
            with patch.object(c, "_record_daily_submission") as mock_rec:
                res = c.submit_alpha("A2", enforce_checks=False, day_str="2099-01-02")
                self.assertFalse(res["success"])
                self.assertEqual(res["status"], "SUBMIT_REJECTED")
                self.assertTrue(res.get("corr_rejected"))
                mock_rec.assert_not_called()

    def test_complete_poll_with_active_remote_status_succeeds_and_vault_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            c = self._make_client(td)
            c.session = MagicMock()
            c.session.post.return_value = _FakeResp(
                202, headers={"Location": "https://api.worldquantbrain.com/alphas/A3/submit"}
            )
            c.session.get.side_effect = [
                _FakeResp(200, payload={"status": "COMPLETE", "is": {"checks": [{"name": "SELF_CORRELATION", "result": "PASS"}]}}),
                _FakeResp(200, payload={"id": "A3", "status": "ACTIVE"}),
            ]
            with patch.object(c, "_record_daily_submission") as mock_rec:
                res = c.submit_alpha("A3", enforce_checks=False, day_str="2099-01-03")
                self.assertTrue(res["success"])
                self.assertEqual(res["status"], "SUBMITTED")
                mock_rec.assert_called_once()

            # Verify AlphaVault deletion works for rejected high-corr item
            vault = AlphaVault(db_path=str(Path(td) / "vault.db"))
            save_res = vault.save_alpha(
                name="BRAIN_A1",
                expression="rank(close)",
                sharpe=1.9,
                fitness=1.3,
                turnover=0.2,
                margin=0.002,
                status_override="BRAIN_COMPLETE",
            )
            vid = save_res["alpha_id"]
            self.assertTrue(any(a["id"] == vid for a in vault.list_alphas()))
            vault.delete_alpha(int(vid))
            self.assertFalse(any(a["id"] == vid for a in vault.list_alphas()))

    def test_near_miss_refinements_for_self_corr_and_sub_universe_and_flush_rejected_status(self) -> None:
        from worldquant_brain.alpha_miner import ContinuousAlphaMiner
        from worldquant_brain.alpha_research_engines import estimate_local_structural_correlation

        miner = ContinuousAlphaMiner()
        miner.vault.db_path = ":memory:"
        corr_variants = miner.generate_near_miss_refinements(
            "hump(group_neutralize(rank(ts_delta(close, 20)), subindustry), hump=0.01)",
            sharpe=1.55,
            fitness=1.15,
            turnover=0.22,
            failed_checks=["SELF_CORRELATION_FAIL"],
            max_variants=6,
        )
        self.assertTrue(len(corr_variants) >= 2)
        self.assertTrue(any("ts_regression(" in v["expression"] for v in corr_variants))
        self.assertTrue(any("23" in v["expression"] or "signed_power(" in v["expression"] for v in corr_variants))

        sub_variants = miner.generate_near_miss_refinements(
            "rank(ts_zscore(eps_consensus_mean, 40))",
            sharpe=1.45,
            fitness=1.08,
            turnover=0.25,
            failed_checks=["LOW_SUB_UNIVERSE_SHARPE"],
            max_variants=3,
        )
        self.assertTrue(len(sub_variants) >= 2)
        self.assertTrue(any("volume * close" in v["expression"] for v in sub_variants))

        est_corr, shares_raw = estimate_local_structural_correlation(
            "hump(group_neutralize(winsorize(rank(operating_cashflow / (cap + debt + 0.001)), std=4), subindustry), hump=0.01)",
            ["hump(group_neutralize(winsorize(rank(eps_consensus_mean), std=4), industry), hump=0.01)"],
        )
        self.assertLess(est_corr, 0.45)
        self.assertFalse(shares_raw)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_file = str(Path(td) / "flush_rej.db")
            bc = BrainClient(email="u@example.com", password="pw")
            bc.quota_db_path = db_file
            bc.authenticated = True
            bc.stage_candidate_for_daily_selection(
                "REJ_ALPHA_1",
                "rank(operating_cashflow / (cap + 0.001))",
                1.75, 1.30, 0.22, 0.002, day_str="2026-11-19",
            )
            with patch.object(bc, "submit_alpha", return_value={
                "success": False,
                "status": "SUBMIT_REJECTED",
                "alpha_id": "REJ_ALPHA_1",
                "corr_rejected": True,
                "error": "SELF_CORRELATION",
            }):
                res = bc.flush_daily_candidates(day_str="2026-11-19", max_daily_submissions=2)
            self.assertEqual(len(res), 1)
            import sqlite3
            conn = sqlite3.connect(db_file)
            st = conn.execute("SELECT status FROM brain_candidate_pool WHERE alpha_id = 'REJ_ALPHA_1'").fetchone()[0]
            conn.close()
            self.assertEqual(st, "FAILED_CORR")


if __name__ == "__main__":
    unittest.main()
