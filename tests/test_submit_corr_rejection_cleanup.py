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


if __name__ == "__main__":
    unittest.main()
