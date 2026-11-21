"""Offline E2E contract tests for BRAIN Submit (Batch 23-25) & Vertical Protocol Fixes (Batch 26)."""
from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from crypto_trading.bitunix_client import BitunixClient
from notifications.alert_manager import send_webhook_notification
from worldquant_brain.alpha_lifecycle import submission_eligibility
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.pre_submission_gate import evaluate_pre_submission


def _resp(status: int, json_data=None, headers=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.headers = dict(headers or {})
    r.content = b"{}" if json_data is not None else b""
    r.text = text or (str(json_data) if json_data is not None else "")
    r.json.return_value = json_data if json_data is not None else {}
    return r


class TestBrainSubmitAndProtocolE2E(unittest.TestCase):
    def test_submit_201_without_location_polls_submit_endpoint_and_succeeds(self) -> None:
        mock_sess = MagicMock()
        mock_sess.post.return_value = _resp(201, None, headers={"Retry-After": "0.1"})
        mock_sess.get.side_effect = [
            _resp(200, {"status": "SUBMITTED", "is": {"checks": [{"name": "SELF_CORRELATION", "result": "PASS"}]}}),
            _resp(200, {"id": "A_201", "status": "ACTIVE"}),
        ]
        with tempfile.TemporaryDirectory() as td:
            c = BrainClient(email="u@e.com", password="pw", poll_interval=0.05, session=mock_sess)
            c.authenticated = True
            c.quota_db_path = str(Path(td) / "vault.db")
            res = c.submit_alpha("A_201", enforce_checks=False, max_wait_seconds=5.0)
            self.assertTrue(res["success"], res)
            self.assertEqual(res["status"], "SUBMITTED")
            # Must have polled GET /alphas/A_201/submit
            first_get_url = mock_sess.get.call_args_list[0].args[0]
            self.assertTrue(first_get_url.endswith("/alphas/A_201/submit"))

    def test_submit_202_without_location_polls_and_succeeds(self) -> None:
        mock_sess = MagicMock()
        mock_sess.post.return_value = _resp(202, None, headers={})
        mock_sess.get.side_effect = [
            _resp(200, {"status": "COMPLETE"}),
            _resp(200, {"id": "A_202", "status": "ACTIVE"}),
        ]
        with tempfile.TemporaryDirectory() as td:
            c = BrainClient(email="u@e.com", password="pw", poll_interval=0.05, session=mock_sess)
            c.authenticated = True
            c.quota_db_path = str(Path(td) / "vault.db")
            res = c.submit_alpha("A_202", enforce_checks=False, max_wait_seconds=5.0)
            self.assertTrue(res["success"], res)

    def test_submit_403_rule_rejection_does_not_reauthenticate(self) -> None:
        mock_sess = MagicMock()
        mock_sess.post.return_value = _resp(
            403,
            {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL"}]}},
            text='{"is":{"checks":[{"name":"SELF_CORRELATION","result":"FAIL"}]}}',
        )
        c = BrainClient(email="u@e.com", password="pw", poll_interval=0.05, session=mock_sess)
        c.authenticated = True
        with patch.object(c, "authenticate") as mock_auth:
            res = c.submit_alpha("A_403", enforce_checks=False)
            mock_auth.assert_not_called()
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "SUBMIT_REJECTED")
        self.assertTrue(res["corr_rejected"])
        self.assertIn("SELF_CORRELATION", res["failed_checks"])

    def test_flush_daily_candidates_marks_submitted_and_never_resubmits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_p = str(Path(td) / "vault.db")
            c = BrainClient(email="u@e.com", password="pw", poll_interval=0.05, session=MagicMock())
            c.authenticated = True
            c.quota_db_path = db_p
            c.stage_candidate_for_daily_selection("A_ONCE", "rank(close)", 1.8, 1.3, 0.20, 0.002, day_str="2026-11-21")

            with patch.object(c, "submit_alpha", return_value={"success": True, "status": "SUBMITTED", "alpha_id": "A_ONCE"}) as m_sub:
                r1 = c.flush_daily_candidates(day_str="2026-11-21", max_daily_submissions=2)
                self.assertEqual(len(r1), 1)
                self.assertTrue(r1[0]["success"])

                conn = sqlite3.connect(db_p)
                st = conn.execute("SELECT status FROM brain_candidate_pool WHERE alpha_id='A_ONCE'").fetchone()[0]
                conn.close()
                self.assertEqual(st, "SUBMITTED")

                # Second flush MUST NOT resubmit A_ONCE
                r2 = c.flush_daily_candidates(day_str="2026-11-21", max_daily_submissions=2)
                self.assertEqual(r2, [])
                self.assertEqual(m_sub.call_count, 1)

    def test_pre_submission_gate_inclusive_bounds_and_lifecycle_ready(self) -> None:
        elig = submission_eligibility(
            {
                "status": "SUBMISSION_CANDIDATE",
                "sharpe": 1.25,
                "fitness": 1.0,
                "turnover": 0.01,
                "margin_bps": 12.0,
                "drawdown": 0.05,
                "is_check_status": "PASS",
                "self_correlation": 0.40,
                "brain_simulation_id": "sim_1",
                "brain_alpha_id": "A_READY",
            }
        )
        self.assertTrue(elig.ready, elig.reasons)

        gate = evaluate_pre_submission(
            {
                "brain_alpha_id": "A_BOUND",
                "official_simulation_id": "sim_1",
                "metrics_source": "WORLDQUANT_BRAIN",
                "sharpe": 1.25,
                "fitness": 1.0,
                "turnover": 0.01,
                "self_correlation": 0.40,
                "is_checks_passed": True,
                "submit_ready": True,
                "good_ready": True,
            }
        )
        self.assertTrue(gate.eligible, gate.reasons)

    def test_bitunix_official_fapi_url_double_sha256_and_sign_header(self) -> None:
        client = BitunixClient(api_key="my_key", api_secret="my_secret")
        self.assertEqual(client.base_url, "https://fapi.bitunix.com")

        nonce = "0123456789abcdef0123456789abcdef"
        ts = "1700000000000"
        body = '{"symbol":"BTCUSDT"}'
        expected_first = hashlib.sha256(f"{nonce}{ts}my_key{body}".encode("utf-8")).hexdigest()
        expected_sign = hashlib.sha256(f"{expected_first}my_secret".encode("utf-8")).hexdigest()
        self.assertEqual(client._generate_signature(nonce, ts, "", body), expected_sign)

        hdrs = client._get_headers("", body)
        self.assertIn("sign", hdrs)
        self.assertNotIn("signature", hdrs)
        self.assertEqual(len(hdrs["nonce"]), 32)
        self.assertEqual(hdrs["api-key"], "my_key")

    def test_send_webhook_notification_posts_json_payload(self) -> None:
        with patch("requests.post", return_value=_resp(204)) as m_post:
            ok = send_webhook_notification("https://discord.com/api/webhooks/test", "台股晨報", "內容摘要")
            self.assertTrue(ok)
            m_post.assert_called_once()
            self.assertIn("content", m_post.call_args.kwargs["json"])

    def test_sync_vault_snapshot_to_drive_mirrors_sqlite_and_pending_json(self) -> None:
        import json
        from scripts.headless_brain_miner import sync_vault_snapshot_to_drive

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            local_db = td_p / "local_state" / "alpha_vault.db"
            local_db.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(local_db))
            conn.execute(
                "CREATE TABLE brain_candidate_pool ("
                "alpha_id TEXT PRIMARY KEY, expression TEXT, sharpe REAL, fitness REAL, "
                "turnover REAL, margin REAL, score REAL, candidate_date TEXT, status TEXT, created_at TEXT)"
            )
            conn.execute(
                "INSERT INTO brain_candidate_pool VALUES "
                "('A_P1', 'rank(close)', 1.5, 1.2, 0.1, 0.002, 82.0, '2026-11-21', 'PENDING', '2026-11-21T01:00:00Z'),"
                "('A_S1', 'rank(open)', 1.4, 1.1, 0.1, 0.002, 75.0, '2026-11-21', 'SUBMITTED', '2026-11-21T00:00:00Z')"
            )
            conn.commit()
            conn.close()

            mirror_dir = td_p / "drive_mirror"
            status_json = td_p / "drive_status" / "brain_pending_status.json"
            res = sync_vault_snapshot_to_drive(
                local_db,
                mirror_dir=mirror_dir,
                status_json_path=status_json,
                miner_stats={"mined": 10, "accepted": 1},
            )
            self.assertTrue(res["db_mirrored"])
            self.assertTrue(res["status_written"])
            self.assertEqual(res["pending_count"], 1)
            self.assertTrue((mirror_dir / "alpha_vault.db").is_file())

            data = json.loads(status_json.read_text(encoding="utf-8"))
            self.assertEqual(data["pending_count"], 1)
            self.assertEqual(data["candidate_pool_counts"]["PENDING"], 1)
            self.assertEqual(data["candidate_pool_counts"]["SUBMITTED"], 1)
            self.assertEqual(data["pending_alphas"][0]["alpha_id"], "A_P1")


if __name__ == "__main__":
    unittest.main()
