"""Regression tests for P34 (Built-In Auto-Submit 5 bps Alignment + Unsubmitted Backlog Harvester)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class _FakeResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8")
        self.headers = {}

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, get_map: dict[str, _FakeResp]):
        self.get_map = get_map
        self.cookies = {}
        self.headers = {}

    def get(self, url: str, params=None, timeout=None):
        for k, resp in self.get_map.items():
            if k in url:
                return resp
        return _FakeResp(404, {})


def test_verify_alpha_accepts_752bps_P05o9gqw_and_rejects_101pct_turnover(monkeypatch):
    monkeypatch.delenv("BRAIN_MIN_MARGIN", raising=False)
    from worldquant_brain.brain_client import BrainClient

    # P05o9gqw-like passing alpha with 7.52 bps margin (0.000752)
    sess_ok = _FakeSession({
        "/alphas/P05o9gqw": _FakeResp(200, {
            "id": "P05o9gqw",
            "settings": {"delay": 1, "neutralization": "SUBINDUSTRY", "region": "USA"},
            "regular": {"code": "rank(ts_regression(ts_backfill(unsystematic_risk_last_360_days, 20), returns, 60, lag=2, rettype=0))"},
            "is": {"sharpe": 1.67, "fitness": 1.02, "turnover": 0.2218, "margin": 0.000752, "checks": [{"name": "LOW_SHARPE", "result": "PASS"}]},
        }),
    })
    client = BrainClient(email="a@b.com", password="pw", session=sess_ok)
    client.authenticated = True
    client.check_local_active_self_correlation = lambda *a, **k: {"blocked": False}
    res_ok = client.verify_alpha_for_submission("P05o9gqw")
    assert res_ok["passed"] is True, f"Expected 7.52 bps alpha to PASS, got: {res_ok}"

    # 101% turnover (1.01) must NOT be mis-normalized to 0.0101!
    sess_bad_to = _FakeSession({
        "/alphas/BADTO": _FakeResp(200, {
            "id": "BADTO",
            "settings": {"delay": 1, "neutralization": "SUBINDUSTRY", "region": "USA"},
            "regular": {"code": "rank(close)"},
            "is": {"sharpe": 2.0, "fitness": 1.2, "turnover": 1.01, "margin": 0.002, "checks": []},
        }),
    })
    client2 = BrainClient(email="a@b.com", password="pw", session=sess_bad_to)
    client2.authenticated = True
    res_bad = client2.verify_alpha_for_submission("BADTO")
    assert res_bad["passed"] is False
    assert any("Turnover" in r for r in res_bad["reasons"])


def test_harvest_unsubmitted_candidates_stages_qualified_and_skips_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DB_PATH", str(tmp_path / "vault.db"))
    monkeypatch.delenv("BRAIN_MIN_MARGIN", raising=False)
    monkeypatch.setenv("BRAIN_RESERVE_MIN_SHARPE", "99")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_FITNESS", "99")
    monkeypatch.setenv("BRAIN_SUBMIT_ORDER", "best_first")
    from worldquant_brain.brain_client import BrainClient

    sess = _FakeSession({
        "/users/self/alphas": _FakeResp(200, {
            "results": [
                {
                    "id": "vRAOGpeG",
                    "settings": {"delay": 1},
                    "regular": {"code": "group_neutralize(rank(liabilities/assets), subindustry)"},
                    "is": {"sharpe": 1.52, "fitness": 1.33, "turnover": 0.0137, "margin": 0.014023, "checks": [{"name": "LOW_SHARPE", "result": "PASS"}]},
                },
                {
                    "id": "58JQGZ7n",
                    "settings": {"delay": 1},
                    "regular": {"code": "group_neutralize(-ts_std_dev(scl12_buzz, 8), subindustry)"},
                    "is": {"sharpe": 1.78, "fitness": 1.45, "turnover": 0.272, "margin": 0.001326, "checks": [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}]},
                },
                {
                    "id": "P05o9gqw",
                    "settings": {"delay": 1},
                    "regular": {"code": "rank(ts_regression(unsystematic_risk_last_360_days, returns, 60, lag=2, rettype=0))"},
                    "is": {"sharpe": 1.67, "fitness": 1.02, "turnover": 0.2218, "margin": 0.000752, "checks": [{"name": "LOW_SHARPE", "result": "PASS"}]},
                },
            ]
        })
    })
    client = BrainClient(email="a@b.com", password="pw", session=sess)
    client.authenticated = True
    client._quota_db = lambda: str(tmp_path / "vault.db")
    staged = client.harvest_unsubmitted_candidates(limit=50, day_str="2026-11-22")
    assert staged == 2  # vRAOGpeG + P05o9gqw staged; 58JQGZ7n skipped (FAIL check)

    # Idempotent second call must stage 0 duplicates
    assert client.harvest_unsubmitted_candidates(limit=50, day_str="2026-11-22") == 0

    conn = sqlite3.connect(tmp_path / "vault.db")
    ids = {r[0] for r in conn.execute("SELECT alpha_id FROM brain_candidate_pool WHERE status='PENDING'").fetchall()}
    conn.close()
    assert ids == {"vRAOGpeG", "P05o9gqw"}
