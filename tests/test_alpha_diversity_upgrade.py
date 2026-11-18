from __future__ import annotations

import pathlib
import sqlite3
import tempfile
import unittest

from worldquant_brain.alpha_miner import (
    ContinuousAlphaMiner,
    extract_expression_skeleton,
)
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.alpha_research_engines import (
    DomainBanditAllocator,
    FormulaMCTSSearch,
    compute_dico_similarity_penalty,
    warm_start_structural_mutate,
)


class _DummyVault:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def list_alphas(self):
        return [
            {
                "name": "CloneSeed",
                "expression": "ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)",
                "status": "BRAIN_COMPLETE",
            }
        ]


class TestAlphaDiversityUpgrade(unittest.TestCase):
    def test_skeleton_strips_nested_ts_decay_linear_and_numbers(self):
        e1 = "ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)"
        e2 = "ts_decay_linear(ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 3), 12)"
        self.assertEqual(extract_expression_skeleton(e1), extract_expression_skeleton(e2))

    def test_no_nested_ts_decay_linear_and_7_domain_rotation(self):
        db_file = str(pathlib.Path(tempfile.mkdtemp()) / "div.db")
        miner = ContinuousAlphaMiner()
        miner.vault = _DummyVault(db_file)
        generated = [miner.generate_unique_candidate() for _ in range(14)]
        for g in generated:
            self.assertNotIn("ts_decay_linear(ts_decay_linear(", g["expression"].lower())
        skeletons = {extract_expression_skeleton(g["expression"]) for g in generated}
        self.assertGreaterEqual(len(skeletons), 7)

    def test_candidate_pool_deduplicates_structural_clones(self):
        db_file = str(pathlib.Path(tempfile.mkdtemp()) / "pool.db")
        bc = BrainClient(email="test@example.com", password="secret")
        bc.quota_db_path = db_file
        bc.stage_candidate_for_daily_selection(
            "A1",
            "ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)",
            1.28, 1.05, 0.025, 0.002, day_str="2026-11-19",
        )
        bc.stage_candidate_for_daily_selection(
            "A2",
            "ts_decay_linear(ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 3), 12)",
            1.37, 1.22, 0.020, 0.002, day_str="2026-11-19",
        )
        conn = sqlite3.connect(db_file)
        rows = conn.execute("SELECT alpha_id, sharpe FROM brain_candidate_pool").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "A2")



    def test_think_and_select_best_candidate_prefers_high_quality(self):
        db_file = str(pathlib.Path(tempfile.mkdtemp()) / "think.db")
        miner = ContinuousAlphaMiner()
        miner.vault = _DummyVault(db_file)
        best = miner.think_and_select_best_candidate(batch_size=12)
        self.assertGreaterEqual(best["local_quality_score"], 80.0)
        self.assertNotIn("ts_decay_linear(ts_decay_linear(", best["expression"].lower())


    def test_bandit_ucb_prefers_higher_reward_domain_after_exploration(self):
        db_file = str(pathlib.Path(tempfile.mkdtemp()) / "bandit.db")
        bandit = DomainBanditAllocator(db_path=db_file, exploration_c=0.1)
        for d in bandit.domains:
            bandit.record_outcome(d, sharpe=1.45 if d == "ANALYST_EPS_DISPERSION" else 0.4, passed=(d == "ANALYST_EPS_DISPERSION"))
        self.assertEqual(bandit.select_domain(), "ANALYST_EPS_DISPERSION")

    def test_mcts_and_warm_start_and_dico_penalty(self):
        mcts = FormulaMCTSSearch()
        br = mcts.select_branch("FUNDAMENTAL_CF_EV")
        mcts.backpropagate(br, 0.9)
        self.assertEqual(mcts.visits[br], 1)

        ws = warm_start_structural_mutate("hump(group_neutralize(winsorize(rank(operating_cashflow / (cap + debt + 0.001)), std=4), subindustry), hump=0.01)")
        self.assertIsNotNone(ws)
        self.assertNotIn("ts_decay_linear(ts_decay_linear(", ws.lower())

        pen_high = compute_dico_similarity_penalty(
            "group_neutralize(rank(debt / (abs(equity) + 0.001)), industry)",
            ["ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)"],
        )
        pen_low = compute_dico_similarity_penalty(
            "hump(group_neutralize(winsorize(rank(ts_zscore(opt6_slopeavg1m, 40)), std=4), subindustry), hump=0.01)",
            ["ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)"],
        )
        self.assertGreater(pen_high, pen_low)


    def test_zombie_jobs_expired_on_reload_and_inflight_counted(self):
        from worldquant_brain.brain_job_queue import BrainJobQueue
        db_file = str(pathlib.Path(tempfile.mkdtemp()) / "zombie.db")
        conn = sqlite3.connect(db_file)
        conn.execute(
            "CREATE TABLE brain_jobs (job_id TEXT PRIMARY KEY, simulation_key TEXT, expression TEXT, settings_json TEXT, status TEXT, simulation_location TEXT, alpha_id TEXT, metrics_json TEXT, safe_error TEXT, created_at TEXT, priority INTEGER, source TEXT, requested_by TEXT, manual_note TEXT, simulation_only INTEGER, queued_at TEXT, claimed_by TEXT, claimed_at TEXT)"
        )
        conn.execute(
            "INSERT INTO brain_jobs VALUES ('j_zombie', 'k1', 'rank(close)', '{}', 'QUEUED', NULL, NULL, NULL, NULL, '2026-11-18T01:00:00+00:00', 20, 'AUTO_MINING', NULL, NULL, 1, '2026-11-18T01:00:00+00:00', 'Worker-1', '2026-11-18T02:00:00+00:00')"
        )
        conn.commit()
        conn.close()

        class _OfflineClient:
            offline_mode = True

        q = BrainJobQueue(_OfflineClient(), db_path=db_file)
        self.assertEqual(q.queue_depth(), 0)
        self.assertEqual(q.get_job("j_zombie").status.value, "FAILED")
        q._in_flight_ids.add("running_1")
        self.assertEqual(q.queue_depth(), 1)

    def test_cross_day_flush_deduplicates_and_submits_yesterday_pending(self):
        from unittest.mock import patch
        db_file = str(pathlib.Path(tempfile.mkdtemp()) / "cross_day.db")
        bc = BrainClient(email="test@example.com", password="secret")
        bc.quota_db_path = db_file
        bc.authenticated = True
        bc.stage_candidate_for_daily_selection("YEST_TOP", "group_neutralize(rank(debt / (abs(equity) + 0.001)), industry)", 1.36, 1.21, 0.022, 0.002, day_str="2026-11-18")
        # Force insert a second clone from yesterday with lower score
        conn = sqlite3.connect(db_file)
        conn.execute(
            "INSERT INTO brain_candidate_pool VALUES ('YEST_CLONE', 'ts_decay_linear(group_neutralize(rank(debt / (abs(equity) + 0.001)), industry), 5)', 1.28, 1.05, 0.025, 0.002, 1.80, '2026-11-18', 'PENDING', '2026-11-18T02:00:00Z')"
        )
        conn.commit()
        conn.close()

        def fake_sub(alpha_id, max_daily_submissions=2, enforce_checks=True, day_str=None):
            bc._record_daily_submission(alpha_id, day_str=day_str)
            return {"success": True, "status": "SUBMITTED", "alpha_id": alpha_id}

        with patch.object(bc, "submit_alpha", side_effect=fake_sub):
            res = bc.flush_daily_candidates(day_str="2026-11-19", max_daily_submissions=2)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["alpha_id"], "YEST_TOP")


    def test_fetch_self_correlation_polls_retry_after_and_parses_records(self):
        from unittest.mock import MagicMock
        bc = BrainClient(email="test@example.com", password="secret")
        bc.authenticated = True

        r_wait = MagicMock(status_code=202, content=b"", headers={"Retry-After": "0.01"})
        r_records = MagicMock(
            status_code=200,
            content=b"ok",
            headers={},
            json=lambda: {"records": [["OLD_A1", 0.42], ["OLD_A2", 0.79]]},
        )
        r_chk = MagicMock(
            status_code=200,
            content=b"ok",
            headers={},
            json=lambda: {"is": {"checks": [{"name": "SELF_CORRELATION", "result": "FAIL", "value": 0.79}]}},
        )
        r_alpha = MagicMock(status_code=200, content=b"{}", headers={}, json=lambda: {})

        bc.session.get = MagicMock(side_effect=[r_wait, r_records, r_chk, r_alpha])
        out = bc.fetch_self_correlation("NEW_ALPHA_1", max_polls=3)
        self.assertTrue(out["checked"])
        self.assertFalse(out["passed"])
        self.assertAlmostEqual(out["self_correlation"], 0.79, places=4)


    def test_domain_aware_decay_settings(self):
        miner = ContinuousAlphaMiner()
        s_fund = miner.generate_candidate_settings(domain="FUNDAMENTAL_CF_EV")
        s_pv = miner.generate_candidate_settings(domain="PV_RESIDUAL_HUMP")
        self.assertIn(s_fund["decay"], (0, 2, 4))
        self.assertIn(s_pv["decay"], (4, 6, 8, 10))


    def test_risk_manager_sell_00164_and_iso_utc_timestamp_passes(self):
        from datetime import datetime, timezone
        from crypto_trading.risk_manager import RiskManager
        rm = RiskManager(initial_capital=5000.0)
        res = rm.evaluate_order(
            symbol="BTC/USDT",
            side="SELL",
            amount=0.0164,
            price=64000.0,
            order_type="MARKET",
            context={
                "data_source": "EXCHANGE_REST",
                "data_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_quality_passed": True,
                "reconciliation_fresh": True,
                "exchange_connected": True,
                "step_size": 0.001,
                "tick_size": 0.01,
            },
        )
        self.assertTrue(res.allowed, f"Expected approved=True, got {res}")

if __name__ == "__main__":
    unittest.main()
