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

if __name__ == "__main__":
    unittest.main()
