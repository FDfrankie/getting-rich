"""Unit tests for Single-Failure Bounded Parameter Binary Search."""
from __future__ import annotations

import unittest

from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.param_binary_search import (
    BinarySearchBracket,
    generate_single_failure_binary_search,
)


class TestParamBinarySearch(unittest.TestCase):
    def test_bracket_midpoints(self) -> None:
        br = BinarySearchBracket("decay", low=2.0, high=16.0, current=4.0, is_int=True)
        self.assertEqual(br.next_midpoints("UPPER"), [10])
        self.assertEqual(br.next_midpoints("LOWER"), [3])
        self.assertEqual(br.next_midpoints("BOTH"), [3, 10])

    def test_skips_when_multiple_failures_or_weak_sharpe(self) -> None:
        self.assertEqual(
            generate_single_failure_binary_search(
                "rank(close - open)", {"decay": 4}, failed_checks=["LOW_SHARPE", "LOW_FITNESS"], sharpe=1.15
            ),
            [],
        )
        self.assertEqual(
            generate_single_failure_binary_search(
                "rank(close - open)", {"decay": 4}, failed_checks=["LOW_SHARPE"], sharpe=0.70
            ),
            [],
        )

    def test_single_high_turnover_binary_searches_decay_and_hump(self) -> None:
        cands = generate_single_failure_binary_search(
            "hump(rank(close - open), hump=0.008)",
            {"decay": 4, "truncation": 0.05},
            failed_checks=["HIGH_TURNOVER"],
            sharpe=1.55,
            turnover=0.52,
        )
        self.assertGreaterEqual(len(cands), 2)
        self.assertEqual(cands[0]["settings"]["decay"], 10)
        self.assertIn("hump=0.016", cands[1]["expression"])
        self.assertFalse(cands[0]["structural_novelty"])

    def test_single_low_sharpe_binary_searches_window(self) -> None:
        cands = generate_single_failure_binary_search(
            "group_neutralize(ts_zscore(close, 20), industry)",
            {"decay": 4},
            failed_checks=["LOW_SHARPE"],
            sharpe=1.18,
        )
        self.assertEqual(len(cands), 2)
        self.assertIn(", 15)", cands[0]["expression"])
        self.assertIn(", 30)", cands[1]["expression"])

    def test_alpha_miner_prioritizes_binary_search_on_single_failure(self) -> None:
        miner = ContinuousAlphaMiner()
        miner.has_tried_expression = lambda _e: False
        miner.mark_expression_tried = lambda _e: None
        res = miner.generate_near_miss_refinements(
            "group_neutralize(ts_zscore(close, 20), industry)",
            {"universe": "TOP3000", "neutralization": "INDUSTRY", "decay": 4, "truncation": 0.05},
            sharpe=1.18,
            fitness=1.05,
            turnover=0.22,
            failed_checks=["LOW_SHARPE"],
            max_variants=3,
        )
        self.assertTrue(any(r["name"].startswith("BinSearch_") for r in res))


if __name__ == "__main__":
    unittest.main()
