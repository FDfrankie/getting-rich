"""Tests for ScoreTracker leaderboard.score parsing & Challenge reserve-high-value selector."""
from __future__ import annotations

import json
import unittest

from worldquant_brain.daily_candidate_selector import select_daily_candidates
from worldquant_brain.score_tracker import parse_score_payload


class TestScoreTrackerAndReserveSelector(unittest.TestCase):
    def test_parses_real_brain_challenge_leaderboard_payload(self) -> None:
        payload = json.loads(
            '{"id": "FJ10089", "level": "BRONZE", "leaderboard": {"alphas": 1, "country": "TW", "level": "BRONZE", "rank": 30823, "score": 1886.0}}'
        )
        snap = parse_score_payload(payload, source="AUTHENTICATED_CHALLENGE_API")
        self.assertEqual(snap.total_score, 1886.0)
        self.assertEqual(snap.rank_value, 30823)

    def test_reserve_high_value_picks_lowest_qualifying_alpha_first(self) -> None:
        cands = [
            {
                "alpha_id": "HIGH_GEM",
                "brain_alpha_id": "HIGH_GEM",
                "brain_simulation_id": "sim_hi",
                "official_simulation_id": "sim_hi",
                "metrics_source": "WORLDQUANT_BRAIN",
                "is_check_status": "PASS",
                "checks_passed": True,
                "sharpe": 2.45,
                "fitness": 1.95,
                "turnover": 0.18,
                "margin": 0.0025,
                "self_correlation": 0.15,
                "status": "BRAIN_COMPLETE",
            },
            {
                "alpha_id": "LOW_PASS",
                "brain_alpha_id": "LOW_PASS",
                "brain_simulation_id": "sim_lo",
                "official_simulation_id": "sim_lo",
                "metrics_source": "WORLDQUANT_BRAIN",
                "is_check_status": "PASS",
                "checks_passed": True,
                "sharpe": 1.28,
                "fitness": 1.05,
                "turnover": 0.25,
                "margin": 0.0012,
                "self_correlation": 0.35,
                "status": "BRAIN_COMPLETE",
            },
            {
                "alpha_id": "POOL_DEGRADING_HIGH_CORR",
                "brain_alpha_id": "POOL_DEGRADING_HIGH_CORR",
                "brain_simulation_id": "sim_deg",
                "official_simulation_id": "sim_deg",
                "metrics_source": "WORLDQUANT_BRAIN",
                "is_check_status": "PASS",
                "checks_passed": True,
                "sharpe": 1.26,
                "fitness": 1.01,
                "turnover": 0.45,
                "margin": 0.0008,
                "self_correlation": 0.65,
                "status": "BRAIN_COMPLETE",
            },
        ]
        # Default mode picks HIGH_GEM first
        batch_default = select_daily_candidates(cands, top_k=1, formal_submission_count=0, reserve_high_value=False)
        self.assertEqual(batch_default.selected[0].alpha_id, "HIGH_GEM")

        # Reserve mode skips POOL_DEGRADING_HIGH_CORR (corr >= 0.50), picks LOW_PASS, and saves HIGH_GEM
        batch_reserve = select_daily_candidates(cands, top_k=1, formal_submission_count=0, reserve_high_value=True)
        self.assertEqual(batch_reserve.selected[0].alpha_id, "LOW_PASS")
        self.assertEqual(batch_reserve.selected[0].selection_type, "CHALLENGE_LOWEST_QUALIFYING")
        self.assertIn("HIGH_GEM", batch_reserve.retained_pending_ids)


if __name__ == "__main__":
    unittest.main()
