"""Unit tests for P4 Mining Efficiency Bundle."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from worldquant_brain.multi_sim_client import simulate_multi_alpha
from worldquant_brain.two_stage_funnel import STAGE2_NEUTRALIZATIONS, TwoStageAlphaFunnel


class TestP4MiningEfficiency(unittest.TestCase):
    def test_stage2_neutralizations_exclude_restricted_defaults(self) -> None:
        self.assertNotIn("CROWDING", STAGE2_NEUTRALIZATIONS)
        self.assertNotIn("STATISTICAL", STAGE2_NEUTRALIZATIONS)
        self.assertIn("SUBINDUSTRY", STAGE2_NEUTRALIZATIONS)
        self.assertIn("INDUSTRY", STAGE2_NEUTRALIZATIONS)
        self.assertIn("SECTOR", STAGE2_NEUTRALIZATIONS)
        self.assertIn("MARKET", STAGE2_NEUTRALIZATIONS)

    def test_funnel_dynamically_blacklists_unavailable_neutralization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"))
            funnel.ingest_results(
                [
                    {
                        "success": False,
                        "status": "FAILED",
                        "expression": "rank(close)",
                        "dataset_id": "model16",
                        "stage": "STAGE2",
                        "settings": {"neutralization": "INDUSTRY", "decay": 4, "truncation": 0.08},
                        "error": '{"settings":{"neutralization":["Neutralization INDUSTRY is not available."]}}',
                    }
                ]
            )
            self.assertIn("INDUSTRY", funnel._unavailable_neutralizations)

            # Next near-miss refinement must skip INDUSTRY
            funnel._enqueue_immediate_refinement(
                {
                    "expression": "rank(open)",
                    "sharpe": 1.1,
                    "fitness": 0.8,
                    "turnover": 0.25,
                    "dataset_id": "model16",
                    "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                },
                reason="NEAR_MISS_SHARPE",
            )
            queued_neuts = {
                str(item["settings"].get("neutralization", "")).upper()
                for item in funnel._session_refinement_queue
            }
            self.assertNotIn("INDUSTRY", queued_neuts)

    def test_stage2_pruning_skips_high_decay_for_low_turnover_seed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"))
            funnel._enqueue_immediate_refinement(
                {
                    "expression": "rank(close)",
                    "sharpe": 1.1,
                    "fitness": 0.8,
                    "turnover": 0.04,
                    "dataset_id": "model16",
                    "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                },
                reason="NEAR_MISS_SHARPE",
            )
            queued_decays = {
                int(item["settings"].get("decay", 0))
                for item in funnel._session_refinement_queue
            }
            self.assertNotIn(8, queued_decays)
            self.assertNotIn(15, queued_decays)

    def test_disable_multi_sim_env_bypasses_batch_post_and_caches_bad_neut(self) -> None:
        client = MagicMock()
        client.offline_mode = False
        client.authenticated = True
        client._multi_sim_forbidden = False
        client._unavailable_neutralizations = set()

        def fake_sim(expr, settings=None, **kwargs):
            neut = str((settings or {}).get("neutralization") or "").upper()
            if neut == "CROWDING":
                raise RuntimeError('{"settings":{"neutralization":["Neutralization CROWDING is not available."]}}')
            return {"success": True, "status": "COMPLETED", "alpha_id": "A1", "sharpe": 1.5, "fitness": 1.1}

        client.simulate_alpha.side_effect = fake_sim

        items = [
            {"expression": "rank(a)", "settings": {"neutralization": "CROWDING"}, "stage": "STAGE2", "dataset_id": "m16"},
            {"expression": "rank(b)", "settings": {"neutralization": "CROWDING"}, "stage": "STAGE2", "dataset_id": "m16"},
            {"expression": "rank(c)", "settings": {"neutralization": "SECTOR"}, "stage": "STAGE2", "dataset_id": "m16"},
        ]
        with patch.dict(os.environ, {"BRAIN_DISABLE_MULTI_SIM": "true"}):
            res = simulate_multi_alpha(client, items, fallback_item_delay=0.0)

        # Should NOT call session.request for multi-sim POST
        client.session.request.assert_not_called()
        # Second CROWDING item should be fast-skipped without calling simulate_alpha a 2nd time for CROWDING
        self.assertEqual(client.simulate_alpha.call_count, 2)
        self.assertEqual(res[0]["status"], "FAILED")
        self.assertEqual(res[1]["status"], "SKIPPED")
        self.assertTrue(res[2]["success"])

    def test_headless_miner_line_buffering_and_batch_result_logging(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        src = (repo_root / "scripts/headless_brain_miner.py").read_text(encoding="utf-8")
        self.assertIn("line_buffering=True", src)
        self.assertIn("[BATCH_RESULT]", src)
        self.assertIn("BRAIN_DISABLE_MULTI_SIM", src)


if __name__ == "__main__":
    unittest.main()
