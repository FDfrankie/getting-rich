"""Contract tests for P26: Alpha Quality Overhaul (P05M7vpq High-TO Rescue, High-Coverage Dataset Priority, Vault Protection)."""
from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.multi_sim_client import fetch_available_datasets
from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel, classify_verdict


class TestP26AlphaQualityOverhaul(unittest.TestCase):
    def test_high_sharpe_high_turnover_gem_like_p05m7vpq_is_rescued_and_crushed(self) -> None:
        gem = {
            "expression": "rank(ts_backfill(unsystematic_risk_last_360_days, 20) - last_diff_value(ts_backfill(unsystematic_risk_last_360_days, 20), 1))",
            "sharpe": 2.06,
            "fitness": 0.58,
            "turnover": 1.0095,
            "margin": 0.00016,
            "checks_passed": False,
            "failed_checks": ["LOW_FITNESS", "HIGH_TURNOVER"],
            "dataset_id": "model51",
        }
        verdict, reason = classify_verdict(gem)
        self.assertEqual(verdict, "NEAR_MISS", f"Expected NEAR_MISS for Sharpe=2.06 gem, got {verdict} ({reason})")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            f = TwoStageAlphaFunnel(Path(td) / "vault.db")
            added = f._enqueue_immediate_refinement(gem, reason)
            self.assertGreaterEqual(added, 3)
            q = f.next_stage2_batch(batch_size=4)
            exprs = [x["expression"] for x in q]
            decays = [x["settings"].get("decay") for x in q]
            self.assertTrue(any("ts_decay_linear(" in e and d == 10 for e, d in zip(exprs, decays)))
            self.assertTrue(any("trade_when(volume > adv20" in e for e in exprs))
            gc.collect()

    def test_fetch_available_datasets_prioritizes_high_coverage_over_sparse(self) -> None:
        client = MagicMock()
        client.offline_mode = False
        client.is_authenticated = True
        client.BASE_URL = "https://api.worldquantbrain.com"
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "results": [
                {"id": "fundamental2", "valueScore": 1, "alphaCount": 149066, "fieldCount": 766, "coverage": 0.44},
                {"id": "model16", "valueScore": 1, "alphaCount": 14644, "fieldCount": 24, "coverage": 0.56},
                {"id": "model51", "valueScore": 1, "alphaCount": 46443, "fieldCount": 16, "coverage": 0.96},
                {"id": "option9", "valueScore": 1, "alphaCount": 96540, "fieldCount": 74, "coverage": 0.98},
            ]
        }
        client.session.request.return_value = resp
        ds_items = fetch_available_datasets(client, max_datasets=4)
        ds_ids = [d["id"] for d in ds_items]
        self.assertEqual(ds_ids[:2], ["model51", "option9"])

    def test_alpha_vault_never_deletes_submitted_alpha_on_skeleton_collision(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            v = AlphaVault(db_path=str(Path(td) / "vault.db"))
            v.save_alpha(
                name="BRAIN_QP0ae1QG",
                expression="group_neutralize(rank(liabilities / (abs(assets) + 0.001)), subindustry)",
                sharpe=1.50,
                fitness=1.25,
                turnover=0.0155,
                margin=0.0111,
                status_override="SUBMITTED",
                brain_alpha_id="QP0ae1QG",
            )
            v.save_alpha(
                name="DRAFT_CLONE",
                expression="group_neutralize(rank(liabilities / (abs(assets) + 0.002)), subindustry)",
                sharpe=2.10,
                fitness=1.50,
                turnover=0.02,
                margin=0.012,
                status_override="LOCAL_VERIFIED",
            )
            all_ids = [a.get("brain_alpha_id") for a in v.list_alphas()]
            self.assertIn("QP0ae1QG", all_ids)
            gc.collect()


if __name__ == "__main__":
    unittest.main()
