"""Contract tests for P23: WorldQuant BRAIN 4-pillar de-correlation rescue for good Alphas."""
from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.alpha_yield_booster import generate_decorrelation_variants
from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel


class TestP23DecorrelationRescue(unittest.TestCase):
    def test_generate_decorrelation_variants_covers_official_4_pillars(self) -> None:
        expr = "rank(ts_mean(close / (assets + 0.001), 20))"
        sett = {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "SUBINDUSTRY", "decay": 4}
        variants = generate_decorrelation_variants(expr, sett, max_variants=5)
        self.assertGreaterEqual(len(variants), 4)
        reasons = {v["reason"] for v in variants}
        # Pillar 1: field sub (close -> vwap/open)
        self.assertTrue(any("FIELD_SUBSTITUTION" in r for r in reasons))
        # Pillar 2: operator sub (ts_mean -> ts_median)
        self.assertTrue(any("OP_SUBSTITUTION" in r for r in reasons))
        # Pillar 3: neutralization rotation (SUBINDUSTRY -> INDUSTRY)
        self.assertTrue(any("NEUTRALIZATION_" in r for r in reasons))
        # Pillar 4: orthogonal residual / combo
        self.assertTrue(any("ORTHOG_" in r for r in reasons))

    def test_mark_submit_failed_corr_auto_enqueues_decorrelation_rescue(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            f = TwoStageAlphaFunnel(Path(td) / "vault.db")
            f.mark_submit_failed(
                alpha_id="A_GOOD_HIGH_CORR",
                reason="SUBMISSION_CHECK_FAILED:SELF_CORRELATION=0.81",
                expression="rank(ts_backfill(fnd6_ebitda, 63))",
                dataset_id="fundamental6",
                probe="rank_bf",
            )
            self.assertGreaterEqual(f.pending_refinement_count(), 2)
            q = f.next_stage2_batch(batch_size=5)
            self.assertTrue(any(it["settings"].get("neutralization") != "SUBINDUSTRY" for it in q))
            self.assertTrue(any("ts_regression" in it["expression"] or "zscore(" in it["expression"] for it in q))
            gc.collect()

    def test_headless_miner_wires_decorr_rescue_on_local_self_corr_skip(self) -> None:
        src = Path("scripts/headless_brain_miner.py").read_text(encoding="utf-8")
        self.assertIn("enqueue_decorrelation_rescue", src)
        self.assertIn("DECORR_RESCUE_QUEUED", src)


if __name__ == "__main__":
    unittest.main()
