"""Unit tests for P8 Smart Alpha Search Bundle."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worldquant_brain.two_stage_funnel import (
    TwoStageAlphaFunnel,
    _field_family_prefix,
    _log_window_variants,
)


class TestP8SmartAlphaSearch(unittest.TestCase):
    def test_log_window_variants_are_strictly_integer_and_skip_current(self) -> None:
        expr = "ts_zscore(ts_backfill(model16_alpha, 20), 60)"
        variants = _log_window_variants(expr, prefer_larger=True)
        self.assertTrue(variants)
        # Note: _TS_WIN_RE matches ts_zscore(..., 60) -> should jump to 120, 252
        self.assertTrue(any("120" in v or "252" in v for v in variants))
        for v in variants:
            self.assertNotIn(".", v.split(",")[-1])

    def test_pairwise_synergy_enqueued_after_two_strong_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"))
            funnel.ingest_results(
                [
                    {
                        "success": True,
                        "status": "COMPLETED",
                        "stage": "STAGE1",
                        "dataset_id": "model16",
                        "expression": "rank(ts_backfill(mdl16_a, 20))",
                        "sharpe": 0.92,
                        "fitness": 0.70,
                        "turnover": 0.25,
                        "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                    },
                    {
                        "success": True,
                        "status": "COMPLETED",
                        "stage": "STAGE1",
                        "dataset_id": "model16",
                        "expression": "rank(ts_backfill(mdl16_b, 20))",
                        "sharpe": 0.95,
                        "fitness": 0.72,
                        "turnover": 0.22,
                        "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                    },
                ]
            )
            probes = [item.get("probe") for item in funnel._session_refinement_queue]
            self.assertIn("pairwise_synergy", probes)
            exprs = [item.get("expression", "") for item in funnel._session_refinement_queue]
            self.assertTrue(any("mdl16_a" in e and "mdl16_b" in e for e in exprs))

    def test_field_family_prefix_early_stop_skips_dead_prefix(self) -> None:
        self.assertEqual(_field_family_prefix("news18_sent_7d"), "news18_sent")
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"))
            # 3 weak results in news18_sent family -> should trigger early-stop
            funnel.ingest_results(
                [
                    {
                        "success": True,
                        "status": "COMPLETED",
                        "stage": "STAGE1",
                        "dataset_id": "news18",
                        "expression": f"rank(ts_backfill(news18_sent_{d}d, 20))",
                        "sharpe": 0.10,
                        "fitness": 0.05,
                        "turnover": 0.30,
                        "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                    }
                    for d in (1, 3, 7)
                ]
            )
            batch = funnel.next_batch(
                {"news18": ["news18_sent_30d", "news18_sent_90d", "news18_vol_20d"]},
                batch_size=3,
            )
            # Only news18_vol_20d should be scheduled; dead news18_sent_* family must be skipped
            self.assertTrue(all("news18_sent_" not in item["expression"] for item in batch))
            self.assertTrue(any("news18_vol_20d" in item["expression"] for item in batch))

    def test_diagnostic_margin_booster_queued_on_low_margin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            funnel = TwoStageAlphaFunnel(str(Path(td) / "vault.db"))
            funnel._enqueue_immediate_refinement(
                {
                    "expression": "rank(ts_backfill(mdl16_x, 20))",
                    "sharpe": 1.30,
                    "fitness": 1.05,
                    "turnover": 0.12,
                    "margin": 0.0004,
                    "dataset_id": "model16",
                    "settings": {"neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                },
                reason="MARGIN_TOO_LOW",
            )
            exprs = [item["expression"] for item in funnel._session_refinement_queue]
            self.assertTrue(any("signed_power(" in e for e in exprs))

    def test_headless_miner_wires_local_pv_screener(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        src = (repo_root / "scripts/headless_brain_miner.py").read_text(encoding="utf-8")
        self.assertIn("LocalCrossSectionalScreener", src)
        self.assertIn("BRAIN_LOCAL_PV_SCREENING", src)
        self.assertIn("[LOCAL_PV_SCREEN]", src)


if __name__ == "__main__":
    unittest.main()
