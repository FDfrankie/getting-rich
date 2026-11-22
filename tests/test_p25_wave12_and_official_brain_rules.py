"""Contract tests for P25: Wave 12 CI/Security Fixes & Official WorldQuant BRAIN KB Rules."""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from worldquant_brain.brain_client import BrainClient
from worldquant_brain.two_stage_funnel import classify_verdict, compute_subuniverse_sharpe_cutoff


class TestP25Wave12AndOfficialBrainRules(unittest.TestCase):
    def test_wave12_workflows_and_requirements_hardened(self) -> None:
        bm = Path(".github/workflows/brain_miner.yml").read_text(encoding="utf-8")
        self.assertNotIn("--queue-depth", bm)
        self.assertIn("--batch-size 5", bm)
        self.assertIn("actions/upload-artifact@v4", bm)
        self.assertNotIn("git add worldquant_brain/alpha_vault.db", bm)

        ct = Path(".github/workflows/crypto_and_taiwan_cron.yml").read_text(encoding="utf-8")
        self.assertIn('cron: "40 8 * * 1-5"', ct)
        self.assertIn("adapter.fetch_positions()", ct)
        self.assertIn("adapter.fetch_balance()", ct)

        ci = Path(".github/workflows/future-offline-ci.yml").read_text(encoding="utf-8")
        self.assertIn("branches: [main]", ci)
        self.assertIn("us_stock", ci)
        self.assertIn("model_governance", ci)

        req = Path("requirements.txt").read_text(encoding="utf-8").lower()
        self.assertIn("apscheduler", req)
        self.assertIn("pytest", req)

        rl = Path("crypto_trading/rl_agent.py").read_text(encoding="utf-8")
        self.assertIn("weights_only=True", rl)

        gd = Path("data_persistence/gdrive_sync.py").read_text(encoding="utf-8")
        self.assertIn("timeout=30.0", gd)
        self.assertIn("finally:", gd)

    def test_official_kb_delay0_fitness_1_3_and_subuniverse_formula(self) -> None:
        # Official KB 14220836385687: Delay 0 requires Fitness >= 1.30
        d0_low_fit = {
            "sharpe": 2.20,
            "fitness": 1.15,
            "turnover": 0.25,
            "margin": 0.0020,
            "checks_passed": True,
            "failed_checks": [],
            "settings": {"delay": 0},
        }
        v, reason = classify_verdict(d0_low_fit)
        self.assertNotEqual(v, "PASSED")
        self.assertIn("1.30", reason)

        d0_good = dict(d0_low_fit, fitness=1.35)
        v_ok, _ = classify_verdict(d0_good)
        self.assertEqual(v_ok, "PASSED")

        # Official KB 6568644868375: 0.75 * sqrt(2000/3000) * 2.0 = 1.2247
        cutoff = compute_subuniverse_sharpe_cutoff(2.0, 3000, 2000)
        self.assertAlmostEqual(cutoff, round(0.75 * math.sqrt(2000 / 3000) * 2.0, 4), places=4)

    def test_official_kb_verify_blocks_delay0_low_fit_and_neut_none(self) -> None:
        client = BrainClient(email="u@test.com", password="pw")
        client.authenticated = True
        r_detail = MagicMock()
        r_detail.status_code = 200
        r_detail.content = b"{}"
        r_detail.json.return_value = {
            "id": "A_NONE",
            "regular": {"code": "rank(close)"},
            "settings": {"delay": 0, "neutralization": "NONE", "region": "USA"},
            "is": {"sharpe": 2.30, "fitness": 1.10, "turnover": 0.20, "margin": 0.0020, "checks": []},
        }
        r_chk = MagicMock()
        r_chk.status_code = 200
        r_chk.headers = {}
        r_chk.content = b"{}"
        r_chk.json.return_value = {"is": {"checks": []}}
        sess = MagicMock()
        sess.get.side_effect = [r_detail, r_chk]
        client.session = sess

        res = client.verify_alpha_for_submission("A_NONE")
        self.assertFalse(res["passed"])
        blob = " | ".join(res["reasons"])
        self.assertIn("1.30", blob)
        self.assertIn("NEUTRALIZATION_NONE_WITHOUT_GROUP_NEUTRALIZE", blob)


if __name__ == "__main__":
    unittest.main()
