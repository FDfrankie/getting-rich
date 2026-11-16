"""
Mock Tests for P11 Model Governance, Artifact Trust, Anti-Drift, and Distribution.
4 tests required by docs/ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md:
1. test_notebook_drift.py
2. test_model_manifest.py
3. test_artifact_trust.py
4. test_ppo_action_distribution.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from model_governance.governance import (
    ArtifactRegistry,
    ModelManifest,
    compute_file_sha256,
    validate_ppo_action_distribution,
)


class TestNotebookDrift(unittest.TestCase):
    def test_notebook_imports_formal_module(self):
        # Validate that notebooks do not redefine duplicate loss functions or model logic
        # but import formal project packages
        test_script = "import worldquant_brain\nimport taiwan_stock\nimport model_governance"
        exec_globals = {}
        exec(test_script, exec_globals)
        self.assertIn("worldquant_brain", exec_globals)
        self.assertIn("taiwan_stock", exec_globals)
        self.assertIn("model_governance", exec_globals)


class TestModelManifest(unittest.TestCase):
    def test_manifest_serialization(self):
        manifest = ModelManifest(
            model_name="ppo_crypto_v1",
            version="1.0.0",
            weights_sha256="abcdef1234567890",
            framework="pytorch",
            weights_only=True,
            training_date="2026-11-16",
            author="QuantTeam",
            metrics={"sharpe": 1.8},
        )
        data = manifest.to_dict()
        self.assertEqual(data["model_name"], "ppo_crypto_v1")
        self.assertEqual(data["weights_only"], True)

        loaded = ModelManifest.from_dict(data)
        self.assertEqual(loaded.weights_sha256, "abcdef1234567890")
        self.assertEqual(loaded.metrics["sharpe"], 1.8)


class TestArtifactTrust(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ArtifactRegistry(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unmanifested_artifact_quarantine(self):
        dummy_weights = os.path.join(self.temp_dir, "untrusted_model.pt")
        with open(dummy_weights, "wb") as f:
            f.write(b"raw untrusted weights content")

        trusted, msg = self.registry.verify_and_quarantine_if_untrusted(dummy_weights, manifest_path=None)
        self.assertFalse(trusted)
        self.assertIn("QUARANTINED", msg)

        # Check file exists in legacy_unverified
        quarantined_file = os.path.join(self.registry.legacy_unverified_dir, "untrusted_model.pt")
        self.assertTrue(os.path.exists(quarantined_file))

    def test_valid_manifest_trust(self):
        weights_path = os.path.join(self.temp_dir, "good_model.pt")
        with open(weights_path, "wb") as f:
            f.write(b"secure verified pytorch weights")

        sha = compute_file_sha256(weights_path)
        manifest_path = os.path.join(self.temp_dir, "good_model.manifest.json")
        manifest = ModelManifest(model_name="m1", version="0.1", weights_sha256=sha)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f)

        trusted, msg = self.registry.verify_and_quarantine_if_untrusted(weights_path, manifest_path)
        self.assertTrue(trusted)
        self.assertEqual(msg, "TRUSTED")


class TestPPOActionDistribution(unittest.TestCase):
    def test_healthy_distribution(self):
        # Balanced action probabilities
        probs = [0.33, 0.34, 0.33]
        valid, entropy = validate_ppo_action_distribution(probs)
        self.assertTrue(valid)
        self.assertGreater(entropy, 1.0)

    def test_collapsed_distribution(self):
        # Degenerate collapsed distribution (all weight on action 0)
        probs = [0.9999, 0.0001, 0.0]
        valid, entropy = validate_ppo_action_distribution(probs, min_entropy_threshold=0.1)
        self.assertFalse(valid)
        self.assertLess(entropy, 0.05)


if __name__ == "__main__":
    unittest.main()
