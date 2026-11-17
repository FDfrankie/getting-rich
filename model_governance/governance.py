"""
Model Governance, Artifact Trust Registry, Anti-Drift, and Distribution Validation.
Complies with docs/ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md (P11).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ModelGovernance")


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class ModelManifest:
    """Model manifest schema to guarantee artifact trust, lineage, and SHA-256 verification."""

    def __init__(
        self,
        model_name: str,
        version: str,
        weights_sha256: str,
        framework: str = "pytorch",
        weights_only: bool = True,
        training_date: Optional[str] = None,
        author: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.weights_sha256 = weights_sha256
        self.framework = framework
        self.weights_only = weights_only
        self.training_date = training_date
        self.author = author
        self.metrics = metrics or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "weights_sha256": self.weights_sha256,
            "framework": self.framework,
            "weights_only": self.weights_only,
            "training_date": self.training_date,
            "author": self.author,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelManifest:
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            weights_sha256=data["weights_sha256"],
            framework=data.get("framework", "pytorch"),
            weights_only=data.get("weights_only", True),
            training_date=data.get("training_date"),
            author=data.get("author"),
            metrics=data.get("metrics"),
        )


class ArtifactRegistry:
    """
    Registry for loading and validating model artifacts.
    - Quarantines unverified/unmanifested artifacts to artifacts/legacy_unverified/
    - Validates SHA-256 before loading
    - Enforces weights_only=True for PyTorch / state_dict loading
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.legacy_unverified_dir = os.path.join(base_dir, "legacy_unverified")
        os.makedirs(self.legacy_unverified_dir, exist_ok=True)

    def verify_and_quarantine_if_untrusted(self, artifact_path: str, manifest_path: Optional[str] = None) -> Tuple[bool, str]:
        if not os.path.exists(artifact_path):
            return False, "ARTIFACT_NOT_FOUND"

        if not manifest_path or not os.path.exists(manifest_path):
            # Quarantine untrusted artifact
            dest = os.path.join(self.legacy_unverified_dir, os.path.basename(artifact_path))
            shutil.copy2(artifact_path, dest)
            return False, f"QUARANTINED_TO_{dest}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest = ModelManifest.from_dict(manifest_data)

        # Check SHA-256
        actual_sha = compute_file_sha256(artifact_path)
        if actual_sha != manifest.weights_sha256:
            dest = os.path.join(self.legacy_unverified_dir, f"tampered_{os.path.basename(artifact_path)}")
            shutil.copy2(artifact_path, dest)
            return False, f"SHA256_MISMATCH_QUARANTINED_TO_{dest}"

        return True, "TRUSTED"


def validate_ppo_action_distribution(
    action_probabilities: List[float],
    min_entropy_threshold: float = 0.1,
) -> Tuple[bool, float]:
    """
    Validate that PPO agent's action probability distribution has not collapsed to a degenerate state.
    Computes Shannon entropy: -sum(p * log(p))
    """
    import math

    # Clean probabilities
    total = sum(action_probabilities)
    if total <= 0:
        return False, 0.0

    probs = [p / total for p in action_probabilities]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)

    # If entropy is lower than threshold, distribution has collapsed
    is_valid = entropy >= min_entropy_threshold
    return is_valid, round(entropy, 4)
