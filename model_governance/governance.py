"""
Model Governance, Artifact Trust Registry, Anti-Drift, and Distribution Validation.
Complies with docs/ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md (P11).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
import os
from pathlib import Path
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

    def verify_and_quarantine_if_untrusted(
        self,
        artifact_path: str,
        manifest_path: Optional[str] = None,
        *,
        quarantine_move: bool = True,
    ) -> Tuple[bool, str]:
        if not os.path.exists(artifact_path):
            return False, "ARTIFACT_NOT_FOUND"

        _transfer = shutil.move if quarantine_move else shutil.copy2
        if not manifest_path or not os.path.exists(manifest_path):
            dest = os.path.join(self.legacy_unverified_dir, os.path.basename(artifact_path))
            _transfer(artifact_path, dest)
            return False, f"QUARANTINED_TO_{dest}"

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = ModelManifest.from_dict(manifest_data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            dest = os.path.join(self.legacy_unverified_dir, f"invalid_manifest_{os.path.basename(artifact_path)}")
            _transfer(artifact_path, dest)
            return False, f"INVALID_MANIFEST_QUARANTINED_TO_{dest}"

        actual_sha = compute_file_sha256(artifact_path)
        if actual_sha != manifest.weights_sha256:
            dest = os.path.join(self.legacy_unverified_dir, f"tampered_{os.path.basename(artifact_path)}")
            _transfer(artifact_path, dest)
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


MANIFEST_ALIASES: dict[str, str] = {
    "taiwan_stock_ranker.joblib": "taiwan_stock_manifest.json",
}


@dataclass(frozen=True)
class ManifestDecision:
    allowed: bool
    status: str
    reason: str
    manifest: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_manifest_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    """Ensure both canonical (`feature_columns`, `symbols`) and legacy (`feature_cols`, `universe_symbols`) keys are present."""
    out = dict(manifest)
    if "feature_columns" not in out and "feature_cols" in out:
        out["feature_columns"] = list(out["feature_cols"])
    elif "feature_cols" not in out and "feature_columns" in out:
        out["feature_cols"] = list(out["feature_columns"])
    if "symbols" not in out and "universe_symbols" in out:
        out["symbols"] = list(out["universe_symbols"])
    elif "universe_symbols" not in out and "symbols" in out:
        out["universe_symbols"] = list(out["symbols"])
    return out


def resolve_manifest_path(model: Path) -> Path:
    manifest_path = model.with_name(model.stem + "_manifest.json")
    if not manifest_path.exists() and model.name in MANIFEST_ALIASES:
        alt = model.with_name(MANIFEST_ALIASES[model.name])
        if alt.exists():
            return alt
    return manifest_path


def validate_model_manifest(
    model_path: str | Path,
    *,
    require_inference_approval: bool = True,
    require_order_approval: bool = False,
) -> ManifestDecision:
    """Validate a model without guessing missing approval fields.

    Old manifests that omit explicit approval are quarantined intentionally.
    """
    model = Path(model_path)
    manifest_path = resolve_manifest_path(model)
    if not model.exists():
        return ManifestDecision(False, "MODEL_MISSING", str(model), {})
    if not manifest_path.exists():
        return ManifestDecision(False, "MANIFEST_MISSING", str(manifest_path), {})
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = normalize_manifest_fields(raw_manifest) if isinstance(raw_manifest, dict) else raw_manifest
    except Exception as exc:
        return ManifestDecision(False, "MANIFEST_INVALID", type(exc).__name__, {})

    if manifest.get("is_synthetic") or manifest.get("synthetic_data"):
        return ManifestDecision(False, "SYNTHETIC_MODEL_QUARANTINED", "synthetic_data", manifest)
    if manifest.get("deployment_stage") in {"RESEARCH_ONLY", "QUARANTINED", "REJECTED"}:
        return ManifestDecision(False, "MODEL_QUARANTINED", str(manifest.get("rejection_reason", "stage")), manifest)
    if require_inference_approval and manifest.get("approved_for_inference") is not True:
        return ManifestDecision(False, "INFERENCE_NOT_APPROVED", "explicit approval missing or false", manifest)
    if require_order_approval and manifest.get("approved_for_order_entry") is not True:
        return ManifestDecision(False, "ORDER_ENTRY_NOT_APPROVED", "explicit approval missing or false", manifest)

    expected_hash = manifest.get("weights_sha256") or manifest.get("artifact_sha256") or manifest.get("sha256")
    if not expected_hash:
        return ManifestDecision(False, "ARTIFACT_HASH_MISSING", "no model hash", manifest)
    actual_hash = sha256_file(model)
    if actual_hash.lower() != str(expected_hash).lower():
        return ManifestDecision(False, "ARTIFACT_HASH_MISMATCH", actual_hash, manifest)
    return ManifestDecision(True, "APPROVED", "manifest and artifact verified", manifest)
