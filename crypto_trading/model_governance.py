"""Fail-closed model-manifest governance shared by every strategy loader."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_ALIASES = {
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
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
