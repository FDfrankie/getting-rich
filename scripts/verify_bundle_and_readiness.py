#!/usr/bin/env python3
"""Verify BUNDLE_MANIFEST.json SHA-256 hashes and output decoupled 3-domain readiness:
UNIVERSE_READY / MODEL_RESEARCH_READY / UI_READY.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent


def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate_readiness(root: Path = BASE_DIR) -> dict[str, Any]:
    manifest_path = root / "BUNDLE_MANIFEST.json"
    hash_mismatches: list[str] = []
    verified_files = 0

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rel, info in (m.get("files") or {}).items():
            fp = root / rel
            if not fp.exists():
                hash_mismatches.append(f"MISSING:{rel}")
                continue
            if sha256_path(fp) != info.get("sha256"):
                hash_mismatches.append(f"HASH_MISMATCH:{rel}")
            else:
                verified_files += 1

    # Check Universe readiness
    univ_cov_path = root / "reports" / "taiwan_universe_coverage.json"
    univ_status = "READY_TO_MERGE"
    if univ_cov_path.exists():
        uc = json.loads(univ_cov_path.read_text(encoding="utf-8"))
        if uc.get("full_universe_gate") not in {"PASS"}:
            univ_status = "PARTIAL_OR_NETWORK_BLOCKED"

    # Check Model readiness
    lgbm_rep_path = root / "reports" / "lgbm_validation_gate_report.json"
    model_status = "RESEARCH_ONLY"
    if lgbm_rep_path.exists():
        lr = json.loads(lgbm_rep_path.read_text(encoding="utf-8"))
        if lr.get("gate_passed") is True:
            model_status = "PAPER_CANDIDATE_READY"

    ui_status = "READY_TO_MERGE" if (root / "web_ui" / "static" / "index.html").exists() else "BLOCKED"
    bundle_ok = len(hash_mismatches) == 0

    overall = (
        "READY_TO_MERGE_MAIN"
        if (bundle_ok and univ_status == "READY_TO_MERGE" and ui_status == "READY_TO_MERGE")
        else ("BLOCKED" if not bundle_ok else "PARTIAL_READY")
    )
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_hashes_verified": bundle_ok,
        "verified_files_count": verified_files,
        "hash_mismatches": hash_mismatches,
        "universe_status": univ_status,
        "model_status": model_status,
        "ui_status": ui_status,
        "overall_status": overall,
    }
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "merge_readiness_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = evaluate_readiness()
    print(json.dumps(res, indent=2, ensure_ascii=False))
