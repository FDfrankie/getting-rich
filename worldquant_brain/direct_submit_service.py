"""Direct Submit Service for already-simulated BRAIN_COMPLETE Alphas without re-simulating."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from worldquant_brain.pre_submission_gate import GateVerdict, evaluate_pre_submission


class DirectSubmitService:
    """Submits an already-simulated official BRAIN alpha ID directly after passing pre-submission gate."""

    def __init__(self, submit_fn: Callable[[str], Mapping[str, Any]]) -> None:
        self.submit_fn = submit_fn

    def submit_existing_alpha(
        self,
        alpha_record: Mapping[str, Any],
        *,
        enforce_gate: bool = True,
    ) -> dict[str, Any]:
        brain_alpha_id = str(alpha_record.get("brain_alpha_id") or "").strip()
        if not brain_alpha_id:
            return {"success": False, "error": "MISSING_BRAIN_ALPHA_ID"}
        if alpha_record.get("submitted_at") or alpha_record.get("status") == "SUBMITTED":
            return {"success": False, "error": "ALREADY_SUBMITTED_IDEMPOTENT", "brain_alpha_id": brain_alpha_id}

        if enforce_gate:
            gate = evaluate_pre_submission(alpha_record)
            if gate.verdict != GateVerdict.PASS:
                return {
                    "success": False,
                    "error": f"PRE_SUBMISSION_GATE_{gate.verdict.value}",
                    "brain_alpha_id": brain_alpha_id,
                    "gate_result": gate.to_dict(),
                }

        resp = dict(self.submit_fn(brain_alpha_id))
        ok = bool(resp.get("success")) or resp.get("status") in {"SUBMITTED", "SUCCESS"}
        now = datetime.now(timezone.utc).isoformat()
        resp_hash = hashlib.sha256(json.dumps(resp, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {
            "success": ok,
            "brain_alpha_id": brain_alpha_id,
            "status": "SUBMITTED" if ok else "SUBMISSION_CANDIDATE",
            "submitted_at": now if ok else None,
            "submit_response_hash": resp_hash,
            "retained_in_vault": True,
            "raw_response": resp,
        }
