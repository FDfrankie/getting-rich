"""Daily Top-K Utility submission service with audit logging and non-destructive state transitions."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from worldquant_brain.alpha_lifecycle import AlphaState
from worldquant_brain.alpha_repository import AlphaRepository
from worldquant_brain.daily_candidate_selector import SelectionBatch, select_daily_candidates


class SubmissionService:
    def __init__(self, db_path: str | Path, top_k: int = 2) -> None:
        self.repo = AlphaRepository(db_path)
        self.top_k = int(top_k)

    def execute_daily_top_k(
        self,
        candidates: Iterable[Mapping[str, Any]],
        submitter: Callable[[str], Mapping[str, Any]],
        *,
        formal_submission_count: int = 0,
    ) -> dict[str, Any]:
        batch: SelectionBatch = select_daily_candidates(
            candidates, top_k=self.top_k, formal_submission_count=formal_submission_count
        )
        submitted_ids: list[str] = []
        failed_ids: list[str] = []

        for sel in batch.selected:
            alpha_id = sel.alpha_id
            resp = dict(submitter(alpha_id))
            ok = bool(resp.get("success")) or resp.get("status") in {"SUBMITTED", "SUCCESS"}
            now = datetime.now(timezone.utc).isoformat()
            resp_hash = hashlib.sha256(json.dumps(resp, sort_keys=True, default=str).encode("utf-8")).hexdigest()
            with self.repo.store.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO brain_submissions (
                        submission_id, alpha_id, brain_alpha_id, brain_simulation_id,
                        status, submitted_at, submit_acknowledgement, submit_response_hash,
                        official_status, attribution_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        alpha_id,
                        str(resp.get("brain_alpha_id") or alpha_id),
                        resp.get("simulation_id"),
                        "SUBMITTED" if ok else "FAILED",
                        now,
                        "ACK",
                        resp_hash,
                        str(resp.get("status") or ("SUBMITTED" if ok else "FAILED")),
                        "PENDING_NEXT_REFRESH",
                    ),
                )
            if ok:
                submitted_ids.append(alpha_id)
            else:
                failed_ids.append(alpha_id)

        return {
            "selected_count": len(batch.selected),
            "submitted_ids": submitted_ids,
            "failed_ids": failed_ids,
            "retained_pending_ids": list(batch.retained_pending_ids),
            "rejected_ids": list(batch.rejected_ids),
        }
