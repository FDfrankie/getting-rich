"""Simulation Budget, Single-Family Baseline Lock, and Successive Halving Precheck."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from worldquant_brain.alpha_novelty import (
    evaluate_correlation_buffer,
    find_duplicate,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    expression: str
    blueprint_family: str
    data_domain: str
    expected_quality: float = 0.70
    estimated_novelty: float = 0.85
    expected_information_gain: float = 0.75
    domain_diversity: float = 0.80
    local_corr: float = 0.25
    shares_raw_fields: bool = False
    is_vector_field_without_reducer: bool = False


@dataclass(frozen=True, slots=True)
class QueueDecision:
    admitted: bool
    reason: str
    priority_score: float = 0.0


class ThreeMinuteSimulationBudget:
    """Controls BRAIN simulation queue admission using Successive Halving,
    single-baseline-per-family lock, max_pending cap, and optional minimum interval."""

    def __init__(
        self,
        *,
        minimum_interval_seconds: float = 0.0,
        max_pending: int = 3,
        duplicate_threshold: float = 0.86,
    ) -> None:
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.max_pending = int(max_pending)
        self.duplicate_threshold = float(duplicate_threshold)
        self._last_queued_ts: float | None = None
        self._pending_ids: set[str] = set()
        self._active_families: set[str] = set()

    @staticmethod
    def compute_priority(c: Candidate) -> float:
        return round(
            0.35 * float(c.expected_quality)
            + 0.30 * float(c.estimated_novelty)
            + 0.25 * float(c.expected_information_gain)
            + 0.10 * float(c.domain_diversity),
            4,
        )

    def admit(
        self,
        candidate: Candidate,
        *,
        existing_expressions: Iterable[Mapping[str, Any]] = (),
        now_ts: float | None = None,
    ) -> QueueDecision:
        now = time.time() if now_ts is None else float(now_ts)
        if candidate.is_vector_field_without_reducer:
            return QueueDecision(False, "VECTOR_FIELD_MISSING_REDUCER")
        if candidate.candidate_id in self._pending_ids:
            return QueueDecision(False, "DUPLICATE_CANDIDATE_IN_QUEUE")
        if len(self._pending_ids) >= self.max_pending:
            return QueueDecision(False, "MAX_PENDING_REACHED")
        if candidate.blueprint_family in self._active_families:
            return QueueDecision(False, "FAMILY_BASELINE_ALREADY_ACTIVE")
        if self.minimum_interval_seconds > 0 and self._last_queued_ts is not None:
            if (now - self._last_queued_ts) < self.minimum_interval_seconds:
                return QueueDecision(False, "THROTTLED_MIN_INTERVAL")

        dup = find_duplicate(candidate.expression, existing_expressions, threshold=self.duplicate_threshold)
        if dup.blocked:
            return QueueDecision(False, dup.reason)

        corr_dec = evaluate_correlation_buffer(
            candidate.local_corr,
            shares_raw_fields=candidate.shares_raw_fields,
        )
        if not corr_dec.allowed_for_brain:
            return QueueDecision(False, corr_dec.decision)

        prio = self.compute_priority(candidate)
        if corr_dec.decision == "ACCEPT_LOW_PRIORITY":
            prio = round(prio * 0.80, 4)
        elif corr_dec.decision == "QUARANTINE":
            prio = round(prio * 0.55, 4)

        self._pending_ids.add(candidate.candidate_id)
        self._active_families.add(candidate.blueprint_family)
        self._last_queued_ts = now
        return QueueDecision(True, f"ADMITTED_{corr_dec.decision}", prio)

    def release(self, candidate_id: str, blueprint_family: str) -> None:
        self._pending_ids.discard(candidate_id)
        self._active_families.discard(blueprint_family)
