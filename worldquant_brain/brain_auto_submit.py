"""
WorldQuant BRAIN Auto-Submit Engine with Safety Gates, Quota, Cooldown, and Idempotency.
Complies with docs/ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md (P9).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BrainAutoSubmit")

SUBMIT_ACK_ENV = "BRAIN_AUTO_SUBMIT_ACK"
REQUIRED_SUBMIT_ACK = "I_UNDERSTAND_SUBMISSION_IS_IRREVERSIBLE"
KILL_SWITCH_ENV = "BRAIN_SUBMISSION_KILL_SWITCH"


class AlphaStatus(str, Enum):
    GENERATED = "GENERATED"
    LOCAL_SCREENED = "LOCAL_SCREENED"
    DEMO_SAMPLE = "DEMO_SAMPLE"
    LEGACY_UNVERIFIED = "LEGACY_UNVERIFIED"
    SIMULATION_QUEUED = "SIMULATION_QUEUED"
    BRAIN_RUNNING = "BRAIN_RUNNING"
    BRAIN_COMPLETE = "BRAIN_COMPLETE"
    BRAIN_REJECTED = "BRAIN_REJECTED"
    SUBMISSION_CHECK_PENDING = "SUBMISSION_CHECK_PENDING"
    READY_FOR_SUBMIT = "READY_FOR_SUBMIT"
    SUBMISSION_CLAIMED = "SUBMISSION_CLAIMED"
    SUBMITTED = "SUBMITTED"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"


def migrate_alpha_status(alpha_record: Dict[str, Any]) -> str:
    """
    Migrate legacy alpha statuses per P9:
    - Missing alpha_id or Location with VERIFIED status -> LEGACY_UNVERIFIED
    - Local BTC proxy or local backtest -> LOCAL_SCREENED
    - Seed/sample alphas -> DEMO_SAMPLE
    """
    current_status = alpha_record.get("status", "")
    is_seed = alpha_record.get("is_seed", False) or alpha_record.get("source") == "seed"
    if is_seed:
        return AlphaStatus.DEMO_SAMPLE.value

    has_location = bool(alpha_record.get("simulation_location") or alpha_record.get("location"))
    has_alpha_id = bool(alpha_record.get("alpha_id") or alpha_record.get("simulation_id"))
    is_local_proxy = alpha_record.get("is_local_proxy", False) or alpha_record.get("proxy_type") == "BTC"

    if is_local_proxy:
        return AlphaStatus.LOCAL_SCREENED.value

    if current_status in ("VERIFIED", "PASS", "PASSED"):
        if not (has_location or has_alpha_id):
            return AlphaStatus.LEGACY_UNVERIFIED.value
        return AlphaStatus.BRAIN_COMPLETE.value

    return current_status or AlphaStatus.GENERATED.value


class BrainAutoSubmitGate:
    def __init__(
        self,
        min_sharpe: float = 1.25,
        min_fitness: float = 1.0,
        min_margin: float = 0.0010,  # > 10 bps
        min_turnover: float = 0.01,  # 1%
        max_turnover: float = 0.70,  # 70%
        max_self_correlation: float = 0.70,
        max_sub_universe_sharpe_decay: float = 0.50,
        daily_quota: int = 10,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.min_sharpe = min_sharpe
        self.min_fitness = min_fitness
        self.min_margin = min_margin
        self.min_turnover = min_turnover
        self.max_turnover = max_turnover
        self.max_self_correlation = max_self_correlation
        self.max_sub_universe_sharpe_decay = max_sub_universe_sharpe_decay
        self.daily_quota = daily_quota
        self.cooldown_seconds = cooldown_seconds

        self._submitted_today = 0
        self._last_submitted_at: float = 0.0
        self._submitted_keys: set[str] = set()
        self._lock = threading.Lock()
        self.audit_log: List[Dict[str, Any]] = []

    def is_kill_switch_active(self) -> bool:
        val = os.getenv(KILL_SWITCH_ENV, "0").strip().lower()
        return val in ("1", "true", "yes", "on")

    def is_ack_valid(self) -> bool:
        return os.getenv(SUBMIT_ACK_ENV, "").strip() == REQUIRED_SUBMIT_ACK

    def evaluate_alpha(self, metrics: Dict[str, Any], correlation: float = 0.0) -> tuple[bool, str]:
        """Strict validation against WorldQuant BRAIN standards."""
        sharpe = metrics.get("sharpe")
        fitness = metrics.get("fitness")
        turnover = metrics.get("turnover")
        margin = metrics.get("margin")

        if sharpe is None or fitness is None or turnover is None or margin is None:
            return False, "Missing core metrics"

        if sharpe < self.min_sharpe:
            return False, f"Sharpe {sharpe:.2f} < threshold {self.min_sharpe:.2f}"
        if fitness < self.min_fitness:
            return False, f"Fitness {fitness:.2f} < threshold {self.min_fitness:.2f}"
        if turnover < self.min_turnover or turnover > self.max_turnover:
            return False, f"Turnover {turnover:.2%} not within [{self.min_turnover:.0%}, {self.max_turnover:.0%}]"
        if margin < self.min_margin:
            return False, f"Margin {margin:.4f} < threshold {self.min_margin:.4f}"
        if correlation >= self.max_self_correlation:
            return False, f"Correlation {correlation:.2f} >= max {self.max_self_correlation:.2f}"

        return True, "PASSED_HARD_GATE"

    def can_submit(self, submission_key: str, metrics: Dict[str, Any], correlation: float = 0.0) -> tuple[bool, str]:
        if self.is_kill_switch_active():
            return False, "KILL_SWITCH_ACTIVE"

        if not self.is_ack_valid():
            return False, f"MISSING_OR_INVALID_ACK (Requires {SUBMIT_ACK_ENV}={REQUIRED_SUBMIT_ACK})"

        passed, reason = self.evaluate_alpha(metrics, correlation)
        if not passed:
            return False, reason

        with self._lock:
            if submission_key in self._submitted_keys:
                return False, "ALREADY_SUBMITTED_IDEMPOTENT"

            if self._submitted_today >= self.daily_quota:
                return False, f"DAILY_QUOTA_EXCEEDED (Max {self.daily_quota}/day)"

            now = time.time()
            elapsed = now - self._last_submitted_at
            if elapsed < self.cooldown_seconds:
                return False, f"COOLDOWN_ACTIVE (Wait {self.cooldown_seconds - elapsed:.1f}s)"

        return True, "READY"

    def record_submission(self, alpha_id: str, submission_key: str, status: str = "SUBMITTED", error: Optional[str] = None):
        with self._lock:
            self._submitted_keys.add(submission_key)
            self._submitted_today += 1
            self._last_submitted_at = time.time()
            entry = {
                "alpha_id": alpha_id,
                "submission_key": submission_key,
                "status": status,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.audit_log.append(entry)
            return entry
