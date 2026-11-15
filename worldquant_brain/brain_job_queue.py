"""
Brain Job Queue for asynchronous simulation processing and deduplication.
Complies with docs/BRAIN_WEB_API_ASYNC_FIX_SPEC.md.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BrainJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    OFFLINE = "OFFLINE"


class BrainSimulationJob:
    def __init__(
        self,
        job_id: str,
        simulation_key: str,
        expression: str,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.job_id: str = job_id
        self.simulation_key: str = simulation_key
        self.expression: str = expression
        self.settings: dict[str, Any] = settings or {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 4,
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
        }
        self.status: BrainJobStatus = BrainJobStatus.QUEUED
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.simulation_location: str | None = None
        self.alpha_id: str | None = None
        self.last_polled_at: str | None = None
        self.next_poll_at: str | None = None
        self.metrics: dict[str, float | None] = {
            "sharpe": None,
            "fitness": None,
            "turnover": None,
            "margin": None,
        }
        self.safe_error: str | None = None
        self.is_duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value
            if isinstance(self.status, BrainJobStatus)
            else str(self.status),
            "simulation_location": self.simulation_location,
            "alpha_id": self.alpha_id,
            "last_polled_at": self.last_polled_at,
            "next_poll_at": self.next_poll_at,
            "metrics": self.metrics,
            "safe_error": self.safe_error,
        }


def compute_simulation_key(
    expression: str, settings: Mapping[str, Any] | None = None
) -> str:
    data = {"expression": expression.strip(), "settings": dict(settings or {})}
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sanitize_error(exc: Exception) -> str:
    msg = str(exc)
    # Mask any common token/secret patterns
    for sensitive in ["password", "secret", "token", "authorization", "key"]:
        if sensitive in msg.lower():
            return "Simulation request failed due to authentication/security error."
    # Truncate to reasonable length
    if len(msg) > 200:
        msg = msg[:197] + "..."
    return msg or "Internal simulation error"


class BrainJobQueue:
    def __init__(self, brain_client: Any, default_poll_interval: float = 2.0) -> None:
        self.brain_client = brain_client
        self.default_poll_interval = float(default_poll_interval)
        self._jobs: dict[str, BrainSimulationJob] = {}
        self._key_to_job_id: dict[str, str] = {}
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._pending_queue: list[str] = []
        self._running: bool = True

        self._start_worker()

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def create_job(
        self,
        expression: str,
        settings: Mapping[str, Any] | None = None,
    ) -> tuple[BrainSimulationJob, bool]:
        key = compute_simulation_key(expression, settings)
        with self._lock:
            if key in self._key_to_job_id:
                existing_id = self._key_to_job_id[key]
                job = self._jobs[existing_id]
                return job, False

            job_id = f"job_{key[:16]}_{int(time.time())}"
            job = BrainSimulationJob(
                job_id=job_id,
                simulation_key=key,
                expression=expression,
                settings=dict(settings or {}),
            )
            self._jobs[job_id] = job
            self._key_to_job_id[key] = job_id
            self._pending_queue.append(job_id)
            return job, True

    def get_job(self, job_id: str) -> BrainSimulationJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _worker_loop(self) -> None:
        while self._running:
            job_to_run = None
            with self._lock:
                if self._pending_queue:
                    job_id = self._pending_queue.pop(0)
                    job_to_run = self._jobs.get(job_id)

            if job_to_run:
                self._execute_job(job_to_run)
            else:
                time.sleep(0.2)

    def _execute_job(self, job: BrainSimulationJob) -> None:
        with self._lock:
            job.status = BrainJobStatus.RUNNING
            job.last_polled_at = datetime.now(timezone.utc).isoformat()

        # Handle offline mode immediately
        if getattr(self.brain_client, "offline_mode", True):
            try:
                res = self.brain_client.simulate_alpha(
                    job.expression, settings=job.settings
                )
                with self._lock:
                    job.status = BrainJobStatus.OFFLINE
                    job.simulation_location = res.get("location")
                    job.alpha_id = res.get("simulation_id")
                    job.metrics = {
                        "sharpe": res.get("sharpe"),
                        "fitness": res.get("fitness"),
                        "turnover": res.get("turnover"),
                        "margin": res.get("margin"),
                    }
                    job.safe_error = None
            except Exception as e:
                with self._lock:
                    job.status = BrainJobStatus.FAILED
                    job.safe_error = sanitize_error(e)
            return

        # Online mode: submit simulation
        try:
            if not self.brain_client.authenticated:
                self.brain_client.authenticate()

            payload = {
                "type": "REGULAR",
                "settings": job.settings,
                "regular": job.expression,
            }
            # Post simulation to get Location
            resp = self.brain_client.session.post(
                f"{self.brain_client.BASE_URL}/simulations",
                json=payload,
                timeout=self.brain_client.timeout,
            )
            if resp.status_code not in (200, 201, 202):
                resp.raise_for_status()

            location = resp.headers.get("Location") or resp.headers.get("location")
            if not location:
                raise RuntimeError("BRAIN /simulations did not return Location header")
            if location.startswith("/"):
                location = f"{self.brain_client.BASE_URL}{location}"

            with self._lock:
                job.simulation_location = location
                job.alpha_id = location.rstrip("/").split("/")[-1]

            # Poll simulation in background without blocking HTTP requests
            deadline = time.monotonic() + 1800.0
            poll_interval = self.default_poll_interval

            while time.monotonic() <= deadline:
                with self._lock:
                    job.last_polled_at = datetime.now(timezone.utc).isoformat()
                    job.next_poll_at = datetime.fromtimestamp(
                        time.time() + poll_interval, tz=timezone.utc
                    ).isoformat()

                poll_resp = self.brain_client.session.get(
                    location, timeout=self.brain_client.timeout
                )
                if poll_resp.status_code == 429:
                    retry_after = float(
                        poll_resp.headers.get("Retry-After", poll_interval)
                    )
                    time.sleep(max(poll_interval, retry_after))
                    continue

                poll_resp.raise_for_status()
                poll_data = poll_resp.json() if poll_resp.content else {}
                state = str(
                    poll_data.get("status") or poll_data.get("state") or ""
                ).upper()

                if state == "COMPLETE":
                    with self._lock:
                        job.status = BrainJobStatus.COMPLETE
                        job.metrics = {
                            "sharpe": self.brain_client._number(poll_data, "sharpe"),
                            "fitness": self.brain_client._number(poll_data, "fitness"),
                            "turnover": self.brain_client._number(
                                poll_data, "turnover"
                            ),
                            "margin": self.brain_client._number(poll_data, "margin"),
                        }
                    return

                if state in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                    raise RuntimeError(f"Simulation {state}: {poll_data}")

                retry_after = poll_resp.headers.get("Retry-After")
                sleep_dur = float(retry_after) if retry_after else poll_interval
                time.sleep(sleep_dur)

            raise TimeoutError("BRAIN simulation polling exceeded 1800 seconds")

        except Exception as e:
            with self._lock:
                job.status = BrainJobStatus.FAILED
                job.safe_error = sanitize_error(e)
