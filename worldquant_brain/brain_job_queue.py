"""
Brain Job Queue for asynchronous simulation processing and deduplication.
Complies with docs/BRAIN_WEB_API_ASYNC_FIX_SPEC.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class BrainJobStatus(str, Enum):
    QUEUED = "QUEUED"
    GENERATED_LOCAL = "GENERATED_LOCAL"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    AUTH_FAILED = "AUTH_FAILED"
    TIMEOUT = "TIMEOUT"
    OFFLINE = "OFFLINE"
    COMPLETE = "COMPLETED"  # Backwards compatibility alias


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
        self.vault_id: int | None = None
        self.auto_submit: bool = False
        self.submitted_to_brain: bool = False
        self.priority: int = 20
        self.source: str = "AUTO_MINING"
        self.requested_by: str | None = None
        self.manual_note: str | None = None
        self.simulation_only: bool = True
        self.queued_at: str = self.created_at
        self.claimed_by: str | None = None
        self.claimed_at: str | None = None
        self.submit_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        sharpe = self.metrics.get("sharpe")
        fitness = self.metrics.get("fitness")
        turnover = self.metrics.get("turnover")
        margin = self.metrics.get("margin")
        status_str = self.status.value if isinstance(self.status, BrainJobStatus) else str(self.status)
        is_completed = (
            status_str == "COMPLETED"
            and (sharpe is not None)
            and (fitness is not None)
            and bool(self.simulation_location or self.alpha_id)
        )
        is_submitted = status_str in ("SUBMITTED", "RUNNING", "COMPLETED")

        res: dict[str, Any] = {
            "job_id": self.job_id,
            "status": status_str,
            "success": is_completed,
            "submitted": is_submitted,
            "simulation_id": self.alpha_id,
            "alpha_id": self.alpha_id,
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "margin": margin,
            "simulation_location": self.simulation_location,
            "last_polled_at": self.last_polled_at,
            "next_poll_at": self.next_poll_at,
            "metrics": self.metrics,
            "safe_error": self.safe_error,
            "message": "WorldQuant 線上回測完成"
            if is_completed
            else ("WorldQuant 線上模擬未完成" if status_str == "OFFLINE" else None),
            "error": None if is_completed else (self.safe_error or f"Job status is {status_str}"),
            "error_code": None
            if is_completed
            else (
                "OFFLINE_MODE"
                if status_str == "OFFLINE"
                else ("AUTH_ERROR" if status_str == "AUTH_FAILED" else "SIMULATION_FAILED")
            ),
            "retryable": status_str in ("TIMEOUT", "FAILED"),
            "vault_id": self.vault_id,
            "brain_submitted": self.submitted_to_brain,
            "brain_submit": self.submit_result,
            "priority": self.priority, "source": self.source,
            "requested_by": self.requested_by, "manual_note": self.manual_note,
            "simulation_only": self.simulation_only, "queued_at": self.queued_at,
            "claimed_by": self.claimed_by, "claimed_at": self.claimed_at,
        }
        return res


def compute_simulation_key(expression: str, settings: Mapping[str, Any] | None = None) -> str:
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
    def __init__(
        self,
        brain_client: Any,
        default_poll_interval: float = 2.0,
        autostart: bool = True,
        on_complete_callback: Any = None,
        on_fail_callback: Any = None,
        db_path: str | None = None,
        num_workers: int | None = None,
    ) -> None:
        self.brain_client = brain_client
        self.default_poll_interval = float(default_poll_interval)
        self.on_complete_callback = on_complete_callback
        self.on_fail_callback = on_fail_callback
        self.db_path = db_path
        self._jobs: dict[str, BrainSimulationJob] = {}
        self._key_to_job_id: dict[str, str] = {}
        self._lock = threading.Lock()
        self.num_workers = max(1, int(num_workers if num_workers is not None else os.getenv("BRAIN_QUEUE_WORKERS", "3")))
        self._worker_thread: threading.Thread | None = None
        self._worker_threads: list[threading.Thread] = []
        self._pending_queue: list[str] = []
        self._in_flight_ids: set[str] = set()
        self._running: bool = False
        self._init_persistence()
        self._restore_jobs()
        if autostart:
            self.start()

    def _init_persistence(self) -> None:
        if not self.db_path or self.db_path == ":memory:":
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            with conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS brain_jobs (
                        job_id TEXT PRIMARY KEY,
                        simulation_key TEXT NOT NULL,
                        expression TEXT NOT NULL,
                        settings_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        simulation_location TEXT,
                        alpha_id TEXT,
                        metrics_json TEXT,
                        safe_error TEXT,
                        created_at TEXT NOT NULL,
                        priority INTEGER NOT NULL DEFAULT 20,
                        source TEXT NOT NULL DEFAULT 'AUTO_MINING',
                        requested_by TEXT, manual_note TEXT,
                        simulation_only INTEGER NOT NULL DEFAULT 1,
                        queued_at TEXT, claimed_by TEXT, claimed_at TEXT
                    )"""
                )
                existing = {row[1] for row in conn.execute("PRAGMA table_info(brain_jobs)")}
                for declaration in (
                    "priority INTEGER NOT NULL DEFAULT 20",
                    "source TEXT NOT NULL DEFAULT 'AUTO_MINING'", "requested_by TEXT",
                    "manual_note TEXT", "simulation_only INTEGER NOT NULL DEFAULT 1",
                    "queued_at TEXT", "claimed_by TEXT", "claimed_at TEXT",
                ):
                    if declaration.split()[0] not in existing:
                        conn.execute(f"ALTER TABLE brain_jobs ADD COLUMN {declaration}")
                conn.execute("UPDATE brain_jobs SET queued_at=COALESCE(queued_at, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS ix_brain_jobs_priority ON brain_jobs(status, priority DESC, queued_at ASC)")
        finally:
            conn.close()

    def _persist_job(self, job: BrainSimulationJob) -> None:
        if not self.db_path or self.db_path == ":memory:":
            return
        last_err = None
        for attempt in range(5):
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=20.0)
                conn.execute("PRAGMA busy_timeout=15000")
                with conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO brain_jobs
                        (job_id, simulation_key, expression, settings_json, status, simulation_location, alpha_id, metrics_json, safe_error, created_at, priority, source, requested_by, manual_note, simulation_only, queued_at, claimed_by, claimed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            job.job_id,
                            job.simulation_key,
                            job.expression,
                            json.dumps(job.settings, ensure_ascii=False),
                            job.status.value if isinstance(job.status, BrainJobStatus) else str(job.status),
                            job.simulation_location,
                            job.alpha_id,
                            json.dumps(job.metrics, ensure_ascii=False),
                            job.safe_error, job.created_at, job.priority, job.source,
                            job.requested_by, job.manual_note, int(job.simulation_only),
                            job.queued_at, job.claimed_by, job.claimed_at,
                        ),
                    )
                return
            except sqlite3.OperationalError as e:
                last_err = e
                time.sleep(0.15 * (attempt + 1))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        if last_err is not None:
            print(f"⚠️ [BrainJobQueue] _persist_job SQLite retry exhausted ({job.job_id}): {last_err}")

    def _restore_jobs(self) -> None:
        if not self.db_path or self.db_path == ":memory:" or not os.path.exists(self.db_path):
            return
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM brain_jobs ORDER BY created_at DESC LIMIT 200").fetchall()
            for r in rows:
                job = BrainSimulationJob(
                    job_id=r["job_id"],
                    simulation_key=r["simulation_key"],
                    expression=r["expression"],
                    settings=json.loads(r["settings_json"] or "{}"),
                )
                job.created_at = r["created_at"]
                job.simulation_location = r["simulation_location"]
                job.alpha_id = r["alpha_id"]
                job.safe_error = r["safe_error"]
                keys = set(r.keys())
                job.priority = int(r["priority"]) if "priority" in keys else 20
                job.source = r["source"] if "source" in keys else "AUTO_MINING"
                job.requested_by = r["requested_by"] if "requested_by" in keys else None
                job.manual_note = r["manual_note"] if "manual_note" in keys else None
                job.simulation_only = bool(r["simulation_only"]) if "simulation_only" in keys else True
                job.queued_at = (r["queued_at"] if "queued_at" in keys else None) or job.created_at
                job.claimed_by = r["claimed_by"] if "claimed_by" in keys else None
                job.claimed_at = r["claimed_at"] if "claimed_at" in keys else None
                if r["metrics_json"]:
                    job.metrics = json.loads(r["metrics_json"])
                try:
                    job.status = BrainJobStatus(r["status"])
                except ValueError:
                    job.status = BrainJobStatus.FAILED
                if job.status in (BrainJobStatus.QUEUED, BrainJobStatus.SUBMITTED, BrainJobStatus.RUNNING):
                    # Expire zombie jobs that were already claimed in a previous run or queued > 2 hours ago
                    is_stale = bool(job.claimed_at)
                    if not is_stale and job.created_at:
                        try:
                            dt = datetime.fromisoformat(str(job.created_at).replace("Z", "+00:00"))
                            if (datetime.now(timezone.utc) - dt).total_seconds() > 7200:
                                is_stale = True
                        except Exception:
                            pass
                    if is_stale:
                        job.status = BrainJobStatus.FAILED
                        job.safe_error = "EXPIRED_ZOMBIE_JOB_ON_RESTART"
                        try:
                            conn.execute("UPDATE brain_jobs SET status = ?, safe_error = ? WHERE job_id = ?", (job.status.value, job.safe_error, job.job_id))
                            conn.commit()
                        except Exception:
                            pass
                    else:
                        self._pending_queue.append(job.job_id)
                self._jobs[job.job_id] = job
                self._key_to_job_id[job.simulation_key] = job.job_id
        except Exception:
            pass
        finally:
            conn.close()

    def _start_worker(self) -> None:
        self._worker_threads = [t for t in self._worker_threads if t.is_alive()]
        while len(self._worker_threads) < self.num_workers:
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._worker_threads.append(t)
        self._worker_thread = self._worker_threads[0] if self._worker_threads else None

    def create_job(
        self,
        expression: str,
        settings: Mapping[str, Any] | None = None,
        vault_id: int | None = None,
        auto_submit: bool = False,
        priority: int = 20, source: str = "AUTO_MINING",
        requested_by: str | None = None, manual_note: str | None = None,
        simulation_only: bool = True,
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
            job.vault_id = vault_id
            job.auto_submit = bool(auto_submit) and not bool(simulation_only)
            job.priority = int(priority)
            job.source = str(source or "AUTO_MINING")
            job.requested_by = requested_by
            job.manual_note = manual_note
            job.simulation_only = bool(simulation_only)
            job.queued_at = datetime.now(timezone.utc).isoformat()
            self._jobs[job_id] = job
            self._key_to_job_id[key] = job_id
            self._pending_queue.append(job_id)
            self._persist_job(job)
            return job, True

    def start(self) -> None:
        with self._lock:
            if any(t.is_alive() for t in self._worker_threads):
                return
            self._running = True
            self._start_worker()

    def stop(self) -> None:
        self._running = False

    def join(self, timeout: float = 2.0) -> None:
        for t in list(self._worker_threads):
            if t and t.is_alive():
                t.join(timeout=timeout)

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._pending_queue) + len(self._in_flight_ids)

    def worker_alive(self) -> bool:
        return bool(self._running and any(t.is_alive() for t in self._worker_threads))

    def get_job(self, job_id: str) -> BrainSimulationJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _worker_loop(self) -> None:
        while self._running:
          try:
            job_to_run = None
            with self._lock:
                if self._pending_queue:
                    self._pending_queue.sort(key=lambda jid: (
                        -int(getattr(self._jobs.get(jid), "priority", 20)),
                        str(getattr(self._jobs.get(jid), "queued_at", "")),
                    ))
                    job_id = self._pending_queue.pop(0)
                    job_to_run = self._jobs.get(job_id)
                    if job_to_run is not None:
                        self._in_flight_ids.add(job_id)
                        job_to_run.claimed_by = threading.current_thread().name
                        job_to_run.claimed_at = datetime.now(timezone.utc).isoformat()
                        self._persist_job(job_to_run)

            if job_to_run:
                try:
                    self._execute_job(job_to_run)
                finally:
                    with self._lock:
                        self._in_flight_ids.discard(job_to_run.job_id)
            else:
                time.sleep(0.02)
          except Exception as exc:
            print(f"⚠️ [BrainJobQueue] _worker_loop caught transient error (continuing): {exc}")
            time.sleep(0.5)

    def _execute_job(self, job: BrainSimulationJob) -> None:
        with self._lock:
            job.status = BrainJobStatus.RUNNING
            job.last_polled_at = datetime.now(timezone.utc).isoformat()

        # Handle offline mode immediately
        if getattr(self.brain_client, "offline_mode", True):
            try:
                simulate_fn = getattr(self.brain_client, "simulate_alpha", None)
                if callable(simulate_fn):
                    res = simulate_fn(job.expression, settings=job.settings)
                else:
                    res = {
                        "status": "OFFLINE",
                        "sharpe": None,
                        "fitness": None,
                        "turnover": None,
                        "margin": None,
                        "location": None,
                        "simulation_id": None,
                    }
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
                    job.safe_error = res.get("error") or "WorldQuant 線上模擬未完成（離線模式）"
            except Exception as e:
                with self._lock:
                    job.status = BrainJobStatus.FAILED
                    job.safe_error = sanitize_error(e)
            return

        # Online mode: submit simulation
        try:
            if not getattr(self.brain_client, "authenticated", False):
                try:
                    self.brain_client.authenticate()
                except Exception as auth_err:
                    with self._lock:
                        job.status = BrainJobStatus.AUTH_FAILED
                        job.safe_error = f"WorldQuant 認證失敗: {sanitize_error(auth_err)}"
                    return

            payload = {
                "type": "REGULAR",
                "settings": job.settings,
                "regular": job.expression,
            }
            # Post simulation to get Location (with 429 concurrent-limit linear backoff)
            resp = None
            for _attempt in range(3):
                resp = self.brain_client.session.post(
                    f"{self.brain_client.BASE_URL}/simulations",
                    json=payload,
                    timeout=self.brain_client.timeout,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After") or (10 * (_attempt + 1)))
                    time.sleep(min(retry_after, 30.0))
                    continue
                break
            if resp.status_code in (401, 403):
                with self._lock:
                    job.status = BrainJobStatus.AUTH_FAILED
                    job.safe_error = f"WorldQuant 授權失敗 (HTTP {resp.status_code})"
                return

            if resp.status_code not in (200, 201, 202):
                resp.raise_for_status()

            with self._lock:
                job.status = BrainJobStatus.SUBMITTED

            location = resp.headers.get("Location") or resp.headers.get("location")
            if not location:
                body_json = {}
                try:
                    body_json = resp.json() if resp.content else {}
                except Exception:
                    pass
                loc = body_json.get("location") or body_json.get("id")
                if loc:
                    location = loc if str(loc).startswith("http") else f"{self.brain_client.BASE_URL}/simulations/{loc}"

            if not location:
                raise RuntimeError("WorldQuant /simulations 未回傳 Location header 或 ID")
            if location.startswith("/"):
                location = f"{self.brain_client.BASE_URL}{location}"

            with self._lock:
                job.simulation_location = location
                job.alpha_id = location.rstrip("/").split("/")[-1]
                job.status = BrainJobStatus.RUNNING

            # Poll simulation in background without blocking HTTP requests
            deadline = time.monotonic() + 1800.0
            poll_interval = self.default_poll_interval

            while time.monotonic() <= deadline:
                with self._lock:
                    job.last_polled_at = datetime.now(timezone.utc).isoformat()
                    job.next_poll_at = datetime.fromtimestamp(time.time() + poll_interval, tz=timezone.utc).isoformat()

                poll_resp = self.brain_client.session.get(location, timeout=self.brain_client.timeout)
                if poll_resp.status_code == 429:
                    retry_after = float(poll_resp.headers.get("Retry-After", poll_interval))
                    time.sleep(max(poll_interval, retry_after))
                    continue
                elif poll_resp.status_code in (502, 503, 504):
                    time.sleep(poll_interval)
                    continue

                if poll_resp.status_code in (401, 403):
                    with self._lock:
                        job.status = BrainJobStatus.AUTH_FAILED
                        job.safe_error = f"WorldQuant 輪詢認證失敗 (HTTP {poll_resp.status_code})"
                    return

                poll_resp.raise_for_status()
                poll_data = poll_resp.json() if poll_resp.content else {}
                state = str(poll_data.get("status") or poll_data.get("state") or "").upper()

                if state in ("COMPLETE", "COMPLETED"):
                    real_alpha_id = poll_data.get("alpha") or poll_data.get("alpha_id")
                    metrics_source = dict(poll_data)
                    checks_passed = True
                    failed_check_names: list[str] = []
                    if real_alpha_id:
                        try:
                            alpha_resp = self.brain_client.session.get(
                                f"{self.brain_client.BASE_URL}/alphas/{real_alpha_id}",
                                timeout=self.brain_client.timeout,
                            )
                            if getattr(alpha_resp, "status_code", None) == 200 and getattr(alpha_resp, "content", None):
                                alpha_json = alpha_resp.json()
                                if isinstance(alpha_json, dict):
                                    metrics_source.update(alpha_json)
                                    is_blk = alpha_json.get("is")
                                    if isinstance(is_blk, dict):
                                        metrics_source.update(is_blk)
                                        if isinstance(is_blk.get("checks"), list):
                                            for chk in is_blk["checks"]:
                                                if isinstance(chk, dict) and str(chk.get("result") or "").upper() in ("FAIL", "FAILED", "ERROR"):
                                                    checks_passed = False
                                                    failed_check_names.append(str(chk.get("name") or "CHECK"))
                        except Exception:
                            pass

                    sharpe = self.brain_client._number(metrics_source, "sharpe")
                    fitness = self.brain_client._number(metrics_source, "fitness")
                    turnover = self.brain_client._number(metrics_source, "turnover")
                    margin = self.brain_client._number(metrics_source, "margin")

                    has_metrics = (sharpe is not None) and (fitness is not None)
                    with self._lock:
                        if real_alpha_id:
                            job.alpha_id = str(real_alpha_id)
                        job.status = BrainJobStatus.COMPLETED if has_metrics else BrainJobStatus.FAILED
                        job.metrics = {
                            "sharpe": sharpe,
                            "fitness": fitness,
                            "turnover": turnover,
                            "margin": margin,
                            "checks_passed": checks_passed,
                            "failed_checks": failed_check_names,
                        }
                        if not has_metrics:
                            job.safe_error = "線上模擬完成但缺少有效 Sharpe 或 Fitness 指標"
                        self._persist_job(job)
                    if has_metrics and callable(getattr(self, "on_complete_callback", None)):
                        try:
                            self.on_complete_callback(job)
                        except Exception:
                            pass
                    return

                if state in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                    err_msg = poll_data.get("message") or poll_data.get("error") or f"Simulation {state}"
                    raise RuntimeError(f"Simulation {state}: {err_msg}")

                retry_after = poll_resp.headers.get("Retry-After")
                sleep_dur = float(retry_after) if retry_after else poll_interval
                time.sleep(sleep_dur)

            with self._lock:
                job.status = BrainJobStatus.TIMEOUT
                job.safe_error = "BRAIN simulation polling exceeded 1800 seconds"

        except Exception as e:
            with self._lock:
                job.safe_error = sanitize_error(e)
            if callable(getattr(self, "on_fail_callback", None)):
                try:
                    self.on_fail_callback(job)
                except Exception:
                    pass
            with self._lock:
                job.status = BrainJobStatus.FAILED
