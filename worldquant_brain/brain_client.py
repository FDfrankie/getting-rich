"""WorldQuant BRAIN API client with asynchronous simulation polling, exponential backoff, and strict security logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import traceback
from collections.abc import Mapping
from typing import Any

import requests
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl

logger = logging.getLogger("BrainClient")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def mask_sensitive_data(text: Any) -> str:
    """Mask tokens, passwords, cookies, and authorization data from string."""
    if text is None:
        return ""
    text_str = str(text)
    # Mask basic auth credentials in URLs or strings (e.g. user:password@host)
    text_str = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", text_str)
    # Mask common sensitive keys in JSON / headers / logs
    patterns = [
        (r'(?i)(password|passwd|pwd)[\'"]?\s*[:=]\s*[\'"][^\'"]+[\'"]', r'\1: "***"'),
        (r'(?i)(authorization|bearer|token|secret)[\'"]?\s*[:=]\s*[\'"][^\'"]+[\'"]', r'\1: "***"'),
        (r'(?i)(cookie[\'"]?\s*[:=]\s*[\'"]?)[^\'";\r\n]+', r"\1***"),
        (r'(?i)(session_id[\'"]?\s*[:=]\s*)[^\'";\s]+', r"\1***"),
        (r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]+", r"\1***"),
    ]
    for p, r in patterns:
        text_str = re.sub(p, r, text_str)
    return text_str


class BrainBiometricRequired(RuntimeError):
    error_code = "BRAIN_BIOMETRIC_REQUIRED"


class BrainWAFBlocked(RuntimeError):
    error_code = "BRAIN_WAF_BLOCKED"


def _classify_auth_challenge(response) -> str | None:
    authenticate = response.headers.get("WWW-Authenticate", "").lower()
    body = (response.text or "").lower()
    if response.status_code in {401, 403}:
        if "persona" in authenticate or "biometric" in body:
            return "BRAIN_BIOMETRIC_REQUIRED"
        if "cloudflare" in body or "waf" in body or "captcha" in body:
            return "BRAIN_WAF_BLOCKED"
    return None



def _parse_retry_after(header_val: str | None, default: float = 2.0, cap: float = 15.0) -> float:
    if not header_val:
        return default
    raw = str(header_val).strip()
    try:
        return min(max(float(raw), 0.1), cap)
    except ValueError:
        try:
            from datetime import datetime, timezone
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(raw)
            delta = (dt - datetime.now(dt.tzinfo or timezone.utc)).total_seconds()
            return min(max(delta, 0.5), cap)
        except Exception:
            return default


class BrainClient:
    BASE_URL = "https://api.worldquantbrain.com"

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
        poll_interval: float = 5.0,
        session: requests.Session | None = None,
    ) -> None:
        self.email = (os.getenv("BRAIN_EMAIL") or os.getenv("BRAIN_USERNAME")) if email is None else email
        self.password = os.getenv("BRAIN_PASSWORD") if password is None else password
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self._fixed_session: requests.Session | None = session
        self._thread_local = threading.local()
        self._auth_lock = threading.RLock()
        self._shared_cookies: dict[str, str] = {}
        self._shared_headers: dict[str, str] = {}
        self._auth_generation: int = 0
        self.authenticated = False
        from collections import OrderedDict
        self._submitted_max_size: int = 2000
        self._submitted: OrderedDict[str, dict[str, Any]] = OrderedDict()


    def _cache_submitted(self, key: str, result: dict[str, Any]) -> None:
        self._submitted[key] = result
        try:
            self._submitted.move_to_end(key)
            while len(self._submitted) > self._submitted_max_size:
                self._submitted.popitem(last=False)
        except Exception:
            pass

    @property
    def session(self) -> requests.Session:
        """Return a thread-local requests.Session (or injected fixed session) with synced auth cookies."""
        if self._fixed_session is not None:
            return self._fixed_session
        sess = getattr(self._thread_local, "session", None)
        gen = getattr(self._thread_local, "auth_generation", -1)
        if sess is None:
            sess = requests.Session()
            self._thread_local.session = sess
            gen = -1
        if gen != self._auth_generation:
            with self._auth_lock:
                if self._shared_cookies:
                    sess.cookies.update(self._shared_cookies)
                if self._shared_headers:
                    sess.headers.update(self._shared_headers)
                self._thread_local.auth_generation = self._auth_generation
        return sess

    @session.setter
    def session(self, value: requests.Session | None) -> None:
        self._fixed_session = value

    @property
    def offline_mode(self) -> bool:
        return not bool(self.email and self.password)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.authenticated and not self.offline_mode)

    def authenticate(self) -> bool:
        if self.offline_mode:
            return False
        t0 = time.monotonic()
        try:
            r = self.session.post(
                f"{self.BASE_URL}/authentication",
                auth=(self.email, self.password),
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - t0
            if r.status_code != 200 and r.status_code != 201:
                logger.error(
                    "WorldQuant authentication failed: status=%s, content_type=%s, elapsed=%.2fs, body_prefix=%s",
                    r.status_code,
                    r.headers.get("Content-Type"),
                    elapsed,
                    mask_sensitive_data(r.text[:500]),
                )
                challenge = _classify_auth_challenge(r)
                if challenge == "BRAIN_BIOMETRIC_REQUIRED":
                    raise BrainBiometricRequired(
                        "WorldQuant BRAIN requires interactive verification. Complete verification in the official browser session and retry."
                    )
                if challenge == "BRAIN_WAF_BLOCKED":
                    raise BrainWAFBlocked("WorldQuant BRAIN request blocked by WAF/anti-bot challenge.")
                r.raise_for_status()
            with self._auth_lock:
                if self._fixed_session is None:
                    try:
                        self._shared_cookies.update(self.session.cookies.get_dict())
                        auth_hdr = self.session.headers.get("Authorization")
                        if auth_hdr:
                            self._shared_headers["Authorization"] = str(auth_hdr)
                        self._auth_generation += 1
                        self._thread_local.auth_generation = self._auth_generation
                    except Exception as _sync_exc:
                        logger.debug("Could not sync auth cookies across threads: %s", _sync_exc)
                self.authenticated = True
            logger.info("WorldQuant authentication successful in %.2fs", elapsed)
            return True
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "WorldQuant authentication exception: type=%s, message=%s, elapsed=%.2fs\n%s",
                type(e).__name__,
                mask_sensitive_data(str(e)),
                elapsed,
                mask_sensitive_data(traceback.format_exc()),
            )
            raise

    @staticmethod
    def _key(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

    @staticmethod
    def _number(obj: Mapping[str, Any], *names: str) -> float | None:
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                lowered = {str(k).lower(): v for k, v in cur.items()}
                for n in names:
                    if n.lower() in lowered:
                        val = lowered[n.lower()]
                        if val is not None:
                            try:
                                f_val = float(val)
                                import math

                                if not math.isnan(f_val) and not math.isinf(f_val):
                                    return f_val
                            except (TypeError, ValueError):
                                pass
                stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
            elif isinstance(cur, list):
                stack.extend(cur)
        return None

    def _offline_result(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "status": "OFFLINE",
            "submitted": False,
            "simulation_id": None,
            "alpha_id": None,
            "sharpe": None,
            "fitness": None,
            "turnover": None,
            "margin": None,
            "message": "WorldQuant 線上模擬未完成",
            "error": "BRAIN_EMAIL/BRAIN_PASSWORD 未設定，未送出任何網路請求。",
            "error_code": "OFFLINE_CREDENTIALS_MISSING",
            "retryable": False,
            "request": dict(payload),
        }

    def submit_and_poll_simulation(
        self,
        payload: Mapping[str, Any],
        poll_interval: float | None = None,
        max_wait_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        body = dict(payload)
        key = self._key(body)
        if key in self._submitted:
            cached = dict(self._submitted[key])
            cached["deduplicated"] = True
            return cached
        if self.offline_mode:
            result = self._offline_result(body)
            self._submitted[key] = result
            return dict(result)

        if not self.authenticated:
            self.authenticate()

        # Submit simulation with retry/exponential backoff for 429/5xx
        submit_url = f"{self.BASE_URL}/simulations"
        response = None
        max_retries = 3
        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                response = self.session.post(submit_url, json=body, timeout=self.timeout)
                elapsed = time.monotonic() - t0
            except Exception as e:
                elapsed = time.monotonic() - t0
                logger.error(
                    "WorldQuant POST /simulations connection failed (attempt %d): type=%s, message=%s, elapsed=%.2fs",
                    attempt,
                    type(e).__name__,
                    mask_sensitive_data(str(e)),
                    elapsed,
                )
                if attempt >= max_retries:
                    raise
                time.sleep(2**attempt)
                continue

            if response.status_code in (204, 401) and attempt < max_retries:
                logger.warning("POST /simulations returned HTTP %d (session expired); clearing cookies and re-authenticating...", response.status_code)
                try:
                    self.session.cookies.clear()
                except Exception:
                    pass
                self.authenticated = False
                if self.authenticate():
                    continue
                response.raise_for_status()
            if response.status_code == 429:
                hdr_ra = _parse_retry_after(response.headers.get("Retry-After"), default=float(2 ** (attempt + 1)), cap=30.0)
                retry_after = min(30.0, max(hdr_ra, float(2 ** (attempt + 1))))
                logger.warning("POST /simulations hit 429 rate limit. Retry-After: %.1fs", retry_after)
                if attempt >= max_retries:
                    response.raise_for_status()
                time.sleep(retry_after)
                continue
            elif response.status_code in (502, 503, 504):
                logger.warning(
                    "POST /simulations server error %d. Retrying in %ds...",
                    response.status_code,
                    2**attempt,
                )
                if attempt >= max_retries:
                    response.raise_for_status()
                time.sleep(2**attempt)
                continue
            break

        if response is None:
            raise RuntimeError("WorldQuant /simulations response is None")

        if response.status_code not in (200, 201, 202):
            logger.error(
                "WorldQuant /simulations failed: status=%s, content_type=%s, body_prefix=%s",
                response.status_code,
                response.headers.get("Content-Type"),
                mask_sensitive_data(response.text[:500]),
            )
            response.raise_for_status()

        location = response.headers.get("Location") or response.headers.get("location")
        if not location:
            # Check if simulation_id or location in response body
            body_dict = {}
            try:
                body_dict = response.json() if response.content else {}
            except Exception as _silent_exc:
                logger.debug("worldquant_brain/brain_client.py:L258 suppressed exception: %s", _silent_exc)
            loc = body_dict.get("location") or body_dict.get("id")
            if loc:
                location = loc if str(loc).startswith("http") else f"{self.BASE_URL}/simulations/{loc}"

        if not location:
            raise RuntimeError(
                f"WorldQuant /simulations 未回傳 Location header 或 ID (status={response.status_code}, body={mask_sensitive_data(response.text[:500])})"
            )

        if location.startswith("/"):
            location = f"{self.BASE_URL}{location}"

        interval = self.poll_interval if poll_interval is None else float(poll_interval)
        deadline = time.monotonic() + float(max_wait_seconds)
        last: dict[str, Any] = {}
        poll_net_errors = 0
        poll_reauth_used = False

        # Poll simulation
        while True:
            if time.monotonic() > deadline:
                logger.error("WorldQuant simulation polling TIMEOUT; Location=%s", location)
                raise TimeoutError(f"BRAIN simulation polling 超過 {max_wait_seconds} 秒；Location={location}")

            t0 = time.monotonic()
            try:
                poll = self.session.get(location, timeout=self.timeout)
                poll_net_errors = 0
            except Exception as poll_exc:
                poll_net_errors += 1
                logger.warning(
                    "WorldQuant simulation poll connection error (%d/4): %s",
                    poll_net_errors,
                    mask_sensitive_data(str(poll_exc)),
                )
                if poll_net_errors >= 4:
                    raise
                time.sleep(min(15.0, interval * poll_net_errors))
                continue
            elapsed = time.monotonic() - t0

            if poll.status_code in (204, 401) and not poll_reauth_used:
                poll_reauth_used = True
                logger.warning("Polling hit HTTP %d (session expired); clearing cookies and re-authenticating...", poll.status_code)
                try:
                    self.session.cookies.clear()
                except Exception:
                    pass
                self.authenticated = False
                if self.authenticate():
                    time.sleep(1.0)
                    continue
            if poll.status_code == 429:
                retry_header = poll.headers.get("Retry-After")
                sleep_sec = _parse_retry_after(retry_header, default=interval, cap=30.0)
                logger.warning("Polling hit 429. Retry-After: %.1fs", sleep_sec)
                time.sleep(max(interval, sleep_sec))
                continue
            elif poll.status_code in (502, 503, 504):
                logger.warning("Polling server error %d. Retrying in %.1fs", poll.status_code, interval)
                time.sleep(interval)
                continue

            poll.raise_for_status()
            last = poll.json() if poll.content else {}
            status = str(last.get("status") or last.get("state") or "").upper()

            if status in ("COMPLETE", "COMPLETED"):
                sim_id = last.get("id") or location.rstrip("/").split("/")[-1]
                alpha_id = last.get("alpha") or last.get("alpha_id") or sim_id
                
                # If metrics are not in the simulation object, fetch the alpha object (with transient retry)
                alpha_data = {}
                if last.get("alpha"):
                    alpha_url = f"{self.BASE_URL}/alphas/{last['alpha']}"
                    for _a_try in range(3):
                        try:
                            alpha_resp = self.session.get(alpha_url, timeout=self.timeout)
                            if alpha_resp.status_code == 200 and getattr(alpha_resp, "content", None):
                                alpha_data = alpha_resp.json() or {}
                                break
                            if alpha_resp.status_code in (429, 502, 503, 504) and _a_try < 2:
                                time.sleep(2.0 * (_a_try + 1))
                                continue
                            break
                        except Exception as e:
                            logger.warning("Failed to fetch alpha record %s (try %d/3): %s", last["alpha"], _a_try + 1, e)
                            if _a_try < 2:
                                time.sleep(2.0 * (_a_try + 1))

                metrics_source = dict(last)
                if alpha_data:
                    metrics_source.update(alpha_data)
                    if "is" in alpha_data and isinstance(alpha_data["is"], dict):
                        metrics_source.update(alpha_data["is"])

                sharpe = self._number(metrics_source, "sharpe")
                fitness = self._number(metrics_source, "fitness")
                turnover = self._number(metrics_source, "turnover")
                margin = self._number(metrics_source, "margin")
                returns_val = self._number(metrics_source, "returns")
                drawdown_val = self._number(metrics_source, "drawdown")

                # P21 vertical fix: extract official IS-checks so single-sim fallback populates checks_passed/failed_checks
                checks_raw = metrics_source.get("checks")
                failed_checks: list[str] = []
                if isinstance(checks_raw, list):
                    for chk in checks_raw:
                        if isinstance(chk, Mapping):
                            res_st = str(chk.get("result") or "").upper()
                            if res_st in ("FAIL", "FAILED", "ERROR"):
                                failed_checks.append(str(chk.get("name") or chk.get("message") or "CHECK_FAILED"))
                    checks_passed = len(failed_checks) == 0
                else:
                    checks_passed = False

                has_valid_metrics = (sharpe is not None) and (fitness is not None)
                result = {
                    "success": has_valid_metrics,
                    "status": "COMPLETED" if has_valid_metrics else "FAILED",
                    "submitted": True,
                    "simulation_id": sim_id,
                    "alpha_id": alpha_id,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "turnover": turnover,
                    "margin": margin,
                    "returns": returns_val,
                    "drawdown": drawdown_val,
                    "checks_passed": checks_passed,
                    "failed_checks": failed_checks,
                    "message": "Alpha 模擬成功完成"
                    if has_valid_metrics
                    else "模擬完成但未取得有效 Sharpe/Fitness 指標",
                    "error": None if has_valid_metrics else "Missing Sharpe or Fitness in simulation result",
                    "raw": alpha_data or last,
                    "location": location,
                    "deduplicated": False,
                }
                self._submitted[key] = result
                return dict(result)

            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                err_msg = last.get("message") or last.get("error") or f"Simulation {status}"
                logger.error(
                    "WorldQuant simulation failed: status=%s, location=%s, body_prefix=%s",
                    status,
                    location,
                    mask_sensitive_data(str(last)[:500]),
                )
                raise RuntimeError(f"WorldQuant simulation {status}: {err_msg}")

            retry = poll.headers.get("Retry-After")
            sleep_time = float(retry) if retry else interval
            time.sleep(sleep_time)

    def simulate_alpha(
        self,
        expression: str,
        settings: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = {
            "type": "REGULAR",
            "settings": dict(
                settings
                or {
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
            ),
            "regular": expression,
        }
        return self.submit_and_poll_simulation(payload, **kwargs)

    def _quota_db(self) -> str:
        custom = getattr(self, "quota_db_path", None)
        if custom:
            return str(custom)
        try:
            from worldquant_brain.alpha_vault import AlphaVault
            return str(AlphaVault.DEFAULT_DB_PATH)
        except Exception:
            return os.path.join(os.path.dirname(__file__), "alpha_vault.db")

    def get_active_pnl_cache(self):
        """Return ActivePnlCache backed by the local alpha_vault SQLite DB."""
        from worldquant_brain.active_pnl_cache import ActivePnlCache

        return ActivePnlCache(self._quota_db())

    def check_local_active_self_correlation(
        self,
        alpha_id: str,
        *,
        region: str = "USA",
        threshold: float = 0.70,
        candidate_pnl: list[float] | None = None,
    ) -> dict[str, Any]:
        """Pre-check candidate alpha daily PnL against cached ACTIVE alphas to save submission quota."""
        try:
            cache = self.get_active_pnl_cache()
            pnl = list(candidate_pnl) if candidate_pnl is not None else self.fetch_alpha_pnl(alpha_id)
            if len(pnl) < 30:
                return {"checked": False, "blocked": False, "max_corr": 0.0, "matched_alpha_id": None}
            max_corr, worst_id = cache.max_local_self_correlation(pnl, region=region)
            blocked = bool(worst_id is not None and max_corr >= float(threshold))
            return {
                "checked": True,
                "blocked": blocked,
                "max_corr": max_corr,
                "matched_alpha_id": worst_id,
            }
        except Exception as exc:
            logger.debug("check_local_active_self_correlation failed: %s", exc)
            return {"checked": False, "blocked": False, "max_corr": 0.0, "matched_alpha_id": None}

    def cache_submitted_alpha_pnl(
        self,
        alpha_id: str,
        *,
        region: str = "USA",
        cum_pnl: list[float] | None = None,
    ) -> bool:
        """Persist a newly submitted ACTIVE alpha's cumulative PnL into ActivePnlCache."""
        try:
            pnl = list(cum_pnl) if cum_pnl is not None else self.fetch_alpha_pnl(alpha_id)
            if len(pnl) < 30:
                return False
            self.get_active_pnl_cache().upsert_cumulative_pnl(str(alpha_id), str(region).upper(), pnl)
            return True
        except Exception as exc:
            logger.debug("cache_submitted_alpha_pnl failed: %s", exc)
            return False

    def _today_brain_quota_day(self) -> str:
        from datetime import datetime, timedelta, timezone
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        except Exception:
            return datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d")

    def _today_taipei(self) -> str:
        return self._today_brain_quota_day()

    def get_daily_submission_count(self, day_str: str | None = None) -> int:
        import sqlite3
        day = day_str or self._today_taipei()
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        conn = sqlite3.connect(db_p, timeout=10.0)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS brain_daily_submissions (alpha_id TEXT PRIMARY KEY, submitted_date TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            ensure_schema_from_ddl(conn, 'CREATE TABLE IF NOT EXISTS brain_daily_submissions (alpha_id TEXT PRIMARY KEY, submitted_date TEXT NOT NULL, created_at TEXT NOT NULL)')
            cur = conn.execute("SELECT COUNT(*) FROM brain_daily_submissions WHERE submitted_date = ?", (day,))
            return int(cur.fetchone()[0])
        finally:
            conn.close()

    def _record_daily_submission(self, alpha_id: str, day_str: str | None = None) -> None:
        import sqlite3
        from datetime import datetime, timezone
        day = day_str or self._today_taipei()
        now_iso = datetime.now(timezone.utc).isoformat()
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        conn = sqlite3.connect(db_p, timeout=10.0)
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS brain_daily_submissions (alpha_id TEXT PRIMARY KEY, submitted_date TEXT NOT NULL, created_at TEXT NOT NULL)"
                )
                ensure_schema_from_ddl(conn, 'CREATE TABLE IF NOT EXISTS brain_daily_submissions (alpha_id TEXT PRIMARY KEY, submitted_date TEXT NOT NULL, created_at TEXT NOT NULL)')
                conn.execute(
                    "INSERT OR IGNORE INTO brain_daily_submissions (alpha_id, submitted_date, created_at) VALUES (?, ?, ?)",
                    (str(alpha_id), day, now_iso),
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
                )
                ensure_schema_from_ddl(conn, "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)")
                conn.execute("UPDATE brain_candidate_pool SET status = 'SUBMITTED' WHERE alpha_id = ?", (str(alpha_id),))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # P37: Qualified Formula Permanent Archive + Bronze Reverse-Order Submit
    #      + Consultant Reserve Protection (Sharpe >= 1.50 / Fitness >= 1.20)
    # ------------------------------------------------------------------
    @staticmethod
    def _submit_order_mode() -> str:
        """Return 'worst_first' (Bronze default) or 'best_first' (Consultant mode)."""
        if os.getenv("BRAIN_CONSULTANT_MODE", "false").strip().lower() in ("1", "true", "yes"):
            return os.getenv("BRAIN_SUBMIT_ORDER", "best_first").strip().lower() or "best_first"
        mode = os.getenv("BRAIN_SUBMIT_ORDER", "worst_first").strip().lower()
        return mode if mode in ("worst_first", "best_first") else "worst_first"

    @staticmethod
    def _consultant_reserve_thresholds() -> tuple[bool, float, float]:
        """Return (enabled, min_sharpe, min_fitness) for Consultant Reserve protection."""
        if os.getenv("BRAIN_CONSULTANT_MODE", "false").strip().lower() in ("1", "true", "yes"):
            return False, 0.0, 0.0
        min_sh = float(os.getenv("BRAIN_RESERVE_MIN_SHARPE", "1.50") or "1.50")
        min_fit = float(os.getenv("BRAIN_RESERVE_MIN_FITNESS", "1.20") or "1.20")
        enabled = (min_sh > 0.0) or (min_fit > 0.0)
        return enabled, min_sh, min_fit

    @classmethod
    def _is_consultant_reserve(cls, sharpe: float, fitness: float) -> bool:
        """Protect high-scoring alphas from being wasted during unpaid Bronze/Challenge submissions."""
        enabled, min_sh, min_fit = cls._consultant_reserve_thresholds()
        if not enabled:
            return False
        if min_sh > 0.0 and float(sharpe) >= min_sh:
            return True
        if min_fit > 0.0 and float(fitness) >= min_fit:
            return True
        return False

    @staticmethod
    def _ensure_qualified_archive_table(conn: Any) -> None:
        ddl = (
            "CREATE TABLE IF NOT EXISTS qualified_alpha_archive ("
            "alpha_id TEXT PRIMARY KEY, "
            "expression TEXT NOT NULL, "
            "settings_json TEXT NOT NULL DEFAULT '{}', "
            "sharpe REAL NOT NULL, "
            "fitness REAL NOT NULL, "
            "turnover REAL NOT NULL, "
            "margin REAL NOT NULL DEFAULT 0.0, "
            "returns REAL NOT NULL DEFAULT 0.0, "
            "drawdown REAL NOT NULL DEFAULT 0.0, "
            "sub_universe_sharpe REAL, "
            "tier TEXT NOT NULL DEFAULT 'BRONZE_SUBMITTABLE', "
            "submit_status TEXT NOT NULL DEFAULT 'ARCHIVED', "
            "source TEXT NOT NULL DEFAULT 'MINER', "
            "archived_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        conn.execute(ddl)
        ensure_schema_from_ddl(conn, ddl)

    @staticmethod
    def _ensure_rejected_alphas_table(conn: Any) -> None:
        ddl = (
            "CREATE TABLE IF NOT EXISTS brain_rejected_alphas ("
            "alpha_id TEXT PRIMARY KEY, "
            "status TEXT NOT NULL DEFAULT 'FAILED_CHECKS', "
            "reason TEXT NOT NULL DEFAULT '', "
            "rejected_at TEXT NOT NULL)"
        )
        conn.execute(ddl)
        ensure_schema_from_ddl(conn, ddl)

    @classmethod
    def _get_rejected_alpha_ids(cls, conn: Any) -> set[str]:
        rejected: set[str] = set()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "brain_rejected_alphas" in tables:
            for r in conn.execute("SELECT alpha_id FROM brain_rejected_alphas").fetchall():
                if r and r[0]:
                    rejected.add(str(r[0]))
        if "brain_candidate_pool" in tables:
            for r in conn.execute(
                "SELECT alpha_id FROM brain_candidate_pool "
                "WHERE UPPER(status) IN ('FAILED_CORR', 'FAILED_CHECKS', 'SUBMIT_REJECTED')"
            ).fetchall():
                if r and r[0]:
                    rejected.add(str(r[0]))
        return rejected

    def _record_submit_rejection_and_prune(self, alpha_id: str, status: str, reason: str = "") -> None:
        """P43: Persist submit rejection and permanently prune alpha_id from qualified_alpha_archive & alphas."""
        import sqlite3
        from datetime import datetime, timezone

        aid = str(alpha_id or "").strip()
        if not aid:
            return
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(db_p, timeout=10.0)
        try:
            with conn:
                self._ensure_rejected_alphas_table(conn)
                self._ensure_qualified_archive_table(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO brain_rejected_alphas (alpha_id, status, reason, rejected_at) "
                    "VALUES (?, ?, ?, ?)",
                    (aid, str(status or "FAILED_CHECKS"), str(reason or "")[:500], now_iso),
                )
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                if "brain_candidate_pool" in tables:
                    conn.execute("UPDATE brain_candidate_pool SET status = ? WHERE alpha_id = ?", (str(status), aid))
                conn.execute("DELETE FROM qualified_alpha_archive WHERE alpha_id = ?", (aid,))
                if "alphas" in tables:
                    conn.execute(
                        "DELETE FROM alphas WHERE brain_alpha_id = ? OR name = ?",
                        (aid, f"BRAIN_{aid}"),
                    )
        finally:
            conn.close()

    def _sync_vault_exports(self) -> None:
        """P43: Auto-regenerate JSON vaults and unsubmitted backup .xlsx after submit/reject mutations."""
        from pathlib import Path as _Path

        db_p = self._quota_db()
        explicit_xlsx = getattr(self, "vault_xlsx_path", None)
        explicit_jsons = getattr(self, "vault_json_paths", None)

        is_default_db = False
        try:
            from worldquant_brain.alpha_vault import AlphaVault
            is_default_db = _Path(db_p).resolve() == _Path(AlphaVault.DEFAULT_DB_PATH).resolve()
        except Exception:
            is_default_db = False

        if not is_default_db and explicit_xlsx is None and not explicit_jsons:
            return

        root_dir = _Path(__file__).resolve().parent.parent
        json_targets: list[_Path] = []
        if explicit_jsons:
            json_targets = [_Path(p) for p in explicit_jsons]
        elif is_default_db:
            json_targets = [
                root_dir / "observability" / "qualified_alpha_formulas_vault.json",
                root_dir / "data_persistence" / "qualified_alpha_formulas_vault.json",
            ]

        xlsx_target: _Path | None = None
        if explicit_xlsx is not None:
            xlsx_target = _Path(explicit_xlsx)
        elif is_default_db:
            xlsx_target = root_dir / "BRAIN_官方知識庫與研究紀錄" / "worldquant_brain_qualified_alphas_latest.xlsx"

        payload: dict[str, Any] | None = None
        for jp in json_targets:
            try:
                payload = self.export_qualified_formula_vault(output_path=jp)
            except Exception as exc:
                logger.debug("P43 _sync_vault_exports JSON write failed (%s): %s", jp, exc)
        if payload is None:
            try:
                payload = self.export_qualified_formula_vault()
            except Exception as exc:
                logger.debug("P43 _sync_vault_exports payload build failed: %s", exc)
                return

        if xlsx_target is not None:
            try:
                from scripts.backfill_brain_passed_alphas import export_vault_to_xlsx
                export_vault_to_xlsx(payload, xlsx_target)
            except Exception as exc:
                logger.debug("P43 _sync_vault_exports XLSX write failed (%s): %s", xlsx_target, exc)

    def archive_qualified_alpha(
        self,
        alpha_id: str,
        expression: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        margin: float = 0.0,
        *,
        settings: dict[str, Any] | None = None,
        returns: float = 0.0,
        drawdown: float = 0.0,
        sub_universe_sharpe: float | None = None,
        submit_status: str | None = None,
        source: str = "MINER",
    ) -> str:
        """Permanently save every passing alpha's full FastExpr formula, settings, and metrics."""
        import json as _json
        import sqlite3
        from datetime import datetime, timezone

        if not alpha_id or not str(expression or "").strip():
            return "SKIPPED"
        tier = "CONSULTANT_RESERVE" if self._is_consultant_reserve(float(sharpe), float(fitness)) else "BRONZE_SUBMITTABLE"
        st_val = str(submit_status or ("CONSULTANT_RESERVE" if tier == "CONSULTANT_RESERVE" else "ARCHIVED"))
        now_iso = datetime.now(timezone.utc).isoformat()
        sett_json = _json.dumps(settings or {}, ensure_ascii=False, sort_keys=True)
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        conn = sqlite3.connect(db_p, timeout=10.0)
        try:
            with conn:
                self._ensure_qualified_archive_table(conn)
                if st_val.upper() != "SUBMITTED":
                    rejected_ids = self._get_rejected_alpha_ids(conn)
                    if str(alpha_id) in rejected_ids:
                        return "REJECTED_SKIPPED"
                existing = conn.execute(
                    "SELECT archived_at, submit_status, settings_json FROM qualified_alpha_archive WHERE alpha_id = ?",
                    (str(alpha_id),),
                ).fetchone()
                archived_at = str(existing[0]) if existing and existing[0] else now_iso
                if existing and not submit_status and str(existing[1] or "") == "SUBMITTED":
                    st_val = "SUBMITTED"
                if existing and (not settings) and existing[2] and existing[2] != "{}":
                    sett_json = str(existing[2])
                conn.execute(
                    "INSERT OR REPLACE INTO qualified_alpha_archive "
                    "(alpha_id, expression, settings_json, sharpe, fitness, turnover, margin, returns, drawdown, "
                    "sub_universe_sharpe, tier, submit_status, source, archived_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(alpha_id),
                        str(expression).strip(),
                        sett_json,
                        float(sharpe),
                        float(fitness),
                        float(turnover),
                        float(margin or 0.0),
                        float(returns or 0.0),
                        float(drawdown or 0.0),
                        float(sub_universe_sharpe) if sub_universe_sharpe is not None else None,
                        tier,
                        st_val,
                        str(source or "MINER"),
                        archived_at,
                        now_iso,
                    ),
                )
        finally:
            conn.close()
        return tier

    def export_qualified_formula_vault(self, output_path: str | Path | None = None) -> dict[str, Any]:
        """Atomically export all permanently archived passing alpha formulas to JSON."""
        import json as _json
        import sqlite3
        from datetime import datetime, timezone, timedelta
        from pathlib import Path as _Path

        db_p = self._quota_db()
        rows: list[dict[str, Any]] = []
        rejected_ids: set[str] = set()
        if os.path.exists(db_p):
            conn = sqlite3.connect(db_p, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                rejected_ids = self._get_rejected_alpha_ids(conn)
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                if "qualified_alpha_archive" in tables:
                    for r in conn.execute(
                        "SELECT * FROM qualified_alpha_archive ORDER BY sharpe DESC, fitness DESC"
                    ).fetchall():
                        d = dict(r)
                        try:
                            d["settings"] = _json.loads(d.pop("settings_json", "{}") or "{}")
                        except Exception:
                            d["settings"] = {}
                        rows.append(d)
            finally:
                conn.close()

        excluded_unsub = {"SUBMITTED", "FAILED_CORR", "FAILED_CHECKS", "SUBMIT_REJECTED"}
        reserve = [
            r for r in rows
            if r.get("tier") == "CONSULTANT_RESERVE"
            and str(r.get("submit_status") or "").upper() not in excluded_unsub
            and str(r.get("alpha_id") or "") not in rejected_ids
        ]
        submittable = sorted(
            [
                r for r in rows
                if r.get("tier") != "CONSULTANT_RESERVE"
                and str(r.get("submit_status") or "").upper() not in excluded_unsub
                and str(r.get("alpha_id") or "") not in rejected_ids
            ],
            key=lambda x: (float(x.get("fitness") or 0.0), float(x.get("sharpe") or 0.0)),
        )
        submitted = [r for r in rows if str(r.get("submit_status") or "").upper() == "SUBMITTED"]

        enabled, min_sh, min_fit = self._consultant_reserve_thresholds()
        now_utc = datetime.now(timezone.utc)
        now_tpe = now_utc.astimezone(timezone(timedelta(hours=8)))
        payload = {
            "updated_at_utc": now_utc.isoformat(),
            "updated_at_taipei": now_tpe.strftime("%Y-%m-%d %H:%M:%S+08:00"),
            "policy": {
                "submit_order": self._submit_order_mode(),
                "consultant_reserve_enabled": enabled,
                "reserve_min_sharpe": min_sh,
                "reserve_min_fitness": min_fit,
            },
            "summary": {
                "total_archived": len(rows),
                "consultant_reserve_count": len(reserve),
                "bronze_submittable_count": len(submittable),
                "submitted_count": len(submitted),
                "rejected_or_excluded_count": max(0, len(rows) - len(reserve) - len(submittable) - len(submitted)),
            },
            "consultant_reserve": reserve,
            "bronze_submittable_worst_first": submittable,
            "submitted": submitted,
        }
        if output_path is not None:
            out_p = _Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            tmp_p = out_p.parent / f".{out_p.name}.{os.getpid()}.tmp"
            try:
                tmp_p.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp_p, out_p)
            finally:
                if tmp_p.exists():
                    try:
                        tmp_p.unlink()
                    except Exception:
                        pass
        return payload


    def backfill_all_passed_alphas(
        self,
        *,
        max_pages: int = 20,
        page_size: int = 100,
        vault_json_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """P42: Full-history paginated sync of ALL passed Alphas from live BRAIN API.

        Paginates BOTH status=ACTIVE (Submitted) and status=UNSUBMITTED (is.sharpe>=1.25,
        is.fitness>=1.0), verifies 0 FAIL on IS checks, archives every qualifying formula
        into SQLite qualified_alpha_archive (with zero margin-drop bias), and exports JSON.
        """
        if self.offline_mode:
            return {"ok": False, "reason": "offline_mode", "submitted_archived": 0, "unsubmitted_archived": 0, "total_archived": 0}
        if not self.is_authenticated and not self.authenticate():
            return {"ok": False, "reason": "auth_failed", "submitted_archived": 0, "unsubmitted_archived": 0, "total_archived": 0}

        url = f"{self.BASE_URL}/users/self/alphas"
        limit = int(max(1, min(100, page_size)))
        submitted_archived = 0
        unsubmitted_archived = 0
        skipped_failed_checks = 0
        skipped_rejected = 0
        pages_scanned = 0

        import sqlite3
        rejected_ids: set[str] = set()
        db_p = self._quota_db()
        if os.path.exists(db_p):
            try:
                _rc = sqlite3.connect(db_p, timeout=10.0)
                try:
                    rejected_ids = self._get_rejected_alpha_ids(_rc)
                finally:
                    _rc.close()
            except Exception:
                rejected_ids = set()

        scan_specs = [
            ("ACTIVE", {"status": "ACTIVE", "order": "-dateSubmitted"}),
            ("UNSUBMITTED", {"status": "UNSUBMITTED", "order": "-is.sharpe", "is.sharpe>": 1.2499, "is.fitness>": 0.9999}),
        ]

        for status_tag, base_params in scan_specs:
            offset = 0
            for _ in range(int(max(1, max_pages))):
                params = dict(base_params)
                params["limit"] = limit
                params["offset"] = offset
                try:
                    resp = self.session.get(url, params=params, timeout=min(self.timeout, 25.0))
                    if resp.status_code == 401:
                        self.authenticated = False
                        if self.authenticate():
                            resp = self.session.get(url, params=params, timeout=min(self.timeout, 25.0))
                    if resp.status_code != 200 or not resp.content:
                        break
                    payload = resp.json()
                except Exception as exc:
                    logger.debug("backfill_all_passed_alphas (%s offset=%s) failed: %s", status_tag, offset, exc)
                    break

                pages_scanned += 1
                items = payload.get("results") if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
                if not isinstance(items, list) or not items:
                    break

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    aid = str(item.get("id") or "").strip()
                    if not aid:
                        continue
                    is_blk = item.get("is") if isinstance(item.get("is"), dict) else {}
                    sett = item.get("settings") if isinstance(item.get("settings"), dict) else {}
                    reg = item.get("regular") if isinstance(item.get("regular"), dict) else {}
                    expr = str(reg.get("code") or item.get("expression") or "").strip()
                    if not expr:
                        continue
                    sh = self._number(is_blk, "sharpe")
                    fit = self._number(is_blk, "fitness")
                    to = self._number(is_blk, "turnover")
                    mg = self._number(is_blk, "margin") or 0.0
                    ret = self._number(is_blk, "returns") or 0.0
                    dd = self._number(is_blk, "drawdown") or 0.0
                    if sh is None or fit is None or to is None:
                        continue
                    norm_to = to / 100.0 if 2.0 < to <= 100.0 else to
                    checks = is_blk.get("checks") if isinstance(is_blk.get("checks"), list) else []
                    sub_u_sh = None
                    has_fail = False
                    for ch in checks:
                        if not isinstance(ch, dict):
                            continue
                        if str(ch.get("name") or "").upper() == "LOW_SUB_UNIVERSE_SHARPE":
                            val = ch.get("value")
                            if isinstance(val, (int, float)):
                                sub_u_sh = float(val)
                        if str(ch.get("result") or "").upper() in ("FAIL", "FAILED", "ERROR"):
                            has_fail = True
                    if status_tag == "UNSUBMITTED" and aid in rejected_ids:
                        skipped_rejected += 1
                        continue
                    if status_tag == "UNSUBMITTED" and has_fail:
                        skipped_failed_checks += 1
                        continue
                    arch_res = self.archive_qualified_alpha(
                        aid,
                        expr,
                        float(sh),
                        float(fit),
                        float(norm_to),
                        float(mg),
                        settings=sett,
                        returns=float(ret),
                        drawdown=float(dd),
                        sub_universe_sharpe=sub_u_sh,
                        submit_status="SUBMITTED" if status_tag == "ACTIVE" else None,
                        source="P42_FULL_BACKFILL_ACTIVE" if status_tag == "ACTIVE" else "P42_FULL_BACKFILL_UNSUBMITTED",
                    )
                    if arch_res == "REJECTED_SKIPPED":
                        skipped_rejected += 1
                        continue
                    if status_tag == "ACTIVE":
                        submitted_archived += 1
                    else:
                        unsubmitted_archived += 1

                if len(items) < limit:
                    break
                offset += limit

        exported = self.export_qualified_formula_vault(output_path=vault_json_path)
        summary = exported.get("summary") or {}
        return {
            "ok": True,
            "pages_scanned": pages_scanned,
            "submitted_archived": submitted_archived,
            "unsubmitted_archived": unsubmitted_archived,
            "skipped_failed_checks": skipped_failed_checks,
            "total_archived": int(summary.get("total_archived", 0)),
            "consultant_reserve_count": int(summary.get("consultant_reserve_count", 0)),
            "bronze_submittable_count": int(summary.get("bronze_submittable_count", 0)),
            "submitted_count": int(summary.get("submitted_count", 0)),
        }

    def stage_candidate_for_daily_selection(
        self,
        alpha_id: str,
        expression: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        margin: float = 0.0,
        day_str: str | None = None,
        self_correlation: float | None = None,
        *,
        settings: dict[str, Any] | None = None,
        returns: float = 0.0,
        drawdown: float = 0.0,
        sub_universe_sharpe: float | None = None,
        source: str = "MINER",
    ) -> str:
        import sqlite3
        from datetime import datetime, timezone
        day = day_str or self._today_taipei()
        import math

        # P37: Always permanently archive every passing alpha's full FastExpr formula first!
        tier = self.archive_qualified_alpha(
            alpha_id=str(alpha_id),
            expression=str(expression),
            sharpe=float(sharpe),
            fitness=float(fitness),
            turnover=float(turnover),
            margin=float(margin or 0.0),
            settings=settings,
            returns=float(returns or 0.0),
            drawdown=float(drawdown or 0.0),
            sub_universe_sharpe=sub_universe_sharpe,
            source=source,
        )

        norm_m = float(margin) / 10000.0 if float(margin) > 0.5 else float(margin)
        m_bps = norm_m * 10000.0
        s_score = max(0.0, min(1.0, (float(sharpe) - 1.0) / 1.5))
        f_score = max(0.0, min(1.0, (float(fitness) - 0.7) / 1.3))
        m_score = max(0.0, min(1.0, (m_bps - 5.0) / 20.0)) if m_bps > 0 else 0.30
        t_val = max(0.005, min(1.0, float(turnover) / 100.0 if float(turnover) > 2.0 else float(turnover)))
        # Official KB 5968273187735: target Turnover < 30% for low transaction cost
        t_bell = math.exp(-((math.log(t_val) - math.log(0.18)) ** 2) / (2.0 * (0.55 ** 2)))
        if t_val <= 0.30:
            t_bell = min(1.0, t_bell + 0.08)
        _expr_low = str(expression).lower()
        _is_high_yield_domain = any(
            k in _expr_low
            for k in (
                "fnd6_", "anl4_", "opt8_", "opt6_", "mdl16_", "nws12_", "snt1_", "scl12_",
                "est_eps", "ebitda", "cashflow_op", "operating_income",
            )
        )
        e_bonus = 0.85 if _is_high_yield_domain else (0.55 if "ts_regression" in _expr_low else 0.30)
        corr_bonus = max(0.0, min(1.0, (0.70 - abs(float(self_correlation))) / 0.70)) * 5.0 if self_correlation is not None else 0.0
        score = round(100.0 * (0.35 * s_score + 0.25 * f_score + 0.15 * m_score + 0.15 * t_bell + 0.10 * e_bonus) + float(sharpe) * 0.01 + corr_bonus, 4)
        now_iso = datetime.now(timezone.utc).isoformat()
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        conn = sqlite3.connect(db_p, timeout=10.0)
        pool_status = "CONSULTANT_RESERVE" if tier == "CONSULTANT_RESERVE" else "PENDING"
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
                )
                ensure_schema_from_ddl(conn, "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)")
                self.ensure_novelty_survey_tables(conn)
                if pool_status == "CONSULTANT_RESERVE":
                    conn.execute(
                        "INSERT OR REPLACE INTO brain_candidate_pool (alpha_id, expression, sharpe, fitness, turnover, margin, score, candidate_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'CONSULTANT_RESERVE', ?)",
                        (str(alpha_id), str(expression), float(sharpe), float(fitness), float(turnover), float(margin), score, day, now_iso),
                    )
                    logger.info(
                        "[CONSULTANT_RESERVE_PROTECTED] Alpha %s (Sharpe=%.2f, Fitness=%.2f) archived & protected from Bronze submit",
                        alpha_id, float(sharpe), float(fitness),
                    )
                    return "CONSULTANT_RESERVE"
                worst_first = self._submit_order_mode() == "worst_first"
                if worst_first:
                    # P43: In Bronze worst_first mode, only block exact_hash match to already-SUBMITTED alphas;
                    # allow structural/parameter clones into brain_candidate_pool so Bronze quota tests them first.
                    _sub_rows = conn.execute(
                        "SELECT alpha_id, expression FROM brain_candidate_pool WHERE UPPER(status) = 'SUBMITTED'"
                    ).fetchall()
                    from worldquant_brain.alpha_novelty import fingerprint as _fp_fn
                    _new_fp = _fp_fn(expression)
                    for sr in _sub_rows:
                        if str(sr[0]) != str(alpha_id) and _fp_fn(str(sr[1])).exact_hash == _new_fp.exact_hash:
                            return "DUPLICATE_ARCHIVED"
                    conn.execute(
                        "INSERT OR REPLACE INTO brain_candidate_pool (alpha_id, expression, sharpe, fitness, turnover, margin, score, candidate_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)",
                        (str(alpha_id), str(expression), float(sharpe), float(fitness), float(turnover), float(margin), score, day, now_iso),
                    )
                else:
                    _all_existing = conn.execute(
                        "SELECT alpha_id, expression FROM brain_candidate_pool WHERE status IN ('PENDING', 'SUBMITTED')"
                    ).fetchall()
                    _existing_exprs = [{"alpha_id": str(r[0]), "expression": str(r[1])} for r in _all_existing if str(r[0]) != str(alpha_id)]
                    from worldquant_brain.alpha_novelty import find_duplicate
                    dup_chk = find_duplicate(expression, _existing_exprs)
                    if dup_chk.blocked:
                        return "DUPLICATE_ARCHIVED"
                    from worldquant_brain.alpha_miner import extract_expression_skeleton
                    new_skel = extract_expression_skeleton(expression)
                    existing_rows = conn.execute("SELECT alpha_id, expression, score FROM brain_candidate_pool WHERE candidate_date = ? AND status = 'PENDING'", (day,)).fetchall()
                    skip_clone = False
                    for er in existing_rows:
                        if extract_expression_skeleton(er[1]) == new_skel and er[0] != str(alpha_id):
                            if float(er[2]) >= float(score):
                                skip_clone = True
                                break
                            else:
                                conn.execute("DELETE FROM brain_candidate_pool WHERE alpha_id = ?", (er[0],))
                    if not skip_clone:
                        conn.execute(
                            "INSERT OR REPLACE INTO brain_candidate_pool (alpha_id, expression, sharpe, fitness, turnover, margin, score, candidate_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)",
                            (str(alpha_id), str(expression), float(sharpe), float(fitness), float(turnover), float(margin), score, day, now_iso),
                        )
        finally:
            conn.close()
        return pool_status

    def harvest_unsubmitted_candidates(
        self,
        *,
        limit: int = 50,
        day_str: str | None = None,
    ) -> int:
        """Scan BRAIN /users/self/alphas?status=UNSUBMITTED:
        - In Bronze worst_first mode: harvest lowest-fitness qualifying alphas (is.fitness ASC) for submission,
          AND run a second pass (-is.fitness DESC) to permanently archive top gems as CONSULTANT_RESERVE.
        - In Consultant best_first mode: harvest highest-fitness qualifying alphas (-is.fitness DESC).
        """
        if self.offline_mode:
            return 0
        if not self.is_authenticated and not self.authenticate():
            return 0
        import sqlite3

        min_margin = float(os.getenv("BRAIN_MIN_MARGIN", "0.0005") or "0.0005")
        url = f"{self.BASE_URL}/users/self/alphas"
        worst_first = self._submit_order_mode() == "worst_first"
        pass_orders = [
            ("is.fitness", True),   # Pass 1: worst qualifying first for Bronze submit
            ("-is.fitness", False), # Pass 2: archive top UNSUBMITTED gems into CONSULTANT_RESERVE
        ] if worst_first else [("-is.fitness", True)]

        existing_ids: set[str] = set()
        rejected_ids: set[str] = set()
        db_p = self._quota_db()
        if os.path.exists(db_p):
            try:
                c = sqlite3.connect(db_p, timeout=10.0)
                c.execute(
                    "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
                )
                existing_ids = {str(r[0]) for r in c.execute("SELECT alpha_id FROM brain_candidate_pool").fetchall()}
                rejected_ids = self._get_rejected_alpha_ids(c)
                c.close()
            except Exception:
                pass

        staged = 0
        # P42: Keep ACTIVE submitted alphas synced in archive alongside paginated UNSUBMITTED harvest
        try:
            act_resp = self.session.get(url, params={"status": "ACTIVE", "limit": 100, "offset": 0}, timeout=min(self.timeout, 15.0))
            if act_resp.status_code == 200 and act_resp.content:
                act_payload = act_resp.json()
                act_items = act_payload.get("results") if isinstance(act_payload, dict) else []
                for a_it in (act_items or []):
                    if not isinstance(a_it, dict):
                        continue
                    a_id = str(a_it.get("id") or "").strip()
                    a_is = a_it.get("is") if isinstance(a_it.get("is"), dict) else {}
                    a_reg = a_it.get("regular") if isinstance(a_it.get("regular"), dict) else {}
                    a_expr = str(a_reg.get("code") or a_it.get("expression") or "").strip()
                    a_sh = self._number(a_is, "sharpe")
                    a_fit = self._number(a_is, "fitness")
                    a_to = self._number(a_is, "turnover")
                    if a_id and a_expr and a_sh is not None and a_fit is not None and a_to is not None:
                        a_norm_to = a_to / 100.0 if 2.0 < a_to <= 100.0 else a_to
                        self.archive_qualified_alpha(
                            a_id, a_expr, float(a_sh), float(a_fit), float(a_norm_to),
                            float(self._number(a_is, "margin") or 0.0),
                            settings=a_it.get("settings") if isinstance(a_it.get("settings"), dict) else {},
                            returns=float(self._number(a_is, "returns") or 0.0),
                            drawdown=float(self._number(a_is, "drawdown") or 0.0),
                            submit_status="SUBMITTED",
                            source="ACTIVE_SYNC",
                        )
        except Exception:
            pass

        for order_val, allow_pending_stage in pass_orders:
            params: dict[str, Any] = {
                "status": "UNSUBMITTED",
                "limit": int(max(10, min(100, limit))),
                "offset": 0,
                "order": order_val,
                "is.sharpe>": 1.2499,
                "is.fitness>": 0.9999,
            }
            try:
                resp = self.session.get(url, params=params, timeout=min(self.timeout, 20.0))
                if resp.status_code == 401:
                    self.authenticated = False
                    if self.authenticate():
                        resp = self.session.get(url, params=params, timeout=min(self.timeout, 20.0))
                if resp.status_code != 200 or not resp.content:
                    continue
                payload = resp.json()
            except Exception as exc:
                logger.debug("harvest_unsubmitted_candidates (%s) fetch failed: %s", order_val, exc)
                continue

            items = payload.get("results") if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
            if not isinstance(items, list) or not items:
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                aid = str(item.get("id") or "").strip()
                if not aid or aid in rejected_ids:
                    continue
                is_blk = item.get("is") if isinstance(item.get("is"), dict) else {}
                sett = item.get("settings") if isinstance(item.get("settings"), dict) else {}
                sh = self._number(is_blk, "sharpe")
                fit = self._number(is_blk, "fitness")
                to = self._number(is_blk, "turnover")
                mg = self._number(is_blk, "margin")
                ret = self._number(is_blk, "returns") or 0.0
                dd = self._number(is_blk, "drawdown") or 0.0
                if sh is None or fit is None or to is None:
                    continue
                delay = int(sett.get("delay", 1) or 1)
                min_sh = 2.00 if delay == 0 else 1.25
                min_fit = 1.30 if delay == 0 else 1.00
                norm_to = to / 100.0 if 2.0 < to <= 70.0 else to
                if sh < min_sh or fit < min_fit or not (0.01 <= norm_to <= 0.70):
                    continue
                if mg is not None:
                    norm_mg = mg / 10000.0 if mg > 0.5 else mg
                    if norm_mg < min_margin:
                        continue
                checks = is_blk.get("checks") if isinstance(is_blk.get("checks"), list) else []
                if any(
                    isinstance(ch, dict) and str(ch.get("result") or "").upper() in ("FAIL", "FAILED", "ERROR")
                    for ch in checks
                ):
                    continue
                reg = item.get("regular") if isinstance(item.get("regular"), dict) else {}
                expr = str(reg.get("code") or item.get("expression") or "").strip()
                if not expr:
                    continue

                if not allow_pending_stage:
                    # Pass 2: archive-only pass (protects top gems without putting them into PENDING)
                    self.archive_qualified_alpha(
                        aid, expr, float(sh), float(fit), float(norm_to), float(mg or 0.0),
                        settings=sett, returns=float(ret), drawdown=float(dd), source="BACKLOG_ARCHIVE",
                    )
                    continue

                if aid in existing_ids:
                    # Still ensure its formula is permanently archived
                    self.archive_qualified_alpha(
                        aid, expr, float(sh), float(fit), float(norm_to), float(mg or 0.0),
                        settings=sett, returns=float(ret), drawdown=float(dd), source="BACKLOG_SYNC",
                    )
                    continue

                res_status = self.stage_candidate_for_daily_selection(
                    aid,
                    expr,
                    float(sh),
                    float(fit),
                    float(norm_to),
                    float(mg or 0.0),
                    day_str=day_str,
                    settings=sett,
                    returns=float(ret),
                    drawdown=float(dd),
                    source="BACKLOG_HARVEST",
                )
                existing_ids.add(aid)
                if res_status == "PENDING":
                    staged += 1
        return staged

    def sweep_consultant_reserve(self, day_str: str | None = None) -> int:
        """P38: Immediately move any PENDING gems (>=1.50 Sharpe or >=1.20 Fitness) to CONSULTANT_RESERVE
        and permanently archive their formulas — even BEFORE flush_hour=22!"""
        import sqlite3
        db_p = self._quota_db()
        if not os.path.exists(db_p):
            return 0
        enabled, _, _ = self._consultant_reserve_thresholds()
        if not enabled:
            return 0
        day = day_str or self._today_taipei()
        moved = 0
        conn = sqlite3.connect(db_p, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "brain_candidate_pool" not in tables:
                return 0
            self._ensure_qualified_archive_table(conn)
            pending_rows = conn.execute(
                "SELECT alpha_id, expression, sharpe, fitness, turnover, margin "
                "FROM brain_candidate_pool WHERE status = 'PENDING'"
            ).fetchall()
        finally:
            conn.close()

        for row in pending_rows:
            _aid = str(row["alpha_id"])
            _expr = str(row["expression"] or "")
            _sh = float(row["sharpe"] or 0.0)
            _fit = float(row["fitness"] or 0.0)
            _to = float(row["turnover"] or 0.0)
            _mg = float(row["margin"] or 0.0)
            if self._is_consultant_reserve(_sh, _fit):
                self.archive_qualified_alpha(
                    _aid, _expr, _sh, _fit, _to, _mg,
                    submit_status="CONSULTANT_RESERVE", source="STARTUP_RESERVE_SWEEP",
                )
                c_res = sqlite3.connect(db_p, timeout=10.0)
                try:
                    c_res.execute("UPDATE brain_candidate_pool SET status = 'CONSULTANT_RESERVE' WHERE alpha_id = ?", (_aid,))
                    c_res.commit()
                finally:
                    c_res.close()
                moved += 1
                logger.info(
                    "[CONSULTANT_RESERVE_SWEPT] Alpha %s (Sharpe=%.2f, Fitness=%.2f) moved from PENDING to CONSULTANT_RESERVE",
                    _aid, _sh, _fit,
                )
        return moved

    def flush_daily_candidates(self, day_str: str | None = None, max_daily_submissions: int | None = None) -> list[dict[str, Any]]:
        """Submit pending candidates up to the remaining daily quota.
        P37: Defaults to reverse order ('worst_first': fitness ASC, sharpe ASC, score ASC)
        and enforces defense-in-depth Consultant Reserve protection (Sharpe >= 1.50 or Fitness >= 1.20).
        """
        import sqlite3
        from datetime import datetime, timezone
        day = day_str or self._today_taipei()
        limit = int(max_daily_submissions if max_daily_submissions is not None else os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2"))
        remaining = max(0, limit - self.get_daily_submission_count(day))
        if remaining == 0:
            return []
        db_p = self._quota_db()
        if not os.path.exists(db_p):
            return []
        order_mode = self._submit_order_mode()
        order_sql = "ORDER BY fitness ASC, sharpe ASC, score ASC" if order_mode == "worst_first" else "ORDER BY score DESC, sharpe DESC"
        conn = sqlite3.connect(db_p, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
            )
            ensure_schema_from_ddl(conn, "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)")
            self._ensure_qualified_archive_table(conn)
            all_pending = conn.execute(
                f"SELECT alpha_id, expression, score, sharpe, fitness, turnover, margin FROM brain_candidate_pool WHERE candidate_date <= ? AND status = 'PENDING' {order_sql}",
                (day,),
            ).fetchall()
            _ref_rows: list[tuple[str, str, float, float, str]] = []
            for rr in conn.execute(
                "SELECT alpha_id, expression, sharpe, score, status FROM brain_candidate_pool "
                "WHERE UPPER(status) IN ('PENDING', 'SUBMITTED', 'CONSULTANT_RESERVE')"
            ).fetchall():
                _ref_rows.append((str(rr[0]), str(rr[1] or ""), float(rr[2] or 0.0), float(rr[3] or 0.0), str(rr[4] or "").upper()))
            _seen_ref_ids = {x[0] for x in _ref_rows}
            for ar in conn.execute(
                "SELECT alpha_id, expression, sharpe, submit_status FROM qualified_alpha_archive"
            ).fetchall():
                _a_id = str(ar[0])
                if _a_id not in _seen_ref_ids:
                    _ref_rows.append((_a_id, str(ar[1] or ""), float(ar[2] or 0.0), 0.0, str(ar[3] or "").upper()))
                    _seen_ref_ids.add(_a_id)
            from worldquant_brain.alpha_miner import extract_expression_skeleton
            rows = list(all_pending)
            conn.commit()
        finally:
            conn.close()

        # P37 Upfront Reserve Sweep: immediately protect & archive ANY legacy PENDING gems (>=1.50 Sharpe or >=1.20 Fitness)
        eligible_rows = []
        for row in rows:
            _aid = str(row["alpha_id"])
            _expr = str(row["expression"] or "")
            _sh = float(row["sharpe"] or 0.0)
            _fit = float(row["fitness"] or 0.0)
            _to = float(row["turnover"] or 0.0)
            _mg = float(row["margin"] or 0.0)
            if self._is_consultant_reserve(_sh, _fit):
                self.archive_qualified_alpha(
                    _aid, _expr, _sh, _fit, _to, _mg,
                    submit_status="CONSULTANT_RESERVE", source="FLUSH_RESERVE_SWEEP",
                )
                try:
                    c_res = sqlite3.connect(db_p, timeout=10.0)
                    c_res.execute("UPDATE brain_candidate_pool SET status = 'CONSULTANT_RESERVE' WHERE alpha_id = ?", (_aid,))
                    c_res.commit()
                    c_res.close()
                except Exception as _res_exc:
                    logger.debug("Could not move %s to CONSULTANT_RESERVE: %s", _aid, _res_exc)
                logger.info(
                    "[CONSULTANT_RESERVE_BLOCKED_SUBMIT] Protected Alpha %s (Sharpe=%.2f, Fitness=%.2f) from Bronze submit",
                    _aid, _sh, _fit,
                )
            else:
                eligible_rows.append(row)

        # P43: In Bronze worst_first mode, prioritize likely-duplicate / redundant clones FIRST,
        # ordered by (dup_priority DESC, fitness ASC, sharpe ASC, score ASC).
        if order_mode == "worst_first" and len(eligible_rows) > 1:
            from worldquant_brain.alpha_novelty import fingerprint as _fp_fn, near_duplicate_score as _sim_fn

            skel_counts: dict[str, int] = {}
            ref_fps: list[tuple[str, Any, float, float, str]] = []
            for r_id, r_expr, r_sh, r_sc, r_st in _ref_rows:
                sk = extract_expression_skeleton(r_expr)
                if sk:
                    skel_counts[sk] = skel_counts.get(sk, 0) + 1
                try:
                    ref_fps.append((r_id, _fp_fn(r_expr), r_sh, r_sc, r_st))
                except Exception:
                    pass

            def _bronze_sort_key(row_item: Any) -> tuple[int, float, float, float]:
                aid_k = str(row_item["alpha_id"])
                expr_k = str(row_item["expression"] or "")
                fit_k = float(row_item["fitness"] or 0.0)
                sh_k = float(row_item["sharpe"] or 0.0)
                sc_k = float(row_item["score"] or 0.0)
                sk_k = extract_expression_skeleton(expr_k)
                dup_prio = 1 if (sk_k and skel_counts.get(sk_k, 0) >= 2) else 0
                if dup_prio == 0:
                    try:
                        fp_k = _fp_fn(expr_k)
                        for r_id, r_fp, r_sh, r_sc, r_st in ref_fps:
                            if r_id == aid_k:
                                continue
                            is_sub_or_higher = (r_st == "SUBMITTED") or (r_sh > sh_k) or (r_sc > sc_k)
                            if is_sub_or_higher and _sim_fn(fp_k, r_fp) >= 0.70:
                                dup_prio = 1
                                break
                    except Exception:
                        pass
                return (-dup_prio, fit_k, sh_k, sc_k)

            eligible_rows.sort(key=_bronze_sort_key)

        results: list[dict[str, Any]] = []
        submitted_count = 0
        submitted_skels: set[str] = set()
        failed_corr_skels: set[str] = set()
        mutated = False
        for row in eligible_rows:
            if submitted_count >= remaining:
                break
            aid = str(row["alpha_id"])
            expr = str(row["expression"] or "")
            sh_val = float(row["sharpe"] or 0.0)
            fit_val = float(row["fitness"] or 0.0)
            to_val = float(row["turnover"] or 0.0)
            mg_val = float(row["margin"] or 0.0)

            # P37 Defense-in-Depth: NEVER submit Consultant Reserve gems during Bronze mode
            if self._is_consultant_reserve(sh_val, fit_val):
                self.archive_qualified_alpha(
                    aid, expr, sh_val, fit_val, to_val, mg_val,
                    submit_status="CONSULTANT_RESERVE", source="FLUSH_PROTECTED",
                )
                try:
                    c_res = sqlite3.connect(db_p, timeout=10.0)
                    c_res.execute("UPDATE brain_candidate_pool SET status = 'CONSULTANT_RESERVE' WHERE alpha_id = ?", (aid,))
                    c_res.commit()
                    c_res.close()
                except Exception as _res_exc:
                    logger.debug("Could not move %s to CONSULTANT_RESERVE: %s", aid, _res_exc)
                logger.info(
                    "[CONSULTANT_RESERVE_BLOCKED_SUBMIT] Protected Alpha %s (Sharpe=%.2f, Fitness=%.2f) from Bronze submit",
                    aid, sh_val, fit_val,
                )
                continue

            skel = extract_expression_skeleton(expr)
            if skel in submitted_skels:
                continue  # Skip same skeleton only AFTER a sibling succeeded this round
            if skel and skel in failed_corr_skels:
                # P36 + P43: Fast-skip clones whose structural skeleton already failed SELF_CORRELATION and prune from backup
                try:
                    self._record_submit_rejection_and_prune(aid, "FAILED_CORR", "SKELETON_SELF_CORR_CLONE")
                    mutated = True
                except Exception as _skip_exc:
                    logger.debug("Could not mark clone FAILED_CORR for %s: %s", aid, _skip_exc)
                continue

            # Ensure formula is permanently archived before submitting
            self.archive_qualified_alpha(aid, expr, sh_val, fit_val, to_val, mg_val, source="PRE_SUBMIT")

            res = self.submit_alpha(aid, max_daily_submissions=limit, enforce_checks=True, day_str=day)
            results.append(res)
            status_val = str(res.get("status") or "")
            now_iso = datetime.now(timezone.utc).isoformat()
            if res.get("success"):
                submitted_count += 1
                submitted_skels.add(skel)
                mutated = True
                try:
                    c_ok = sqlite3.connect(db_p, timeout=10.0)
                    c_ok.execute("UPDATE brain_candidate_pool SET status = 'SUBMITTED' WHERE alpha_id = ?", (aid,))
                    c_ok.execute(
                        "UPDATE qualified_alpha_archive SET submit_status = 'SUBMITTED', updated_at = ? WHERE alpha_id = ?",
                        (now_iso, aid),
                    )
                    c_ok.commit()
                    c_ok.close()
                except Exception as _ok_exc:
                    logger.debug("Could not mark candidate SUBMITTED for %s: %s", aid, _ok_exc)
            elif status_val in ("DAILY_QUOTA_REACHED", "AUTH_FAILED", "OFFLINE"):
                break
            elif status_val in ("SUBMISSION_CHECK_FAILED", "SUBMIT_REJECTED"):
                err_txt = str(res.get("error") or "").upper()
                is_corr = bool(res.get("corr_rejected")) or ("CORRELATION" in err_txt) or ("CORR" in err_txt)
                if is_corr and skel:
                    failed_corr_skels.add(skel)
                new_status = "FAILED_CORR" if is_corr else "FAILED_CHECKS"
                mutated = True
                try:
                    self._record_submit_rejection_and_prune(aid, new_status, str(res.get("error") or status_val))
                except Exception as _silent_exc:
                    logger.debug("worldquant_brain/brain_client.py:L565 suppressed exception: %s", _silent_exc)
        if mutated:
            self._sync_vault_exports()
        return results

    def fetch_self_correlation(self, alpha_id: str, max_polls: int = 5) -> dict[str, Any]:
        """Query /alphas/{alpha_id}/correlations/self (polling Retry-After & parsing records/max) and /alphas/{alpha_id}/check."""
        if self.offline_mode or not alpha_id:
            return {"checked": False, "passed": None, "self_correlation": None, "prod_correlation": None, "reason": "OFFLINE_OR_EMPTY"}
        if not self.is_authenticated and not self.authenticate():
            return {"checked": False, "passed": None, "self_correlation": None, "prod_correlation": None, "reason": "AUTH_FAILED"}
        self_corr = None
        prod_corr = None
        passed = True
        reasons: list[str] = []

        # 1. Poll /alphas/{alpha_id}/correlations/self (BRAIN often returns Retry-After while computing)
        try:
            url_corr = f"{self.BASE_URL}/alphas/{alpha_id}/correlations/self"
            for _ in range(max(1, int(max_polls))):
                r_corr = self.session.get(url_corr, timeout=min(self.timeout, 15.0))
                retry_hdr = r_corr.headers.get("Retry-After") if hasattr(r_corr, "headers") and r_corr.headers else None
                if r_corr.status_code == 200 and getattr(r_corr, "content", None):
                    cdata = r_corr.json() or {}
                    if isinstance(cdata, dict):
                        val = cdata.get("max") if cdata.get("max") is not None else cdata.get("value")
                        if val is not None:
                            self_corr = float(val)
                        elif isinstance(cdata.get("records"), list) and cdata["records"]:
                            vals = []
                            for row in cdata["records"]:
                                if isinstance(row, (list, tuple)) and len(row) >= 2:
                                    try:
                                        vals.append(float(row[1]))
                                    except (TypeError, ValueError):
                                        pass
                                elif isinstance(row, dict) and row.get("correlation") is not None:
                                    try:
                                        vals.append(float(row["correlation"]))
                                    except (TypeError, ValueError):
                                        pass
                            if vals:
                                self_corr = max(vals)
                        if self_corr is not None or not retry_hdr:
                            break
                if retry_hdr:
                    try:
                        time.sleep(min(float(retry_hdr), 3.0))
                    except Exception:
                        time.sleep(1.0)
                else:
                    break
        except Exception as _silent_exc:
            logger.debug("worldquant_brain/brain_client.py:L616 suppressed exception: %s", _silent_exc)

        # 2. Poll /alphas/{alpha_id}/check (official submission check endpoint with SELF_CORRELATION / PROD_CORRELATION)
        try:
            url_chk = f"{self.BASE_URL}/alphas/{alpha_id}/check"
            for _ in range(max(1, int(max_polls))):
                r_chk = self.session.get(url_chk, timeout=min(self.timeout, 15.0))
                retry_hdr = r_chk.headers.get("Retry-After") if hasattr(r_chk, "headers") and r_chk.headers else None
                if r_chk.status_code == 200 and getattr(r_chk, "content", None):
                    j = r_chk.json() or {}
                    is_blk = j.get("is") if isinstance(j.get("is"), dict) else j
                    checks = is_blk.get("checks") if isinstance(is_blk, dict) else None
                    if isinstance(checks, list):
                        for chk_item in checks:
                            if not isinstance(chk_item, dict):
                                continue
                            cname = str(chk_item.get("name") or "").upper()
                            cres = str(chk_item.get("result") or "").upper()
                            cval = chk_item.get("value")
                            if "SELF_CORRELATION" in cname and cval is not None and self_corr is None:
                                try:
                                    self_corr = float(cval)
                                except (TypeError, ValueError):
                                    pass
                            if "PROD_CORRELATION" in cname and cval is not None:
                                try:
                                    prod_corr = float(cval)
                                except (TypeError, ValueError):
                                    pass
                            if "CORRELATION" in cname and cres in ("FAIL", "FAILED", "ERROR"):
                                passed = False
                                reasons.append(f"{cname}={cres} (value={cval})")
                        break
                if retry_hdr:
                    try:
                        time.sleep(min(float(retry_hdr), 3.0))
                    except Exception:
                        time.sleep(1.0)
                else:
                    break
        except Exception as _silent_exc:
            logger.debug("worldquant_brain/brain_client.py:L657 suppressed exception: %s", _silent_exc)

        # 3. Fallback to /alphas/{alpha_id} IS-checks
        chk = self.verify_alpha_for_submission(alpha_id)
        for r in chk.get("reasons", []):
            if "CORRELATION" in str(r).upper() and str(r) not in reasons:
                passed = False
                reasons.append(str(r))
        corr_limit = 0.50 if getattr(self, "power_pool_mode", False) else 0.70
        if self_corr is not None and self_corr >= corr_limit:
            passed = False
            msg = f"SELF_CORRELATION={self_corr:.4f} >= {corr_limit:.2f}"
            if msg not in reasons:
                reasons.append(msg)
        if prod_corr is not None and prod_corr >= 0.70:
            passed = False
            msg = f"PROD_CORRELATION={prod_corr:.4f} >= 0.70"
            if msg not in reasons:
                reasons.append(msg)
        return {
            "checked": True,
            "alpha_id": alpha_id,
            "passed": passed,
            "self_correlation": self_corr,
            "prod_correlation": prod_corr,
            "reasons": reasons,
        }

    def verify_alpha_for_submission(self, alpha_id: str) -> dict[str, Any]:
        """Verify official /alphas/{alpha_id} and /alphas/{alpha_id}/check IS-checks (0 FAILs) before calling /submit."""
        try:
            resp = self.session.get(f"{self.BASE_URL}/alphas/{alpha_id}", timeout=self.timeout)
        except Exception as e:
            return {"passed": False, "status": "CHECK_FETCH_ERROR", "reasons": [mask_sensitive_data(str(e))]}

        # Also inspect /alphas/{alpha_id}/check for authoritative SELF_CORRELATION / PROD_CORRELATION
        check_reasons: list[str] = []
        has_side_effect = getattr(getattr(self.session, "get", None), "side_effect", None) is not None
        try:
            for _poll_chk in range(8):
                r_chk = self.session.get(f"{self.BASE_URL}/alphas/{alpha_id}/check", timeout=min(self.timeout, 15.0))
                retry_hdr = r_chk.headers.get("Retry-After") if hasattr(r_chk, "headers") and r_chk.headers else None
                if isinstance(getattr(r_chk, "status_code", None), int) and r_chk.status_code == 200 and getattr(r_chk, "content", None):
                    cj = r_chk.json() or {}
                    c_is = cj.get("is") if isinstance(cj.get("is"), dict) else cj
                    c_list = c_is.get("checks") if isinstance(c_is, dict) else None
                    if isinstance(c_list, list):
                        for chk in c_list:
                            if isinstance(chk, dict) and str(chk.get("result") or "").upper() in ("FAIL", "FAILED", "ERROR"):
                                cname = str(chk.get("name") or "UNKNOWN_CHECK")
                                cval = chk.get("value")
                                check_reasons.append(f"{cname}={chk.get('result')} (value={cval})")
                        break
                    if not retry_hdr:
                        break
                if retry_hdr:
                    try:
                        time.sleep(min(float(retry_hdr), 3.0))
                    except Exception:
                        time.sleep(1.0)
                elif isinstance(getattr(r_chk, "status_code", None), int) and r_chk.status_code == 200 and not getattr(r_chk, "content", None) and not has_side_effect:
                    time.sleep(1.0)
                else:
                    break
        except Exception as _silent_exc:
            logger.debug("worldquant_brain/brain_client.py:L716 suppressed exception: %s", _silent_exc)

        if not isinstance(getattr(resp, "status_code", None), int) or resp.status_code != 200 or not getattr(resp, "content", None):
            return {
                "passed": False,
                "status": "SUBMISSION_CHECK_FAILED",
                "reasons": check_reasons or [f"ALPHA_DETAIL_UNAVAILABLE_HTTP_{getattr(resp, 'status_code', 'ERR')}"],
            }

        try:
            data = resp.json()
        except Exception:
            return {
                "passed": False,
                "status": "SUBMISSION_CHECK_FAILED",
                "reasons": check_reasons or ["ALPHA_DETAIL_NON_JSON"],
            }

        if not isinstance(data, dict):
            return {
                "passed": False,
                "status": "SUBMISSION_CHECK_FAILED",
                "reasons": check_reasons or ["ALPHA_DETAIL_NON_DICT"],
            }

        is_block = data.get("is") if isinstance(data.get("is"), dict) else data
        reasons: list[str] = list(check_reasons)

        checks = is_block.get("checks") if isinstance(is_block, dict) else None
        has_pending_self_corr = False
        if isinstance(checks, list):
            failed_checks = []
            for chk in checks:
                if isinstance(chk, dict):
                    res_str = str(chk.get("result") or "").upper()
                    name_str = str(chk.get("name") or "UNKNOWN_CHECK")
                    if res_str in ("FAIL", "FAILED", "ERROR"):
                        val_info = f" (value={chk.get('value')}, limit={chk.get('limit')})" if "value" in chk else ""
                        failed_checks.append(f"{name_str}={res_str}{val_info}")
                    elif name_str == "SELF_CORRELATION" and res_str == "PENDING":
                        has_pending_self_corr = True
            if failed_checks:
                reasons.append("Official IS-Check failed: " + ", ".join(failed_checks))
        if has_pending_self_corr and not check_reasons and not has_side_effect:
            corr_res = self.fetch_self_correlation(alpha_id, max_polls=6)
            if corr_res.get("checked") and corr_res.get("passed") is False:
                for cr in corr_res.get("reasons", []):
                    if cr not in reasons:
                        reasons.append(cr)

        sharpe = self._number(is_block, "sharpe")
        fitness = self._number(is_block, "fitness")
        turnover = self._number(is_block, "turnover")
        margin = self._number(is_block, "margin")

        settings_blk = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        try:
            delay_val = int(settings_blk.get("delay", 1))
        except (TypeError, ValueError):
            delay_val = 1
        min_sharpe = 2.00 if delay_val == 0 else 1.25
        min_fitness = 1.30 if delay_val == 0 else 1.00  # Official BRAIN KB 14220836385687
        if sharpe is not None and sharpe < min_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < {min_sharpe:.2f} (delay={delay_val})")
        if fitness is not None and fitness < min_fitness:
            reasons.append(f"Fitness {fitness:.2f} < {min_fitness:.2f} (delay={delay_val})")
        neut_val = str(settings_blk.get("neutralization") or "").strip().upper()
        reg_expr = str((data.get("regular") or {}).get("code") or data.get("expression") or "").lower()
        if neut_val == "NONE" and "group_neutralize(" not in reg_expr and "group_normalize(" not in reg_expr:
            reasons.append("NEUTRALIZATION_NONE_WITHOUT_GROUP_NEUTRALIZE (Official BRAIN KB 13306223024151)")
        if turnover is not None:
            norm_to = turnover / 100.0 if 2.0 < turnover <= 70.0 else turnover
            if not (0.01 <= norm_to <= 0.70):
                reasons.append(f"Turnover {norm_to:.4f} outside [0.01, 0.70]")
        if margin is not None:
            min_margin = float(os.getenv("BRAIN_MIN_MARGIN", "0.0005") or "0.0005")
            norm_margin = margin / 10000.0 if margin > 0.5 else margin
            if norm_margin < min_margin:
                reasons.append(f"Margin {norm_margin:.5f} < {min_margin:.4f} ({min_margin * 10000:.1f} bps)")

        if len(reasons) == 0 and not has_side_effect and not self.offline_mode:
            region_val = str(settings_blk.get("region") or "USA").upper()
            local_sc = self.check_local_active_self_correlation(alpha_id, region=region_val, threshold=0.70)
            if local_sc.get("blocked"):
                reasons.append(
                    f"LOCAL_ACTIVE_SELF_CORRELATION={local_sc['max_corr']:.4f} >= 0.70 (vs {local_sc['matched_alpha_id']})"
                )

        return {
            "passed": len(reasons) == 0,
            "status": "PASSED" if len(reasons) == 0 else "SUBMISSION_CHECK_FAILED",
            "reasons": reasons,
            "metrics": {"sharpe": sharpe, "fitness": fitness, "turnover": turnover, "margin": margin},
        }

    def submit_alpha(
        self,
        alpha_id: str,
        max_wait_seconds: float = 60.0,
        max_daily_submissions: int | None = None,
        enforce_checks: bool = True,
        day_str: str | None = None,
    ) -> dict[str, Any]:
        """Submit a simulated alpha to WorldQuant BRAIN (/alphas/{alpha_id}/submit) with strict IS-check gate and daily quota cap (default 2/day)."""
        if not alpha_id or not str(alpha_id).strip():
            return {"success": False, "status": "INVALID_ALPHA_ID", "error": "alpha_id is required"}

        limit = int(max_daily_submissions if max_daily_submissions is not None else os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2"))
        today_cnt = self.get_daily_submission_count(day_str)
        if today_cnt >= limit:
            return {
                "success": False,
                "status": "DAILY_QUOTA_REACHED",
                "alpha_id": alpha_id,
                "daily_limit": limit,
                "submitted_today": today_cnt,
                "error": f"已達每日提交上限 ({today_cnt}/{limit})，今日不再送出",
            }

        if self.offline_mode:
            return {
                "success": False,
                "status": "OFFLINE",
                "alpha_id": alpha_id,
                "error": "BRAIN_EMAIL/BRAIN_PASSWORD 未設定，未送出任何網路請求。",
            }
        if not self.is_authenticated and not self.authenticate():
            return {"success": False, "status": "AUTH_FAILED", "alpha_id": alpha_id, "error": "BRAIN 登入失敗"}

        if enforce_checks:
            chk = self.verify_alpha_for_submission(str(alpha_id).strip())
            if not chk.get("passed", True):
                return {
                    "success": False,
                    "status": "SUBMISSION_CHECK_FAILED",
                    "alpha_id": alpha_id,
                    "reasons": chk.get("reasons", []),
                    "error": "未通過 WorldQuant BRAIN 提交前檢核: " + "; ".join(chk.get("reasons", [])),
                }

        url = f"{self.BASE_URL}/alphas/{alpha_id}/submit"
        deadline = time.monotonic() + float(max(max_wait_seconds, 90.0))
        resp = self.session.post(url, timeout=self.timeout)

        # Only re-authenticate on 401 (or 403 explicitly complaining about auth credentials)
        if resp.status_code == 401 or (
            resp.status_code == 403
            and any(tok in (resp.text or "").lower() for tok in ("authentication", "credentials", "unauthorized"))
        ):
            self.authenticated = False
            if self.authenticate():
                resp = self.session.post(url, timeout=self.timeout)

        if resp.status_code == 429:
            time.sleep(_parse_retry_after(resp.headers.get("Retry-After"), default=self.poll_interval, cap=15.0))
            resp = self.session.post(url, timeout=self.timeout)

        def _extract_failed_checks(payload: Any) -> list[str]:
            if not isinstance(payload, dict):
                return []
            checks: list[Any] = []
            is_blk = payload.get("is")
            if isinstance(is_blk, dict) and isinstance(is_blk.get("checks"), list):
                checks.extend(is_blk["checks"])
            if isinstance(payload.get("checks"), list):
                checks.extend(payload["checks"])
            failed: list[str] = []
            for chk in checks:
                if isinstance(chk, dict) and str(chk.get("result") or "").upper() in ("FAIL", "FAILED", "ERROR"):
                    failed.append(str(chk.get("name") or chk.get("message") or "CHECK_FAILED"))
            return failed

        def _verify_not_unsubmitted() -> tuple[bool, str, dict[str, Any]]:
            try:
                g = self.session.get(f"{self.BASE_URL}/alphas/{alpha_id}", timeout=min(self.timeout, 15.0))
                if g.status_code == 200 and g.content:
                    rdata = g.json() if hasattr(g, "json") else {}
                    if isinstance(rdata, dict) and rdata.get("status"):
                        rst = str(rdata.get("status") or "").upper()
                        if rst in ("UNSUBMITTED", "REJECTED", "FAIL", "FAILED"):
                            return False, rst, rdata
            except Exception as _silent_exc:
                logger.debug("worldquant_brain/brain_client.py verify_not_unsubmitted exception: %s", _silent_exc)
            return True, "", {}

        # Immediate 400/403 rule rejection on POST (preserve exact BRAIN rejection reasons without re-auth)
        if resp.status_code in (400, 403):
            try:
                err_data = resp.json() if getattr(resp, "content", None) else {}
            except Exception:
                err_data = {}
            failed_checks = _extract_failed_checks(err_data)
            detail_msg = str(err_data.get("detail") or "") if isinstance(err_data, dict) else ""
            corr_rej = any("CORR" in c.upper() for c in failed_checks) or ("CORR" in detail_msg.upper())
            return {
                "success": False,
                "status": "SUBMIT_REJECTED",
                "alpha_id": alpha_id,
                "http_status": resp.status_code,
                "failed_checks": failed_checks,
                "corr_rejected": corr_rej,
                "error": "BRAIN 提交被拒: " + (", ".join(failed_checks) if failed_checks else (detail_msg or f"HTTP {resp.status_code}")),
                "raw": err_data,
            }

        # CRITICAL FIX: BRAIN POST /alphas/{id}/submit returns 201/202 with Retry-After (often WITHOUT Location header).
        # Poll `Location` if present, otherwise poll `GET /alphas/{alpha_id}/submit` until Retry-After clears!
        location = resp.headers.get("Location") or resp.headers.get("location")
        init_retry = resp.headers.get("Retry-After") if hasattr(resp, "headers") and resp.headers else None
        needs_poll = (
            (resp.status_code in (200, 201, 202) and bool(location))
            or resp.status_code == 202
            or (resp.status_code == 201 and (bool(init_retry) or not getattr(resp, "content", None)))
        )

        if needs_poll:
            poll_url = (f"{self.BASE_URL}{location}" if location and location.startswith("/") else (location or url))
            if init_retry:
                time.sleep(_parse_retry_after(init_retry, default=self.poll_interval, cap=10.0))
            while time.monotonic() < deadline:
                poll = self.session.get(poll_url, timeout=self.timeout)
                if poll.status_code == 401:
                    self.authenticated = False
                    if self.authenticate():
                        continue
                    return {"success": False, "status": "AUTH_FAILED", "alpha_id": alpha_id, "error": "輪詢期間登入失效"}
                if poll.status_code == 429:
                    time.sleep(_parse_retry_after(poll.headers.get("Retry-After"), default=self.poll_interval, cap=15.0))
                    continue
                if poll.status_code in (200, 201, 202):
                    retry_hdr = poll.headers.get("Retry-After") if hasattr(poll, "headers") and poll.headers else None
                    if retry_hdr and (not getattr(poll, "content", None) or poll.status_code in (201, 202)):
                        time.sleep(_parse_retry_after(retry_hdr, default=self.poll_interval, cap=10.0))
                        continue
                    data = poll.json() if getattr(poll, "content", None) else {}
                    if isinstance(data, dict) and data.get("detail") and not data.get("status") and not data.get("state"):
                        detail_msg = str(data.get("detail") or "")
                        return {
                            "success": False,
                            "status": "SUBMIT_REJECTED",
                            "alpha_id": alpha_id,
                            "failed_checks": [],
                            "corr_rejected": ("CORR" in detail_msg.upper()),
                            "error": f"BRAIN 提交被拒: {detail_msg}",
                            "raw": data,
                        }
                    failed_checks = _extract_failed_checks(data)
                    st = str(
                        (data.get("status") or data.get("state") or ("PENDING" if poll.status_code == 202 else "SUBMITTED"))
                        if isinstance(data, dict)
                        else ""
                    ).upper()
                    if failed_checks or st in ("FAIL", "FAILED", "ERROR"):
                        corr_rej = any("CORR" in c.upper() for c in failed_checks) or ("CORR" in str(data).upper() and bool(failed_checks))
                        return {
                            "success": False,
                            "status": "SUBMIT_REJECTED",
                            "alpha_id": alpha_id,
                            "failed_checks": failed_checks,
                            "corr_rejected": corr_rej,
                            "error": "BRAIN 提交被拒: " + (", ".join(failed_checks) if failed_checks else st),
                            "raw": data,
                        }
                    if st in ("COMPLETE", "COMPLETED", "SUBMITTED", "PASS", "ACTIVE") or (poll.status_code == 200 and not retry_hdr):
                        ok_remote, remote_st, remote_raw = _verify_not_unsubmitted()
                        if not ok_remote:
                            r_failed = _extract_failed_checks(remote_raw)
                            return {
                                "success": False,
                                "status": "SUBMIT_REJECTED",
                                "alpha_id": alpha_id,
                                "failed_checks": r_failed or [f"REMOTE_STATUS_{remote_st}"],
                                "corr_rejected": any("CORR" in c.upper() for c in r_failed),
                                "error": f"官網狀態仍為 {remote_st}，未正式送出",
                                "raw": remote_raw or data,
                            }
                        self._record_daily_submission(alpha_id, day_str=day_str)
                        return {"success": True, "status": "SUBMITTED", "alpha_id": alpha_id, "raw": data}
                elif poll.status_code >= 400:
                    try:
                        err_data = poll.json() if getattr(poll, "content", None) else {}
                    except Exception:
                        err_data = {}
                    failed_checks = _extract_failed_checks(err_data)
                    corr_rej = any("CORR" in c.upper() for c in failed_checks) or ("CORR" in str(err_data).upper())
                    is_rejected = bool(failed_checks) or poll.status_code in (400, 403)
                    return {
                        "success": False,
                        "status": "SUBMIT_REJECTED" if is_rejected else "SUBMIT_ERROR",
                        "alpha_id": alpha_id,
                        "http_status": poll.status_code,
                        "failed_checks": failed_checks,
                        "corr_rejected": corr_rej,
                        "error": "BRAIN 提交被拒: " + (", ".join(failed_checks) if failed_checks else f"HTTP {poll.status_code}"),
                        "raw": err_data,
                    }
                time.sleep(min(self.poll_interval, 5.0))
            return {
                "success": False,
                "status": "SUBMIT_TIMEOUT",
                "alpha_id": alpha_id,
                "error": "BRAIN 提交輪詢逾時，請稍後查詢官網狀態避免重複送出",
            }
        if resp.status_code in (200, 201):
            data = resp.json() if resp.content else {}
            failed_checks = _extract_failed_checks(data)
            if failed_checks:
                corr_rej = any("CORR" in c.upper() for c in failed_checks)
                return {
                    "success": False,
                    "status": "SUBMIT_REJECTED",
                    "alpha_id": alpha_id,
                    "failed_checks": failed_checks,
                    "corr_rejected": corr_rej,
                    "error": "BRAIN 提交被拒: " + ", ".join(failed_checks),
                    "raw": data,
                }
            ok_remote, remote_st, remote_raw = _verify_not_unsubmitted()
            if not ok_remote:
                r_failed = _extract_failed_checks(remote_raw)
                return {
                    "success": False,
                    "status": "SUBMIT_REJECTED",
                    "alpha_id": alpha_id,
                    "failed_checks": r_failed or [f"REMOTE_STATUS_{remote_st}"],
                    "corr_rejected": any("CORR" in c.upper() for c in r_failed),
                    "error": f"官網狀態仍為 {remote_st}，未正式送出",
                    "raw": remote_raw or data,
                }
            self._record_daily_submission(alpha_id, day_str=day_str)
            return {"success": True, "status": "SUBMITTED", "alpha_id": alpha_id, "raw": data}
        return {
            "success": False,
            "status": "SUBMIT_ERROR",
            "alpha_id": alpha_id,
            "http_status": resp.status_code,
            "error": mask_sensitive_data((resp.text or "")[:300]),
        }

    def fetch_challenge_score(self) -> dict[str, Any]:
        """Fetch authenticated user's Challenge score, daily score, rank, and progress toward 10,000 points
        corresponding to https://platform.worldquantbrain.com/competition/challenge.
        """
        if self.offline_mode:
            return {
                "success": False,
                "status": "OFFLINE",
                "error": "BRAIN_EMAIL/BRAIN_PASSWORD 未設定，無法連線抓取 Challenge 分數",
            }
        if not self.is_authenticated and not self.authenticate():
            return {"success": False, "status": "AUTH_FAILED", "error": "BRAIN 登入認證失敗"}

        candidate_endpoints = (
            f"{self.BASE_URL}/users/self/competitions/challenge",
            f"{self.BASE_URL}/competitions/challenge",
            f"{self.BASE_URL}/users/self",
        )
        merged_payload: dict[str, Any] = {}
        last_status = 0
        hit_endpoint = None

        for url in candidate_endpoints:
            try:
                resp = self.session.get(url, timeout=self.timeout)
                last_status = resp.status_code
                if resp.status_code in (401, 403):
                    self.authenticated = False
                    if self.authenticate():
                        resp = self.session.get(url, timeout=self.timeout)
                        last_status = resp.status_code
                if resp.status_code == 200 and resp.content:
                    data = resp.json()
                    if isinstance(data, dict):
                        merged_payload.update(data)
                        if hit_endpoint is None:
                            hit_endpoint = url
                        # Stop early if we found explicit score fields
                        if any(k in data for k in ("totalScore", "total_score", "score", "dailyScore", "daily_score")):
                            break
            except Exception:
                continue

        if not merged_payload:
            return {
                "success": False,
                "status": "ENDPOINT_UNAVAILABLE",
                "http_status": last_status,
                "error": f"無法從 BRAIN Challenge API 取得分數資料 (HTTP {last_status})",
            }

        return {
            "success": True,
            "status": "OK",
            "endpoint": hit_endpoint,
            "payload": merged_payload,
        }



    def ensure_novelty_survey_tables(self, conn) -> None:
        """Create the 6 idempotent novelty, survey, budget, and correlation precheck tables."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public_expression_denylist (
                exact_hash TEXT PRIMARY KEY,
                normalized_hash TEXT NOT NULL,
                structural_hash TEXT NOT NULL,
                expression TEXT NOT NULL,
                source_title TEXT,
                source_url TEXT,
                license TEXT,
                fields_json TEXT,
                operators_json TEXT,
                family TEXT,
                imported_at TEXT NOT NULL
            )
            """
        )
        ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS public_expression_denylist (\n                exact_hash TEXT PRIMARY KEY,\n                normalized_hash TEXT NOT NULL,\n                structural_hash TEXT NOT NULL,\n                expression TEXT NOT NULL,\n                source_title TEXT,\n                source_url TEXT,\n                license TEXT,\n                fields_json TEXT,\n                operators_json TEXT,\n                family TEXT,\n                imported_at TEXT NOT NULL\n            )\n            ')
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alpha_fingerprints (
                alpha_id TEXT PRIMARY KEY,
                exact_hash TEXT NOT NULL,
                normalized_hash TEXT NOT NULL,
                structural_hash TEXT NOT NULL,
                fields_json TEXT,
                operators_json TEXT,
                family TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS alpha_fingerprints (\n                alpha_id TEXT PRIMARY KEY,\n                exact_hash TEXT NOT NULL,\n                normalized_hash TEXT NOT NULL,\n                structural_hash TEXT NOT NULL,\n                fields_json TEXT,\n                operators_json TEXT,\n                family TEXT,\n                created_at TEXT NOT NULL\n            )\n            ')
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS field_catalog (
                field_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                category TEXT,
                field_type TEXT,
                description TEXT,
                region TEXT,
                universe TEXT,
                delay INTEGER,
                coverage REAL,
                alpha_count INTEGER,
                user_count INTEGER,
                update_frequency TEXT,
                availability TEXT,
                metadata_hash TEXT NOT NULL,
                last_probed_at TEXT,
                vector_reducer_required INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS field_catalog (\n                field_id TEXT PRIMARY KEY,\n                dataset_id TEXT,\n                category TEXT,\n                field_type TEXT,\n                description TEXT,\n                region TEXT,\n                universe TEXT,\n                delay INTEGER,\n                coverage REAL,\n                alpha_count INTEGER,\n                user_count INTEGER,\n                update_frequency TEXT,\n                availability TEXT,\n                metadata_hash TEXT NOT NULL,\n                last_probed_at TEXT,\n                vector_reducer_required INTEGER NOT NULL DEFAULT 0\n            )\n            ')
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS field_probe_cache (
                field_id TEXT PRIMARY KEY,
                metadata_hash TEXT NOT NULL,
                long_count INTEGER,
                short_count INTEGER,
                turnover REAL,
                nan_ratio REAL,
                probed_at TEXT NOT NULL
            )
            """
        )
        ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS field_probe_cache (\n                field_id TEXT PRIMARY KEY,\n                metadata_hash TEXT NOT NULL,\n                long_count INTEGER,\n                short_count INTEGER,\n                turnover REAL,\n                nan_ratio REAL,\n                probed_at TEXT NOT NULL\n            )\n            ')
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_queue_decisions (
                decision_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                blueprint_family TEXT,
                admitted INTEGER NOT NULL,
                reason TEXT NOT NULL,
                priority_score REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS simulation_queue_decisions (\n                decision_id TEXT PRIMARY KEY,\n                candidate_id TEXT NOT NULL,\n                blueprint_family TEXT,\n                admitted INTEGER NOT NULL,\n                reason TEXT NOT NULL,\n                priority_score REAL,\n                created_at TEXT NOT NULL\n            )\n            ')
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS correlation_prechecks (
                precheck_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                local_corr REAL NOT NULL,
                shares_raw_fields INTEGER NOT NULL DEFAULT 0,
                decision TEXT NOT NULL,
                allowed_for_brain INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        ensure_schema_from_ddl(conn, '\n            CREATE TABLE IF NOT EXISTS correlation_prechecks (\n                precheck_id TEXT PRIMARY KEY,\n                candidate_id TEXT NOT NULL,\n                local_corr REAL NOT NULL,\n                shares_raw_fields INTEGER NOT NULL DEFAULT 0,\n                decision TEXT NOT NULL,\n                allowed_for_brain INTEGER NOT NULL,\n                created_at TEXT NOT NULL\n            )\n            ')
        try:
            from worldquant_brain.alpha_novelty import seed_public_denylist
            seed_public_denylist(conn)
        except Exception as _silent_exc:
            logger.debug("worldquant_brain/brain_client.py:L1152 suppressed exception: %s", _silent_exc)

    def fetch_data_fields(
        self,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Fetch available data fields from WorldQuant BRAIN /data-fields endpoint."""
        if self.offline_mode:
            return []
        if not self.is_authenticated and not self.authenticate():
            return []
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/data-fields",
                params={
                    "instrumentType": "EQUITY",
                    "region": region,
                    "universe": universe,
                    "delay": int(delay),
                    "limit": int(limit),
                    "offset": int(offset),
                },
                timeout=min(self.timeout, 20.0),
            )
            if resp.status_code != 200:
                return []
            payload = resp.json() or {}
            results = payload.get("results") if isinstance(payload, dict) else []
            return list(results) if isinstance(results, list) else []
        except Exception:
            return []

    def fetch_alpha_pnl(self, alpha_id: str) -> list[float]:
        """Fetch daily PnL series from /alphas/{alpha_id}/recordsets/pnl for local correlation prechecks."""
        if self.offline_mode or not alpha_id:
            return []
        if not self.is_authenticated and not self.authenticate():
            return []
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/alphas/{alpha_id}/recordsets/pnl",
                timeout=min(self.timeout, 20.0),
            )
            if resp.status_code != 200:
                return []
            data = resp.json() or {}
            records = data.get("records") if isinstance(data, dict) else []
            out: list[float] = []
            for row in (records or []):
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    try:
                        out.append(float(row[1]))
                    except (TypeError, ValueError):
                        pass
            return out
        except Exception:
            return []


WorldQuantBrainClient = BrainClient
