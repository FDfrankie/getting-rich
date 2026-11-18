"""WorldQuant BRAIN API client with asynchronous simulation polling, exponential backoff, and strict security logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import traceback
from collections.abc import Mapping
from typing import Any

import requests

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
        self.email = email or os.getenv("BRAIN_EMAIL") or os.getenv("BRAIN_USERNAME")
        self.password = password or os.getenv("BRAIN_PASSWORD")
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.session = session or requests.Session()
        self.authenticated = False
        self._submitted: dict[str, dict[str, Any]] = {}

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

    def simulate_alpha(
        self,
        expression: str,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "type": "REGULAR",
            "settings": dict(settings or {}),
            "regular": expression,
        }
        return self.submit_and_poll_simulation(payload)

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

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2**attempt))
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
            except Exception:
                pass
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

        # Poll simulation
        while True:
            if time.monotonic() > deadline:
                logger.error("WorldQuant simulation polling TIMEOUT; Location=%s", location)
                raise TimeoutError(f"BRAIN simulation polling 超過 {max_wait_seconds} 秒；Location={location}")

            t0 = time.monotonic()
            poll = self.session.get(location, timeout=self.timeout)
            elapsed = time.monotonic() - t0

            if poll.status_code == 429:
                retry_header = poll.headers.get("Retry-After")
                sleep_sec = float(retry_header) if retry_header else interval
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
                
                # If metrics are not in the simulation object, fetch the alpha object
                alpha_data = {}
                if last.get("alpha"):
                    try:
                        alpha_url = f"{self.BASE_URL}/alphas/{last['alpha']}"
                        alpha_resp = self.session.get(alpha_url, timeout=self.timeout)
                        if alpha_resp.status_code == 200:
                            alpha_data = alpha_resp.json()
                    except Exception as e:
                        logger.warning("Failed to fetch alpha record %s: %s", last["alpha"], e)

                metrics_source = dict(last)
                if alpha_data:
                    metrics_source.update(alpha_data)
                    if "is" in alpha_data and isinstance(alpha_data["is"], dict):
                        metrics_source.update(alpha_data["is"])

                sharpe = self._number(metrics_source, "sharpe")
                fitness = self._number(metrics_source, "fitness")
                turnover = self._number(metrics_source, "turnover")
                margin = self._number(metrics_source, "margin")

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
                    "message": "Alpha 模擬成功完成"
                    if has_valid_metrics
                    else "模擬完成但未取得有效 Sharpe/Fitness 指標",
                    "error": None if has_valid_metrics else "Missing Sharpe or Fitness in simulation result",
                    "raw": last,
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
        return getattr(self, "quota_db_path", None) or os.path.join(os.path.dirname(__file__), "alpha_vault.db")

    def _today_taipei(self) -> str:
        from datetime import datetime, timedelta, timezone
        return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

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
                conn.execute(
                    "INSERT OR IGNORE INTO brain_daily_submissions (alpha_id, submitted_date, created_at) VALUES (?, ?, ?)",
                    (str(alpha_id), day, now_iso),
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
                )
                conn.execute("UPDATE brain_candidate_pool SET status = 'SUBMITTED' WHERE alpha_id = ?", (str(alpha_id),))
        finally:
            conn.close()

    def stage_candidate_for_daily_selection(
        self,
        alpha_id: str,
        expression: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        margin: float = 0.0,
        day_str: str | None = None,
    ) -> None:
        import sqlite3
        from datetime import datetime, timezone
        day = day_str or self._today_taipei()
        import math
        norm_m = float(margin) / 10000.0 if float(margin) > 0.5 else float(margin)
        m_bps = norm_m * 10000.0
        s_score = max(0.0, min(1.0, (float(sharpe) - 1.0) / 1.5))
        f_score = max(0.0, min(1.0, (float(fitness) - 0.7) / 1.3))
        m_score = max(0.0, min(1.0, (m_bps - 5.0) / 20.0)) if m_bps > 0 else 0.30
        t_val = max(0.005, min(1.0, float(turnover) / 100.0 if float(turnover) > 2.0 else float(turnover)))
        t_bell = math.exp(-((math.log(t_val) - math.log(0.20)) ** 2) / (2.0 * (0.65 ** 2)))
        e_bonus = 0.85 if ("opt6_" in str(expression) or "scl12_" in str(expression) or "ts_regression" in str(expression)) else 0.35
        score = round(100.0 * (0.35 * s_score + 0.25 * f_score + 0.15 * m_score + 0.15 * t_bell + 0.10 * e_bonus) + float(sharpe) * 0.01, 4)
        now_iso = datetime.now(timezone.utc).isoformat()
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        conn = sqlite3.connect(db_p, timeout=10.0)
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
                )
                self.ensure_novelty_survey_tables(conn)
                from worldquant_brain.alpha_novelty import find_duplicate
                dup_chk = find_duplicate(expression, [])
                if dup_chk.blocked and dup_chk.reason == "PUBLIC_EXPRESSION_DENYLIST":
                    return
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

    def flush_daily_candidates(self, day_str: str | None = None, max_daily_submissions: int | None = None) -> list[dict[str, Any]]:
        """Submit top-scoring pending candidates from today's pool up to the remaining daily quota."""
        import sqlite3
        day = day_str or self._today_taipei()
        limit = int(max_daily_submissions if max_daily_submissions is not None else os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2"))
        remaining = max(0, limit - self.get_daily_submission_count(day))
        if remaining == 0:
            return []
        db_p = self._quota_db()
        if not os.path.exists(db_p):
            return []
        conn = sqlite3.connect(db_p, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS brain_candidate_pool (alpha_id TEXT PRIMARY KEY, expression TEXT NOT NULL, sharpe REAL NOT NULL, fitness REAL NOT NULL, turnover REAL NOT NULL, margin REAL NOT NULL, score REAL NOT NULL, candidate_date TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL)"
            )
            rows = conn.execute(
                "SELECT alpha_id FROM brain_candidate_pool WHERE candidate_date = ? AND status = 'PENDING' ORDER BY score DESC, sharpe DESC",
                (day,),
            ).fetchall()
        finally:
            conn.close()

        results: list[dict[str, Any]] = []
        submitted_count = 0
        for row in rows:
            if submitted_count >= remaining:
                break
            aid = str(row["alpha_id"])
            res = self.submit_alpha(aid, max_daily_submissions=limit, enforce_checks=True, day_str=day)
            results.append(res)
            status_val = str(res.get("status") or "")
            if res.get("success"):
                submitted_count += 1
            elif status_val == "DAILY_QUOTA_REACHED":
                break
            elif status_val == "SUBMISSION_CHECK_FAILED":
                err_txt = str(res.get("error") or "").upper()
                new_status = "FAILED_CORR" if ("CORRELATION" in err_txt or "CORR" in err_txt) else "FAILED_CHECKS"
                try:
                    c2 = sqlite3.connect(db_p, timeout=10.0)
                    c2.execute("UPDATE brain_candidate_pool SET status = ? WHERE alpha_id = ?", (new_status, aid))
                    c2.commit()
                    c2.close()
                except Exception:
                    pass
        return results

    def fetch_self_correlation(self, alpha_id: str) -> dict[str, Any]:
        """Query /alphas/{alpha_id}/correlations/self and /alphas/{alpha_id} IS-checks to detect self/prod correlation failures early."""
        if self.offline_mode or not alpha_id:
            return {"checked": False, "passed": None, "self_correlation": None, "prod_correlation": None, "reason": "OFFLINE_OR_EMPTY"}
        if not self.is_authenticated() and not self.authenticate():
            return {"checked": False, "passed": None, "self_correlation": None, "prod_correlation": None, "reason": "AUTH_FAILED"}
        self_corr = None
        prod_corr = None
        passed = True
        reasons: list[str] = []
        try:
            r_corr = self.session.get(f"{self.BASE_URL}/alphas/{alpha_id}/correlations/self", timeout=min(self.timeout, 15.0))
            if r_corr.status_code == 200:
                cdata = r_corr.json() or {}
                if isinstance(cdata, dict):
                    val = cdata.get("max") if cdata.get("max") is not None else cdata.get("value")
                    if val is not None:
                        self_corr = float(val)
        except Exception:
            pass
        chk = self.verify_alpha_for_submission(alpha_id)
        for r in chk.get("reasons", []):
            if "CORRELATION" in str(r).upper():
                passed = False
                reasons.append(str(r))
        if self_corr is not None and self_corr >= 0.70:
            passed = False
            reasons.append(f"SELF_CORRELATION={self_corr:.4f} >= 0.70")
        return {
            "checked": True,
            "alpha_id": alpha_id,
            "passed": passed,
            "self_correlation": self_corr,
            "prod_correlation": prod_corr,
            "reasons": reasons,
        }

    def verify_alpha_for_submission(self, alpha_id: str) -> dict[str, Any]:
        """Verify official /alphas/{alpha_id} IS-checks (0 FAILs) and metrics before calling /submit."""
        try:
            resp = self.session.get(f"{self.BASE_URL}/alphas/{alpha_id}", timeout=self.timeout)
        except Exception as e:
            return {"passed": False, "status": "CHECK_FETCH_ERROR", "reasons": [mask_sensitive_data(str(e))]}

        if not isinstance(getattr(resp, "status_code", None), int) or resp.status_code != 200 or not getattr(resp, "content", None):
            return {"passed": True, "status": "SKIPPED_NO_DETAIL", "reasons": []}

        try:
            data = resp.json()
        except Exception:
            return {"passed": True, "status": "SKIPPED_NON_JSON", "reasons": []}

        if not isinstance(data, dict):
            return {"passed": True, "status": "SKIPPED_NON_DICT", "reasons": []}

        is_block = data.get("is") if isinstance(data.get("is"), dict) else data
        reasons: list[str] = []

        checks = is_block.get("checks") if isinstance(is_block, dict) else None
        if isinstance(checks, list):
            failed_checks = []
            for chk in checks:
                if isinstance(chk, dict):
                    res_str = str(chk.get("result") or "").upper()
                    name_str = str(chk.get("name") or "UNKNOWN_CHECK")
                    if res_str in ("FAIL", "FAILED", "ERROR"):
                        val_info = f" (value={chk.get('value')}, limit={chk.get('limit')})" if "value" in chk else ""
                        failed_checks.append(f"{name_str}={res_str}{val_info}")
            if failed_checks:
                reasons.append("Official IS-Check failed: " + ", ".join(failed_checks))

        sharpe = self._number(is_block, "sharpe")
        fitness = self._number(is_block, "fitness")
        turnover = self._number(is_block, "turnover")
        margin = self._number(is_block, "margin")

        if sharpe is not None and sharpe < 1.25:
            reasons.append(f"Sharpe {sharpe:.2f} < 1.25")
        if fitness is not None and fitness < 1.0:
            reasons.append(f"Fitness {fitness:.2f} < 1.00")
        if turnover is not None:
            norm_to = turnover / 100.0 if 2.0 < turnover <= 70.0 else turnover
            if not (0.01 <= norm_to <= 0.70):
                reasons.append(f"Turnover {norm_to:.4f} outside [0.01, 0.70]")
        if margin is not None:
            norm_margin = margin / 10000.0 if margin > 0.5 else margin
            if norm_margin < 0.0010:
                reasons.append(f"Margin {norm_margin:.5f} < 0.0010 (10 bps)")

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
        deadline = time.monotonic() + float(max_wait_seconds)
        resp = self.session.post(url, timeout=self.timeout)
        if resp.status_code in (401, 403):
            self.authenticated = False
            if self.authenticate():
                resp = self.session.post(url, timeout=self.timeout)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After") or self.poll_interval)
            time.sleep(min(retry_after, 10.0))
            resp = self.session.post(url, timeout=self.timeout)

        location = resp.headers.get("Location") or resp.headers.get("location")
        if resp.status_code in (200, 201, 202) and location:
            if location.startswith("/"):
                location = f"{self.BASE_URL}{location}"
            while time.monotonic() < deadline:
                poll = self.session.get(location, timeout=self.timeout)
                if poll.status_code == 429:
                    time.sleep(float(poll.headers.get("Retry-After") or self.poll_interval))
                    continue
                if poll.status_code in (200, 201):
                    data = poll.json() if poll.content else {}
                    st = str(data.get("status") or data.get("state") or "SUBMITTED").upper()
                    if st in ("COMPLETE", "COMPLETED", "SUBMITTED", "PASS", "ACTIVE"):
                        self._record_daily_submission(alpha_id, day_str=day_str)
                        return {"success": True, "status": "SUBMITTED", "alpha_id": alpha_id, "raw": data}
                    if st in ("FAIL", "FAILED", "ERROR"):
                        return {"success": False, "status": "SUBMIT_REJECTED", "alpha_id": alpha_id, "raw": data}
                elif poll.status_code >= 400:
                    return {"success": False, "status": "SUBMIT_ERROR", "alpha_id": alpha_id, "http_status": poll.status_code}
                time.sleep(min(self.poll_interval, 5.0))
        if resp.status_code in (200, 201):
            data = resp.json() if resp.content else {}
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
        try:
            from worldquant_brain.alpha_novelty import seed_public_denylist
            seed_public_denylist(conn)
        except Exception:
            pass

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
        if not self.is_authenticated() and not self.authenticate():
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
        if not self.is_authenticated() and not self.authenticate():
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
