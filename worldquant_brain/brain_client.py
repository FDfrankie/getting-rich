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
                sharpe = self._number(last, "sharpe")
                fitness = self._number(last, "fitness")
                turnover = self._number(last, "turnover")
                margin = self._number(last, "margin")

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
