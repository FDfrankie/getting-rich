"""WorldQuant BRAIN API client with asynchronous simulation polling and offline fallback."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import requests


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
        r = self.session.post(
            f"{self.BASE_URL}/authentication",
            auth=(self.email, self.password),
            timeout=self.timeout,
        )
        r.raise_for_status()
        self.authenticated = True
        return True

    @staticmethod
    def _key(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
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
                        try:
                            return float(lowered[n.lower()])
                        except (TypeError, ValueError):
                            pass
                stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
            elif isinstance(cur, list):
                stack.extend(cur)
        return None

    def _offline_result(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "OFFLINE",
            "offline": True,
            "submitted": False,
            "deduplicated": False,
            "reason": "BRAIN_EMAIL/BRAIN_PASSWORD 未設定，未送出任何網路請求。",
            "request": dict(payload),
            "sharpe": None,
            "fitness": None,
            "turnover": None,
            "margin": None,
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
        response = self.session.post(
            f"{self.BASE_URL}/simulations", json=body, timeout=self.timeout
        )
        if response.status_code not in (200, 201, 202):
            response.raise_for_status()
        location = response.headers.get("Location") or response.headers.get("location")
        if not location:
            raise RuntimeError("BRAIN /simulations 未回傳 Location header")
        if location.startswith("/"):
            location = f"{self.BASE_URL}{location}"
        interval = self.poll_interval if poll_interval is None else float(poll_interval)
        deadline = time.monotonic() + float(max_wait_seconds)
        last: dict[str, Any] = {}
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"BRAIN simulation polling 超過 {max_wait_seconds} 秒；Location={location}"
                )
            poll = self.session.get(location, timeout=self.timeout)
            if poll.status_code == 429:
                time.sleep(
                    max(interval, float(poll.headers.get("Retry-After", interval)))
                )
                continue
            poll.raise_for_status()
            last = poll.json() if poll.content else {}
            status = str(last.get("status") or last.get("state") or "").upper()
            if status == "COMPLETE":
                result = {
                    "status": "COMPLETE",
                    "offline": False,
                    "submitted": True,
                    "deduplicated": False,
                    "location": location,
                    "simulation_id": last.get("id")
                    or location.rstrip("/").split("/")[-1],
                    "sharpe": self._number(last, "sharpe"),
                    "fitness": self._number(last, "fitness"),
                    "turnover": self._number(last, "turnover"),
                    "margin": self._number(last, "margin"),
                    "raw": last,
                }
                self._submitted[key] = result
                return dict(result)
            if status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                raise RuntimeError(f"BRAIN simulation {status}: {last}")
            retry = poll.headers.get("Retry-After")
            time.sleep(float(retry) if retry else interval)

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
