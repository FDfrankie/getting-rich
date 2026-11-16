"""Cached WorldQuant BRAIN connectivity probe and auth failure classifier."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class BrainConnectivity:
    service_available: bool
    configured: bool
    reachable: bool
    authenticated: bool
    biometric_required: bool
    terms_required: bool
    waf_blocked: bool
    rate_limited: bool
    simulation_enabled: bool
    auto_submit_enabled: bool
    local_alpha_worker_alive: bool
    brain_worker_alive: bool
    queue_depth: int
    last_http_status: int | None
    latency_ms: int | None
    checked_at: str
    safe_error_code: str | None


class BrainAuthFailureClassifier:
    """Classifies HTTP status, headers, and sanitized body fingerprints without leaking tokens."""

    @staticmethod
    def classify(status_code: int | None, headers: dict[str, Any] | None, body: str | None) -> str | None:
        if status_code in (200, 201):
            return None
        hdrs = {str(k).lower(): str(v).lower() for k, v in (headers or {}).items()}
        text = (body or "").lower()
        auth_hdr = hdrs.get("www-authenticate", "")
        ctype = hdrs.get("content-type", "")

        if status_code == 429:
            return "RATE_LIMITED"
        if status_code in (502, 503, 504):
            return "UPSTREAM_UNAVAILABLE"
        if status_code in (401, 403):
            if "persona" in auth_hdr or "biometric" in text:
                return "BRAIN_BIOMETRIC_REQUIRED"
            if "terms" in text or "agreement" in text:
                return "TERMS_REQUIRED"
            if "cloudflare" in text or "cf-ray" in hdrs or "waf" in text or "captcha" in text:
                return "BRAIN_WAF_BLOCKED"
            if "expired" in text:
                return "SESSION_EXPIRED"
            if "text/html" in ctype:
                return "HTML_RESPONSE_UNKNOWN"
            return "AUTH_FAILED"
        return f"HTTP_{status_code}" if status_code else "UNREACHABLE"


class BrainConnectivityService:
    """Maintains cached BRAIN connectivity state so 3-second UI refreshes never spam /authentication."""

    def __init__(self, brain_client: Any, job_queue: Any, settings_getter: Any) -> None:
        self.brain_client = brain_client
        self.job_queue = job_queue
        self.settings_getter = settings_getter
        self._last_http_status: int | None = None
        self._last_latency_ms: int | None = None
        self._last_error_code: str | None = None
        self._last_reachable: bool = False
        self._last_checked_at: datetime = datetime.now(timezone.utc)

    def record_probe_outcome(
        self,
        *,
        reachable: bool,
        http_status: int | None,
        latency_ms: int | None,
        error_code: str | None = None,
    ) -> None:
        self._last_reachable = reachable
        self._last_http_status = http_status
        self._last_latency_ms = latency_ms
        self._last_error_code = error_code
        self._last_checked_at = datetime.now(timezone.utc)

    def get_status(self, *, local_alpha_worker_alive: bool = False) -> dict[str, Any]:
        offline = bool(getattr(self.brain_client, "offline_mode", True))
        auth_attr = getattr(self.brain_client, "is_authenticated", False)
        authed = bool(auth_attr() if callable(auth_attr) else auth_attr) and not offline
        settings = self.settings_getter() if callable(self.settings_getter) else None
        sim_enabled = bool(getattr(settings, "brain_simulation_enabled", False)) if settings else False
        auto_sub = bool(getattr(settings, "brain_auto_submit_enabled", False)) if settings else False

        q_fn = getattr(self.job_queue, "queue_depth", None)
        q_depth = int(q_fn()) if callable(q_fn) else 0
        bw_fn = getattr(self.job_queue, "worker_alive", None)
        bw_alive = bool(bw_fn()) if callable(bw_fn) else False

        err_code = "BRAIN_CREDENTIALS_MISSING" if offline else self._last_error_code
        conn = BrainConnectivity(
            service_available=True,
            configured=not offline,
            reachable=authed or self._last_reachable,
            authenticated=authed,
            biometric_required=(err_code == "BRAIN_BIOMETRIC_REQUIRED"),
            terms_required=(err_code == "TERMS_REQUIRED"),
            waf_blocked=(err_code == "BRAIN_WAF_BLOCKED"),
            rate_limited=(err_code == "RATE_LIMITED"),
            simulation_enabled=sim_enabled,
            auto_submit_enabled=auto_sub,
            local_alpha_worker_alive=local_alpha_worker_alive,
            brain_worker_alive=bw_alive,
            queue_depth=q_depth,
            last_http_status=self._last_http_status,
            latency_ms=self._last_latency_ms,
            checked_at=self._last_checked_at.isoformat(),
            safe_error_code=err_code,
        )
        return asdict(conn)
