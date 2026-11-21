"""WorldQuant BRAIN Multi-Simulation & Dataset Field Discovery Client.

Supports batching 2-10 REGULAR alpha expressions into a single POST /simulations
Multi-Simulation request, polling the parent batch until COMPLETE, fetching all
child alpha metrics & IS-checks, and querying /data-fields for non-PV screening.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_AUTH_LOCK = threading.Lock()
_LAST_AUTH_FAIL_TS = 0.0
_AUTH_FAIL_COOLDOWN_SEC = 30.0
_AUTH_REFRESH_REUSE_SEC = 5.0


def _is_rate_limit_error(text: str) -> bool:
    low = (text or "").lower()
    return "429" in low or "too many requests" in low or "rate limit" in low


def _is_auth_rate_or_captcha_error(text: str) -> bool:
    """True ONLY for genuine auth/CAPTCHA blocks (never plain /simulations 429 rate limits)."""
    low = (text or "").lower()
    return "captcha" in low or "authentication" in low or "unauthorized" in low


def _ensure_authenticated(client: Any, *, force_refresh: bool = False) -> bool:
    """Serialize authentication/refresh under _AUTH_LOCK with a 30s failure cooldown."""
    global _LAST_AUTH_FAIL_TS
    with _AUTH_LOCK:
        now = time.monotonic()
        fail_ts = float(getattr(client, "_multi_sim_last_auth_fail_ts", 0.0) or 0.0)
        if fail_ts > 0.0 and (now - fail_ts) < _AUTH_FAIL_COOLDOWN_SEC:
            return False

        ok_ts = float(getattr(client, "_multi_sim_last_auth_ok_ts", 0.0) or 0.0)
        if getattr(client, "is_authenticated", False):
            if not force_refresh or (ok_ts > 0.0 and (now - ok_ts) < _AUTH_REFRESH_REUSE_SEC):
                return True

        if force_refresh:
            client.authenticated = False
            _sess = getattr(client, "session", None)
            if _sess is not None and hasattr(_sess, "cookies"):
                try:
                    _sess.cookies.clear()
                except Exception:
                    pass
        ok = bool(client.authenticate())
        now_after = time.monotonic()
        if ok:
            _LAST_AUTH_FAIL_TS = 0.0
            try:
                setattr(client, "_multi_sim_last_auth_fail_ts", 0.0)
                setattr(client, "_multi_sim_last_auth_ok_ts", now_after)
            except Exception as _silent_exc:
                logger.debug("worldquant_brain/multi_sim_client.py:L50 suppressed exception: %s", _silent_exc)
            return True

        _LAST_AUTH_FAIL_TS = now_after
        try:
            setattr(client, "_multi_sim_last_auth_fail_ts", now_after)
        except Exception as _silent_exc:
            logger.debug("worldquant_brain/multi_sim_client.py:L57 suppressed exception: %s", _silent_exc)
        return False

DEFAULT_SIM_SETTINGS: dict[str, Any] = {
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

# Curated high-coverage non-PV fallback fields when /data-fields is unavailable or offline
CURATED_DATASET_FIELDS: dict[str, tuple[str, ...]] = {
    "analyst4": (
        "anl4_afv4_eps_mean",
        "anl4_afv4_eps_high",
        "anl4_afv4_eps_low",
        "anl4_afv4_eps_std",
        "anl4_afv4_eps_number",
        "anl4_afv4_sales_mean",
        "anl4_afv4_ebitda_mean",
        "anl4_afv4_cfps_mean",
        "anl4_tot_assets_mean",
        "anl4_netprofit_mean",
        "anl4_bvps_mean",
        "anl4_roe_mean",
    ),
    "fundamental6": (
        "fnd6_ebit",
        "fnd6_ebitda",
        "fnd6_oancf",
        "fnd6_sale",
        "fnd6_ni",
        "fnd6_at",
        "fnd6_lt",
        "fnd6_ceq",
        "fnd6_che",
        "fnd6_capx",
        "fnd6_invt",
        "fnd6_rect",
        "fnd6_cogs",
        "fnd6_xsga",
    ),
    "model16": (
        "mdl16_value",
        "mdl16_momentum",
        "mdl16_quality",
        "mdl16_growth",
        "mdl16_earnings_quality",
        "mdl16_analyst_revisions",
        "mdl16_price_momentum",
        "mdl16_short_interest",
    ),
    "option8": (
        "opt8_imp_vol_30",
        "opt8_imp_vol_60",
        "opt8_imp_vol_90",
        "opt8_put_call_ratio",
        "opt8_skew_30",
        "opt8_call_volume",
        "opt8_put_volume",
    ),
    "news12": (
        "nws12_prez_1l",
        "nws12_prez_5l",
        "nws12_prez_20l",
        "nws12_mainz_1l",
        "nws12_mainz_5l",
        "nws12_mainz_20l",
    ),
    "sentiment1": (
        "snt1_cored1_score",
        "snt1_d1_score",
        "snt1_d5_score",
        "snt1_d20_score",
    ),
}


def build_sim_payload(expression: str, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_SIM_SETTINGS)
    if settings:
        merged.update(dict(settings))
    return {
        "type": "REGULAR",
        "settings": merged,
        "regular": str(expression).strip(),
    }


def _extract_alpha_metrics(alpha_data: Mapping[str, Any]) -> dict[str, Any]:
    is_blk = alpha_data.get("is") if isinstance(alpha_data.get("is"), Mapping) else alpha_data
    checks = is_blk.get("checks") if isinstance(is_blk, Mapping) else None
    failed_checks: list[str] = []
    if isinstance(checks, list):
        for chk in checks:
            if isinstance(chk, Mapping):
                res = str(chk.get("result") or "").upper()
                if res in ("FAIL", "FAILED", "ERROR"):
                    failed_checks.append(str(chk.get("name") or chk.get("message") or "CHECK_FAILED"))
    sharpe = is_blk.get("sharpe") if isinstance(is_blk, Mapping) else None
    fitness = is_blk.get("fitness") if isinstance(is_blk, Mapping) else None
    turnover = is_blk.get("turnover") if isinstance(is_blk, Mapping) else None
    margin = is_blk.get("margin") if isinstance(is_blk, Mapping) else None
    returns = is_blk.get("returns") if isinstance(is_blk, Mapping) else None
    drawdown = is_blk.get("drawdown") if isinstance(is_blk, Mapping) else None
    return {
        "sharpe": float(sharpe) if sharpe is not None else None,
        "fitness": float(fitness) if fitness is not None else None,
        "turnover": float(turnover) if turnover is not None else None,
        "margin": float(margin) if margin is not None else None,
        "returns": float(returns) if returns is not None else None,
        "drawdown": float(drawdown) if drawdown is not None else None,
        "checks_passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
    }


def _request_with_retry(
    client: Any,
    method: str,
    url: str,
    *,
    json_body: Any = None,
    params: Mapping[str, Any] | None = None,
    max_retries: int = 4,
) -> Any:
    timeout = float(getattr(client, "timeout", 30.0) or 30.0)
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.session.request(method, url, json=json_body, params=params, timeout=timeout)
        except Exception as req_exc:
            if attempt >= max_retries:
                raise
            logger.warning(
                "_request_with_retry transient %s on %s %s (attempt %d/%d): %s",
                type(req_exc).__name__,
                method,
                url,
                attempt + 1,
                max_retries + 1,
                req_exc,
            )
            time.sleep(min(15.0, 2.0 * (attempt + 1)))
            continue
        code = int(getattr(resp, "status_code", 0) or 0)
        if code in (204, 401) and attempt == 0:
            if _ensure_authenticated(client, force_refresh=True):
                continue
            return resp
        if code == 403:
            return resp
        if code == 429:
            resp_text = str(getattr(resp, "text", "") or "")
            if _is_auth_rate_or_captcha_error(resp_text) or attempt >= max_retries:
                return resp
            retry_after = float(resp.headers.get("Retry-After") or (5.0 * (attempt + 1)))
            time.sleep(min(retry_after, 60.0))
            continue
        if code in (500, 502, 503, 504) and attempt < max_retries:
            time.sleep(min(2.0 * (2**attempt), 30.0))
            continue
        return resp
    return resp


def fetch_available_datasets(
    client: Any,
    *,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    min_coverage: float = 0.40,
    max_datasets: int = 30,
) -> list[dict[str, Any]]:
    """Dynamically discover datasets unlocked on the user's BRAIN account via GET /data-sets.

    Prioritizes higher valueScore and lower global alphaCount (less crowded = easier corr pass).
    """
    if getattr(client, "offline_mode", True):
        return []
    if not getattr(client, "is_authenticated", False) and not client.authenticate():
        return []

    base_url = getattr(client, "BASE_URL", "https://api.worldquantbrain.com")
    results: list[Any] = []
    offset = 0
    page_limit = 50
    try:
        while offset < 200:
            params = {
                "instrumentType": "EQUITY",
                "region": region,
                "universe": universe,
                "delay": int(delay),
                "limit": page_limit,
                "offset": offset,
            }
            resp = _request_with_retry(client, "GET", f"{base_url}/data-sets", params=params, max_retries=2)
            if int(getattr(resp, "status_code", 0) or 0) != 200:
                break
            data = resp.json() if getattr(resp, "content", None) else {}
            page_res = data.get("results") if isinstance(data, Mapping) else None
            if not isinstance(page_res, list) or not page_res:
                break
            results.extend(page_res)
            if len(page_res) < page_limit:
                break
            offset += page_limit
    except Exception as exc:
        logger.warning("fetch_available_datasets error: %s", exc)
        if not results:
            return []
    if not results:
        return []

    items: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        ds_id = str(row.get("id") or "").strip()
        if not ds_id:
            continue
        cov = row.get("coverage")
        if cov is not None:
            try:
                if float(cov) < min_coverage:
                    continue
            except (TypeError, ValueError):
                pass
        val_score = float(row.get("valueScore") or 1.0)
        alpha_cnt = int(row.get("alphaCount") or 0)
        field_cnt = int(row.get("fieldCount") or 0)
        cat = row.get("category")
        cat_id = str(cat.get("id") if isinstance(cat, Mapping) else (cat or "")).lower()
        items.append(
            {
                "id": ds_id,
                "name": str(row.get("name") or ds_id),
                "category": cat_id,
                "value_score": val_score,
                "alpha_count": alpha_cnt,
                "field_count": field_cnt,
                "coverage": float(cov) if cov is not None else 1.0,
            }
        )

    # Sort: highest valueScore first, then lowest alpha_count (least crowded), deprioritize pv (>1M alphas)
    items.sort(key=lambda d: (d["alpha_count"] > 1_000_000, -d["value_score"], d["alpha_count"]))
    return items[:max_datasets]


def fetch_dataset_fields(
    client: Any,
    dataset_id: str,
    *,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    min_coverage: float = 0.50,
    limit: int = 100,
    matrix_only: bool = False,
) -> list[str]:
    """Fetch high-coverage MATRIX and VECTOR data fields for a dataset from /data-fields.

    VECTOR fields are prefixed with `VEC:` internally so probes wrap them with vec_avg/vec_sum.
    """
    if getattr(client, "offline_mode", True):
        return list(CURATED_DATASET_FIELDS.get(dataset_id, ()))
    if not getattr(client, "is_authenticated", False) and not client.authenticate():
        return list(CURATED_DATASET_FIELDS.get(dataset_id, ()))

    base_url = getattr(client, "BASE_URL", "https://api.worldquantbrain.com")
    fields: list[str] = []
    offset = 0
    page_size = min(50, max(10, int(limit)))

    while len(fields) < limit:
        params = {
            "dataset.id": dataset_id,
            "delay": int(delay),
            "instrumentType": "EQUITY",
            "region": region,
            "universe": universe,
            "limit": page_size,
            "offset": offset,
        }
        try:
            resp = _request_with_retry(client, "GET", f"{base_url}/data-fields", params=params, max_retries=2)
            if int(getattr(resp, "status_code", 0) or 0) != 200:
                break
            data = resp.json() if getattr(resp, "content", None) else {}
            results = data.get("results") if isinstance(data, Mapping) else None
            if not isinstance(results, list) or not results:
                break
            for row in results:
                if not isinstance(row, Mapping):
                    continue
                fid = str(row.get("id") or "").strip()
                if not fid:
                    continue
                ftype = str(row.get("type") or "MATRIX").upper()
                if matrix_only and ftype != "MATRIX":
                    continue
                if ftype not in ("MATRIX", "VECTOR"):
                    continue
                cov = row.get("coverage")
                if cov is not None:
                    try:
                        if float(cov) < min_coverage:
                            continue
                    except (TypeError, ValueError):
                        pass
                fields.append(f"VEC:{fid}" if ftype == "VECTOR" else fid)
                if len(fields) >= limit:
                    break
            if len(results) < page_size:
                break
            offset += page_size
        except Exception as exc:
            logger.warning("fetch_dataset_fields(%s) error: %s", dataset_id, exc)
            break

    return fields if fields else list(CURATED_DATASET_FIELDS.get(dataset_id, ()))


_UNAVAILABLE_NEUT_RE = re.compile(r"Neutralization\s+([A-Z0-9_]+)\s+is\s+not\s+available", re.IGNORECASE)


def _simulate_single_fallback(
    client: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    max_wait_seconds: float,
    poll_interval: float,
    item_delay_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Sequentially fall back to single-alpha simulation when multi-sim batch POST is rejected."""
    unavailable_neuts: set[str] = getattr(client, "_unavailable_neutralizations", None)
    if unavailable_neuts is None:
        unavailable_neuts = set()
        try:
            setattr(client, "_unavailable_neutralizations", unavailable_neuts)
        except Exception as _attr_exc:
            logger.debug("Could not attach _unavailable_neutralizations to client: %s", _attr_exc)

    results: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        expr = str(it.get("expression") or "")
        sett = dict(it.get("settings") or DEFAULT_SIM_SETTINGS)
        stage = it.get("stage", "STAGE1")
        dataset_id = it.get("dataset_id", "")
        item_neut = str(sett.get("neutralization") or "").upper()
        if item_neut and item_neut in unavailable_neuts:
            results.append(
                {
                    "success": False,
                    "status": "SKIPPED",
                    "expression": expr,
                    "settings": sett,
                    "stage": stage,
                    "dataset_id": dataset_id,
                    "error": f"Neutralization {item_neut} is not available (cached)",
                }
            )
            continue
        try:
            res = client.simulate_alpha(
                expr,
                settings=sett,
                max_wait_seconds=max_wait_seconds,
                poll_interval=poll_interval,
            )
            out = dict(res) if isinstance(res, Mapping) else {"success": False, "status": "FAILED", "error": str(res)}
            out.setdefault("expression", expr)
            out.setdefault("settings", sett)
            out["stage"] = stage
            out["dataset_id"] = dataset_id
            out["probe"] = str(it.get("probe") or "")
            results.append(out)
            err_text = str(out.get("error") or "")
            for m_neut in _UNAVAILABLE_NEUT_RE.finditer(err_text):
                unavailable_neuts.add(m_neut.group(1).upper())
            status_text = str(out.get("status") or "").upper()
            is_auth_blk = status_text == "AUTH_FAILED" or _is_auth_rate_or_captcha_error(err_text)
            is_rl_blk = (not is_auth_blk) and (status_text == "RATE_LIMITED" or _is_rate_limit_error(err_text))
            if is_auth_blk or is_rl_blk:
                abort_status = "AUTH_FAILED" if is_auth_blk else "RATE_LIMITED"
                for rem in items[idx + 1 :]:
                    results.append(
                        {
                            "success": False,
                            "status": abort_status,
                            "expression": str(rem.get("expression") or ""),
                            "settings": dict(rem.get("settings") or DEFAULT_SIM_SETTINGS),
                            "stage": rem.get("stage", "STAGE1"),
                            "dataset_id": rem.get("dataset_id", ""),
                            "error": err_text or ("Authentication blocked" if is_auth_blk else "429 rate-limited during single-sim fallback"),
                        }
                    )
                break
        except Exception as exc:
            err_text = str(exc)
            for m_neut in _UNAVAILABLE_NEUT_RE.finditer(err_text):
                unavailable_neuts.add(m_neut.group(1).upper())
            is_auth = _is_auth_rate_or_captcha_error(err_text)
            is_rl = (not is_auth) and _is_rate_limit_error(err_text)
            fail_status = "AUTH_FAILED" if is_auth else ("RATE_LIMITED" if is_rl else "FAILED")
            results.append(
                {
                    "success": False,
                    "status": fail_status,
                    "expression": expr,
                    "settings": sett,
                    "stage": stage,
                    "dataset_id": dataset_id,
                    "error": err_text,
                }
            )
            if is_auth or is_rl:
                for rem in items[idx + 1 :]:
                    results.append(
                        {
                            "success": False,
                            "status": fail_status,
                            "expression": str(rem.get("expression") or ""),
                            "settings": dict(rem.get("settings") or DEFAULT_SIM_SETTINGS),
                            "stage": rem.get("stage", "STAGE1"),
                            "dataset_id": rem.get("dataset_id", ""),
                            "error": err_text,
                        }
                    )
                break
        if idx + 1 < len(items) and item_delay_seconds > 0:
            time.sleep(item_delay_seconds)
    return results


def simulate_multi_alpha(
    client: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    max_wait_seconds: float = 1800.0,
    poll_interval: float = 5.0,
    fallback_item_delay: float = 2.5,
) -> list[dict[str, Any]]:
    """Run 2-10 alpha expressions in a single WorldQuant BRAIN Multi-Simulation batch.

    Each element of `items` must have `expression` and optional `settings`, `stage`, `dataset_id`.
    Returns a list of result dicts aligned 1-to-1 with `items` (preserving input metadata).
    """
    if not items:
        return []

    if getattr(client, "offline_mode", True):
        return [
            {
                "success": False,
                "status": "OFFLINE",
                "expression": str(it.get("expression") or ""),
                "settings": dict(it.get("settings") or DEFAULT_SIM_SETTINGS),
                "stage": it.get("stage", "STAGE1"),
                "dataset_id": it.get("dataset_id", ""),
                "error": "offline_mode=True",
            }
            for it in items
        ]

    # Bypass Multi-Sim batch POST when explicitly disabled via env or circuit-broken by HTTP 403
    if (os.getenv("BRAIN_DISABLE_MULTI_SIM", "false").lower() == "true" or getattr(client, "_multi_sim_forbidden", False)) and len(items) > 1:
        return _simulate_single_fallback(
            client,
            items,
            max_wait_seconds=max_wait_seconds,
            poll_interval=poll_interval,
            item_delay_seconds=fallback_item_delay,
        )

    # Single-item fallback uses regular single simulation
    if len(items) == 1:
        it = items[0]
        res = client.simulate_alpha(
            str(it.get("expression") or ""),
            settings=it.get("settings") or DEFAULT_SIM_SETTINGS,
            max_wait_seconds=max_wait_seconds,
            poll_interval=poll_interval,
        )
        out = dict(res)
        out.setdefault("expression", str(it.get("expression") or ""))
        out.setdefault("settings", dict(it.get("settings") or DEFAULT_SIM_SETTINGS))
        out["stage"] = it.get("stage", "STAGE1")
        out["dataset_id"] = it.get("dataset_id", "")
        out["probe"] = str(it.get("probe") or "")
        return [out]

    if not _ensure_authenticated(client, force_refresh=False):
        return [
            {
                "success": False,
                "status": "AUTH_FAILED",
                "expression": str(it.get("expression") or ""),
                "settings": dict(it.get("settings") or DEFAULT_SIM_SETTINGS),
                "stage": it.get("stage", "STAGE1"),
                "dataset_id": it.get("dataset_id", ""),
                "error": "BRAIN authentication failed",
            }
            for it in items
        ]

    base_url = getattr(client, "BASE_URL", "https://api.worldquantbrain.com")
    payloads = [build_sim_payload(str(it.get("expression") or ""), it.get("settings")) for it in items]

    resp = _request_with_retry(client, "POST", f"{base_url}/simulations", json_body=payloads)
    code = int(getattr(resp, "status_code", 0) or 0)
    location = resp.headers.get("Location") or resp.headers.get("location") if resp is not None else None
    resp_text = str(getattr(resp, "text", "") or "") if resp is not None else ""

    if code == 401 or _is_auth_rate_or_captcha_error(resp_text):
        err_msg = f"Multi-sim auth/CAPTCHA blocked (HTTP {code})"
        return [
            {
                "success": False,
                "status": "AUTH_FAILED",
                "expression": payloads[idx]["regular"],
                "settings": payloads[idx]["settings"],
                "stage": items[idx].get("stage", "STAGE1"),
                "dataset_id": items[idx].get("dataset_id", ""),
                "probe": str(items[idx].get("probe") or ""),
                "error": err_msg,
            }
            for idx in range(len(items))
        ]
    if code == 429 or _is_rate_limit_error(resp_text):
        err_msg = f"Multi-sim rate-limited (HTTP {code}: 429 Too Many Requests)"
        return [
            {
                "success": False,
                "status": "RATE_LIMITED",
                "expression": payloads[idx]["regular"],
                "settings": payloads[idx]["settings"],
                "stage": items[idx].get("stage", "STAGE1"),
                "dataset_id": items[idx].get("dataset_id", ""),
                "probe": str(items[idx].get("probe") or ""),
                "error": err_msg,
            }
            for idx in range(len(items))
        ]

    if code in (400, 403) or (code in (200, 201, 202) and not location):
        if code == 403:
            try:
                setattr(client, "_multi_sim_forbidden", True)
                setattr(client, "_multi_sim_forbidden_reason", "FEATURE_NOT_ENTITLED")
            except Exception as _silent_exc:
                logger.debug("worldquant_brain/multi_sim_client.py:L542 suppressed exception: %s", _silent_exc)
        logger.warning(
            "Multi-sim POST returned HTTP %s without Location; falling back to single-alpha simulation for %d items",
            code,
            len(items),
        )
        return _simulate_single_fallback(
            client,
            items,
            max_wait_seconds=max_wait_seconds,
            poll_interval=poll_interval,
            item_delay_seconds=fallback_item_delay,
        )

    if code not in (200, 201, 202) or not location:
        err_msg = f"Multi-sim POST failed (HTTP {code})"
        return [
            {
                "success": False,
                "status": "SUBMIT_ERROR",
                "expression": payloads[idx]["regular"],
                "settings": payloads[idx]["settings"],
                "stage": items[idx].get("stage", "STAGE1"),
                "dataset_id": items[idx].get("dataset_id", ""),
                "probe": str(items[idx].get("probe") or ""),
                "error": err_msg,
            }
            for idx in range(len(items))
        ]

    if location.startswith("/"):
        location = f"{base_url}{location}"

    deadline = time.monotonic() + float(max_wait_seconds)
    parent_data: dict[str, Any] = {}
    while time.monotonic() < deadline:
        poll = _request_with_retry(client, "GET", location, max_retries=2)
        p_code = int(getattr(poll, "status_code", 0) or 0)
        if p_code == 200 and getattr(poll, "content", None):
            pdata = poll.json() if hasattr(poll, "json") else {}
            if isinstance(pdata, Mapping):
                parent_data = dict(pdata)
                status = str(parent_data.get("status") or "").upper()
                if status in ("COMPLETE", "COMPLETED", "WARNING", "ERROR", "FAIL", "FAILED"):
                    break
        retry_after = poll.headers.get("Retry-After") if poll is not None else None
        sleep_s = float(retry_after) if retry_after else float(poll_interval)
        time.sleep(min(max(sleep_s, 0.05), 30.0))

    children = parent_data.get("children")
    if not isinstance(children, list) or not children:
        err_msg = str(parent_data.get("message") or parent_data.get("status") or "MULTI_SIM_NO_CHILDREN")
        return [
            {
                "success": False,
                "status": "FAILED",
                "expression": payloads[idx]["regular"],
                "settings": payloads[idx]["settings"],
                "stage": items[idx].get("stage", "STAGE1"),
                "dataset_id": items[idx].get("dataset_id", ""),
                "probe": str(items[idx].get("probe") or ""),
                "error": err_msg,
            }
            for idx in range(len(items))
        ]

    results: list[dict[str, Any]] = []
    for idx, child_id in enumerate(children):
        it = items[idx] if idx < len(items) else {}
        expr = payloads[idx]["regular"] if idx < len(payloads) else ""
        sett = payloads[idx]["settings"] if idx < len(payloads) else dict(DEFAULT_SIM_SETTINGS)
        c_url = child_id if str(child_id).startswith("http") else f"{base_url}/simulations/{child_id}"
        c_resp = _request_with_retry(client, "GET", c_url, max_retries=2)
        c_code = int(getattr(c_resp, "status_code", 0) or 0)
        c_data = c_resp.json() if (c_code == 200 and getattr(c_resp, "content", None)) else {}
        alpha_id = str(c_data.get("alpha") or "").strip() if isinstance(c_data, Mapping) else ""
        c_status = str(c_data.get("status") or "FAILED").upper() if isinstance(c_data, Mapping) else "FAILED"

        if not alpha_id:
            results.append(
                {
                    "success": False,
                    "status": c_status,
                    "alpha_id": None,
                    "expression": expr,
                    "settings": sett,
                    "stage": it.get("stage", "STAGE1"),
                    "dataset_id": it.get("dataset_id", ""),
                    "error": str(c_data.get("message") or f"Child simulation {c_status}"),
                }
            )
            continue

        a_resp = _request_with_retry(client, "GET", f"{base_url}/alphas/{alpha_id}", max_retries=2)
        a_code = int(getattr(a_resp, "status_code", 0) or 0)
        a_data = a_resp.json() if (a_code == 200 and getattr(a_resp, "content", None)) else {}
        metrics = _extract_alpha_metrics(a_data) if isinstance(a_data, Mapping) else {}
        has_valid_metrics = metrics.get("sharpe") is not None and metrics.get("fitness") is not None
        results.append(
            {
                "success": bool(has_valid_metrics),
                "status": "COMPLETED" if has_valid_metrics else "INCOMPLETE_METRICS",
                "alpha_id": alpha_id,
                "expression": expr,
                "settings": sett,
                "stage": it.get("stage", "STAGE1"),
                "dataset_id": it.get("dataset_id", ""),
                "metrics": metrics,
                "sharpe": metrics.get("sharpe"),
                "fitness": metrics.get("fitness"),
                "turnover": metrics.get("turnover"),
                "margin": metrics.get("margin"),
                "checks_passed": metrics.get("checks_passed", False),
                "failed_checks": metrics.get("failed_checks", []),
            }
        )

    return results
