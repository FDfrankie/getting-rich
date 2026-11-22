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
# Live-verified USA TOP3000 D1 fields across all 14 Data Explorer datasets (audited 2026-11-22)
CURATED_DATASET_FIELDS: dict[str, tuple[str, ...]] = {
    "model16": (
        "analyst_revision_rank_derivative",
        "cashflow_efficiency_rank_derivative",
        "composite_factor_score_derivative",
        "earnings_certainty_rank_derivative",
        "fscore_bfl_growth",
        "fscore_bfl_momentum",
        "fscore_bfl_profitability",
        "fscore_bfl_quality",
        "fscore_bfl_value",
        "fscore_bfl_total",
    ),
    "model51": (
        "beta_last_30_days_spy",
        "beta_last_60_days_spy",
        "beta_last_90_days_spy",
        "beta_last_360_days_spy",
        "correlation_last_30_days_spy",
        "correlation_last_90_days_spy",
        "correlation_last_360_days_spy",
        "systematic_risk_last_30_days",
        "systematic_risk_last_90_days",
        "unsystematic_risk_last_30_days",
        "unsystematic_risk_last_90_days",
        "unsystematic_risk_last_360_days",
    ),
    "socialmedia8": (
        "snt_social_value",
        "snt_social_value_fast_d1",
        "snt_social_volume",
        "snt_social_volume_fast_d1",
    ),
    "socialmedia12": (
        "scl12_buzz",
        "scl12_buzz_fast_d1",
        "scl12_sentiment",
        "scl12_sentiment_fast_d1",
        "snt_buzz",
        "snt_buzz_bfl",
        "snt_buzz_ret",
        "VEC:scl12_sentvec",
        "VEC:scl12_buzzvec",
    ),
    "option8": (
        "historical_volatility_10",
        "historical_volatility_30",
        "historical_volatility_90",
        "historical_volatility_180",
        "implied_volatility_call_10",
        "implied_volatility_call_120",
        "implied_volatility_call_180",
        "implied_volatility_call_1080",
    ),
    "option9": (
        "pcr_oi_10",
        "pcr_oi_30",
        "pcr_oi_60",
        "pcr_oi_90",
        "pcr_oi_180",
        "pcr_oi_360",
        "pcr_oi_all",
        "call_breakeven_10",
        "call_breakeven_20",
        "put_breakeven_10",
    ),
    "pv13": (
        "pv13_custretsig_retsig",
        "pv13_com_page_rank",
        "pv13_com_rk_au",
        "primary_sector_focused_company_count",
    ),
    "news12": (
        "news_max_up_ret",
        "news_tot_ticks",
        "news_atr14",
        "news_pe_ratio",
        "news_mins_20_pct_dn",
        "VEC:nws12_prez_4l",
        "VEC:nws12_afterhsz_sl",
    ),
    "news18": (
        "VEC:nws18_ghc_lna",
        "VEC:nws18_nip",
        "rp_css_earnings",
        "rp_ess_earnings",
        "rp_css_ratings",
        "rp_ess_ratings",
        "rp_ess_mna",
        "rp_css_revenue",
    ),
    "analyst4": (
        "est_eps",
        "anl4_qf_az_eps_number",
        "anl4_ebit_value",
        "anl4_ebitda_value",
        "anl4_adjusted_netincome_ft",
        "anl4_epsr_flag",
        "anl4_netdebt_flag",
        "anl4_ptp_flag",
        "anl4_bvps_flag",
    ),
    "fundamental6": (
        "assets",
        "assets_curr",
        "bookvalue_ps",
        "capex",
        "cash",
        "cashflow_op",
        "cogs",
        "current_ratio",
        "debt",
        "debt_lt",
    ),
    "fundamental2": (
        "fn_accum_oth_income_loss_net_of_tax_a",
        "fn_comp_not_rec_a",
        "fn_op_lease_rent_exp_a",
        "fn_op_lease_min_pay_due_a",
        "fn_eff_income_tax_rate_continuing_operations_a",
        "fn_liab_fair_val_l1_a",
    ),
}


# Verified live Bronze account constraints (platform.worldquantbrain.com/simulate & /learn/operators, 2026-11-22)
BRONZE_ALLOWED_NEUTRALIZATIONS: frozenset[str] = frozenset({
    "NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY",
})
CONSULTANT_ONLY_NEUTRALIZATIONS: frozenset[str] = frozenset({
    "SLOW", "FAST", "SLOW_AND_FAST", "CROWDING", "STATISTICAL", "REVERSION_AND_MOMENTUM",
})
DEPRECATED_DATA_FIELDS: frozenset[str] = frozenset({
    "opt6_nexterntod", "opt6_lasterntod",
})
# Exact 66 unlocked operators on Bronze accounts (Arithmetic 15, Logical 11, TS 24, CS 6, Vector 2, Transform 2, Group 6)
BRONZE_UNLOCKED_OPERATORS: frozenset[str] = frozenset({
    # Arithmetic (15)
    "abs", "add", "densify", "divide", "inverse", "log", "max", "min", "multiply",
    "power", "reverse", "sign", "signed_power", "sqrt", "subtract",
    # Logical (11) -> named functions
    "and", "if_else", "is_nan", "not", "or",
    # Time Series (24)
    "days_from_last_change", "hump", "kth_element", "last_diff_value",
    "ts_arg_max", "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_corr",
    "ts_count_nans", "ts_covariance", "ts_decay_linear", "ts_delay", "ts_delta",
    "ts_mean", "ts_product", "ts_quantile", "ts_rank", "ts_regression",
    "ts_scale", "ts_std_dev", "ts_step", "ts_sum", "ts_zscore",
    # Cross Sectional (6)
    "normalize", "quantile", "rank", "scale", "winsorize", "zscore",
    # Vector (2)
    "vec_avg", "vec_sum",
    # Transformational (2)
    "bucket", "trade_when",
    # Group (6)
    "group_backfill", "group_mean", "group_neutralize", "group_rank", "group_scale", "group_zscore",
})
# Official BRAIN Neutralization doc dataset-to-neutralization defaults
DATASET_NEUTRALIZATION_DEFAULTS: dict[str, str] = {
    "fundamental": "INDUSTRY",
    "analyst": "INDUSTRY",
    "earnings": "INDUSTRY",
    "short": "INDUSTRY",
    "option": "SECTOR",
    "pv": "MARKET",
    "news": "SUBINDUSTRY",
    "social": "INDUSTRY",
    "sentiment": "SUBINDUSTRY",
    "model": "INDUSTRY",
    "institution": "SECTOR",
}
_FASTEXPR_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def is_consultant_mode() -> bool:
    return os.getenv("BRAIN_CONSULTANT_MODE", "false").strip().lower() in {"1", "true", "yes"}


def default_neutralization_for_dataset(dataset_id: str = "") -> str:
    ds = (dataset_id or "").strip().lower()
    for prefix, neut in DATASET_NEUTRALIZATION_DEFAULTS.items():
        if prefix in ds:
            return neut
    return "SUBINDUSTRY"

# P36: Empirical USA TOP3000 D1 dataset simulation profiles (wq-alpha-research & AgonAlpha 2026)
DATASET_SIM_PROFILES: dict[str, dict[str, Any]] = {
    "fundamental": {"neutralization": "SUBINDUSTRY", "decay": 0, "nanHandling": "ON", "truncation": 0.08},
    "analyst": {"neutralization": "SUBINDUSTRY", "decay": 2, "nanHandling": "ON", "truncation": 0.08},
    "earnings": {"neutralization": "INDUSTRY", "decay": 2, "nanHandling": "ON", "truncation": 0.08},
    "option": {"neutralization": "INDUSTRY", "decay": 2, "nanHandling": "ON", "truncation": 0.08},
    "model": {"neutralization": "INDUSTRY", "decay": 4, "nanHandling": "ON", "truncation": 0.08},
    "news": {"neutralization": "SUBINDUSTRY", "decay": 4, "nanHandling": "ON", "truncation": 0.08},
    "social": {"neutralization": "INDUSTRY", "decay": 4, "nanHandling": "ON", "truncation": 0.08},
    "sentiment": {"neutralization": "SUBINDUSTRY", "decay": 4, "nanHandling": "ON", "truncation": 0.08},
    "short": {"neutralization": "INDUSTRY", "decay": 4, "nanHandling": "ON", "truncation": 0.08},
    "pv13": {"neutralization": "SUBINDUSTRY", "decay": 4, "nanHandling": "ON", "truncation": 0.08},
    "pv": {"neutralization": "INDUSTRY", "decay": 12, "nanHandling": "OFF", "truncation": 0.08},
}


def default_sim_profile_for_dataset(dataset_id: str = "") -> dict[str, Any]:
    """Return empirical USA TOP3000 D1 settings (neutralization, decay, nanHandling, truncation) for dataset."""
    ds = (dataset_id or "").strip().lower()
    if "pv13" in ds:
        return dict(DATASET_SIM_PROFILES["pv13"])
    for prefix, prof in DATASET_SIM_PROFILES.items():
        if prefix == "pv13":
            continue
        if prefix in ds:
            return dict(prof)
    return {"neutralization": default_neutralization_for_dataset(ds), "decay": 4, "nanHandling": "ON", "truncation": 0.08}



def sanitize_sim_settings(settings: Mapping[str, Any] | None = None, *, dataset_id: str = "") -> dict[str, Any]:
    """Ensure simulation settings apply dataset-calibrated defaults and Bronze-safe neutralizations."""
    merged = dict(DEFAULT_SIM_SETTINGS)
    if dataset_id:
        merged.update(default_sim_profile_for_dataset(dataset_id))
    if settings:
        merged.update(dict(settings))
    neut = str(merged.get("neutralization") or "SUBINDUSTRY").strip().upper()
    if not is_consultant_mode() and neut not in BRONZE_ALLOWED_NEUTRALIZATIONS:
        fallback = default_neutralization_for_dataset(dataset_id)
        logger.debug("Mapping unavailable neutralization %s -> %s (Bronze mode)", neut, fallback)
        merged["neutralization"] = fallback
    else:
        merged["neutralization"] = neut
    return merged


def find_locked_bronze_operators(expression: str) -> list[str]:
    """Return sorted list of function-call operators in expression not unlocked on Bronze accounts."""
    if is_consultant_mode():
        return []
    calls = _FASTEXPR_CALL_RE.findall(str(expression or ""))
    return sorted({c for c in calls if c.lower() not in BRONZE_UNLOCKED_OPERATORS})


_POWER_POOL_EXCLUDED_GROUPS: frozenset[str] = frozenset({
    "currency", "country", "exchange", "sector", "industry", "subindustry", "market",
    "split", "adjfactor", "pv13_1l_scibr", "pv13_h2_sector", "pv13_h_f1_sector",
})
_POWER_POOL_KWARGS: frozenset[str] = frozenset({
    "std", "hump", "driver", "range", "lag", "rettype", "k", "ignore", "dense", "true", "false", "nan",
})


def check_power_pool_eligibility(expression: str) -> dict[str, Any]:
    """Official BRAIN Power Pool structural check: <= 8 operators (excl. backfill) and <= 3 unique data fields."""
    expr = str(expression or "")
    calls = [c.lower() for c in _FASTEXPR_CALL_RE.findall(expr)]
    counted_ops = [c for c in calls if c not in {"ts_backfill", "group_backfill"}]
    op_set = set(calls)
    all_idents = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr)
    unique_fields = sorted({
        tok for tok in all_idents
        if tok.lower() not in op_set
        and tok.lower() not in _POWER_POOL_EXCLUDED_GROUPS
        and tok.lower() not in _POWER_POOL_KWARGS
    })
    return {
        "eligible": len(counted_ops) <= 8 and len(unique_fields) <= 3,
        "operator_count": len(counted_ops),
        "unique_field_count": len(unique_fields),
        "unique_fields": unique_fields,
    }


def build_sim_payload(expression: str, settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = sanitize_sim_settings(settings)
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
        except (OSError, TimeoutError):
            if attempt < max_retries:
                time.sleep(min(1.0 * (2**attempt), 15.0))
                continue
            raise
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

    # P36 Sort: prioritize empirical high-pass-rate tiers (Fundamental 40% > Analyst/Option > Model/News/PV13 > PV1/Other)
    def _dataset_tier(d: Mapping[str, Any]) -> int:
        ds_id_l = str(d.get("id") or "").lower()
        if ds_id_l in {"fundamental6", "fundamental2"} or ds_id_l.startswith("fundamental"):
            return 0
        if ds_id_l.startswith("analyst") or ds_id_l in {"option8", "option9"} or ds_id_l.startswith("option"):
            return 1
        if ds_id_l in {"model16", "model51", "pv13", "news12", "news18", "socialmedia12"}:
            return 2
        return 3

    items.sort(key=lambda d: (_dataset_tier(d), float(d.get("coverage", 1.0)) < 0.65, -d["value_score"], d["alpha_count"]))
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
    """Concurrently fall back to single-alpha simulations (Bronze 3-slot pool) when multi-sim POST is rejected."""
    from concurrent.futures import ThreadPoolExecutor

    unavailable_neuts: set[str] = getattr(client, "_unavailable_neutralizations", None)
    if unavailable_neuts is None:
        unavailable_neuts = set()
        try:
            setattr(client, "_unavailable_neutralizations", unavailable_neuts)
        except Exception as _attr_exc:
            logger.debug("Could not attach _unavailable_neutralizations to client: %s", _attr_exc)

    n_items = len(items)
    if n_items == 0:
        return []

    raw_conc = int(os.getenv("BRAIN_BRONZE_CONCURRENCY", "3") or "3")
    max_workers = max(1, min(3, raw_conc, n_items))
    # Preserve deterministic sequential execution when item_delay_seconds == 0 and mock/test expects strict ordering,
    # or run concurrently across up to 3 Bronze simulation slots.
    neut_lock = threading.Lock()
    abort_event = threading.Event()
    abort_info: dict[str, str] = {"status": "", "error": ""}

    def _run_one(idx: int, it: Mapping[str, Any]) -> dict[str, Any]:
        expr = str(it.get("expression") or "")
        dataset_id = str(it.get("dataset_id") or "")
        sett = sanitize_sim_settings(it.get("settings"), dataset_id=dataset_id)
        stage = it.get("stage", "STAGE1")

        if abort_event.is_set():
            return {
                "success": False,
                "status": abort_info["status"] or "RATE_LIMITED",
                "expression": expr,
                "settings": sett,
                "stage": stage,
                "dataset_id": dataset_id,
                "error": abort_info["error"] or "Aborted due to upstream auth/rate-limit",
            }

        item_neut = str(sett.get("neutralization") or "").upper()
        with neut_lock:
            neut_blocked = bool(item_neut and item_neut in unavailable_neuts)
        if neut_blocked:
            return {
                "success": False,
                "status": "SKIPPED",
                "expression": expr,
                "settings": sett,
                "stage": stage,
                "dataset_id": dataset_id,
                "error": f"Neutralization {item_neut} is not available (cached)",
            }

        if idx > 0 and item_delay_seconds > 0:
            time.sleep(min(item_delay_seconds * (idx % max_workers), 2.5))

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
            if it.get("probe") is not None:
                out.setdefault("probe", it.get("probe"))
            err_text = str(out.get("error") or "")
            with neut_lock:
                for m_neut in _UNAVAILABLE_NEUT_RE.finditer(err_text):
                    unavailable_neuts.add(m_neut.group(1).upper())
            status_text = str(out.get("status") or "").upper()
            is_auth_blk = status_text == "AUTH_FAILED" or _is_auth_rate_or_captcha_error(err_text)
            is_rl_blk = (not is_auth_blk) and (status_text == "RATE_LIMITED" or _is_rate_limit_error(err_text))
            if is_auth_blk or is_rl_blk:
                abort_info["status"] = "AUTH_FAILED" if is_auth_blk else "RATE_LIMITED"
                abort_info["error"] = err_text or ("Authentication blocked" if is_auth_blk else "429 rate-limited during single-sim fallback")
                abort_event.set()
            return out
        except Exception as exc:
            err_text = str(exc)
            with neut_lock:
                for m_neut in _UNAVAILABLE_NEUT_RE.finditer(err_text):
                    unavailable_neuts.add(m_neut.group(1).upper())
            is_auth = _is_auth_rate_or_captcha_error(err_text)
            is_rl = (not is_auth) and _is_rate_limit_error(err_text)
            fail_status = "AUTH_FAILED" if is_auth else ("RATE_LIMITED" if is_rl else "FAILED")
            if is_auth or is_rl:
                abort_info["status"] = fail_status
                abort_info["error"] = err_text
                abort_event.set()
            return {
                "success": False,
                "status": fail_status,
                "expression": expr,
                "settings": sett,
                "stage": stage,
                "dataset_id": dataset_id,
                "error": err_text,
            }

    if max_workers == 1:
        return [_run_one(idx, it) for idx, it in enumerate(items)]

    results: list[dict[str, Any] | None] = [None] * n_items
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [(idx, pool.submit(_run_one, idx, it)) for idx, it in enumerate(items)]
        for idx, fut in futs:
            results[idx] = fut.result()
    return [r for r in results if r is not None]


def bronze_max_concurrent_sims() -> int:
    """Official BRAIN Help Center (32/32 helpful): Regular/Bronze max 3 concurrent sims, Consultants max 8."""
    if os.getenv("BRAIN_CONSULTANT_MODE", "false").strip().lower() in ("1", "true", "yes"):
        return int(os.getenv("BRAIN_MAX_CONCURRENT_SIMS", "8") or "8")
    return min(3, max(1, int(os.getenv("BRAIN_MAX_CONCURRENT_SIMS", "3") or "3")))


def api_min_spacing_seconds() -> float:
    """Official BRAIN API rate-limit community consensus: >= 2.0s spacing between requests."""
    return max(2.0, float(os.getenv("BRAIN_API_MIN_SPACING_SEC", "2.0") or "2.0"))


def simulate_multi_alpha(
    client: Any,
    items: Sequence[Mapping[str, Any]],
    *,
    max_wait_seconds: float | None = None,
    poll_interval: float = 5.0,
    fallback_item_delay: float = 2.5,
) -> list[dict[str, Any]]:
    """Run 2-10 alpha expressions in a single WorldQuant BRAIN Multi-Simulation batch.

    Each element of `items` must have `expression` and optional `settings`, `stage`, `dataset_id`.
    Returns a list of result dicts aligned 1-to-1 with `items` (preserving input metadata).
    """
    if max_wait_seconds is None:
        max_wait_seconds = float(os.getenv("BRAIN_SIM_TIMEOUT_SEC", "300.0") or "300.0")
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
