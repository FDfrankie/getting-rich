"""Official TWSE/TPEx Monthly Revenue (PIT-safe) and TWSE Margin Trading (MI_MARGN) fetchers."""
from __future__ import annotations

from datetime import date
from typing import Any

import requests

TWSE_REV_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
TPEX_REV_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"
TWSE_MARGIN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"


def _to_float(val: Any) -> float | None:
    s = str(val or "").replace(",", "").replace("%", "").strip()
    if not s or s in {"-", "--", "N/A"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def roc_ym_to_pit_effective_date(roc_ym: str) -> str | None:
    """Convert ROC year-month (e.g. '11510' -> 2026-10) to PIT-safe effective date (next month 11th)."""
    digits = "".join(ch for ch in str(roc_ym or "") if ch.isdigit())
    if len(digits) < 4:
        return None
    roc_y = int(digits[:-2])
    month = int(digits[-2:])
    if not (1 <= month <= 12):
        return None
    ad_y = roc_y + 1911
    eff_y, eff_m = (ad_y + 1, 1) if month == 12 else (ad_y, month + 1)
    return date(eff_y, eff_m, 11).isoformat()


def parse_monthly_revenue_rows(rows: list[dict[str, Any]], market: str = "TWSE") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows or []:
        sym = str(r.get("公司代號") or r.get("SecuritiesCompanyCode") or "").strip()
        if not sym:
            continue
        roc_ym = str(r.get("資料年月") or "").strip()
        out[sym] = {
            "market": market,
            "roc_ym": roc_ym,
            "pit_effective_date": roc_ym_to_pit_effective_date(roc_ym),
            "rev_curr": _to_float(r.get("營業收入-當月營收")),
            "rev_mom_pct": _to_float(r.get("營業收入-上月比較增減(%)")),
            "rev_yoy_pct": _to_float(r.get("營業收入-去年同月增減(%)")),
            "cum_rev_yoy_pct": _to_float(r.get("累計營業收入-前期比較增減(%)")),
        }
    return out


def fetch_monthly_revenue_snapshot(timeout: int = 20) -> dict[str, dict[str, Any]]:
    """Fetch latest TWSE + TPEx monthly revenue snapshot with YoY%, MoM% and PIT effective date."""
    out: dict[str, dict[str, Any]] = {}
    for market, url in (("TWSE", TWSE_REV_URL), ("TPEX", TPEX_REV_URL)):
        resp = requests.get(url, headers={"User-Agent": "AntigravityQuant/1.0"}, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, list):
            out.update(parse_monthly_revenue_rows(rows, market=market))
    return out


def parse_twse_margin_payload(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        return {}
    tables = payload.get("tables") or []
    target_table = next((t for t in tables if any("代號" in str(f) for f in (t.get("fields") or []))), None)
    if not target_table:
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in target_table.get("data") or []:
        if len(row) < 13:
            continue
        sym = str(row[0]).strip()
        if not sym:
            continue
        prev_margin = _to_float(row[5]) or 0.0
        curr_margin = _to_float(row[6]) or 0.0
        prev_short = _to_float(row[11]) or 0.0
        curr_short = _to_float(row[12]) or 0.0
        out[sym] = {
            "margin_balance_lots": curr_margin,
            "margin_change_lots": curr_margin - prev_margin,
            "short_balance_lots": curr_short,
            "short_change_lots": curr_short - prev_short,
            "short_margin_ratio": round(curr_short / (curr_margin + 1e-6), 4),
        }
    return out


def fetch_twse_margin_trading(date_yyyymmdd: str, timeout: int = 20) -> dict[str, dict[str, float]]:
    """Fetch TWSE MI_MARGN per-stock margin/short balances (unit: lots / 張). Returns {} on holidays."""
    resp = requests.get(
        TWSE_MARGIN_URL,
        params={"date": date_yyyymmdd, "selectType": "ALL", "response": "json"},
        headers={"User-Agent": "AntigravityQuant/1.0"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_twse_margin_payload(resp.json())
