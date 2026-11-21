"""Official TWSE T86 daily institutional investor flows fetcher (unit: shares)."""
from __future__ import annotations

from typing import Any

import requests

TWSE_T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"


def _parse_tw_int(val: Any) -> int:
    s = str(val or "").replace(",", "").strip()
    if not s or s in {"-", "--", "N/A"}:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_twse_t86_payload(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Parse TWSE T86 JSON payload without double-counting foreign/dealer sub-columns."""
    if not isinstance(payload, dict) or payload.get("stat") != "OK":
        return {}
    raw_fields = payload.get("fields")
    raw_data = payload.get("data")
    if not raw_fields or not raw_data:
        return {}

    fields = [str(f).strip() for f in raw_fields]
    out: dict[str, dict[str, int]] = {}
    for row in raw_data:
        rec = dict(zip(fields, row))
        sym = str(rec.get("證券代號", "")).strip()
        if not sym:
            continue
        foreign_main = _parse_tw_int(rec.get("外陸資買賣超股數(不含外資自營商)"))
        foreign_dealer = _parse_tw_int(rec.get("外資自營商買賣超股數"))
        trust_net = _parse_tw_int(rec.get("投信買賣超股數"))
        dealer_net = _parse_tw_int(rec.get("自營商買賣超股數"))
        total_net = _parse_tw_int(rec.get("三大法人買賣超股數"))
        out[sym] = {
            "foreign_net_shares": foreign_main + foreign_dealer,
            "trust_net_shares": trust_net,
            "dealer_net_shares": dealer_net,
            "total_inst_net_shares": total_net,
        }
    return out


def fetch_twse_t86_institutional(date_yyyymmdd: str, timeout: int = 20) -> dict[str, dict[str, int]]:
    """Fetch official TWSE T86 daily institutional flows (unit: shares; divide by 1000 for lots).

    Returns {} on non-trading days (stat != 'OK').
    """
    resp = requests.get(
        TWSE_T86_URL,
        params={"date": date_yyyymmdd, "selectType": "ALL", "response": "json"},
        headers={"User-Agent": "AntigravityQuant/1.0"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return parse_twse_t86_payload(resp.json())


def parse_twse_t86_row(fields: list[str], row: list[str]) -> dict[str, float | str] | None:
    """Safely parse a TWSE T86 19-column row into lots (1000 shares) without double-counting dealer/foreign sub-columns."""
    fmap = {str(k).strip(): str(v).strip() for k, v in zip(fields, row)}
    code = fmap.get("證券代號", "").strip()
    if not code:
        return None

    def _shares_to_lots(col_name: str) -> float:
        raw = fmap.get(col_name, "").replace(",", "").replace("+", "").strip()
        if not raw or raw in {"--", "---", "X"}:
            return 0.0
        try:
            return float(raw) / 1000.0
        except ValueError:
            return 0.0

    foreign_ex_dealer = _shares_to_lots("外陸資買賣超股數(不含外資自營商)")
    foreign_dealer = _shares_to_lots("外資自營商買賣超股數")
    trust_net = _shares_to_lots("投信買賣超股數")
    dealer_prop = _shares_to_lots("自營商買賣超股數(自行買賣)")
    dealer_hedge = _shares_to_lots("自營商買賣超股數(避險)")
    official_total = _shares_to_lots("三大法人買賣超股數")

    # Directional institutional flow: foreign + trust + dealer proprietary (excludes mechanical warrant/CB hedging)
    smart_institutional_lots = (foreign_ex_dealer + foreign_dealer) + trust_net + dealer_prop
    return {
        "code": code,
        "foreign_net_lots": foreign_ex_dealer + foreign_dealer,
        "trust_net_lots": trust_net,
        "dealer_prop_lots": dealer_prop,
        "dealer_hedge_lots": dealer_hedge,
        "official_total_lots": official_total,
        "smart_institutional_lots": smart_institutional_lots,
    }
