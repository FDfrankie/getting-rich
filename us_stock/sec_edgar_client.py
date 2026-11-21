"""Point-in-Time US Equity Fundamentals via official free SEC EDGAR CompanyFacts API."""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

GAAP_TAGS: dict[str, tuple[str, ...]] = {
    "assets": ("Assets",),
    "gross_profit": ("GrossProfit",),
    "revenues": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "ocf": ("NetCashProvidedByUsedInOperatingActivities",),
    "liabilities": ("Liabilities",),
    "equity": ("StockholdersEquity",),
}


def parse_company_facts_pit(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse SEC EDGAR CompanyFacts JSON into a `filed`-date indexed PIT DataFrame."""
    us_gaap = (payload.get("facts") or {}).get("us-gaap") or {}
    rows: dict[str, dict[str, float]] = {}
    for canonical, tags in GAAP_TAGS.items():
        for tag in tags:
            units = (us_gaap.get(tag) or {}).get("units") or {}
            usd_items = units.get("USD") or []
            if not usd_items:
                continue
            for item in usd_items:
                form = str(item.get("form") or "")
                if form not in {"10-K", "10-Q"}:
                    continue
                filed = str(item.get("filed") or "")
                val = item.get("val")
                if filed and isinstance(val, (int, float)):
                    rows.setdefault(filed, {})[canonical] = float(val)
            break

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill()

    if "gross_profit" in df.columns and "assets" in df.columns:
        df["gross_profitability"] = df["gross_profit"] / df["assets"].replace(0.0, pd.NA)
    if "ocf" in df.columns and "net_income" in df.columns and "assets" in df.columns:
        df["cash_accrual_quality"] = (df["ocf"] - df["net_income"]) / df["assets"].replace(0.0, pd.NA)
    if "operating_income" in df.columns and "revenues" in df.columns:
        df["operating_margin"] = df["operating_income"] / df["revenues"].replace(0.0, pd.NA)
    return df


class SecEdgarClient:
    """Rate-limited (<= 8 req/s) SEC EDGAR client enforcing true `filed`-date Point-in-Time alignment."""

    def __init__(self, user_agent: str | None = None, min_interval_sec: float = 0.14) -> None:
        self.user_agent = user_agent or os.getenv("SEC_EDGAR_USER_AGENT", "AntigravityQuant research@example.com")
        self.min_interval_sec = float(min_interval_sec)
        self._last_req = 0.0
        self._ticker_to_cik: dict[str, int] | None = None

    def _get_json(self, url: str, timeout: int = 20) -> dict[str, Any]:
        wait = self.min_interval_sec - (time.monotonic() - self._last_req)
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
        )
        self._last_req = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    def load_ticker_cik_map(self) -> dict[str, int]:
        if self._ticker_to_cik is None:
            raw = self._get_json(SEC_TICKERS_URL)
            self._ticker_to_cik = {
                str(v["ticker"]).upper(): int(v["cik_str"])
                for v in raw.values()
                if "ticker" in v and "cik_str" in v
            }
        return self._ticker_to_cik

    def fetch_pit_fundamentals(self, ticker: str) -> pd.DataFrame:
        cik_map = self.load_ticker_cik_map()
        cik = cik_map.get(ticker.upper())
        if cik is None:
            return pd.DataFrame()
        return parse_company_facts_pit(self._get_json(SEC_FACTS_URL.format(cik=cik)))


def compute_pead_sue_score(
    quarterly_eps_records: list[dict[str, object]],
    as_of_date: str,
) -> float | None:
    """Compute strictly Point-in-Time Standardized Unexpected Earnings (SUE) using filed < as_of_date."""
    import math

    as_of = str(as_of_date).strip()[:10]
    eligible = [
        r for r in quarterly_eps_records
        if str(r.get("filed") or "").strip()[:10] < as_of and r.get("eps") is not None
    ]
    # Deduplicate by period end date keeping the latest filed restatement available as of as_of_date
    by_end: dict[str, dict[str, object]] = {}
    for r in eligible:
        end_key = str(r.get("end") or r.get("filed") or "").strip()[:10]
        prev = by_end.get(end_key)
        if prev is None or str(r.get("filed") or "") >= str(prev.get("filed") or ""):
            by_end[end_key] = r
    ordered = [by_end[k] for k in sorted(by_end.keys())]
    if len(ordered) < 5:
        return None
    eps_vals = [float(r["eps"]) for r in ordered]
    # Seasonal YoY EPS changes (q - q_lag4) over trailing up to 8 quarters
    yoy_diffs = [eps_vals[i] - eps_vals[i - 4] for i in range(4, len(eps_vals))]
    recent = yoy_diffs[-8:]
    if len(recent) < 2:
        return None
    mean_d = sum(recent) / len(recent)
    var_d = sum((x - mean_d) ** 2 for x in recent) / max(1, len(recent) - 1)
    std_d = math.sqrt(var_d)
    if std_d < 1e-9:
        return 0.0
    return round((recent[-1] - mean_d) / std_d, 4)


def compute_form4_cluster_buy_score(
    filings: list[dict[str, object]],
    as_of_date: str,
    *,
    lookback_days: int = 30,
) -> dict[str, float | int]:
    """Compute strictly PIT SEC Form 4 open-market insider cluster buy score (code=='P', non-10b5-1)."""
    import math
    from datetime import date, timedelta

    as_of_dt = date.fromisoformat(str(as_of_date).strip()[:10])
    cutoff_dt = as_of_dt - timedelta(days=max(1, int(lookback_days)))

    distinct_insiders: set[str] = set()
    total_usd = 0.0
    for f in filings:
        filed_s = str(f.get("filed") or "").strip()[:10]
        if not filed_s:
            continue
        try:
            f_dt = date.fromisoformat(filed_s)
        except ValueError:
            continue
        if f_dt > as_of_dt or f_dt < cutoff_dt:
            continue
        if str(f.get("transactionCode") or "").upper() != "P":
            continue
        if bool(f.get("is_10b5_1")):
            continue
        insider = str(f.get("insider_name") or f.get("rptOwnerCik") or "").strip()
        if insider:
            distinct_insiders.add(insider)
        shares = float(f.get("shares") or 0.0)
        price = float(f.get("price") or 0.0)
        if shares > 0 and price > 0:
            total_usd += shares * price

    n_insiders = len(distinct_insiders)
    cluster_score = round(math.log1p(total_usd) * (1.0 + 0.25 * (n_insiders - 2)), 4) if n_insiders >= 2 else 0.0
    return {
        "distinct_insiders": n_insiders,
        "total_buy_usd": round(total_usd, 2),
        "cluster_buy_score": cluster_score,
    }
