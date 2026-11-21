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
