#!/usr/bin/env python3
"""Audit Taiwan Equity Universe Completeness (Gates U1-U7) against Official TWSE + TPEx Masters."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from taiwan_stock.universe_registry import (
    DEFAULT_POPULAR_WATCHLIST,
    POPULAR_WATCHLIST_IS_FULL_UNIVERSE,
    TaiwanUniverseRegistry,
    to_yahoo_symbol,
)


def run_audit() -> dict[str, Any]:
    reg = TaiwanUniverseRegistry()
    source_status = {"TWSE": "OK", "TPEX": "OK"}
    official_records = []
    network_error = None
    try:
        official_records = reg.fetch_current(include_emerging=False)
    except Exception as exc:
        network_error = f"{type(exc).__name__}: {exc}"
        source_status = {"TWSE": "NETWORK_BLOCKED", "TPEX": "NETWORK_BLOCKED"}

    common = [r for r in official_records if r.security_type == "COMMON_STOCK"]
    twse_general = [r for r in common if r.market == "TWSE"]
    tib_stocks = [r for r in common if r.market == "TIB" or r.board == "TIB"]
    tpex_stocks = [r for r in common if r.market == "TPEX"]

    official_codes = {r.symbol for r in common}
    pop_codes = {str(x["code"]) for x in DEFAULT_POPULAR_WATCHLIST if not str(x["industry"]).startswith("ETF")}
    matched = pop_codes & official_codes
    missing = sorted(official_codes - pop_codes)
    extra = sorted(pop_codes - official_codes)

    wrong_suffix = sum(
        1 for r in common
        if r.yahoo_symbol != to_yahoo_symbol(r.symbol, r.market)
    )
    seen = set()
    dups = 0
    for r in common:
        k = (r.market, r.symbol)
        if k in seen:
            dups += 1
        seen.add(k)

    coverage_ratio = round(len(matched) / max(len(official_codes), 1), 4) if official_codes else 0.0
    gate_passed = (
        network_error is None
        and len(twse_general) + len(tib_stocks) >= 900
        and len(tpex_stocks) >= 750
        and len(common) >= 1700
        and wrong_suffix == 0
        and dups == 0
    )
    gate_status = "NOT_VERIFIED (NETWORK_BLOCKED)" if network_error else ("PASS" if gate_passed else "FAIL")

    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "source_status": source_status,
        "network_error": network_error,
        "official_twse_count": len(twse_general),
        "official_tib_count": len(tib_stocks),
        "official_tpex_count": len(tpex_stocks),
        "official_equity_total": len(common),
        "popular_watchlist_count": len(DEFAULT_POPULAR_WATCHLIST),
        "popular_watchlist_common_count": len(pop_codes),
        "popular_watchlist_is_full_universe": POPULAR_WATCHLIST_IS_FULL_UNIVERSE,
        "as_of_date": datetime.now(timezone.utc).date().isoformat(),
        "official_raw_count": len(official_records),
        "watchlist_overlap_count": len(matched),
        "watchlist_missing_from_official": sorted(extra),
        "popular_watchlist_official_matches": len(matched),
        "popular_watchlist_missing_count": len(missing),
        "popular_watchlist_extra_count": len(extra),
        "coverage_ratio": coverage_ratio,
        "wrong_suffix_count": wrong_suffix,
        "duplicate_count": dups,
        "full_universe_gate": gate_status,
    }
    out_dir = BASE_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "taiwan_universe_coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# Taiwan Universe Coverage & Completeness Audit",
        f"- FULL_UNIVERSE_GATE: **{gate_status}**",
        f"- POPULAR_WATCHLIST_COUNT: **{len(DEFAULT_POPULAR_WATCHLIST)}**",
        f"- POPULAR_WATCHLIST_IS_FULL_UNIVERSE: **{POPULAR_WATCHLIST_IS_FULL_UNIVERSE}**",
        f"- Official TWSE + TIB Common Equities: **{len(twse_general) + len(tib_stocks)}** (TIB: {len(tib_stocks)})",
        f"- Official TPEx Common Equities: **{len(tpex_stocks)}**",
        f"- Official Total Ordinary Equities: **{len(common)}**",
        f"- Popular Watchlist Coverage Ratio: **{coverage_ratio * 100:.2f}%**",
    ]
    (out_dir / "taiwan_universe_coverage.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_audit()
    print(json.dumps(res, indent=2, ensure_ascii=False))
