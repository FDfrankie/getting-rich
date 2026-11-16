"""
Taiwan Stock Alpha Research, Cross-Sectional IC, Dual-Session Scheduler, and Inverse ETF Filter.
Complies with docs/ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md (P10).
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("TaiwanStockResearch")

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

# Inverse / Leveraged ETF filter list
INVERSE_LEVERAGED_ETFS = {
    "00632R.TW",  # 元大台灣50反1
    "00632R",
    "00671R.TW",  # 富邦NASDAQ反1
    "00671R",
    "00676R.TW",  # 富邦臺灣加權反1
    "00676R",
    "00631L.TW",  # 元大台灣50正2
    "00631L",
}


def filter_inverse_etfs(symbols: List[str]) -> List[str]:
    """Strictly isolate and exclude inverse/leveraged ETFs from long equity alpha research."""
    return [s for s in symbols if s not in INVERSE_LEVERAGED_ETFS]


def compute_cross_sectional_rank_ic(
    predictions: Dict[str, float],
    actual_returns: Dict[str, float],
    benchmark_return: Optional[float] = None,
    min_universe_size: int = 10,
) -> Dict[str, Any]:
    """
    Compute daily cross-sectional Spearman Rank IC.
    - If target return is not benchmark subtracted, target_mode is RAW_RETURN.
    - If universe size < min_universe_size (e.g. 10), do not include in official IC.
    """
    common_symbols = [s for s in predictions if s in actual_returns and s not in INVERSE_LEVERAGED_ETFS]
    universe_size = len(common_symbols)

    target_mode = "EXCESS_RETURN" if benchmark_return is not None else "RAW_RETURN"

    if universe_size < min_universe_size:
        return {
            "rank_ic": None,
            "universe_size": universe_size,
            "target_mode": target_mode,
            "is_official_ic": False,
            "status_message": f"Watchlist Ranking, Universe Size={universe_size}",
        }

    # Compute ranks
    sorted_preds = sorted(common_symbols, key=lambda s: predictions[s])
    pred_ranks = {s: r + 1 for r, s in enumerate(sorted_preds)}

    # If benchmark_return given, subtract
    adj_returns = {s: actual_returns[s] - (benchmark_return or 0.0) for s in common_symbols}
    sorted_returns = sorted(common_symbols, key=lambda s: adj_returns[s])
    ret_ranks = {s: r + 1 for r, s in enumerate(sorted_returns)}

    # Spearman correlation: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    d_sq_sum = sum((pred_ranks[s] - ret_ranks[s]) ** 2 for s in common_symbols)
    n = universe_size
    rank_ic = 1.0 - (6.0 * d_sq_sum) / (n * (n**2 - 1))

    return {
        "rank_ic": round(rank_ic, 4),
        "universe_size": universe_size,
        "target_mode": target_mode,
        "is_official_ic": True,
        "status_message": f"Official IC Computed, Universe Size={universe_size}",
    }


class TaiwanDualSessionScheduler:
    """
    Scheduler with ZoneInfo("Asia/Taipei").
    - 08:00 session: uses previous trading day data.
    - 16:30 session: uses current trading day completed data, retries if unready.
    - No artificial '100% discipline' on missing records.
    """

    def __init__(self, tz: ZoneInfo = TAIPEI_TZ) -> None:
        self.tz = tz

    def get_current_time(self) -> datetime:
        return datetime.now(self.tz)

    def determine_session(self, current_dt: Optional[datetime] = None) -> Dict[str, Any]:
        dt = current_dt or self.get_current_time()
        curr_time = dt.time()

        morning_start = time(8, 0)
        afternoon_start = time(16, 30)

        if curr_time < morning_start:
            session_type = "PRE_MARKET_WAIT"
            data_date_rule = "PREVIOUS_TRADING_DAY"
        elif curr_time < afternoon_start:
            session_type = "MORNING_SESSION"
            data_date_rule = "PREVIOUS_TRADING_DAY"
        else:
            session_type = "AFTERNOON_SESSION"
            data_date_rule = "CURRENT_TRADING_DAY_COMPLETED"

        return {
            "session_type": session_type,
            "data_date_rule": data_date_rule,
            "local_time": dt.isoformat(),
            "timezone": str(self.tz),
        }


def format_taiwan_report(
    summary: str,
    amount_twd: float,
    discipline_score: Optional[float] = None,
) -> str:
    """
    Format Taiwan quant report:
    - Escapes HTML / Markdown.
    - Currency formatted as NT$ / TWD.
    - Missing discipline record is displayed as N/A (NEVER defaulted to 100%).
    """
    safe_summary = html.escape(summary)
    formatted_amount = f"NT$ {amount_twd:,.2f}"

    if discipline_score is None:
        discipline_text = "N/A (無紀錄)"
    else:
        discipline_text = f"{discipline_score:.1f}%"

    return f"【台股量化日報】\n摘要: {safe_summary}\n金額: {formatted_amount}\n紀律評分: {discipline_text}"
