"""DST-aware America/New_York score refresh and batch score attribution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
TAIPEI = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class BatchAttribution:
    batch_date: str
    submitted_alpha_ids: tuple[str, ...]
    score_before: float | None
    score_after: float | None
    score_delta: float | None
    attribution_status: str
    attribution_confidence: str
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_refresh_at_for_taipei_date(
    taipei_date: str,
    refresh_hour_new_york: int = 3,
) -> datetime:
    """Compute Taipei datetime corresponding to 03:00 America/New_York using ZoneInfo (DST-safe)."""
    taipei_midnight = datetime.fromisoformat(f"{taipei_date}T00:00:00").replace(tzinfo=TAIPEI)
    new_york_date = taipei_midnight.astimezone(NEW_YORK).date()
    new_york_refresh = datetime(
        year=new_york_date.year,
        month=new_york_date.month,
        day=new_york_date.day,
        hour=refresh_hour_new_york,
        minute=0,
        second=0,
        tzinfo=NEW_YORK,
    )
    return new_york_refresh.astimezone(TAIPEI)


def build_batch_attribution(
    submissions: Iterable[Mapping[str, Any]],
    *,
    score_before: float | None,
    score_after: float | None,
    observed_at: datetime,
) -> BatchAttribution:
    rows = list(submissions)
    ids = tuple(
        str(row.get("brain_alpha_id") or row.get("alpha_id") or row.get("id"))
        for row in rows
    )

    if score_before is None or score_after is None:
        score_delta = None
        status = "SCORE_UNAVAILABLE"
        confidence = "NONE"
    else:
        score_delta = float(score_after) - float(score_before)
        if len(ids) == 1:
            status = "SINGLE_SUBMISSION_BATCH"
            confidence = "MEDIUM"
        elif len(ids) > 1:
            status = "BATCH_ATTRIBUTION"
            confidence = "LOW"
        else:
            status = "NO_SUBMISSIONS"
            confidence = "NONE"

    batch_date = (
        str(rows[0].get("batch_date"))
        if rows and rows[0].get("batch_date")
        else observed_at.astimezone(TAIPEI).date().isoformat()
    )

    return BatchAttribution(
        batch_date=batch_date,
        submitted_alpha_ids=ids,
        score_before=score_before,
        score_after=score_after,
        score_delta=score_delta,
        attribution_status=status,
        attribution_confidence=confidence,
        observed_at=observed_at.isoformat(),
    )


def estimate_days_to_10k(
    daily_observations: Iterable[Mapping[str, Any]],
    *,
    target_score: float = 10000.0,
    min_observations: int = 7,
) -> dict[str, Any]:
    """Estimate completion date toward 10,000 points ONLY when >= 7 valid daily observations exist."""
    valid = [
        row for row in daily_observations
        if row.get("total_score") is not None and row.get("daily_score") is not None
    ]
    if len(valid) < min_observations:
        return {
            "ready": False,
            "reason": "INSUFFICIENT_OBSERVATIONS",
            "observations": len(valid),
            "required_observations": min_observations,
            "estimated_days": None,
            "optimistic_days": None,
            "conservative_days": None,
        }
    latest_total = float(valid[-1]["total_score"])
    remaining = max(0.0, target_score - latest_total)
    if remaining <= 0.0:
        return {
            "ready": True,
            "reason": "TARGET_REACHED",
            "observations": len(valid),
            "estimated_days": 0,
            "optimistic_days": 0,
            "conservative_days": 0,
        }
    deltas = [max(0.0, float(r["daily_score"])) for r in valid]
    mean_gain = sum(deltas) / len(deltas)
    if mean_gain <= 1e-6:
        return {
            "ready": False,
            "reason": "NON_POSITIVE_DAILY_GAIN",
            "observations": len(valid),
            "estimated_days": None,
            "optimistic_days": None,
            "conservative_days": None,
        }
    sorted_d = sorted(deltas)
    p25 = max(1.0, sorted_d[max(0, int(len(sorted_d) * 0.25))])
    p75 = max(mean_gain, sorted_d[min(len(sorted_d) - 1, int(len(sorted_d) * 0.75))])
    est_days = int(round(remaining / mean_gain))
    opt_days = int(round(remaining / p75))
    con_days = int(round(remaining / p25))
    return {
        "ready": True,
        "reason": "OK",
        "observations": len(valid),
        "remaining_points": round(remaining, 2),
        "mean_daily_gain": round(mean_gain, 2),
        "estimated_days": est_days,
        "optimistic_days": opt_days,
        "conservative_days": con_days,
    }
