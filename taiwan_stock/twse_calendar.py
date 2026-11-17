"""Official TWSE trading calendar (2025-2027 holiday schedule + optional exchange_calendars XTAI)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    TAIPEI_TZ = ZoneInfo("Asia/Taipei")
except Exception:
    TAIPEI_TZ = timezone(timedelta(hours=8))

try:
    import exchange_calendars as xcals
    _XTAI_CAL = xcals.get_calendar("XTAI")
except Exception:
    _XTAI_CAL = None

# Verified TWSE market closures (national holidays & lunar new year closures 2025-2027)
TWSE_CLOSED_DATES: set[date] = {
    # 2025
    date(2025, 1, 1), date(2025, 1, 23), date(2025, 1, 24), date(2025, 1, 27),
    date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30), date(2025, 1, 31),
    date(2025, 2, 28), date(2025, 4, 3), date(2025, 4, 4), date(2025, 5, 1),
    date(2025, 5, 30), date(2025, 10, 6), date(2025, 10, 10),
    # 2026
    date(2026, 1, 1), date(2026, 2, 12), date(2026, 2, 13), date(2026, 2, 16),
    date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20),
    date(2026, 2, 27), date(2026, 4, 3), date(2026, 4, 6), date(2026, 5, 1),
    date(2026, 6, 19), date(2026, 9, 25), date(2026, 10, 9),
    # 2027
    date(2027, 1, 1), date(2027, 2, 3), date(2027, 2, 4), date(2027, 2, 5),
    date(2027, 2, 8), date(2027, 2, 9), date(2027, 2, 10), date(2027, 3, 1),
    date(2027, 4, 5), date(2027, 4, 30), date(2027, 6, 9), date(2027, 9, 15),
    date(2027, 10, 11),
}

CALENDAR_MIN_YEAR = 2025
CALENDAR_MAX_YEAR = 2027


def is_twse_trading_day(dt: date | datetime | None = None) -> bool:
    """Returns True iff the given Taipei date is a valid TWSE trading session. Fails closed outside bounds."""
    if dt is None:
        target = datetime.now(TAIPEI_TZ).date()
    elif isinstance(dt, datetime):
        target = dt.astimezone(TAIPEI_TZ).date() if dt.tzinfo else dt.date()
    else:
        target = dt

    if _XTAI_CAL is not None:
        try:
            return bool(_XTAI_CAL.is_session(target.isoformat()))
        except Exception:
            pass

    if not (CALENDAR_MIN_YEAR <= target.year <= CALENDAR_MAX_YEAR):
        # Fail closed when outside verified static calendar bounds
        return False
    if target.weekday() >= 5:
        return False
    return target not in TWSE_CLOSED_DATES
