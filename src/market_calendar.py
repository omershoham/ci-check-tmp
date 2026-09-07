"""US stock-market (NYSE) trading-day check, so scheduled runs skip weekends and
holidays. Uses pandas_market_calendars; falls back to a weekday-only check if the
library is unavailable.

The deep (~07:00 ET) and light (~16:15 ET) runs both fall on
the same calendar date in US time, so checking today's local date is correct.
"""
from __future__ import annotations

import datetime


def is_trading_day(d: datetime.date | None = None) -> bool:
    d = d or datetime.date.today()
    if d.weekday() >= 5:  # Saturday/Sunday
        return False
    try:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("NYSE").schedule(start_date=str(d), end_date=str(d))
        return not sched.empty
    except Exception:  # noqa: BLE001 - library missing/error → weekday-only fallback
        return True


def reason_closed(d: datetime.date | None = None) -> str:
    d = d or datetime.date.today()
    if d.weekday() >= 5:
        return "weekend"
    return "market holiday"


if __name__ == "__main__":
    today = datetime.date.today()
    print(f"{today}: {'OPEN' if is_trading_day(today) else 'CLOSED (' + reason_closed(today) + ')'}")
