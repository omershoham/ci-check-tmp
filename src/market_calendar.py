"""US stock-market (NYSE) trading-day check, so scheduled runs skip weekends and
holidays. Uses pandas_market_calendars; falls back to a weekday-only check if the
library is unavailable.

The deep (~07:00 ET) and light (~16:15 ET) runs both fall on
the same calendar date in US time, so checking today's local date is correct.
"""
from __future__ import annotations

import datetime

import pandas as pd


def trading_days(start: datetime.date, end: datetime.date) -> pd.DatetimeIndex:
    """Actual NYSE sessions in [start, end], tz-naive at midnight.

    Exists because a horizon is counted in SESSIONS everywhere else: the bar-indexed
    windows in performance._fwd_return step over real trading days, so a rebalance grid
    built from business days would be measuring a different clock (~9 US holidays a
    year is a 3-4% error on a 10-day horizon, and it drifts).

    Falls back to business days when pandas_market_calendars is missing — the same
    degradation is_trading_day makes — but says so out loud, because a measurement
    harness silently switching its date convention is worse than a noisy one.
    """
    try:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("NYSE").schedule(start_date=str(start), end_date=str(end))
        idx = pd.DatetimeIndex(sched.index)
        # mcal has returned both tz-aware and tz-naive indexes across versions; an
        # unconditional tz_localize(None) raises on the naive one and would drop us
        # into the business-day fallback for no reason.
        idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        # Force nanosecond resolution: mcal returns microsecond-unit timestamps, and a
        # us-unit index used as backtest dict keys against ns-unit price bars is a
        # lookup miss waiting to happen.
        return pd.DatetimeIndex(idx.to_numpy(dtype="datetime64[ns]"))
    except Exception as exc:  # noqa: BLE001 - library missing/error
        print(f"[market_calendar] NYSE schedule unavailable ({exc}); falling back to "
              f"business days — horizons will be ~1% long", flush=True)
        return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")


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
    """Why the market is closed on `d` -- or "open" if it isn't. Callers used to have
    to remember this was only meaningful after is_trading_day() returned False."""
    d = d or datetime.date.today()
    if is_trading_day(d):
        return "open"
    if d.weekday() >= 5:
        return "weekend"
    return "market holiday"


if __name__ == "__main__":
    today = datetime.date.today()
    print(f"{today}: {'OPEN' if is_trading_day(today) else 'CLOSED (' + reason_closed(today) + ')'}")
