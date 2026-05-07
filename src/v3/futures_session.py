from __future__ import annotations

import pandas as pd

from .config import EASTERN_TZ


_SESSION_ROLLOVER_HOUR = 18


def ensure_eastern_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(EASTERN_TZ)
    return ts.tz_convert(EASTERN_TZ)


def futures_session_date(value: pd.Timestamp) -> pd.Timestamp:
    """Map a timestamp to its CME-style futures session date.

    18:00 ET starts the next session date. 00:00-16:59 ET stays on the same
    calendar date. The 17:00-18:00 maintenance break is mapped to the current
    calendar date so grouping remains deterministic when those timestamps exist.
    """
    ts = ensure_eastern_timestamp(value)
    session_date = ts.normalize()
    if ts.hour >= _SESSION_ROLLOVER_HOUR:
        session_date = session_date + pd.Timedelta(days=1)
    return session_date.tz_localize(None)


def group_trades_by_futures_session_day(trades) -> dict[pd.Timestamp, list]:
    grouped: dict[pd.Timestamp, list] = {}
    for trade in sorted(trades, key=lambda item: item.exit_time):
        day = futures_session_date(trade.exit_time)
        grouped.setdefault(day, []).append(trade)
    return grouped
