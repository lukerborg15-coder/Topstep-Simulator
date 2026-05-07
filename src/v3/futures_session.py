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
    """Return the CME index futures session date for a timestamp in ET.

    Session semantics:
    - 18:00 ET through 23:59 ET belongs to the next futures session date.
    - 00:00 ET through 16:59 ET belongs to the same calendar date.
    - 17:00 ET through 17:59 ET is the maintenance break. When present in data,
      we map it to the current calendar date so day-grouping stays deterministic.
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


__all__ = ["ensure_eastern_timestamp", "futures_session_date", "group_trades_by_futures_session_day"]
