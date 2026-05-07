from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from v3.futures_session import futures_session_date


TZ = ZoneInfo("America/New_York")


def test_futures_session_date_mapping_examples() -> None:
    assert futures_session_date(pd.Timestamp(2024, 1, 2, 17, 30, tz=TZ)) == pd.Timestamp("2024-01-02")
    assert futures_session_date(pd.Timestamp(2024, 1, 2, 18, 0, tz=TZ)) == pd.Timestamp("2024-01-03")
    assert futures_session_date(pd.Timestamp(2024, 1, 3, 0, 30, tz=TZ)) == pd.Timestamp("2024-01-03")
    assert futures_session_date(pd.Timestamp(2024, 1, 3, 16, 59, tz=TZ)) == pd.Timestamp("2024-01-03")
