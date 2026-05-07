from __future__ import annotations

import pandas as pd

from v3.futures_session import futures_session_date


def test_futures_session_date_handles_rollover_and_maintenance_edges() -> None:
    assert futures_session_date(pd.Timestamp("2025-06-03 16:59", tz="America/New_York")) == pd.Timestamp("2025-06-03")
    assert futures_session_date(pd.Timestamp("2025-06-03 17:30", tz="America/New_York")) == pd.Timestamp("2025-06-03")
    assert futures_session_date(pd.Timestamp("2025-06-03 18:00", tz="America/New_York")) == pd.Timestamp("2025-06-04")
    assert futures_session_date(pd.Timestamp("2025-06-03 23:59", tz="America/New_York")) == pd.Timestamp("2025-06-04")
    assert futures_session_date(pd.Timestamp("2025-06-04 00:00", tz="America/New_York")) == pd.Timestamp("2025-06-04")
