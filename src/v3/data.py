from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_DIR, EASTERN_TZ, SESSION_END, SESSION_START, DateWindow


def load_ohlcv(
    instrument: str = "mnq",
    timeframe: str = "5min",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    session_only: bool = True,
) -> pd.DataFrame:
    """Load full available OHLCV first; callers slice dates after this."""
    path = Path(data_dir) / f"{instrument.lower()}_{timeframe.lower()}_databento.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    frame = pd.read_csv(path)
    if "datetime" not in frame.columns:
        raise ValueError(f"{path} must include a datetime column")
    index = pd.to_datetime(frame.pop("datetime"), utc=True).dt.tz_convert(EASTERN_TZ)
    frame.index = pd.DatetimeIndex(index, name="datetime")
    frame = frame.sort_index()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            raise ValueError(f"{path} missing required column: {column}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if session_only:
        frame = frame.between_time(SESSION_START, SESSION_END, inclusive="both")
    return frame


def slice_window(frame: pd.DataFrame, window: DateWindow) -> pd.DataFrame:
    """Slice frame to the given DateWindow.

    Handles both tz-aware and tz-naive indexes: if the frame index has no
    timezone, boundary timestamps are made tz-naive to avoid comparison errors.
    Real data loaded via load_ohlcv() is always tz-aware (America/New_York);
    synthetic test frames may be tz-naive.
    """
    if frame.index.tz is not None:
        start = pd.Timestamp(window.start, tz=EASTERN_TZ)
        end = pd.Timestamp(window.end, tz=EASTERN_TZ) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        start = pd.Timestamp(window.start)
        end = pd.Timestamp(window.end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return frame.loc[(frame.index >= start) & (frame.index <= end)].copy()


def assert_full_history_loaded(frame: pd.DataFrame, timeframe: str) -> None:
    if frame.empty:
        raise ValueError(f"{timeframe} data loaded empty")
    if frame.index.min().date() > pd.Timestamp("2022-09-01").date():
        raise ValueError(f"{timeframe} data does not include the expected 2022 start")
    if frame.index.max().date() < pd.Timestamp("2026-03-01").date():
        raise ValueError(f"{timeframe} data does not include the expected 2026 end")
