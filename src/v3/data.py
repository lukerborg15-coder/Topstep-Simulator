from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_DIR, EASTERN_TZ, SESSION_END, SESSION_START, TIMEFRAME_MINUTES, DateWindow


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    session_only: bool = False,
) -> pd.DataFrame:
    """Resample 1-minute OHLCV data to a higher timeframe.

    Args:
        frame: DataFrame with 1-minute OHLCV data, indexed by datetime (tz-aware).
        timeframe: Target timeframe name (e.g., '5min', '1h', '4h').
        session_only: If True, filter to the regular session before resampling
            and anchor derived bars to each day's 09:30 ET session open.

    Returns:
        DataFrame resampled to the target timeframe with OHLCV columns.

    Raises:
        ValueError: If timeframe is not supported.
    """
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe!r}. "
            f"Supported: {list(TIMEFRAME_MINUTES.keys())}"
        )

    period_minutes = TIMEFRAME_MINUTES[timeframe]
    working = frame.sort_index()

    if session_only:
        working = filter_regular_session(working)
        if period_minutes == 1:
            return working.copy()
        return _resample_session_anchored(working, period_minutes)

    if period_minutes == 1:
        return working.copy()

    resampled = working.resample(f"{period_minutes}min", closed="left", label="left").agg(_OHLCV_AGG)
    return resampled.dropna(subset=["open", "high", "low", "close"])


def load_ohlcv(
    instrument: str = "mnq",
    timeframe: str = "5min",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    session_only: bool = True,
) -> pd.DataFrame:
    """Load full available OHLCV first; callers slice dates after this.

    For timeframes that are not directly available on disk, the function
    will attempt to resample from 1-minute data if available.

    Args:
        instrument: Instrument symbol (e.g., 'mnq', 'mes').
        timeframe: Timeframe name (e.g., '1min', '5min', '1h').
        data_dir: Directory containing data files.
        session_only: If True, filter to regular trading session hours.

    Returns:
        DataFrame with OHLCV data, indexed by datetime (tz-aware).

    Raises:
        FileNotFoundError: If required data file is missing.
        ValueError: If timeframe is unsupported or data is malformed.
    """
    data_dir_path = Path(data_dir)
    path = data_dir_path / f"{instrument.lower()}_{timeframe.lower()}_databento.csv"
    src_path = data_dir_path / f"{instrument.lower()}_1min_databento.csv"

    if timeframe == "1min":
        if not src_path.exists():
            raise FileNotFoundError(f"Missing data file: {src_path}")
        frame = _load_ohlcv_from_csv(src_path)
        return filter_regular_session(frame) if session_only else frame

    if path.exists():
        frame = _load_ohlcv_from_csv(path)
        return filter_regular_session(frame) if session_only else frame

    if src_path.exists():
        frame = _load_ohlcv_from_csv(src_path)
        return resample_ohlcv(frame, timeframe, session_only=session_only)

    raise FileNotFoundError(
        f"Missing data file: {path}. "
        f"Also tried 1min source ({src_path}) but it doesn't exist."
    )


def _load_ohlcv_from_csv(path: Path) -> pd.DataFrame:
    """Internal helper to load and validate OHLCV from a CSV file."""
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
    return frame


def filter_regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the 09:30-16:00 ET regular session as [09:30, 16:00)."""
    return frame.between_time(SESSION_START, SESSION_END, inclusive="left").copy()


def format_ohlcv_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Format a datetime-indexed OHLCV frame for CSV persistence."""
    payload = frame.copy()
    payload["datetime"] = payload.index.map(_format_datetime)
    return payload.reset_index(drop=True)[["datetime", "open", "high", "low", "close", "volume"]]


def _resample_session_anchored(frame: pd.DataFrame, period_minutes: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    period = pd.Timedelta(minutes=period_minutes)

    for _, day_frame in frame.groupby(frame.index.normalize(), sort=True):
        if day_frame.empty:
            continue
        session_anchor = day_frame.index[0].normalize() + pd.Timedelta(hours=9, minutes=30)
        bucket_numbers = ((day_frame.index - session_anchor) // period).astype("int64")
        bucket_index = pd.DatetimeIndex(session_anchor + (bucket_numbers * period), name="datetime")
        resampled = day_frame.groupby(bucket_index).agg(_OHLCV_AGG)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        if not resampled.empty:
            pieces.append(resampled)

    if not pieces:
        return frame.iloc[0:0].copy()

    return pd.concat(pieces).sort_index()


def _format_datetime(value: pd.Timestamp) -> str:
    offset = value.strftime("%z")
    if len(offset) == 5:
        offset = offset[:3] + ":" + offset[3:]
    return value.strftime("%Y-%m-%d %H:%M:%S") + offset


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
