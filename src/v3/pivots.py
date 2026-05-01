from __future__ import annotations

import numpy as np
import pandas as pd


PIVOT_LEVEL_COLUMNS = (
    "camarilla_h3",
    "camarilla_h4",
    "camarilla_s3",
    "camarilla_s4",
    "session_asia_high",
    "session_asia_low",
    "session_london_high",
    "session_london_low",
    "session_premarket_high",
    "session_premarket_low",
    "session_ny_am_high",
    "session_ny_am_low",
    "prev_day_high",
    "prev_day_low",
    "prev_day_close",
    "prev_week_high",
    "prev_week_low",
)


def compute_camarilla_levels(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.groupby(frame.index.normalize()).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    previous = daily.shift(1)
    previous_range = previous["high"] - previous["low"]
    daily_levels = pd.DataFrame(index=daily.index)
    daily_levels["camarilla_h3"] = previous["close"] + previous_range * 0.275
    daily_levels["camarilla_h4"] = previous["close"] + previous_range * 0.55
    daily_levels["camarilla_s3"] = previous["close"] - previous_range * 0.275
    daily_levels["camarilla_s4"] = previous["close"] - previous_range * 0.55
    return _map_daily_levels(frame, daily_levels)


def compute_session_levels(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index, dtype=float)
    for column in (
        "session_asia_high",
        "session_asia_low",
        "session_london_high",
        "session_london_low",
        "session_premarket_high",
        "session_premarket_low",
        "session_ny_am_high",
        "session_ny_am_low",
    ):
        result[column] = np.nan

    day_groups = {day: day_frame for day, day_frame in frame.groupby(frame.index.normalize(), sort=True)}
    for day, day_frame in day_groups.items():
        prior_day = day - pd.Timedelta(days=1)
        prior_frame = day_groups.get(prior_day)

        asia_parts = []
        if prior_frame is not None and not prior_frame.empty:
            prior_evening = prior_frame.between_time("20:00", "23:59", inclusive="both")
            if not prior_evening.empty:
                asia_parts.append(prior_evening)
        current_early = day_frame.between_time("00:00", "02:00", inclusive="both")
        if not current_early.empty:
            asia_parts.append(current_early)
        if asia_parts:
            asia = pd.concat(asia_parts)
            result.loc[day_frame.index, "session_asia_high"] = asia["high"].max()
            result.loc[day_frame.index, "session_asia_low"] = asia["low"].min()

        _assign_completed_session(result, day_frame.index, day_frame, "london", "02:00", "07:00")
        _assign_completed_session(result, day_frame.index, day_frame, "premarket", "07:00", "09:30", inclusive="left")

        ny_am = day_frame.between_time("09:30", "12:00", inclusive="both")
        for i, ts in enumerate(ny_am.index):
            if i == 0:
                continue
            prior_bars = ny_am.iloc[:i]
            result.at[ts, "session_ny_am_high"] = prior_bars["high"].max()
            result.at[ts, "session_ny_am_low"] = prior_bars["low"].min()

    return result


def compute_previous_day_week_levels(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.groupby(frame.index.normalize()).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    previous = daily.shift(1)
    previous_daily = pd.DataFrame(index=daily.index)
    previous_daily["prev_day_high"] = previous["high"]
    previous_daily["prev_day_low"] = previous["low"]
    previous_daily["prev_day_close"] = previous["close"]
    mapped = _map_daily_levels(frame, previous_daily)

    week_keys = pd.MultiIndex.from_arrays(
        [frame.index.isocalendar().year.astype(int), frame.index.isocalendar().week.astype(int)],
        names=("year", "week"),
    )
    weekly = frame.groupby(week_keys).agg(high=("high", "max"), low=("low", "min"))
    previous_week = weekly.shift(1)
    lookup_high = previous_week["high"].to_dict()
    lookup_low = previous_week["low"].to_dict()
    bar_keys = list(zip(frame.index.isocalendar().year.astype(int), frame.index.isocalendar().week.astype(int)))
    mapped["prev_week_high"] = [lookup_high.get(key, np.nan) for key in bar_keys]
    mapped["prev_week_low"] = [lookup_low.get(key, np.nan) for key in bar_keys]
    return mapped


def compute_pivot_levels(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute timestamp-safe pivot/session levels from the full raw frame."""
    levels = pd.concat(
        [
            compute_camarilla_levels(raw_frame),
            compute_session_levels(raw_frame),
            compute_previous_day_week_levels(raw_frame),
        ],
        axis=1,
    )
    return levels.loc[:, list(PIVOT_LEVEL_COLUMNS)]


def attach_pivot_levels(
    execution_frame: pd.DataFrame,
    raw_frame: pd.DataFrame | None = None,
    pivot_levels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pivot_levels is None:
        if raw_frame is None:
            raise ValueError("attach_pivot_levels requires raw_frame or precomputed pivot_levels")
        pivot_levels = compute_pivot_levels(raw_frame)
    levels = pivot_levels.loc[:, list(PIVOT_LEVEL_COLUMNS)]
    aligned = levels.reindex(execution_frame.index)
    return execution_frame.join(aligned, how="left")


def _assign_completed_session(
    result: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    day_frame: pd.DataFrame,
    name: str,
    start: str,
    end: str,
    inclusive: str = "both",
) -> None:
    bars = day_frame.between_time(start, end, inclusive=inclusive)
    if bars.empty:
        return
    result.loc[target_index, f"session_{name}_high"] = bars["high"].max()
    result.loc[target_index, f"session_{name}_low"] = bars["low"].min()


def _map_daily_levels(frame: pd.DataFrame, daily_levels: pd.DataFrame) -> pd.DataFrame:
    day_series = pd.Series(frame.index.normalize(), index=frame.index)
    result = pd.DataFrame(index=frame.index)
    for column in daily_levels.columns:
        result[column] = day_series.map(daily_levels[column].to_dict())
    return result
