from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import EASTERN_TZ


ALLOWED_REQUIRES = {"pivot_levels"}
PIVOT_LEVEL_COLUMNS = (
    "camarilla_h4",
    "camarilla_s4",
    "camarilla_h3",
    "camarilla_s3",
    "prev_day_high",
    "prev_day_low",
    "prev_week_high",
    "prev_week_low",
    "session_asia_high",
    "session_asia_low",
    "session_london_high",
    "session_london_low",
    "session_premarket_high",
    "session_premarket_low",
    "session_ny_am_high",
    "session_ny_am_low",
)


class StrategyValidationError(Exception):
    pass


def validate_strategy_spec(spec: Any) -> None:
    if not isinstance(getattr(spec, "name", None), str) or not spec.name.strip():
        raise StrategyValidationError("spec.name must be a non-empty string")
    if not callable(getattr(spec, "generate", None)):
        raise StrategyValidationError("spec.generate must be callable")
    if not isinstance(getattr(spec, "default_params", None), dict):
        raise StrategyValidationError("spec.default_params must be a dict")
    if not isinstance(getattr(spec, "param_grid", None), dict):
        raise StrategyValidationError("spec.param_grid must be a dict")

    max_signals_per_day = getattr(spec, "max_signals_per_day", None)
    _validate_param_grid(spec.default_params, spec.param_grid)
    _validate_max_signals_per_day(max_signals_per_day)

    requires = getattr(spec, "requires", None)
    if not isinstance(requires, tuple):
        raise StrategyValidationError("spec.requires must be a tuple of strings")
    for requirement in requires:
        if not isinstance(requirement, str):
            raise StrategyValidationError("spec.requires must be a tuple of strings")
        if requirement not in ALLOWED_REQUIRES:
            raise StrategyValidationError(f"unsupported requirement {requirement!r}")

    filter_of = getattr(spec, "filter_of", None)
    if filter_of is not None and not isinstance(filter_of, str):
        raise StrategyValidationError("spec.filter_of must be None or a string")

    synthetic_df = _synthetic_ohlcv(spec.session_start, spec.session_end, requires)
    try:
        signals = spec.generate(synthetic_df, dict(spec.default_params))
    except Exception as exc:  # noqa: BLE001 - include original failure in validation message.
        raise StrategyValidationError(f"smoke run failed: {exc}") from exc

    if not isinstance(signals, list):
        raise StrategyValidationError("smoke run must return a list")
    for signal in signals:
        _validate_signal(signal)
    _validate_signal_cap(signals, max_signals_per_day)


def validate_filter_references(strategies: dict) -> None:
    for key, spec in strategies.items():
        filter_of = getattr(spec, "filter_of", None)
        if filter_of is not None and filter_of not in strategies:
            raise StrategyValidationError(f"strategy {key!r} filter_of references missing strategy {filter_of!r}")


def _validate_param_grid(default_params: dict[str, Any], param_grid: dict[str, Any]) -> None:
    default_keys = set(default_params)
    grid_keys = set(param_grid)
    if grid_keys != default_keys:
        missing = sorted(default_keys - grid_keys)
        extra = sorted(grid_keys - default_keys)
        raise StrategyValidationError(
            f"param_grid keys must exactly match default_params keys; missing={missing!r}, extra={extra!r}"
        )

    for key, values in param_grid.items():
        if not isinstance(values, tuple):
            raise StrategyValidationError(f"param_grid entry {key!r} must be a tuple")
        if len(values) < 3:
            raise StrategyValidationError(f"param_grid entry {key!r} must contain at least 3 values")
        if default_params[key] not in values:
            raise StrategyValidationError(f"default value for {key!r} must be included in its param_grid tuple")
        if _is_numeric_grid(values) and list(values) != sorted(values):
            raise StrategyValidationError(f"numeric param_grid entry {key!r} must be ordered smallest to largest")


def _is_numeric_grid(values: tuple[Any, ...]) -> bool:
    return all(
        isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
        for value in values
    )


def _validate_max_signals_per_day(max_signals_per_day: Any) -> None:
    if max_signals_per_day is None:
        return
    if isinstance(max_signals_per_day, bool) or not isinstance(max_signals_per_day, int):
        raise StrategyValidationError("spec.max_signals_per_day must be a non-negative int or None")
    if max_signals_per_day < 0:
        raise StrategyValidationError("spec.max_signals_per_day must be a non-negative int or None")


def _validate_signal_cap(signals: list[Any], max_signals_per_day: int | None) -> None:
    if max_signals_per_day is None:
        return

    daily_counts: dict[pd.Timestamp, int] = {}
    for signal in signals:
        raw_time = getattr(signal, "time", None)
        if not isinstance(raw_time, pd.Timestamp) or pd.isna(raw_time) or raw_time.tzinfo is None:
            raise StrategyValidationError(
                "signal time must be a valid timezone-aware pd.Timestamp for max_signals_per_day validation"
            )
        day = raw_time.normalize()
        daily_counts[day] = daily_counts.get(day, 0) + 1
        if daily_counts[day] > max_signals_per_day:
            raise StrategyValidationError(f"smoke run exceeded max_signals_per_day={max_signals_per_day} on {day.date()}")


def _synthetic_ohlcv(session_start: str, session_end: str, requires: tuple[str, ...]) -> pd.DataFrame:
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        index = _session_index(session_start, session_end, bars=200)
        walk = 100.0 + np.cumsum(np.random.normal(0.0, 0.35, len(index)))
        close = walk
        open_ = np.r_[walk[0], walk[:-1]] + np.random.normal(0.0, 0.08, len(index))
        high = np.maximum(open_, close) + np.abs(np.random.normal(0.25, 0.08, len(index)))
        low = np.minimum(open_, close) - np.abs(np.random.normal(0.25, 0.08, len(index)))
        volume = np.random.randint(750, 4_000, len(index))
    finally:
        np.random.set_state(rng_state)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    if "pivot_levels" in requires:
        _add_synthetic_pivot_levels(df)
    return df


def _session_index(session_start: str, session_end: str, bars: int) -> pd.DatetimeIndex:
    ranges: list[pd.DatetimeIndex] = []
    day = pd.Timestamp("2024-01-02")
    while sum(len(day_range) for day_range in ranges) < bars:
        start = pd.Timestamp(f"{day.date()} {session_start}", tz=EASTERN_TZ)
        end = pd.Timestamp(f"{day.date()} {session_end}", tz=EASTERN_TZ)
        ranges.append(pd.date_range(start=start, end=end, freq="5min"))
        day += pd.Timedelta(days=1)
    return ranges[0].append(ranges[1:])[:bars]


def _add_synthetic_pivot_levels(df: pd.DataFrame) -> None:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    df["camarilla_h4"] = close + 1.25
    df["camarilla_s4"] = close - 1.25
    df["camarilla_h3"] = close + 0.85
    df["camarilla_s3"] = close - 0.85
    df["prev_day_high"] = high + 1.75
    df["prev_day_low"] = low - 1.75
    df["prev_week_high"] = high + 2.5
    df["prev_week_low"] = low - 2.5
    df["session_asia_high"] = high + 0.95
    df["session_asia_low"] = low - 0.95
    df["session_london_high"] = high + 1.1
    df["session_london_low"] = low - 1.1
    df["session_premarket_high"] = high + 0.65
    df["session_premarket_low"] = low - 0.65
    df["session_ny_am_high"] = high + 0.45
    df["session_ny_am_low"] = low - 0.45


def _validate_signal(signal: Any) -> None:
    direction = getattr(signal, "direction", None)
    entry = getattr(signal, "entry", None)
    stop = getattr(signal, "stop", None)
    target = getattr(signal, "target", None)
    if direction not in {"long", "short"}:
        raise StrategyValidationError(f"signal direction must be 'long' or 'short', got {direction!r}")
    if direction == "long" and not (stop < entry < target):
        raise StrategyValidationError("long signal must satisfy stop < entry < target")
    if direction == "short" and not (target < entry < stop):
        raise StrategyValidationError("short signal must satisfy target < entry < stop")


validate_strategy = validate_strategy_spec
