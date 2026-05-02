from __future__ import annotations

import importlib
from pathlib import Path
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import SESSION_END, SESSION_START
from .indicators import atr, linreg_value, rsi


@dataclass(frozen=True)
class TradeSignal:
    time: pd.Timestamp
    direction: str
    entry: float
    stop: float
    target: float
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


StrategyFn = Callable[[pd.DataFrame, dict[str, Any]], list[TradeSignal]]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    generate: StrategyFn
    default_params: dict[str, Any]
    param_grid: dict[str, tuple[Any, ...]]
    max_signals_per_day: int | None
    session_start: str = SESSION_START
    session_end: str = SESSION_END
    requires: tuple[str, ...] = ()
    filter_of: str | None = None

    def grid(self) -> list[dict[str, Any]]:
        if not self.param_grid:
            return [dict(self.default_params)]
        keys = list(self.param_grid)
        values = [self.param_grid[key] for key in keys]
        return [{**self.default_params, **dict(zip(keys, combo, strict=True))} for combo in product(*values)]


def _append_signal(
    signals: list[TradeSignal],
    ts: pd.Timestamp,
    direction: int,
    entry: float,
    stop: float,
    target: float,
    strategy: str,
    params: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    if not np.isfinite([entry, stop, target]).all():
        return
    if direction > 0 and not (stop < entry < target):
        return
    if direction < 0 and not (target < entry < stop):
        return
    signals.append(
        TradeSignal(
            time=ts,
            direction="long" if direction > 0 else "short",
            entry=float(entry),
            stop=float(stop),
            target=float(target),
            strategy=strategy,
            params=dict(params),
            metadata=metadata or {},
        )
    )


def _daily_limiter(max_per_day: int) -> dict[pd.Timestamp, int]:
    return {}


def connors_rsi2(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
    close = df["close"]
    atr_values = atr(df, int(params["atr_period"]))
    trend = close.rolling(int(params["trend_ma"]), min_periods=int(params["trend_ma"])).mean()
    exit_ma = close.rolling(int(params["exit_ma"]), min_periods=int(params["exit_ma"])).mean()
    rsi_values = rsi(close, int(params["rsi_period"]))
    signals: list[TradeSignal] = []
    position = 0
    for i in range(max(int(params["trend_ma"]), int(params["atr_period"])) + 1, len(df)):
        ts = df.index[i]
        price = close.iloc[i]
        day = ts.normalize()
        if position == 1 and (rsi_values.iloc[i] > params["rsi_exit"] or price > exit_ma.iloc[i]):
            position = 0
            continue
        if position == -1 and (rsi_values.iloc[i] < params["rsi_entry"] or price < exit_ma.iloc[i]):
            position = 0
            continue
        if position:
            continue
        prev_rsi = rsi_values.iloc[i - 1]
        a = atr_values.iloc[i]
        if not np.isfinite([price, trend.iloc[i], prev_rsi, rsi_values.iloc[i], a]).all():
            continue
        if price > trend.iloc[i] and prev_rsi >= params["rsi_entry"] and rsi_values.iloc[i] < params["rsi_entry"]:
            _append_signal(signals, ts, 1, price, price - params["stop_mult"] * a, price + params["target_atr_mult"] * a, "connors_rsi2", params)
            position = 1
        elif price < trend.iloc[i] and prev_rsi <= params["rsi_exit"] and rsi_values.iloc[i] > params["rsi_exit"]:
            _append_signal(signals, ts, -1, price, price + params["stop_mult"] * a, price - params["target_atr_mult"] * a, "connors_rsi2", params)
            position = -1
    return signals


def ttm_squeeze(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    bb_period = int(params["bb_period"])
    kc_period = int(params["kc_period"])
    mom_period = int(params["momentum_period"])
    bb_mid = close.rolling(bb_period, min_periods=bb_period).mean()
    bb_std = close.rolling(bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + float(params["bb_std"]) * bb_std
    bb_lower = bb_mid - float(params["bb_std"]) * bb_std
    kc_mid = close.ewm(span=kc_period, adjust=False, min_periods=kc_period).mean()
    atr_kc = atr(df, kc_period)
    kc_upper = kc_mid + float(params["kc_mult"]) * atr_kc
    kc_lower = kc_mid - float(params["kc_mult"]) * atr_kc
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    midpoint = (high.rolling(mom_period, min_periods=mom_period).max() + low.rolling(mom_period, min_periods=mom_period).min()) / 2
    delta = close - ((midpoint + close.rolling(mom_period, min_periods=mom_period).mean()) / 2)
    momentum = linreg_value(delta, mom_period)
    atr_values = atr(df, int(params["atr_period"]))
    squeeze_count = squeeze.astype(int).groupby((squeeze != squeeze.shift()).cumsum()).cumsum()
    signals: list[TradeSignal] = []
    for i in range(1, len(df)):
        if not bool(squeeze.iloc[i - 1]) or bool(squeeze.iloc[i]):
            continue
        if squeeze_count.iloc[i - 1] < params["min_squeeze_bars"]:
            continue
        mom = momentum.iloc[i]
        prev_mom = momentum.iloc[i - 1]
        a = atr_values.iloc[i]
        if not np.isfinite([mom, prev_mom, a, close.iloc[i]]).all():
            continue
        if mom > 0 and mom > prev_mom:
            _append_signal(signals, df.index[i], 1, close.iloc[i], close.iloc[i] - params["stop_mult"] * a, close.iloc[i] + params["target_mult"] * a, "ttm_squeeze", params)
        elif mom < 0 and mom < prev_mom:
            _append_signal(signals, df.index[i], -1, close.iloc[i], close.iloc[i] + params["stop_mult"] * a, close.iloc[i] - params["target_mult"] * a, "ttm_squeeze", params)
    return signals


def _opening_range_signals(df: pd.DataFrame, params: dict[str, Any], strategy: str) -> list[TradeSignal]:
    atr_values = atr(df, int(params["atr_period"]))
    signals: list[TradeSignal] = []
    daily_count = _daily_limiter(int(params["max_signals_per_day"]))
    for day, day_df in df.groupby(df.index.normalize()):
        if len(day_df) < 3:
            continue
        range_minutes = int(params.get("or_minutes", params.get("ib_minutes", 60)))
        range_end = day_df.index[0] + pd.Timedelta(minutes=range_minutes)
        range_bars = day_df.loc[day_df.index <= range_end]
        if range_bars.empty:
            continue
        range_high = float(range_bars["high"].max())
        range_low = float(range_bars["low"].min())
        range_size = range_high - range_low
        if range_size <= 0:
            continue
        day_key = day.date()
        for ts, row in day_df.loc[day_df.index > range_end].iterrows():
            if ts.time() > pd.Timestamp("11:00").time():
                break
            if daily_count.get(day_key, 0) >= int(params["max_signals_per_day"]):
                break
            a = atr_values.loc[ts]
            if not np.isfinite(a) or a <= 0:
                continue
            min_range_atr = float(params.get("min_range_atr", params.get("min_ib_atr", 0.0)))
            if min_range_atr and range_size < min_range_atr * a:
                continue
            entry = float(row["close"])
            atr_window = atr_values.loc[:ts].tail(int(params.get("atr_lookback", 1))).dropna()
            if strategy == "orb_volatility_filtered":
                if len(atr_window) < int(params["atr_lookback"]):
                    continue
                pct = float((atr_window <= a).mean() * 100)
                if pct < params["min_atr_pct"] or pct > params["max_atr_pct"]:
                    continue
            if strategy == "orb_wick_rejection":
                bar_range = float(row["high"] - row["low"])
                if bar_range <= 0:
                    continue
                body_pct = abs(float(row["close"] - row["open"])) / bar_range
                if body_pct < float(params["min_body_pct"]):
                    continue
                if params.get("require_directional_body", False):
                    if entry > range_high and row["close"] <= row["open"]:
                        continue
                    if entry < range_low and row["close"] >= row["open"]:
                        continue
            if entry > range_high:
                if strategy == "orb_ib":
                    target = range_high + float(params["extension_mult"]) * range_size
                else:
                    target = entry + float(params["target_mult"]) * range_size
                stop = max(range_low, entry - float(params["stop_atr_mult"]) * a)
                _append_signal(signals, ts, 1, entry, stop, target, strategy, params)
                daily_count[day_key] = daily_count.get(day_key, 0) + 1
                break
            if entry < range_low:
                if strategy == "orb_ib":
                    target = range_low - float(params["extension_mult"]) * range_size
                else:
                    target = entry - float(params["target_mult"]) * range_size
                stop = min(range_high, entry + float(params["stop_atr_mult"]) * a)
                _append_signal(signals, ts, -1, entry, stop, target, strategy, params)
                daily_count[day_key] = daily_count.get(day_key, 0) + 1
                break
    return signals


def orb_ib(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
    return _opening_range_signals(df, params, "orb_ib")


def orb_volatility_filtered(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
    return _opening_range_signals(df, params, "orb_volatility_filtered")


def orb_wick_rejection(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
    return _opening_range_signals(df, params, "orb_wick_rejection")


def session_pivot_rejection(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
    atr_values = atr(df, int(params["atr_period"]))
    signals: list[TradeSignal] = []
    daily_count: dict[Any, int] = {}
    support_levels = (
        "camarilla_s4",
        "camarilla_s3",
        "session_asia_low",
        "session_london_low",
        "session_premarket_low",
        "session_ny_am_low",
        "prev_day_low",
        "prev_week_low",
    )
    resistance_levels = (
        "camarilla_h4",
        "camarilla_h3",
        "session_asia_high",
        "session_london_high",
        "session_premarket_high",
        "session_ny_am_high",
        "prev_day_high",
        "prev_week_high",
    )
    for i in range(1, len(df)):
        ts = df.index[i]
        day_key = ts.date()
        if daily_count.get(day_key, 0) >= int(params["max_signals_per_day"]):
            continue
        row = df.iloc[i]
        a = atr_values.iloc[i]
        if not np.isfinite(a):
            continue
        proximity = float(params["proximity_atr"]) * a
        for level_name in support_levels:
            if level_name not in df.columns:
                continue
            level = df[level_name].iloc[i]
            if np.isfinite(level) and row["low"] <= level + proximity and row["close"] > level:
                _append_signal(signals, ts, 1, row["close"], level - 0.5 * a, row["close"] + float(params["target_atr_mult"]) * a, "session_pivot_rejection", params, {"level_hit": level_name})
                daily_count[day_key] = daily_count.get(day_key, 0) + 1
                break
        else:
            for level_name in resistance_levels:
                if level_name not in df.columns:
                    continue
                level = df[level_name].iloc[i]
                if np.isfinite(level) and row["high"] >= level - proximity and row["close"] < level:
                    _append_signal(signals, ts, -1, row["close"], level + 0.5 * a, row["close"] - float(params["target_atr_mult"]) * a, "session_pivot_rejection", params, {"level_hit": level_name})
                    daily_count[day_key] = daily_count.get(day_key, 0) + 1
                    break
    return signals


def session_pivot_break(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
    atr_values = atr(df, int(params["atr_period"]))
    signals: list[TradeSignal] = []
    daily_count: dict[Any, int] = {}
    for i in range(1, len(df)):
        ts = df.index[i]
        day_key = ts.date()
        if daily_count.get(day_key, 0) >= int(params["max_signals_per_day"]):
            continue
        row = df.iloc[i]
        prev_close = df["close"].iloc[i - 1]
        a = atr_values.iloc[i]
        h4 = df["camarilla_h4"].iloc[i] if "camarilla_h4" in df.columns else np.nan
        s4 = df["camarilla_s4"].iloc[i] if "camarilla_s4" in df.columns else np.nan
        if not np.isfinite([a, h4, s4, prev_close, row["close"]]).all():
            continue
        if row["close"] > h4 and prev_close <= h4:
            _append_signal(signals, ts, 1, row["close"], row["close"] - float(params["stop_atr_mult"]) * a, row["close"] + float(params["target_atr_mult"]) * a, "session_pivot_break", params, {"level_hit": "camarilla_h4"})
            daily_count[day_key] = daily_count.get(day_key, 0) + 1
        elif row["close"] < s4 and prev_close >= s4:
            _append_signal(signals, ts, -1, row["close"], row["close"] + float(params["stop_atr_mult"]) * a, row["close"] - float(params["target_atr_mult"]) * a, "session_pivot_break", params, {"level_hit": "camarilla_s4"})
            daily_count[day_key] = daily_count.get(day_key, 0) + 1
    return signals


STRATEGIES: dict[str, StrategySpec] = {
    "connors_rsi2": StrategySpec(
        "connors_rsi2",
        connors_rsi2,
        {"rsi_period": 2, "rsi_entry": 10, "rsi_exit": 90, "exit_ma": 5, "trend_ma": 200, "stop_mult": 1.5, "target_atr_mult": 1.0, "atr_period": 14},
        {
            "rsi_period": (2, 2, 2),
            "rsi_entry": (5, 10, 15),
            "rsi_exit": (85, 90, 95),
            "exit_ma": (5, 5, 5),
            "trend_ma": (200, 200, 200),
            "stop_mult": (1.0, 1.25, 1.5),
            "target_atr_mult": (1.0, 1.25, 1.5),
            "atr_period": (14, 14, 14),
        },
        None,
    ),
    "ttm_squeeze": StrategySpec(
        "ttm_squeeze",
        ttm_squeeze,
        {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_mult": 2.0, "min_squeeze_bars": 5, "momentum_period": 12, "stop_mult": 2.0, "target_mult": 2.0, "atr_period": 14},
        {
            "bb_period": (20, 20, 20),
            "bb_std": (2.0, 2.0, 2.0),
            "kc_period": (20, 20, 20),
            "kc_mult": (1.5, 1.75, 2.0),
            "min_squeeze_bars": (4, 5, 6),
            "momentum_period": (12, 12, 12),
            "stop_mult": (1.5, 1.75, 2.0),
            "target_mult": (1.5, 1.75, 2.0),
            "atr_period": (14, 14, 14),
        },
        None,
    ),
    "orb_ib": StrategySpec(
        "orb_ib",
        orb_ib,
        {"ib_minutes": 60, "extension_mult": 1.5, "stop_atr_mult": 1.5, "atr_period": 14, "max_signals_per_day": 1, "min_ib_atr": 1.0},
        {
            "ib_minutes": (30, 45, 60),
            "extension_mult": (1.0, 1.25, 1.5),
            "stop_atr_mult": (1.0, 1.25, 1.5),
            "atr_period": (14, 14, 14),
            "max_signals_per_day": (1, 1, 1),
            "min_ib_atr": (0.5, 0.75, 1.0),
        },
        1,
    ),
    "orb_volatility_filtered": StrategySpec(
        "orb_volatility_filtered",
        orb_volatility_filtered,
        {"or_minutes": 10, "stop_atr_mult": 1.5, "target_mult": 1.0, "atr_period": 14, "atr_lookback": 100, "min_atr_pct": 25, "max_atr_pct": 85, "max_signals_per_day": 1, "min_range_atr": 0.0},
        {
            "or_minutes": (10, 12, 15),
            "stop_atr_mult": (1.5, 1.5, 1.5),
            "target_mult": (1.0, 1.25, 1.5),
            "atr_period": (14, 14, 14),
            "atr_lookback": (100, 100, 100),
            "min_atr_pct": (15, 20, 25),
            "max_atr_pct": (75, 80, 85),
            "max_signals_per_day": (1, 1, 1),
            "min_range_atr": (0.0, 0.0, 0.0),
        },
        1,
        filter_of="orb_ib",
    ),
    "orb_wick_rejection": StrategySpec(
        "orb_wick_rejection",
        orb_wick_rejection,
        {"or_minutes": 10, "stop_atr_mult": 1.5, "target_mult": 1.0, "atr_period": 14, "max_signals_per_day": 1, "min_body_pct": 0.55, "require_directional_body": False},
        {
            "or_minutes": (10, 12, 15),
            "stop_atr_mult": (1.5, 1.5, 1.5),
            "target_mult": (1.0, 1.25, 1.5),
            "atr_period": (14, 14, 14),
            "max_signals_per_day": (1, 1, 1),
            "min_body_pct": (0.45, 0.55, 0.65),
            "require_directional_body": (False, False, False),
        },
        1,
        filter_of="orb_ib",
    ),
    "session_pivot_rejection": StrategySpec(
        "session_pivot_rejection",
        session_pivot_rejection,
        {"proximity_atr": 0.5, "atr_period": 14, "target_atr_mult": 2.0, "max_signals_per_day": 2},
        {
            "proximity_atr": (0.25, 0.375, 0.5),
            "atr_period": (14, 14, 14),
            "target_atr_mult": (1.5, 1.75, 2.0),
            "max_signals_per_day": (2, 2, 2),
        },
        2,
        requires=("pivot_levels",),
    ),
    "session_pivot_break": StrategySpec(
        "session_pivot_break",
        session_pivot_break,
        {"atr_period": 14, "stop_atr_mult": 1.5, "target_atr_mult": 2.0, "max_signals_per_day": 2},
        {
            "atr_period": (14, 14, 14),
            "stop_atr_mult": (1.0, 1.25, 1.5),
            "target_atr_mult": (1.5, 1.75, 2.0),
            "max_signals_per_day": (2, 2, 2),
        },
        2,
        requires=("pivot_levels",),
    ),
}


def generate_signals(strategy_name: str, df: pd.DataFrame, params: dict[str, Any] | None = None) -> list[TradeSignal]:
    spec = STRATEGIES[strategy_name]
    active_params = dict(spec.default_params)
    if params:
        active_params.update(params)
    return spec.generate(df, active_params)


def register_strategy(spec: StrategySpec) -> None:
    from .validator import StrategyValidationError, validate_strategy_spec

    validate_strategy_spec(spec)
    if spec.name in STRATEGIES:
        raise StrategyValidationError(f"strategy {spec.name!r} is already registered")
    STRATEGIES[spec.name] = spec


def load_user_strategies() -> None:
    from .validator import validate_filter_references

    user_strategy_dir = Path(__file__).resolve().parent / "user_strategies"
    for path in sorted(user_strategy_dir.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        importlib.import_module(f"{__package__}.user_strategies.{path.stem}")
    validate_filter_references(STRATEGIES)


def _install_bundled_user_strategies() -> None:
    """Shipped user strategy registers on import — must run after ``register_strategy`` is defined."""

    importlib.import_module(f"{__package__}.user_strategies.hl2_sma_retrace_atr")


_install_bundled_user_strategies()
