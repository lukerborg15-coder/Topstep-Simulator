"""HL2 SMA retrace with ATR stop/target — user strategy.

See plan: HL2 = (high+low)/2, SMA(HL2), touch + isolation filter, directional tap,
fixed ATR multiples for stop/target. Position spacing matches evaluator bar priority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v3.config import SESSION_END
from v3.indicators import atr
from v3.strategies import StrategySpec, _append_signal, register_strategy

STRATEGY_KEY = "hl2_sma_retrace_atr"


def _exit_bar_index(
    df: pd.DataFrame,
    entry_idx: int,
    *,
    stop: float,
    target: float,
    is_long: bool,
) -> int:
    """Exit bar following evaluator.simulate_trades ordering (stop before target same bar)."""
    session_end_time = pd.Timestamp(SESSION_END).time()
    for j in range(entry_idx + 1, len(df)):
        row = df.iloc[j]
        ts = df.index[j]
        if is_long:
            if row["low"] <= stop:
                return j
            if row["high"] >= target:
                return j
        else:
            if row["high"] >= stop:
                return j
            if row["low"] <= target:
                return j
        if ts.time() >= session_end_time:
            return j
    return len(df) - 1


def hl2_sma_retrace_generate(df: pd.DataFrame, params: dict) -> list:
    ma_len = int(params["ma_length"])
    atr_period = int(params["atr_period"])
    untouched = int(params["untouched_lookback"])
    stop_mult = float(params["stop_atr_mult"])
    target_mult = float(params["target_atr_mult"])

    hl2 = (df["high"] + df["low"]) / 2.0
    ma = hl2.rolling(ma_len, min_periods=ma_len).mean()
    atr_series = atr(df, atr_period)

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    ma_arr = ma.to_numpy(dtype=float)

    signals: list = []
    start = max(ma_len, atr_period, untouched + 1) + 1
    next_allowed = 0

    for i in range(start, len(df)):
        if i < next_allowed:
            continue

        m = ma_arr[i]
        a = float(atr_series.iloc[i])
        if not np.isfinite(m) or not np.isfinite(a) or a <= 0:
            continue

        if not (lows[i] <= m <= highs[i]):
            continue

        ok_iso = True
        for k in range(1, untouched + 1):
            j = i - k
            mj = ma_arr[j]
            if not np.isfinite(mj):
                ok_iso = False
                break
            if lows[j] <= mj <= highs[j]:
                ok_iso = False
                break
        if not ok_iso:
            continue

        prev_c = closes[i - 1]
        pm = ma_arr[i - 1]
        if not np.isfinite(pm):
            continue

        ts = df.index[i]
        entry = float(m)
        long_bias = prev_c > pm
        short_bias = prev_c < pm
        if long_bias:
            stop = entry - stop_mult * a
            targ = entry + target_mult * a
            _append_signal(signals, ts, 1, entry, stop, targ, STRATEGY_KEY, params)
            next_allowed = _exit_bar_index(df, i, stop=stop, target=targ, is_long=True) + 1
        elif short_bias:
            stop = entry + stop_mult * a
            targ = entry - target_mult * a
            _append_signal(signals, ts, -1, entry, stop, targ, STRATEGY_KEY, params)
            next_allowed = _exit_bar_index(df, i, stop=stop, target=targ, is_long=False) + 1

    return signals


_MA_GRID = tuple(range(15, 31))
_UNTOUCH_GRID = tuple(range(5, 16))

register_strategy(
    StrategySpec(
        name=STRATEGY_KEY,
        generate=hl2_sma_retrace_generate,
        default_params={
            "ma_length": 21,
            "atr_period": 14,
            "untouched_lookback": 8,
            "stop_atr_mult": 1.0,
            "target_atr_mult": 2.0,
        },
        param_grid={
            "ma_length": _MA_GRID,
            "atr_period": (10, 14, 20),
            "untouched_lookback": _UNTOUCH_GRID,
            "stop_atr_mult": (0.5, 1.0, 1.5, 2.0),
            "target_atr_mult": (0.5, 1.0, 1.5, 2.0),
        },
        max_signals_per_day=None,
    )
)
