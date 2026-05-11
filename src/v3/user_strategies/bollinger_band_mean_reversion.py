"""Bollinger Band Mean Reversion strategy — user strategy.

Multi-timeframe strategy using:
- HTF RSI for trend direction
- HTF ADX for trend strength
- LTF ADX for entry confirmation
- LTF Supertrend for entries
- LTF Bollinger Bands for mean reversion targets/exits
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from v3.config import SESSION_END
from v3.data import load_ohlcv, resample_ohlcv, TIMEFRAME_MINUTES
from v3.indicators import atr, rsi, adx, supertrend
from v3.strategies import StrategySpec, _append_signal, register_strategy

STRATEGY_KEY = "bollinger_band_mean_reversion"


def _exit_bar_index(
    df: pd.DataFrame,
    entry_idx: int,
    *,
    stop: float,
    target: float,
    is_long: bool,
    session_end_time,
) -> int:
    """Exit bar following evaluator.simulate_trades ordering (stop before target same bar)."""
    high = df["high"]
    low = df["low"]
    for j in range(entry_idx + 1, len(df)):
        ts = df.index[j]
        if is_long:
            if low.iloc[j] <= stop:
                return j
            if high.iloc[j] >= target:
                return j
        else:
            if high.iloc[j] >= stop:
                return j
            if low.iloc[j] <= target:
                return j
        if ts.time() >= session_end_time:
            return j
    return len(df) - 1


def _load_htf_data(htf_timeframe: str, instrument: str = "mnq") -> pd.DataFrame | None:
    """Load HTF data directly from file if available, else resample from 1min."""
    try:
        return load_ohlcv(instrument, htf_timeframe)
    except FileNotFoundError:
        return None


def bollinger_band_mean_reversion_generate(df: pd.DataFrame, params: dict) -> list:
    """Generate trade signals for Bollinger Band Mean Reversion strategy.

    Args:
        df: Primary (LT) DataFrame with columns [open, high, low, close, volume]
        params: Strategy parameters dict

    Returns:
        List of TradeSignal objects
    """
    htf_timeframe = str(params["htf_timeframe"])
    ltf_timeframe = str(params["ltf_timeframe"])

    bb_period = int(params["bb_period"])
    bb_std = float(params["bb_std"])
    atr_period = int(params["atr_period"])

    supertrend_period = int(params["supertrend_period"])
    supertrend_mult = float(params["supertrend_mult"])

    htf_rsi_period = int(params["htf_rsi_period"])
    htf_adx_period = int(params["htf_adx_period"])
    ltf_adx_period = int(params["ltf_adx_period"])

    htf_rsi_long = float(params["htf_rsi_long"])
    htf_rsi_short = float(params["htf_rsi_short"])
    htf_adx_thresh = float(params["htf_adx_thresh"])
    ltf_adx_thresh = float(params["ltf_adx_thresh"])

    stop_atr_mult = float(params["stop_atr_mult"])
    exit_mode = str(params["exit_mode"])
    rr_mult = float(params["rr_mult"])
    cooldown_bars = int(params["cooldown_bars"])

    # Load HTF data from file (e.g., 15min from mnq_15min_databento.csv)
    # Fall back to resampling 1min if file not found
    htf_df = _load_htf_data(htf_timeframe)
    if htf_df is None:
        # Resample from primary (LT) data if HTF file not available
        htf_df = resample_ohlcv(df, htf_timeframe, session_only=True)
    
    ltf_df = df  # Primary is already LTF; this strategy only reads from it.
    ltf_close = ltf_df["close"]

    if len(htf_df) < max(htf_rsi_period, htf_adx_period, bb_period) + 1:
        return []

    # HTF indicators
    htf_rsi_series = rsi(htf_df["close"], htf_rsi_period)
    htf_adx_series = adx(htf_df, htf_adx_period)

    # LTF indicators
    ltf_adx_series = adx(ltf_df, ltf_adx_period)
    ltf_atr_series = atr(ltf_df, atr_period)

    # Supertrend
    st_data = supertrend(ltf_df, supertrend_period, supertrend_mult)
    ltf_st = st_data["st"]
    ltf_st_dir = st_data["dir"]

    # Bollinger Bands on LTF
    bb_mid = ltf_close.rolling(bb_period, min_periods=bb_period).mean()
    bb_std_val = ltf_close.rolling(bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val

    # Map HTF values to LTF index
    htf_rsi_reindexed = htf_rsi_series.reindex(ltf_df.index, method="ffill")
    htf_adx_reindexed = htf_adx_series.reindex(ltf_df.index, method="ffill")

    signals: list = []
    next_allowed = 0

    # Warmup period
    warmup = max(htf_rsi_period, htf_adx_period, ltf_adx_period, bb_period, supertrend_period, atr_period) + 1
    session_end_time = pd.Timestamp(SESSION_END).time()

    for i in range(warmup, len(ltf_df)):
        if i < next_allowed:
            continue

        ts = ltf_df.index[i]
        if ts.time() >= session_end_time:
            continue

        # Current values
        htf_rsi = htf_rsi_reindexed.iloc[i]
        htf_adx = htf_adx_reindexed.iloc[i]
        ltf_adx = ltf_adx_series.iloc[i]
        ltf_st_val = ltf_st.iloc[i]
        st_dir = int(ltf_st_dir.iloc[i])
        a = ltf_atr_series.iloc[i]

        # Check for finite values
        if not all(np.isfinite(x) for x in [htf_rsi, htf_adx, ltf_adx, ltf_st_val, a]):
            continue

        if a <= 0:
            continue

        entry_price = float(ltf_close.iloc[i])
        bb_u = bb_upper.iloc[i]
        bb_l = bb_lower.iloc[i]
        bb_m = bb_mid.iloc[i]

        if not np.isfinite(bb_u) or not np.isfinite(bb_l):
            continue

        direction = 0
        stop_price = 0.0
        target_price = 0.0

        # Long conditions: HTF RSI above threshold, HTF ADX above threshold, LTF ADX confirms, Supertrend bullish
        if (
            htf_rsi >= htf_rsi_long
            and htf_adx >= htf_adx_thresh
            and ltf_adx >= ltf_adx_thresh
            and st_dir == 1
        ):
            direction = 1
            stop_price = entry_price - stop_atr_mult * a
            if exit_mode == "bb_band_tp":
                target_price = bb_u
            else:  # fixed_rr
                target_price = entry_price + rr_mult * (entry_price - stop_price)

        # Short conditions: HTF RSI below threshold, HTF ADX above threshold, LTF ADX confirms, Supertrend bearish
        elif (
            htf_rsi <= htf_rsi_short
            and htf_adx >= htf_adx_thresh
            and ltf_adx >= ltf_adx_thresh
            and st_dir == -1
        ):
            direction = -1
            stop_price = entry_price + stop_atr_mult * a
            if exit_mode == "bb_band_tp":
                target_price = bb_l
            else:  # fixed_rr
                target_price = entry_price - rr_mult * (stop_price - entry_price)

        if direction != 0:
            _append_signal(signals, ts, direction, entry_price, stop_price, target_price, STRATEGY_KEY, params)
            next_allowed = _exit_bar_index(
                ltf_df, i,
                stop=stop_price,
                target=target_price,
                is_long=(direction == 1),
                session_end_time=session_end_time,
            ) + cooldown_bars

    return signals


default_params = {
    "htf_timeframe": "15min",
    "ltf_timeframe": "5min",
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "supertrend_period": 10,
    "supertrend_mult": 3.0,
    "htf_rsi_period": 14,
    "htf_adx_period": 14,
    "ltf_adx_period": 14,
    "htf_rsi_long": 60.0,
    "htf_rsi_short": 40.0,
    "htf_adx_thresh": 35.0,
    "ltf_adx_thresh": 20.0,
    "stop_atr_mult": 2.0,
    "exit_mode": "bb_band_tp",
    "rr_mult": 1.5,
    "cooldown_bars": 10,
}

param_grid = {
    "htf_timeframe": ("15min", "30min", "1h"),
    "ltf_timeframe": ("5min", "15min", "30min"),
    "bb_period": (15, 20, 25),
    "bb_std": (1.5, 2.0, 2.5),
    "atr_period": (10, 14, 20),
    "supertrend_period": (7, 10, 14),
    "supertrend_mult": (2.0, 3.0, 4.0),
    "htf_rsi_period": (10, 14, 20),
    "htf_adx_period": (10, 14, 20),
    "ltf_adx_period": (10, 14, 20),
    "htf_rsi_long": (55.0, 60.0, 65.0),
    "htf_rsi_short": (35.0, 40.0, 45.0),
    "htf_adx_thresh": (25.0, 35.0, 45.0),
    "ltf_adx_thresh": (15.0, 20.0, 25.0),
    "stop_atr_mult": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    "exit_mode": ("bb_band_tp", "fixed_rr", "atr_trail"),
    "rr_mult": (1.0, 1.5, 2.0),
    "cooldown_bars": (5, 10, 15),
}

register_strategy(
    StrategySpec(
        name=STRATEGY_KEY,
        generate=bollinger_band_mean_reversion_generate,
        default_params=default_params,
        param_grid=param_grid,
        max_signals_per_day=None,
    )
)
