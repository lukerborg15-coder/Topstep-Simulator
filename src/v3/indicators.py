from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def linreg_value(series: pd.Series, period: int) -> pd.Series:
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        slope = ((x - x_mean) * (values - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return float(slope * (period - 1) + intercept)

    return series.rolling(period, min_periods=period).apply(calc, raw=True)


def directional_efficiency(close: pd.Series, lookback: int = 50) -> pd.Series:
    direction = (close - close.shift(lookback)).abs()
    path = close.diff().abs().rolling(lookback, min_periods=lookback).sum()
    return direction / path.replace(0, np.nan)


def rolling_slope(close: pd.Series, lookback: int = 50) -> pd.Series:
    x = np.arange(lookback, dtype=float)
    x = x - x.mean()
    denom = (x**2).sum()

    def calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values - values.mean()
        return float((x * y).sum() / denom)

    return close.rolling(lookback, min_periods=lookback).apply(calc, raw=True)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX).

    Returns a Series with ADX values. The series index aligns with df.index.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional Movement
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    diff_high = high.diff()
    diff_low = -low.diff()

    up_mask = (diff_high > diff_low) & (diff_high > 0)
    down_mask = (diff_low > diff_high) & (diff_low > 0)

    plus_dm[up_mask] = diff_high[up_mask]
    minus_dm[down_mask] = diff_low[down_mask]

    # Smooth with Wilder's smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / period

    # Calculate smoothed TR, +DM, -DM
    tr = true_range(df)

    smooth_tr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Directional Indicators
    plus_di = 100 * (smooth_plus_dm / smooth_tr.replace(0, np.nan))
    minus_di = 100 * (smooth_minus_dm / smooth_tr.replace(0, np.nan))

    # DX
    di_sum = plus_di + minus_di
    dx = 100 * ((plus_di - minus_di).abs() / di_sum.replace(0, np.nan))

    # ADX = Wilder's smoothing of DX
    adx_series = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx_series.fillna(0)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend indicator.

    Returns a DataFrame with columns:
        - 'st': Supertrend value
        - 'dir': Direction (1 = uptrend, -1 = downtrend)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = true_range(df)
    atr_series = tr.rolling(period, min_periods=period).mean()

    # HL2 basic price
    hl2 = (high + low) / 2.0

    # Upper and lower bands
    upper_band = hl2 + multiplier * atr_series
    lower_band = hl2 - multiplier * atr_series

    # Initialize arrays
    st = np.full(len(df), np.nan)
    direction = np.zeros(len(df), dtype=int)

    # Forward pass
    for i in range(1, len(df)):
        prev_close = close.iloc[i - 1]
        curr_close = close.iloc[i]
        prev_upper = upper_band.iloc[i - 1] if np.isfinite(upper_band.iloc[i - 1]) else upper_band.iloc[i]
        prev_lower = lower_band.iloc[i - 1] if np.isfinite(lower_band.iloc[i - 1]) else lower_band.iloc[i]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]

        if np.isnan(st[i - 1]):
            if curr_close > curr_upper:
                st[i] = curr_lower
                direction[i] = 1
            else:
                st[i] = curr_upper
                direction[i] = -1
        elif direction[i - 1] == 1:
            if curr_close < prev_lower:
                st[i] = curr_upper
                direction[i] = -1
            else:
                st[i] = max(prev_lower, curr_lower)
                direction[i] = 1
        else:  # direction[i-1] == -1
            if curr_close > prev_upper:
                st[i] = curr_lower
                direction[i] = 1
            else:
                st[i] = min(prev_upper, curr_upper)
                direction[i] = -1

    result = pd.DataFrame({"st": st, "dir": direction}, index=df.index)
    return result