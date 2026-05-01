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
