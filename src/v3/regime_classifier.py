"""Per-strategy regime classifier.

Determines whether a strategy's edge leans toward calm or volatile market conditions.
Vol proxy: rolling stdev of session log returns (no external data required).

Trades at entry are bucketed calm (vol <= median) vs volatile (vol > median).
Output: single verdict per strategy run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DateWindow
from .trades import TradeResult

_N_MIN_BUCKET = 10
_EXPECTANCY_EPSILON = 5.0  # min mean_pnl advantage to declare preference


@dataclass(frozen=True)
class BucketStats:
    count: int
    win_rate: float
    expectancy_r: float
    total_net_pnl: float
    mean_net_pnl: float


@dataclass(frozen=True)
class RegimeFitResult:
    verdict: str  # prefers_calm | prefers_volatile | mixed | insufficient_data
    calm_stats: BucketStats
    volatile_stats: BucketStats
    vol_window: int
    total_trades: int
    window_name: str


def _bucket_stats(trades: list[TradeResult]) -> BucketStats:
    if not trades:
        return BucketStats(0, 0.0, 0.0, 0.0, 0.0)
    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    rs = np.array([t.r_multiple for t in trades], dtype=float)
    return BucketStats(
        count=len(trades),
        win_rate=float((pnls > 0).mean()),
        expectancy_r=float(np.mean(rs)),
        total_net_pnl=float(pnls.sum()),
        mean_net_pnl=float(np.mean(pnls)),
    )


def classify_regime_fit(
    frame: pd.DataFrame,
    trades: list[TradeResult],
    window: DateWindow,
    *,
    vol_window: int = 20,
    n_min: int = _N_MIN_BUCKET,
    expectancy_epsilon: float = _EXPECTANCY_EPSILON,
) -> RegimeFitResult:
    """Classify strategy regime preference on provided trades.

    Uses rolling stdev of log returns as volatility proxy. Median of vol scores
    within the evaluation window is the calm/volatile split threshold.
    """
    empty = _bucket_stats([])
    if not trades:
        return RegimeFitResult(
            verdict="insufficient_data",
            calm_stats=empty,
            volatile_stats=empty,
            vol_window=vol_window,
            total_trades=0,
            window_name=window.name,
        )

    log_ret = np.log(frame["close"] / frame["close"].shift(1))
    vol_series = log_ret.rolling(vol_window, min_periods=vol_window).std()

    tz = frame.index.tz
    w_start = pd.Timestamp(window.start, tz=tz)
    w_end = pd.Timestamp(window.end, tz=tz)
    window_mask = (frame.index >= w_start) & (frame.index <= w_end)
    window_vol = vol_series[window_mask].dropna()

    if window_vol.empty:
        return RegimeFitResult(
            verdict="insufficient_data",
            calm_stats=empty,
            volatile_stats=empty,
            vol_window=vol_window,
            total_trades=len(trades),
            window_name=window.name,
        )

    vol_median = float(window_vol.median())

    calm_trades: list[TradeResult] = []
    volatile_trades: list[TradeResult] = []

    for trade in trades:
        entry_ts = trade.entry_time
        if entry_ts in vol_series.index:
            vol_at_entry = vol_series.loc[entry_ts]
        else:
            prior = vol_series.index[vol_series.index <= entry_ts]
            if prior.empty:
                continue
            vol_at_entry = vol_series.loc[prior[-1]]

        if pd.isna(vol_at_entry):
            continue

        if float(vol_at_entry) <= vol_median:
            calm_trades.append(trade)
        else:
            volatile_trades.append(trade)

    calm_stats = _bucket_stats(calm_trades)
    volatile_stats = _bucket_stats(volatile_trades)

    if calm_stats.count < n_min or volatile_stats.count < n_min:
        verdict = "insufficient_data"
    else:
        calm_exp = calm_stats.mean_net_pnl
        vol_exp = volatile_stats.mean_net_pnl
        calm_beats = calm_exp > vol_exp + expectancy_epsilon and calm_exp > 0
        vol_beats = vol_exp > calm_exp + expectancy_epsilon and vol_exp > 0

        if calm_beats:
            verdict = "prefers_calm"
        elif vol_beats:
            verdict = "prefers_volatile"
        else:
            verdict = "mixed"

    return RegimeFitResult(
        verdict=verdict,
        calm_stats=calm_stats,
        volatile_stats=volatile_stats,
        vol_window=vol_window,
        total_trades=len(trades),
        window_name=window.name,
    )


def regime_summary_dict(result: RegimeFitResult) -> dict[str, Any]:
    def _b(s: BucketStats) -> dict[str, Any]:
        return {
            "count": s.count,
            "win_rate": s.win_rate,
            "expectancy_r": s.expectancy_r,
            "total_net_pnl": s.total_net_pnl,
            "mean_net_pnl": s.mean_net_pnl,
        }

    return {
        "regime_verdict": result.verdict,
        "vol_window": result.vol_window,
        "total_trades": result.total_trades,
        "window": result.window_name,
        "calm": _b(result.calm_stats),
        "volatile": _b(result.volatile_stats),
    }


def regime_summary_text(result: RegimeFitResult) -> str:
    c = result.calm_stats
    v = result.volatile_stats
    return (
        f"Regime fit: {result.verdict.upper().replace('_', ' ')}\n"
        f"  vol_window={result.vol_window} bars  total_trades={result.total_trades}"
        f"  window={result.window_name}\n"
        f"  Calm:     n={c.count}  win_rate={c.win_rate:.3f}"
        f"  exp_r={c.expectancy_r:.4f}  mean_pnl={c.mean_net_pnl:.2f}\n"
        f"  Volatile: n={v.count}  win_rate={v.win_rate:.3f}"
        f"  exp_r={v.expectancy_r:.4f}  mean_pnl={v.mean_net_pnl:.2f}"
    )


__all__ = [
    "BucketStats",
    "RegimeFitResult",
    "classify_regime_fit",
    "regime_summary_dict",
    "regime_summary_text",
]
