"""Per-strategy regime classifier.

Determines whether a strategy's edge leans toward calm or volatile market conditions.
Vol proxy: rolling stdev of session log returns (no external data required).

Trades at entry are bucketed calm (vol <= median) vs volatile (vol > median).
Output: single verdict per strategy run.

---

Per-day intraday regime labeler (VP Dual-Mode strategy):
Classifies each trading day into one of four regimes at 9:40 ET based on
the 10-bar opening range (09:30–09:39 1-min bars) and signed directional efficiency.

Regime distribution — de_threshold=0.35, de_lookback=20, full MNQ dataset (2021-03-19 to 2026):
  total_days=1289  trend_up=88 (6.8%)  trend_down=88 (6.8%)  balance=518 (40.2%)  unclear=595 (46.2%)

Note: unclear > 20% because OR-breakout-at-9:40 + DE confirmation is a strict conjunction.
Most "inside range" days have non-zero DE and don't satisfy the balance condition either.
Trend regimes (< 15% each) are real but rare — use regime-conditional sizing rather than
 filtering out unclear days entirely. Run scripts/diagnostics/audit_regimes.py for per-year breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import DateWindow, EASTERN_TZ
from .indicators import directional_efficiency
from .trades import TradeResult

# ---------------------------------------------------------------------------
# Legacy constants
# ---------------------------------------------------------------------------

_N_MIN_BUCKET = 10
_EXPECTANCY_EPSILON = 5.0  # min mean_pnl advantage to declare preference

# ---------------------------------------------------------------------------
# New type alias
# ---------------------------------------------------------------------------

DayRegime = Literal["trend_up", "trend_down", "balance", "unclear"]

_VALID_REGIMES: frozenset[str] = frozenset({"trend_up", "trend_down", "balance", "unclear"})

# ---------------------------------------------------------------------------
# Legacy dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Legacy helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Legacy API — preserved exactly
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# New — per-day intraday regime labeler
# ---------------------------------------------------------------------------


def classify_day_regimes(
    exec_5min: pd.DataFrame,
    raw_1min: pd.DataFrame,
    *,
    de_threshold: float = 0.35,
    de_lookback: int = 20,
) -> pd.Series:
    """Label each 5-min bar with the intraday regime for its trading day.

    The regime is determined at the 9:40 ET bar and forward-filled for the
    remainder of the day. Bars before 9:40 ET receive "unclear".

    Note: this classifier still makes RTH-specific assumptions. The broader
    full-session execution model is handled elsewhere for now.

    Args:
        exec_5min: 5-minute RTH execution frame, tz-aware America/New_York.
        raw_1min: 1-minute bar frame (RTH), tz-aware America/New_York.
        de_threshold: signed DE threshold separating trend from balance.
        de_lookback: lookback bars for directional_efficiency on the 5-min frame.

    Returns:
        pd.Series indexed like exec_5min, values in DayRegime literals.
    """
    result: pd.Series = pd.Series("unclear", index=exec_5min.index, dtype=str)

    # Signed directional efficiency computed once across full history.
    unsigned_de = directional_efficiency(exec_5min["close"], lookback=de_lookback)
    net_disp = exec_5min["close"] - exec_5min["close"].shift(de_lookback)
    signed_de = unsigned_de * np.sign(net_disp)

    # Pre-group both frames by date to avoid O(N) scans per day.
    exec_5min_normalized = exec_5min.index.normalize()
    raw_1min_groups: dict = {
        day: grp for day, grp in raw_1min.groupby(raw_1min.index.normalize())
    }

    for day_date, day_5min in exec_5min.groupby(exec_5min_normalized):
        # A2 audit note: this intentionally mixes 1-min OR bars with the 5-min
        # 09:40 close + signed DE confirmation.
        # Opening range from 1-min bars: 09:30–09:39 (10 bars max).
        day_1min = raw_1min_groups.get(day_date)
        if day_1min is None:
            continue
        or_bars = day_1min.between_time("09:30", "09:39", inclusive="both")
        if len(or_bars) < 5:
            continue

        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())

        # Find the 9:40 ET bar on the 5-min frame.
        bars_940 = day_5min.between_time("09:40", "09:40", inclusive="both")
        if bars_940.empty:
            continue

        ts_940 = bars_940.index[0]
        price_940 = float(bars_940["close"].iloc[0])
        de_940 = float(signed_de.loc[ts_940])

        if pd.isna(de_940):
            continue

        # Classification rules (all conditions explicit — unclear is the catch-all).
        if price_940 > or_high and de_940 > de_threshold:
            label: str = "trend_up"
        elif price_940 < or_low and de_940 < -de_threshold:
            label = "trend_down"
        elif or_low <= price_940 <= or_high and abs(de_940) <= de_threshold:
            label = "balance"
        else:
            label = "unclear"

        # Set label at 9:40 and forward-fill for the rest of the day.
        from_940 = (exec_5min_normalized == day_date) & (exec_5min.index >= ts_940)
        result.loc[from_940] = label

    return result


def attach_day_regimes(
    exec_5min: pd.DataFrame,
    raw_1min: pd.DataFrame,
    *,
    de_threshold: float = 0.35,
    de_lookback: int = 20,
) -> pd.DataFrame:
    """Return copy of exec_5min with 'day_regime' column added.

    Calls assert_regime_mutual_exclusivity before returning.
    """
    regimes = classify_day_regimes(
        exec_5min, raw_1min, de_threshold=de_threshold, de_lookback=de_lookback
    )
    assert_regime_mutual_exclusivity(regimes)
    out = exec_5min.copy()
    out["day_regime"] = regimes
    return out


def audit_regime_distribution(
    exec_5min: pd.DataFrame,
    raw_1min: pd.DataFrame,
    *,
    de_threshold: float = 0.35,
    de_lookback: int = 20,
) -> dict[str, Any]:
    """Count per-day regime labels across the full dataset.

    Only counts days where a 9:40 ET bar exists in exec_5min.

    Returns:
        Dict with total_days, per-regime counts/pcts, and de_threshold_used.
    """
    regimes = classify_day_regimes(
        exec_5min, raw_1min, de_threshold=de_threshold, de_lookback=de_lookback
    )

    daily_labels: dict[str, str] = {}
    for day_date, day_5min in exec_5min.groupby(exec_5min.index.normalize()):
        bars_940 = day_5min.between_time("09:40", "09:40", inclusive="both")
        if bars_940.empty:
            continue
        ts_940 = bars_940.index[0]
        daily_labels[str(day_date.date())] = str(regimes.loc[ts_940])

    total = len(daily_labels)
    if total == 0:
        return {
            "total_days": 0,
            "trend_up": 0, "trend_down": 0, "balance": 0, "unclear": 0,
            "trend_up_pct": 0.0, "trend_down_pct": 0.0,
            "balance_pct": 0.0, "unclear_pct": 0.0,
            "de_threshold_used": de_threshold,
        }

    counts = {
        lbl: sum(1 for v in daily_labels.values() if v == lbl)
        for lbl in ("trend_up", "trend_down", "balance", "unclear")
    }
    return {
        "total_days": total,
        "trend_up": counts["trend_up"],
        "trend_down": counts["trend_down"],
        "balance": counts["balance"],
        "unclear": counts["unclear"],
        "trend_up_pct": counts["trend_up"] / total * 100,
        "trend_down_pct": counts["trend_down"] / total * 100,
        "balance_pct": counts["balance"] / total * 100,
        "unclear_pct": counts["unclear"] / total * 100,
        "de_threshold_used": de_threshold,
    }


def assert_regime_mutual_exclusivity(regime_series: pd.Series) -> None:
    """Assert that each trading day carries at most one regime label at/after 9:40 ET.

    Forward-fill guarantees this, but this function makes the invariant explicit
    and catches any bug that might break it.

    Raises:
        ValueError: if any day has multiple distinct labels at or after 9:40 ET.
    """
    for day_date, day_series in regime_series.groupby(regime_series.index.normalize()):
        bars_from_940 = day_series.between_time("09:40", "16:00", inclusive="both")
        if bars_from_940.empty:
            continue
        unique_labels = set(bars_from_940.unique())
        if len(unique_labels) > 1:
            raise ValueError(
                f"Multiple regime labels on {day_date.date()}: {unique_labels}"
            )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Legacy — post-hoc strategy fit classifier
    "BucketStats",
    "RegimeFitResult",
    "classify_regime_fit",
    "regime_summary_dict",
    "regime_summary_text",
    # New — intraday per-day regime labeler
    "DayRegime",
    "attach_day_regimes",
    "classify_day_regimes",
    "audit_regime_distribution",
    "assert_regime_mutual_exclusivity",
]
