"""Bootstrap Combine pass-rate estimate from a trade list.

Groups trades by calendar exit day, then resamples many random orderings of
those days. Each ordering is flattened into a synthetic timeline so
`simulate_topstep` still sees one trading day per original day — daily loss
and consistency rules stay meaningful. Plain trade-order shuffles would not.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Any

import pandas as pd

from .config import TOPSTEP_50K, TopStepRules
from .topstep import TopStepResult, simulate_topstep
from .trades import TradeResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombineSimResult:
    """Headline and diagnostic output of the bootstrap Combine simulation."""

    n_resamples: int
    pass_rate_pct: float          # 0–100; the headline metric

    # Pass / fail / inconclusive counts
    n_passed: int
    n_failed_drawdown: int
    n_not_passed: int             # hit end of trades without busting or passing

    # Days-to-pass distribution (passed resamples only)
    median_days_to_pass: float | None
    mean_days_to_pass: float | None
    min_days_to_pass: int | None
    max_days_to_pass: int | None

    # Bust-risk diagnostics
    pct_daily_limit_hit: float    # % resamples where at least one day hit daily loss
    mean_max_drawdown: float      # average of per-resample peak drawdown ($)
    worst_max_drawdown: float     # worst single resample peak drawdown ($)

    # Input shape
    n_trades: int
    n_trading_days: int
    n_resamples_requested: int


# ---------------------------------------------------------------------------
# Calendar-day grouping (internal)
# ---------------------------------------------------------------------------


def _group_trades_by_calendar_day(
    trades: list[TradeResult],
) -> list[list[TradeResult]]:
    """Return trades grouped by calendar day (exit), sorted within each day by exit_time."""
    day_map: dict[pd.Timestamp, list[TradeResult]] = {}
    for trade in trades:
        day = trade.exit_time.normalize()
        day_map.setdefault(day, []).append(trade)
    return [
        sorted(day_group, key=lambda t: t.exit_time)
        for day_group in day_map.values()
    ]


def _flatten_day_groups_to_synthetic_timeline(day_groups: list[list[TradeResult]]) -> list[TradeResult]:
    """Map each calendar-day group to consecutive synthetic session days.

    Preserves intra-day ordering; only the calendar date is remapped so
    `simulate_topstep` groups PnL correctly after a shuffle of day order.
    """
    result: list[TradeResult] = []
    epoch = pd.Timestamp("2000-01-03", tz="America/New_York")  # Monday
    for day_idx, day_group in enumerate(day_groups):
        synthetic_date = epoch + pd.Timedelta(days=day_idx)
        for trade in day_group:
            original_time = trade.exit_time
            if original_time.tzinfo is not None:
                original_time_et = original_time.tz_convert("America/New_York")
            else:
                original_time_et = original_time.tz_localize("America/New_York")
            new_ts = synthetic_date.replace(
                hour=original_time_et.hour,
                minute=original_time_et.minute,
                second=original_time_et.second,
            )
            result.append(
                TradeResult(
                    strategy=trade.strategy,
                    entry_time=new_ts - pd.Timedelta(minutes=max(trade.bars_held, 1)),
                    exit_time=new_ts,
                    direction=trade.direction,
                    entry=trade.entry,
                    stop=trade.stop,
                    target=trade.target,
                    exit=trade.exit,
                    contracts=trade.contracts,
                    gross_pnl=trade.gross_pnl,
                    commission=trade.commission,
                    net_pnl=trade.net_pnl,
                    r_multiple=trade.r_multiple,
                    exit_reason=trade.exit_reason,
                    bars_held=trade.bars_held,
                    regime=trade.regime,
                    params=trade.params,
                )
            )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_combine_simulator(
    trades: list[TradeResult],
    rules: TopStepRules = TOPSTEP_50K,
    n_resamples: int = 1_000,
    seed: int | None = 42,
) -> CombineSimResult:
    """Monte Carlo over random orderings of calendar days, then `simulate_topstep` each path."""

    if not trades:
        return CombineSimResult(
            n_resamples=0,
            pass_rate_pct=0.0,
            n_passed=0,
            n_failed_drawdown=0,
            n_not_passed=0,
            median_days_to_pass=None,
            mean_days_to_pass=None,
            min_days_to_pass=None,
            max_days_to_pass=None,
            pct_daily_limit_hit=0.0,
            mean_max_drawdown=0.0,
            worst_max_drawdown=0.0,
            n_trades=0,
            n_trading_days=0,
            n_resamples_requested=n_resamples,
        )

    rng = random.Random(seed)
    day_groups = _group_trades_by_calendar_day(trades)
    n_days = len(day_groups)

    n_passed = 0
    n_failed_drawdown = 0
    n_not_passed = 0
    days_to_pass_list: list[int] = []
    daily_limit_hits = 0
    max_drawdowns: list[float] = []

    for _ in range(n_resamples):
        shuffled = day_groups[:]
        rng.shuffle(shuffled)
        resample_trades = _flatten_day_groups_to_synthetic_timeline(shuffled)
        result: TopStepResult = simulate_topstep(resample_trades, rules)

        if result.passed:
            n_passed += 1
            if result.days_to_pass is not None:
                days_to_pass_list.append(result.days_to_pass)
        elif result.failed:
            n_failed_drawdown += 1
        else:
            n_not_passed += 1

        if result.max_daily_loss >= rules.daily_loss_limit:
            daily_limit_hits += 1
        max_drawdowns.append(result.max_drawdown)

    pass_rate_pct = 100.0 * n_passed / n_resamples

    return CombineSimResult(
        n_resamples=n_resamples,
        pass_rate_pct=pass_rate_pct,
        n_passed=n_passed,
        n_failed_drawdown=n_failed_drawdown,
        n_not_passed=n_not_passed,
        median_days_to_pass=median(days_to_pass_list) if days_to_pass_list else None,
        mean_days_to_pass=mean(days_to_pass_list) if days_to_pass_list else None,
        min_days_to_pass=min(days_to_pass_list) if days_to_pass_list else None,
        max_days_to_pass=max(days_to_pass_list) if days_to_pass_list else None,
        pct_daily_limit_hit=100.0 * daily_limit_hits / n_resamples,
        mean_max_drawdown=mean(max_drawdowns),
        worst_max_drawdown=max(max_drawdowns),
        n_trades=len(trades),
        n_trading_days=n_days,
        n_resamples_requested=n_resamples,
    )


def combine_sim_summary_dict(result: CombineSimResult) -> dict[str, Any]:
    """Flat dict suitable for logging, CSV export, or verdict input."""
    return {
        "combine_pass_rate_pct": result.pass_rate_pct,
        "combine_n_resamples": result.n_resamples,
        "combine_n_passed": result.n_passed,
        "combine_n_failed_drawdown": result.n_failed_drawdown,
        "combine_n_not_passed": result.n_not_passed,
        "combine_median_days_to_pass": result.median_days_to_pass,
        "combine_mean_days_to_pass": result.mean_days_to_pass,
        "combine_min_days_to_pass": result.min_days_to_pass,
        "combine_max_days_to_pass": result.max_days_to_pass,
        "combine_pct_daily_limit_hit": result.pct_daily_limit_hit,
        "combine_mean_max_drawdown": result.mean_max_drawdown,
        "combine_worst_max_drawdown": result.worst_max_drawdown,
        "combine_n_trades": result.n_trades,
        "combine_n_trading_days": result.n_trading_days,
    }


__all__ = [
    "CombineSimResult",
    "run_combine_simulator",
    "combine_sim_summary_dict",
]
