from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import TOPSTEP_50K, TopStepRules
from .trades import TradeResult


@dataclass(frozen=True)
class TopStepResult:
    passed: bool
    failed: bool
    days_to_pass: int | None
    final_balance: float
    max_drawdown: float
    max_daily_loss: float
    consistency_ok: bool
    reason: str

    def score(self) -> float:
        if self.failed:
            return -1000.0
        speed = 0.0 if self.days_to_pass is None else max(0.0, 30.0 - self.days_to_pass)
        return (500.0 if self.passed else 0.0) + speed - self.max_drawdown * 0.05 - self.max_daily_loss * 0.05


def _group_trades_by_day(trades: list[TradeResult]) -> dict[pd.Timestamp, list[TradeResult]]:
    day_trades: dict[pd.Timestamp, list[TradeResult]] = {}
    for trade in sorted(trades, key=lambda t: t.exit_time):
        day = trade.exit_time.normalize()
        day_trades.setdefault(day, []).append(trade)
    return day_trades


def simulate_topstep_with_termination(
    trades: list[TradeResult],
    rules: TopStepRules = TOPSTEP_50K,
) -> tuple[TopStepResult, pd.Timestamp | None]:
    """Same rules as simulate_topstep; also returns the normalized calendar day that ended this eval.

    For sequential eval chains: the next eval includes only trades on days **strictly after**
    that termination day (same convention as walk-forward OOS chaining).
    """
    day_trades = _group_trades_by_day(trades)
    if not day_trades:
        return (
            TopStepResult(
                False, False, None, rules.account_size,
                0.0, 0.0,
                True, "not_passed",
            ),
            None,
        )

    balance = rules.account_size
    peak_eod = balance
    floor = rules.account_size - rules.max_drawdown
    max_drawdown = 0.0
    max_daily_loss = 0.0
    daily_pnl: dict[pd.Timestamp, float] = {}
    trading_days: set[pd.Timestamp] = set()
    term_day: pd.Timestamp | None = None

    for day, day_trade_list in sorted(day_trades.items()):
        term_day = day
        day_net = 0.0
        busted = False
        day_locked = False

        for trade in day_trade_list:
            day_net += trade.net_pnl
            balance += trade.net_pnl
            current_dd = peak_eod - balance
            max_drawdown = max(max_drawdown, current_dd)
            if balance <= floor:
                busted = True
                break
            if day_net <= -rules.daily_loss_limit:
                day_locked = True
                break

        if busted:
            return (
                TopStepResult(
                    False, True, None, balance, max_drawdown,
                    max(max_daily_loss, -day_net), False, "drawdown",
                ),
                term_day,
            )

        trading_days.add(day)
        daily_pnl[day] = day_net
        max_daily_loss = max(max_daily_loss, -day_net)

        peak_eod = max(peak_eod, balance)
        floor = max(floor, peak_eod - rules.max_drawdown)

        profit = balance - rules.account_size
        best_day = max(daily_pnl.values())
        consistency_ok = best_day <= rules.consistency_pct_of_target * rules.profit_target
        if profit >= rules.profit_target and consistency_ok:
            return (
                TopStepResult(
                    True, False, len(trading_days), balance,
                    max_drawdown, max_daily_loss, consistency_ok, "passed",
                ),
                term_day,
            )

    best_day = max(daily_pnl.values()) if daily_pnl else 0.0
    consistency_ok = best_day <= rules.consistency_pct_of_target * rules.profit_target
    return (
        TopStepResult(
            False, False, None, balance, max_drawdown,
            max_daily_loss, consistency_ok, "not_passed",
        ),
        term_day,
    )


def simulate_topstep(trades: list[TradeResult], rules: TopStepRules = TOPSTEP_50K) -> TopStepResult:
    """Simulate a single TopStep 50k Combine pass/fail.

    Trailing drawdown floor ratchets from EOD balance only — intraday peaks
    do NOT advance the floor.  The min_trading_days field on TopStepRules is
    a *payout* requirement, not a Combine pass gate, so it is not checked here
    (days_to_pass is still reported for informational use).
    """
    result, _ = simulate_topstep_with_termination(trades, rules)
    return result


def count_sequential_eval_passes(
    trades: list[TradeResult],
    rules: TopStepRules = TOPSTEP_50K,
) -> tuple[int, list[TopStepResult]]:
    """Chained evals on one OOS sample: after each eval pass/fail, start again next calendar day."""
    ordered = sorted(trades, key=lambda t: t.exit_time)
    passes = 0
    log: list[TopStepResult] = []
    remaining: list[TradeResult] = ordered[:]

    while remaining:
        result, term_day = simulate_topstep_with_termination(remaining, rules)
        log.append(result)
        if result.passed:
            passes += 1
        if term_day is None:
            break
        next_remaining = [t for t in remaining if t.exit_time.normalize() > term_day]
        if not next_remaining:
            break
        remaining = next_remaining

    return passes, log


def topstep_summary_dict(result: TopStepResult) -> dict[str, Any]:
    return {
        "topstep_passed": result.passed,
        "topstep_failed": result.failed,
        "topstep_days_to_pass": result.days_to_pass,
        "topstep_final_balance": result.final_balance,
        "topstep_max_drawdown": result.max_drawdown,
        "topstep_max_daily_loss": result.max_daily_loss,
        "topstep_consistency_ok": result.consistency_ok,
        "topstep_reason": result.reason,
        "topstep_score": result.score(),
    }


__all__ = [
    "TopStepResult",
    "count_sequential_eval_passes",
    "simulate_topstep",
    "simulate_topstep_with_termination",
    "topstep_summary_dict",
]
