"""Express-funded-style multi-stint simulator with breach resets (holdout-oriented).

Distinct from Combine ``simulate_topstep``: no profit-target / consistency early exit.
Matches sequential-eval continuation: after a breach, remaining trades exclude the full
termination calendar day (same convention as ``count_sequential_eval_passes``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from .config import DEFAULT_FUNDED_EXPRESS_SIM, FundedExpressSimRules
from .topstep import _group_trades_by_day
from .trades import TradeResult


@dataclass(frozen=True)
class ExpressFundedSimResult:
    accrued_pnl_bank: float
    funded_accounts_failed: int
    stints_opened: int
    funded_accounts_used: int
    max_nominal_peak_balance: float
    worst_daily_drawdown: float
    worst_stint_peak_to_trough_drawdown_from_peak_balance: float
    current_account_active: bool
    current_account_pnl: float | None
    current_max_drawdown: float | None
    current_win_rate_pct: float | None
    current_avg_r_multiple: float | None
    current_profit_factor: float | None
    current_sharpe_annualized: float | None
    total_win_rate_pct: float
    total_avg_r_multiple: float
    best_trade_pnl: float
    worst_trade_pnl: float
    stints_summary: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accrued_pnl_bank": self.accrued_pnl_bank,
            "funded_accounts_failed": self.funded_accounts_failed,
            "stints_simulated": self.stints_opened,
            "funded_accounts_used": self.funded_accounts_used,
            "stints_opened": self.stints_opened,
            "max_nominal_peak_balance": self.max_nominal_peak_balance,
            "worst_daily_drawdown": self.worst_daily_drawdown,
            "worst_stint_peak_to_trough_drawdown_from_peak_balance": (
                self.worst_stint_peak_to_trough_drawdown_from_peak_balance
            ),
            "current_account_active": self.current_account_active,
            "current_account_pnl": self.current_account_pnl,
            "current_max_drawdown": self.current_max_drawdown,
            "current_win_rate_pct": self.current_win_rate_pct,
            "current_avg_r_multiple": self.current_avg_r_multiple,
            "current_profit_factor": self.current_profit_factor,
            "current_sharpe_annualized": self.current_sharpe_annualized,
            "total_win_rate_pct": self.total_win_rate_pct,
            "total_avg_r_multiple": self.total_avg_r_multiple,
            "best_trade_pnl": self.best_trade_pnl,
            "worst_trade_pnl": self.worst_trade_pnl,
            "stints_summary": list(self.stints_summary),
        }


def express_funded_reset_sim_summary_dict(result: ExpressFundedSimResult) -> dict[str, Any]:
    """JSON-friendly projection (alias for stable CLI / scripts)."""

    return result.to_dict()


@dataclass
class _StintOutcome:
    breached: bool
    termination_day: pd.Timestamp | None
    terminal_balance: float
    trades_applied_count: int
    trades_applied: tuple[TradeResult, ...]
    first_trade_day: pd.Timestamp | None
    last_trade_day: pd.Timestamp | None
    peak_nominal_balance: float
    worst_daily_dd: float
    worst_peak_to_trough: float


@dataclass(frozen=True)
class _TradeMetrics:
    win_rate_pct: float
    avg_r_multiple: float
    best_trade_pnl: float
    worst_trade_pnl: float
    profit_factor: float
    sharpe_annualized: float


def _metrics_for_trades(trades: tuple[TradeResult, ...] | list[TradeResult]) -> _TradeMetrics:
    if not trades:
        return _TradeMetrics(
            win_rate_pct=0.0,
            avg_r_multiple=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
            profit_factor=math.inf,
            sharpe_annualized=0.0,
        )

    pnls = [float(t.net_pnl) for t in trades]
    r_values = [float(t.r_multiple) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate_pct = len(wins) / len(pnls) * 100.0
    avg_r_multiple = sum(r_values) / len(r_values)
    best_trade_pnl = max(pnls)
    worst_trade_pnl = min(pnls)

    loss_sum = sum(losses)
    profit_factor = math.inf if loss_sum >= 0 else sum(wins) / abs(loss_sum)

    variance = sum((r - avg_r_multiple) ** 2 for r in r_values) / len(r_values)
    std_r = math.sqrt(variance)
    sharpe_annualized = (avg_r_multiple / std_r * math.sqrt(252.0)) if std_r > 0 else 0.0

    return _TradeMetrics(
        win_rate_pct=win_rate_pct,
        avg_r_multiple=avg_r_multiple,
        best_trade_pnl=best_trade_pnl,
        worst_trade_pnl=worst_trade_pnl,
        profit_factor=profit_factor,
        sharpe_annualized=sharpe_annualized,
    )


def _survival_days(stint: _StintOutcome) -> int:
    if stint.first_trade_day is None:
        return 0
    end_day = stint.termination_day if stint.breached else stint.last_trade_day
    if end_day is None:
        return 0
    return int((end_day - stint.first_trade_day).days)


def _simulate_express_single_stint(
    trades_sorted: list[TradeResult],
    rules: FundedExpressSimRules,
) -> tuple[_StintOutcome, list[TradeResult]]:
    """Run one stint; on breach leave ``leftover`` = trades strictly after termination day."""

    remaining_input = trades_sorted[:]
    if not trades_sorted:
        raise ValueError("_simulate_express_single_stint expects non-empty trades")

    day_trades = _group_trades_by_day(remaining_input)
    if not day_trades:
        sz = rules.account_size
        return (
            _StintOutcome(False, None, sz, 0, (), None, None, sz, 0.0, 0.0),
            [],
        )

    balance = rules.account_size
    peak_eod = balance
    floor = rules.account_size - rules.max_drawdown
    trail_locked = False

    trades_applied_count = 0
    trades_applied: list[TradeResult] = []
    first_trade_day: pd.Timestamp | None = None
    last_trade_day: pd.Timestamp | None = None
    nominal_peak_stint = balance
    worst_daily_dd_stint = 0.0
    balance_high_water = balance
    worst_excursion_stint = 0.0
    termination_day: pd.Timestamp | None = None

    for cal_day, day_trade_list in sorted(day_trades.items()):
        termination_day = cal_day
        day_net = 0.0
        busted = False
        day_locked = False
        intraday_peak = balance
        intraday_min = balance

        for trade in day_trade_list:
            if first_trade_day is None:
                first_trade_day = cal_day
            last_trade_day = cal_day
            day_net += trade.net_pnl
            balance += trade.net_pnl
            trades_applied_count += 1
            trades_applied.append(trade)
            nominal_peak_stint = max(nominal_peak_stint, balance)
            intraday_peak = max(intraday_peak, balance)
            intraday_min = min(intraday_min, balance)

            balance_high_water = max(balance_high_water, balance)
            trough_dd = balance_high_water - balance
            worst_excursion_stint = max(worst_excursion_stint, trough_dd)

            if balance <= floor:
                busted = True
                break
            if day_net <= -rules.daily_loss_limit:
                day_locked = True
                break

        daily_range_dd = intraday_peak - intraday_min
        worst_daily_dd_stint = max(worst_daily_dd_stint, daily_range_dd)

        if busted:
            leftover = [t for t in remaining_input if t.exit_time.normalize() > termination_day]
            assert termination_day is not None
            return (
                _StintOutcome(
                    True,
                    termination_day,
                    balance,
                    trades_applied_count,
                    tuple(trades_applied),
                    first_trade_day,
                    last_trade_day,
                    nominal_peak_stint,
                    worst_daily_dd_stint,
                    worst_excursion_stint,
                ),
                leftover,
            )

        peak_eod = max(peak_eod, balance)
        if trail_locked:
            pass
        elif peak_eod >= rules.lock_trigger_balance:
            trail_locked = True
            floor = rules.locked_floor_balance
        else:
            floor = max(floor, peak_eod - rules.max_drawdown)

        balance_high_water = max(balance_high_water, balance)

    leftover: list[TradeResult] = []
    return (
        _StintOutcome(
            False,
            None,
            balance,
            trades_applied_count,
            tuple(trades_applied),
            first_trade_day,
            last_trade_day,
            nominal_peak_stint,
            worst_daily_dd_stint,
            worst_excursion_stint,
        ),
        leftover,
    )


def simulate_express_funded_resets(
    trades: list[TradeResult],
    rules: FundedExpressSimRules = DEFAULT_FUNDED_EXPRESS_SIM,
) -> ExpressFundedSimResult:
    """Chronological funded Express-style sim with resets on maximum-loss breaches."""

    accrued = 0.0
    breaches = 0
    stints_opened = 0
    global_nominal_peak = rules.account_size
    worst_daily_global = 0.0
    worst_stint_peaks_trough_global = 0.0

    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    stint_rows: list[dict[str, Any]] = []

    if not sorted_trades:
        return ExpressFundedSimResult(
            accrued_pnl_bank=0.0,
            funded_accounts_failed=0,
            stints_opened=0,
            funded_accounts_used=0,
            max_nominal_peak_balance=rules.account_size,
            worst_daily_drawdown=0.0,
            worst_stint_peak_to_trough_drawdown_from_peak_balance=0.0,
            current_account_active=False,
            current_account_pnl=None,
            current_max_drawdown=None,
            current_win_rate_pct=None,
            current_avg_r_multiple=None,
            current_profit_factor=None,
            current_sharpe_annualized=None,
            total_win_rate_pct=0.0,
            total_avg_r_multiple=0.0,
            best_trade_pnl=0.0,
            worst_trade_pnl=0.0,
            stints_summary=(),
        )

    remaining = sorted_trades[:]
    all_applied_trades: list[TradeResult] = []
    final_stint: _StintOutcome | None = None

    while remaining:
        stints_opened += 1

        stint, leftover = _simulate_express_single_stint(remaining, rules)
        final_stint = stint
        all_applied_trades.extend(stint.trades_applied)
        stint_bank_incr = stint.terminal_balance - rules.account_size
        accrued += stint_bank_incr
        global_nominal_peak = max(global_nominal_peak, stint.peak_nominal_balance)
        worst_daily_global = max(worst_daily_global, stint.worst_daily_dd)
        worst_stint_peaks_trough_global = max(
            worst_stint_peaks_trough_global,
            stint.worst_peak_to_trough,
        )
        metrics = _metrics_for_trades(stint.trades_applied)

        row: dict[str, Any] = {
            "stint_index": stints_opened - 1,
            "breached": stint.breached,
            "termination_day": str(stint.termination_day) if stint.termination_day is not None else None,
            "terminal_balance": stint.terminal_balance,
            "bank_increment": stint_bank_incr,
            "trades_applied_count": stint.trades_applied_count,
            "survival_days": _survival_days(stint),
            "win_rate_pct": metrics.win_rate_pct,
            "avg_r_multiple": metrics.avg_r_multiple,
            "best_trade_pnl": metrics.best_trade_pnl,
            "worst_trade_pnl": metrics.worst_trade_pnl,
            "profit_factor": metrics.profit_factor,
            "stint_peak_balance": stint.peak_nominal_balance,
            "stint_worst_daily_dd": stint.worst_daily_dd,
            "stint_worst_peak_to_trough": stint.worst_peak_to_trough,
        }
        stint_rows.append(row)

        if stint.breached:
            breaches += 1

        remaining = leftover
        if not stint.breached:
            break

    funded_accounts_used = 1 + breaches
    total_metrics = _metrics_for_trades(all_applied_trades)
    current_account_active = final_stint is not None and not final_stint.breached
    if current_account_active and final_stint is not None:
        current_metrics = _metrics_for_trades(final_stint.trades_applied)
        current_account_pnl = final_stint.terminal_balance - rules.account_size
        current_max_drawdown = final_stint.worst_peak_to_trough
        current_win_rate_pct = current_metrics.win_rate_pct
        current_avg_r_multiple = current_metrics.avg_r_multiple
        current_profit_factor = current_metrics.profit_factor
        current_sharpe_annualized = current_metrics.sharpe_annualized
    else:
        current_account_pnl = None
        current_max_drawdown = None
        current_win_rate_pct = None
        current_avg_r_multiple = None
        current_profit_factor = None
        current_sharpe_annualized = None

    return ExpressFundedSimResult(
        accrued_pnl_bank=accrued,
        funded_accounts_failed=breaches,
        stints_opened=stints_opened,
        funded_accounts_used=funded_accounts_used,
        max_nominal_peak_balance=global_nominal_peak,
        worst_daily_drawdown=worst_daily_global,
        worst_stint_peak_to_trough_drawdown_from_peak_balance=worst_stint_peaks_trough_global,
        current_account_active=current_account_active,
        current_account_pnl=current_account_pnl,
        current_max_drawdown=current_max_drawdown,
        current_win_rate_pct=current_win_rate_pct,
        current_avg_r_multiple=current_avg_r_multiple,
        current_profit_factor=current_profit_factor,
        current_sharpe_annualized=current_sharpe_annualized,
        total_win_rate_pct=total_metrics.win_rate_pct,
        total_avg_r_multiple=total_metrics.avg_r_multiple,
        best_trade_pnl=total_metrics.best_trade_pnl,
        worst_trade_pnl=total_metrics.worst_trade_pnl,
        stints_summary=tuple(stint_rows),
    )


__all__ = [
    "ExpressFundedSimResult",
    "express_funded_reset_sim_summary_dict",
    "simulate_express_funded_resets",
]

