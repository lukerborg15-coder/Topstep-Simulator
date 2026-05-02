"""Position sizing optimization for walk-forward and holdout periods.

Two optimization objectives:
1. Speed: Find risk that minimizes days_to_pass while maintaining pass_rate
2. Longevity: Find risk that maximizes funded account survival while hitting min profit/trade

Both functions use grid search over risk levels and return Pareto-ranked candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MAX_CONTRACTS,
    MNQ,
    TOPSTEP_50K,
    Instrument,
    TopStepRules,
    FundedExpressSimRules,
    DEFAULT_FUNDED_EXPRESS_SIM,
)
from .evaluator import _contracts_for_fixed_risk
from .funded_express_sim import simulate_express_funded_resets
from .topstep import count_sequential_eval_passes
from .trades import TradeResult


@dataclass(frozen=True)
class SpeedOptimizationResult:
    """Optimal sizing for walk-forward: minimize days_to_pass while maintaining pass_rate."""

    strategy: str
    window: str
    pass_floor_pct: float
    pass_target_pct: float

    # Top candidate
    optimal_risk_dollars: float
    pass_rate_pct: float
    mean_days_to_pass: float
    std_days_to_pass: float
    min_contracts_used: int
    max_contracts_used: int

    # All viable candidates (Pareto-ranked: speed first, then consistency)
    candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class LongevityOptimizationResult:
    """Optimal sizing for holdout: maximize account longevity while hitting min profit/trade."""

    strategy: str
    window: str
    min_profit_per_trade: float

    # Top candidate
    optimal_risk_dollars: float
    avg_pnl_per_trade: float
    total_pnl: float
    funded_accounts_used: int
    accounts_blown: int
    total_trades_executed: int
    longevity_score: float

    # All viable candidates (Pareto-ranked: longevity first, then profit/trade)
    candidates: tuple[dict[str, Any], ...] = ()


def _get_valid_risk_levels(
    trades: list[TradeResult],
    risk_levels: list[float],
    instrument: Instrument = MNQ,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> list[float]:
    """Filter risk levels: reject any that produce zero contracts on all trades.

    Ensures every risk level generates at least one trade with at least one contract.
    """
    if not trades:
        return []

    valid = []
    for risk in risk_levels:
        produces_contracts = False
        for trade in trades:
            contracts = _contracts_for_fixed_risk(
                trade, instrument, risk, max_contracts
            )
            if contracts > 0:
                produces_contracts = True
                break

        if produces_contracts:
            valid.append(risk)

    return valid


def _measure_contracts_range(
    trades: list[TradeResult],
    risk_dollars: float,
    instrument: Instrument = MNQ,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> tuple[int, int]:
    """Return (min_contracts, max_contracts) used at this risk level."""
    if not trades:
        return (0, 0)

    contract_counts = []
    for trade in trades:
        contracts = _contracts_for_fixed_risk(
            trade, instrument, risk_dollars, max_contracts
        )
        if contracts > 0:
            contract_counts.append(contracts)

    if not contract_counts:
        return (0, 0)

    return (min(contract_counts), max(contract_counts))


def optimize_for_speed_wf(
    trades: list[TradeResult],
    strategy: str,
    window: str,
    risk_levels: list[float] | None = None,
    pass_floor_pct: float = 40.0,
    pass_target_pct: float = 75.0,
    instrument: Instrument = MNQ,
    rules: TopStepRules = TOPSTEP_50K,
) -> SpeedOptimizationResult:
    """
    Find risk per trade minimizing days_to_pass while maintaining pass_rate.

    Process:
    1. Filter risk_levels to those producing at least one contract
    2. For each risk level:
       - Resize trades
       - Run sequential eval pass counting
       - Compute pass_rate_pct and mean_days_to_pass (only on passes)
    3. Filter to candidates with pass_rate_pct >= pass_floor_pct
    4. Rank by (mean_days_to_pass ASC, pass_rate_pct DESC)
    5. Return top 5 candidates

    Args:
        trades: All trades from walk-forward test period
        strategy: Strategy name (for logging)
        window: Window name e.g. "WF1_test"
        risk_levels: Dollar risk per trade to test. Default: $50, $75, $100, $150, $200, $300, $400, $500
        pass_floor_pct: Reject candidates with pass_rate < this
        pass_target_pct: Target pass rate (informational)
        instrument: MNQ, ES, etc.
        rules: TopStepRules (profit target, max DD, etc.)

    Returns:
        SpeedOptimizationResult with optimal risk and top 5 candidates
    """
    if risk_levels is None:
        risk_levels = [50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]

    if not trades:
        return SpeedOptimizationResult(
            strategy=strategy,
            window=window,
            pass_floor_pct=pass_floor_pct,
            pass_target_pct=pass_target_pct,
            optimal_risk_dollars=0.0,
            pass_rate_pct=0.0,
            mean_days_to_pass=0.0,
            std_days_to_pass=0.0,
            min_contracts_used=0,
            max_contracts_used=0,
            candidates=(),
        )

    # Filter to valid risk levels
    valid_risks = _get_valid_risk_levels(trades, risk_levels, instrument)
    if not valid_risks:
        return SpeedOptimizationResult(
            strategy=strategy,
            window=window,
            pass_floor_pct=pass_floor_pct,
            pass_target_pct=pass_target_pct,
            optimal_risk_dollars=0.0,
            pass_rate_pct=0.0,
            mean_days_to_pass=0.0,
            std_days_to_pass=0.0,
            min_contracts_used=0,
            max_contracts_used=0,
            candidates=(),
        )

    results: list[dict[str, Any]] = []

    for risk in valid_risks:
        # Resize trades for this risk level
        resized_trades = []
        for trade in trades:
            contracts = _contracts_for_fixed_risk(
                trade, instrument, risk, DEFAULT_MAX_CONTRACTS
            )
            if contracts > 0:
                # Scale trade P&L by contract count ratio
                original_contracts = trade.contracts
                scale_factor = contracts / original_contracts if original_contracts > 0 else 1.0

                scaled_trade = replace(
                    trade,
                    gross_pnl=trade.gross_pnl * scale_factor,
                    commission=trade.commission * scale_factor,
                    net_pnl=trade.net_pnl * scale_factor,
                    contracts=contracts,
                )
                resized_trades.append(scaled_trade)

        if not resized_trades:
            continue

        # Count passes and collect days_to_pass
        passes, log = count_sequential_eval_passes(resized_trades, rules)
        days_to_passes = [r.days_to_pass for r in log if r.passed and r.days_to_pass is not None]

        if not days_to_passes:
            pass_rate = 0.0
            mean_days = float("inf")
            std_days = 0.0
        else:
            pass_rate = (len(days_to_passes) / len(log)) * 100.0
            mean_days = float(np.mean(days_to_passes))
            std_days = float(np.std(days_to_passes)) if len(days_to_passes) > 1 else 0.0

        min_c, max_c = _measure_contracts_range(resized_trades, risk, instrument)

        results.append({
            "risk_dollars": risk,
            "pass_rate_pct": pass_rate,
            "mean_days_to_pass": mean_days,
            "std_days_to_pass": std_days,
            "num_passes": len(days_to_passes),
            "num_evals": len(log),
            "min_contracts": min_c,
            "max_contracts": max_c,
        })

    # Filter to pass_floor and rank by Pareto (speed first, then consistency)
    viable = [r for r in results if r["pass_rate_pct"] >= pass_floor_pct]

    if not viable:
        # No candidates meet threshold; return empty result
        return SpeedOptimizationResult(
            strategy=strategy,
            window=window,
            pass_floor_pct=pass_floor_pct,
            pass_target_pct=pass_target_pct,
            optimal_risk_dollars=0.0,
            pass_rate_pct=0.0,
            mean_days_to_pass=0.0,
            std_days_to_pass=0.0,
            min_contracts_used=0,
            max_contracts_used=0,
            candidates=tuple(results),  # Return all results so user sees why nothing passed
        )

    # Sort by mean_days_to_pass (ascending), then by pass_rate (descending)
    viable_sorted = sorted(
        viable,
        key=lambda r: (r["mean_days_to_pass"], -r["pass_rate_pct"]),
    )

    # Take top 5 for output
    top_5 = viable_sorted[:5]

    best = top_5[0]

    return SpeedOptimizationResult(
        strategy=strategy,
        window=window,
        pass_floor_pct=pass_floor_pct,
        pass_target_pct=pass_target_pct,
        optimal_risk_dollars=best["risk_dollars"],
        pass_rate_pct=best["pass_rate_pct"],
        mean_days_to_pass=best["mean_days_to_pass"],
        std_days_to_pass=best["std_days_to_pass"],
        min_contracts_used=best["min_contracts"],
        max_contracts_used=best["max_contracts"],
        candidates=tuple(top_5),
    )


def optimize_for_longevity_holdout(
    trades: list[TradeResult],
    strategy: str,
    window: str = "holdout",
    risk_levels: list[float] | None = None,
    min_profit_per_trade: float = 150.0,
    instrument: Instrument = MNQ,
    rules: FundedExpressSimRules = DEFAULT_FUNDED_EXPRESS_SIM,
) -> LongevityOptimizationResult:
    """
    Find risk per trade maximizing funded account longevity while hitting min profit/trade.

    Process:
    1. Filter risk_levels to those producing at least one contract
    2. For each risk level:
       - Resize trades
       - Run express_funded_resets (continuous account sim)
       - Compute avg_pnl_per_trade, longevity_score, accounts_blown
    3. Filter to candidates with avg_pnl_per_trade >= min_profit_per_trade
    4. Rank by (longevity_score DESC, avg_pnl_per_trade DESC)
    5. Return top 5 candidates

    Longevity score = funded_accounts_used + (accrued_pnl_bank / 50000)
    - Accounts used: 1 = never blown, 2 = blown once, 3 = blown twice, etc.
    - PnL buffer: accounts_used + pnl_fraction lets accounts with same longevity
      but higher PnL rank higher

    Args:
        trades: All trades from holdout period
        strategy: Strategy name (for logging)
        window: Window name, typically "holdout"
        risk_levels: Dollar risk per trade to test. Default: same as speed
        min_profit_per_trade: Reject candidates with avg_pnl_per_trade < this
        instrument: MNQ, ES, etc.
        rules: FundedExpressSimRules (max DD, daily loss limit, etc.)

    Returns:
        LongevityOptimizationResult with optimal risk and top 5 candidates
    """
    if risk_levels is None:
        risk_levels = [50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]

    if not trades:
        return LongevityOptimizationResult(
            strategy=strategy,
            window=window,
            min_profit_per_trade=min_profit_per_trade,
            optimal_risk_dollars=0.0,
            avg_pnl_per_trade=0.0,
            total_pnl=0.0,
            funded_accounts_used=0,
            accounts_blown=0,
            total_trades_executed=0,
            longevity_score=0.0,
            candidates=(),
        )

    # Filter to valid risk levels
    valid_risks = _get_valid_risk_levels(trades, risk_levels, instrument)
    if not valid_risks:
        return LongevityOptimizationResult(
            strategy=strategy,
            window=window,
            min_profit_per_trade=min_profit_per_trade,
            optimal_risk_dollars=0.0,
            avg_pnl_per_trade=0.0,
            total_pnl=0.0,
            funded_accounts_used=0,
            accounts_blown=0,
            total_trades_executed=0,
            longevity_score=0.0,
            candidates=(),
        )

    results: list[dict[str, Any]] = []

    for risk in valid_risks:
        # Resize trades for this risk level
        resized_trades = []
        for trade in trades:
            contracts = _contracts_for_fixed_risk(
                trade, instrument, risk, DEFAULT_MAX_CONTRACTS
            )
            if contracts > 0:
                original_contracts = trade.contracts
                scale_factor = contracts / original_contracts if original_contracts > 0 else 1.0

                scaled_trade = replace(
                    trade,
                    gross_pnl=trade.gross_pnl * scale_factor,
                    commission=trade.commission * scale_factor,
                    net_pnl=trade.net_pnl * scale_factor,
                    contracts=contracts,
                )
                resized_trades.append(scaled_trade)

        if not resized_trades:
            continue

        # Simulate continuous funded account
        sim = simulate_express_funded_resets(resized_trades, rules)

        total_pnl = sim.accrued_pnl_bank
        total_trades = sum(s.get("trades_applied_count", 0) for s in sim.stints_summary)

        if total_trades > 0:
            avg_pnl = total_pnl / total_trades
        else:
            avg_pnl = 0.0

        # Longevity score: accounts used + PnL as fraction of starting balance
        longevity_score = sim.funded_accounts_used + (total_pnl / rules.account_size)

        min_c, max_c = _measure_contracts_range(resized_trades, risk, instrument)

        results.append({
            "risk_dollars": risk,
            "avg_pnl_per_trade": avg_pnl,
            "total_pnl": total_pnl,
            "funded_accounts_used": sim.funded_accounts_used,
            "accounts_blown": sim.funded_accounts_failed,
            "total_trades_executed": total_trades,
            "longevity_score": longevity_score,
            "min_contracts": min_c,
            "max_contracts": max_c,
        })

    # Filter to min_profit_per_trade and rank by Pareto (longevity first, then profit)
    viable = [r for r in results if r["avg_pnl_per_trade"] >= min_profit_per_trade]

    if not viable:
        # No candidates meet threshold; return empty result but include all results for context
        return LongevityOptimizationResult(
            strategy=strategy,
            window=window,
            min_profit_per_trade=min_profit_per_trade,
            optimal_risk_dollars=0.0,
            avg_pnl_per_trade=0.0,
            total_pnl=0.0,
            funded_accounts_used=0,
            accounts_blown=0,
            total_trades_executed=0,
            longevity_score=0.0,
            candidates=tuple(results),
        )

    # Sort by longevity_score (descending), then by avg_pnl_per_trade (descending)
    viable_sorted = sorted(
        viable,
        key=lambda r: (-r["longevity_score"], -r["avg_pnl_per_trade"]),
    )

    # Take top 5 for output
    top_5 = viable_sorted[:5]

    best = top_5[0]

    return LongevityOptimizationResult(
        strategy=strategy,
        window=window,
        min_profit_per_trade=min_profit_per_trade,
        optimal_risk_dollars=best["risk_dollars"],
        avg_pnl_per_trade=best["avg_pnl_per_trade"],
        total_pnl=best["total_pnl"],
        funded_accounts_used=best["funded_accounts_used"],
        accounts_blown=best["accounts_blown"],
        total_trades_executed=best["total_trades_executed"],
        longevity_score=best["longevity_score"],
        candidates=tuple(top_5),
    )


__all__ = [
    "SpeedOptimizationResult",
    "LongevityOptimizationResult",
    "optimize_for_speed_wf",
    "optimize_for_longevity_holdout",
]
