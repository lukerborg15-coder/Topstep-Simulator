"""Compare optimizer-selected sizing vs fixed risk/contract sizing across folds and holdout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    DEFAULT_MAX_CONTRACTS,
    MNQ,
    TOPSTEP_50K,
    FundedExpressSimRules,
    DEFAULT_FUNDED_EXPRESS_SIM,
    Instrument,
    TopStepRules,
)
from .evaluator import _contracts_for_fixed_risk
from .position_sizing import (
    SpeedOptimizationAggregateResult,
    LongevityOptimizationMCResult,
    _count_sequential_eval_passes_capped,
    _resize_trades_for_risk,
    _resize_trades_for_fixed_contracts,
)
from .funded_express_sim import simulate_express_funded_resets
from .trades import TradeResult


@dataclass(frozen=True)
class SizingComparisonResult:
    """Result of comparing optimizer vs fixed-sizing approaches."""

    strategy: str
    fixed_risk_dollars: float | None
    fixed_contracts: int | None
    track_a_optimizer: dict[str, Any]      # {eval: {risk, pass_rate, median_days, ...}, holdout: {risk, longevity_score, ...}}
    track_b_fixed_risk: dict[str, Any] | None
    track_c_fixed_contracts: dict[str, Any] | None
    deltas: dict[str, Any]                  # vs optimizer baseline
    sanity_flags: tuple[str, ...]           # human-readable flags


def run_sizing_comparison(
    fold_trade_pairs: list[tuple[list[TradeResult], list[TradeResult]]],   # WF train+test
    holdout_trades: list[TradeResult],
    optimizer_speed_result: SpeedOptimizationAggregateResult,
    optimizer_longevity_result: LongevityOptimizationMCResult,
    fixed_risk_dollars: float | None = None,
    fixed_contracts: int | None = None,
    *,
    rules_topstep: TopStepRules = TOPSTEP_50K,
    rules_funded: FundedExpressSimRules = DEFAULT_FUNDED_EXPRESS_SIM,
    instrument: Instrument = MNQ,
    attempt_budget: int = 10,
) -> SizingComparisonResult:
    """
    Compare optimizer-selected sizing vs fixed risk/contracts.

    Track A: Optimizer (using optimal_risk_dollars from speed and longevity results)
    Track B: Fixed risk dollars (if fixed_risk_dollars provided)
    Track C: Fixed contracts (if fixed_contracts provided)

    Args:
        fold_trade_pairs: List of (train_trades, test_trades) per fold
        holdout_trades: Holdout period trades
        optimizer_speed_result: Speed optimization result with optimal_risk_dollars
        optimizer_longevity_result: Longevity optimization result
        fixed_risk_dollars: Fixed risk to test (Track B)
        fixed_contracts: Fixed contracts to test (Track C)
        rules_topstep: TopStepRules for eval phase
        rules_funded: FundedExpressSimRules for funded phase
        instrument: MNQ, ES, etc.
        attempt_budget: Max sequential eval attempts

    Returns:
        SizingComparisonResult
    """
    sanity_flags: list[str] = []

    # Track A: Optimizer baseline
    optimizer_eval_risk = optimizer_speed_result.optimal_risk_dollars
    optimizer_funded_risk = optimizer_longevity_result.optimal_risk_dollars

    track_a_optimizer = {
        "eval_track": {
            "risk_dollars": optimizer_eval_risk,
            "pass_rate_pct": optimizer_speed_result.median_oos_pass_rate_pct,
            "median_days_to_pass": optimizer_speed_result.median_oos_median_days_to_pass,
            "utility": optimizer_speed_result.median_oos_utility,
        },
        "holdout_track": {
            "risk_dollars": optimizer_funded_risk,
            "longevity_score": optimizer_longevity_result.median_longevity_score,
            "accounts_used": int(optimizer_longevity_result.median_accounts_used),
            "accounts_blown": int(optimizer_longevity_result.median_accounts_blown),
        },
    }

    # Track B: Fixed risk dollars
    track_b_fixed_risk = None
    if fixed_risk_dollars is not None:
        oos_pass_rates = []
        oos_median_days = []

        # Eval on OOS folds
        for train_trades, test_trades in fold_trade_pairs:
            resized_test = _resize_trades_for_risk(test_trades, fixed_risk_dollars, instrument, DEFAULT_MAX_CONTRACTS)
            if not resized_test:
                continue
            passes, log, _ = _count_sequential_eval_passes_capped(resized_test, rules_topstep, attempt_budget)
            pass_rate = (passes / min(attempt_budget, len(log))) * 100.0 if log else 0.0
            days_list = [r.days_to_pass for r in log if r.passed and r.days_to_pass is not None]
            median_days = float(np.median(days_list)) if days_list else float("inf")
            oos_pass_rates.append(pass_rate)
            oos_median_days.append(median_days)

        # Holdout funded sim
        resized_holdout = _resize_trades_for_risk(holdout_trades, fixed_risk_dollars, instrument, DEFAULT_MAX_CONTRACTS)
        if resized_holdout:
            holdout_sim = simulate_express_funded_resets(resized_holdout, rules_funded)
            holdout_longevity = 1.0 + holdout_sim.accrued_pnl_bank / rules_funded.account_size
        else:
            holdout_longevity = 0.0

        track_b_fixed_risk = {
            "risk_dollars": fixed_risk_dollars,
            "eval_track": {
                "pass_rate_pct": float(np.median(oos_pass_rates)) if oos_pass_rates else 0.0,
                "median_days_to_pass": float(np.median(oos_median_days)) if oos_median_days else float("inf"),
            },
            "holdout_track": {
                "longevity_score": holdout_longevity,
                "accounts_used": int(holdout_sim.funded_accounts_used) if resized_holdout else 0,
                "accounts_blown": int(holdout_sim.funded_accounts_failed) if resized_holdout else 0,
            },
        }

        # Sanity checks
        if track_b_fixed_risk["eval_track"]["pass_rate_pct"] > optimizer_speed_result.median_oos_pass_rate_pct + 5.0:
            sanity_flags.append("Fixed risk beats optimizer eval pass_rate by >5pp")
        if track_b_fixed_risk["holdout_track"]["longevity_score"] > optimizer_longevity_result.median_longevity_score + 0.10:
            sanity_flags.append("Fixed risk beats optimizer holdout longevity_score by >0.10")

    # Track C: Fixed contracts
    track_c_fixed_contracts = None
    if fixed_contracts is not None:
        fixed_contracts_capped = min(fixed_contracts, DEFAULT_MAX_CONTRACTS)
        oos_pass_rates_c = []
        oos_median_days_c = []

        # Eval on OOS folds
        for train_trades, test_trades in fold_trade_pairs:
            fixed_con_trades = _resize_trades_for_fixed_contracts(test_trades, fixed_contracts_capped, DEFAULT_MAX_CONTRACTS)

            if not fixed_con_trades:
                continue
            passes, log, _ = _count_sequential_eval_passes_capped(fixed_con_trades, rules_topstep, attempt_budget)
            pass_rate = (passes / min(attempt_budget, len(log))) * 100.0 if log else 0.0
            days_list = [r.days_to_pass for r in log if r.passed and r.days_to_pass is not None]
            median_days = float(np.median(days_list)) if days_list else float("inf")
            oos_pass_rates_c.append(pass_rate)
            oos_median_days_c.append(median_days)

        # Holdout funded sim
        holdout_fixed_con = _resize_trades_for_fixed_contracts(holdout_trades, fixed_contracts_capped, DEFAULT_MAX_CONTRACTS)

        if holdout_fixed_con:
            holdout_sim_c = simulate_express_funded_resets(holdout_fixed_con, rules_funded)
            holdout_longevity_c = 1.0 + holdout_sim_c.accrued_pnl_bank / rules_funded.account_size
        else:
            holdout_sim_c = None
            holdout_longevity_c = 0.0

        track_c_fixed_contracts = {
            "fixed_contracts": fixed_contracts_capped,
            "eval_track": {
                "pass_rate_pct": float(np.median(oos_pass_rates_c)) if oos_pass_rates_c else 0.0,
                "median_days_to_pass": float(np.median(oos_median_days_c)) if oos_median_days_c else float("inf"),
            },
            "holdout_track": {
                "longevity_score": holdout_longevity_c,
                "accounts_used": int(holdout_sim_c.funded_accounts_used) if holdout_sim_c else 0,
                "accounts_blown": int(holdout_sim_c.funded_accounts_failed) if holdout_sim_c else 0,
            },
        }

        # Sanity checks
        if track_c_fixed_contracts["eval_track"]["pass_rate_pct"] > optimizer_speed_result.median_oos_pass_rate_pct + 5.0:
            sanity_flags.append("Fixed contracts beats optimizer eval pass_rate by >5pp")
        if track_c_fixed_contracts["holdout_track"]["longevity_score"] > optimizer_longevity_result.median_longevity_score + 0.10:
            sanity_flags.append("Fixed contracts beats optimizer holdout longevity_score by >0.10")

    # Additional sanity flags
    if optimizer_speed_result.per_fold_oos:
        viable_fold_count = len(set(p["fold_idx"] for p in optimizer_speed_result.per_fold_oos))
        if viable_fold_count < len(fold_trade_pairs):
            sanity_flags.append(f"Optimizer chose risk viable in only {viable_fold_count}/{len(fold_trade_pairs)} folds")

    if len(holdout_trades) < 50:
        sanity_flags.append(f"Holdout sample size N={len(holdout_trades)} is small (< 50 trades)")

    # Compute deltas
    deltas = {}
    if track_b_fixed_risk:
        deltas["fixed_risk_vs_optimizer"] = {
            "eval_pass_rate_delta": track_b_fixed_risk["eval_track"]["pass_rate_pct"] - track_a_optimizer["eval_track"]["pass_rate_pct"],
            "eval_days_delta": track_b_fixed_risk["eval_track"]["median_days_to_pass"] - track_a_optimizer["eval_track"]["median_days_to_pass"],
            "holdout_longevity_delta": track_b_fixed_risk["holdout_track"]["longevity_score"] - track_a_optimizer["holdout_track"]["longevity_score"],
        }
    if track_c_fixed_contracts:
        deltas["fixed_contracts_vs_optimizer"] = {
            "eval_pass_rate_delta": track_c_fixed_contracts["eval_track"]["pass_rate_pct"] - track_a_optimizer["eval_track"]["pass_rate_pct"],
            "eval_days_delta": track_c_fixed_contracts["eval_track"]["median_days_to_pass"] - track_a_optimizer["eval_track"]["median_days_to_pass"],
            "holdout_longevity_delta": track_c_fixed_contracts["holdout_track"]["longevity_score"] - track_a_optimizer["holdout_track"]["longevity_score"],
        }

    return SizingComparisonResult(
        strategy=optimizer_speed_result.strategy,
        fixed_risk_dollars=fixed_risk_dollars,
        fixed_contracts=fixed_contracts,
        track_a_optimizer=track_a_optimizer,
        track_b_fixed_risk=track_b_fixed_risk,
        track_c_fixed_contracts=track_c_fixed_contracts,
        deltas=deltas,
        sanity_flags=tuple(sanity_flags),
    )


__all__ = [
    "SizingComparisonResult",
    "run_sizing_comparison",
]
