"""Test run_sizing_comparison."""

from __future__ import annotations

import pandas as pd
import pytest

from v3.sizing_comparison import run_sizing_comparison
from v3.position_sizing import (
    SpeedOptimizationAggregateResult,
    LongevityOptimizationMCResult,
)
from v3.trades import TradeResult


def _make_trade(day: int, net_pnl: float, contracts: int = 1) -> TradeResult:
    """Helper to create a TradeResult."""
    entry_time = pd.Timestamp(f"2024-01-{day:02d} 10:00:00")
    exit_time = pd.Timestamp(f"2024-01-{day:02d} 11:00:00")
    return TradeResult(
        strategy="test",
        entry_time=entry_time,
        exit_time=exit_time,
        direction="long",
        entry=100.0,
        stop=99.0,
        target=101.0,
        exit=100.5 if net_pnl > 0 else 99.5,
        contracts=contracts,
        gross_pnl=net_pnl + 1.40,
        commission=1.40,
        net_pnl=net_pnl,
        r_multiple=net_pnl / 100.0,
        exit_reason="target" if net_pnl > 0 else "stop",
        bars_held=1,
    )


def test_run_sizing_comparison_optimizer_only():
    """Test with only optimizer (no fixed tracks)."""
    fold_pairs = [
        ([_make_trade(i, 200.0) for i in range(1, 8)], [_make_trade(i, 150.0) for i in range(8, 12)]),
    ]
    holdout = [_make_trade(i, 200.0) for i in range(20, 30)]

    speed_result = SpeedOptimizationAggregateResult(
        strategy="test",
        pass_floor_pct=40.0,
        speed_target_days=5.0,
        attempt_budget=10,
        n_folds=1,
        viable_folds=1,
        optimal_risk_dollars=100.0,
        median_oos_utility=0.75,
        min_oos_utility=0.75,
        median_oos_pass_rate_pct=75.0,
        median_oos_median_days_to_pass=3.0,
        per_fold_oos=(),
    )

    longevity_result = LongevityOptimizationMCResult(
        strategy="test",
        window="holdout",
        min_profit_per_trade=150.0,
        min_profit_factor=1.2,
        weights={"survival_score": 0.4, "drawdown_score": 0.2, "efficiency_score": 0.2, "capital_score": 0.2},
        mc_iterations=500,
        mc_block_size=5,
        bootstrap_iterations=1000,
        optimal_risk_dollars=75.0,
        median_longevity_score=0.85,
        p05_longevity_score=0.70,
        median_components={"survival_score": 0.9, "drawdown_score": 0.8, "efficiency_score": 0.9, "capital_score": 0.8},
        p05_components={"survival_score": 0.7, "drawdown_score": 0.6, "efficiency_score": 0.7, "capital_score": 0.6},
        median_avg_pnl_per_trade=200.0,
        p05_avg_pnl_per_trade=180.0,
        median_accounts_used=1.0,
        median_accounts_blown=0.0,
        per_account_survival_days=(25,),
        per_account_summary=({"survival_days": 25},),
    )

    result = run_sizing_comparison(
        fold_pairs,
        holdout,
        speed_result,
        longevity_result,
        fixed_risk_dollars=None,
        fixed_contracts=None,
    )

    assert result.strategy == "test"
    assert result.track_a_optimizer is not None
    assert result.track_b_fixed_risk is None
    assert result.track_c_fixed_contracts is None
    # Small sample size flag is expected with mock data; just confirm no crash and structure is right
    assert isinstance(result.sanity_flags, tuple)


def test_run_sizing_comparison_with_fixed_risk():
    """Test with fixed risk track."""
    fold_pairs = [
        ([_make_trade(i, 200.0) for i in range(1, 8)], [_make_trade(i, 150.0) for i in range(8, 12)]),
    ]
    holdout = [_make_trade(i, 200.0) for i in range(20, 30)]

    speed_result = SpeedOptimizationAggregateResult(
        strategy="test",
        pass_floor_pct=40.0,
        speed_target_days=5.0,
        attempt_budget=10,
        n_folds=1,
        viable_folds=1,
        optimal_risk_dollars=100.0,
        median_oos_utility=0.75,
        min_oos_utility=0.75,
        median_oos_pass_rate_pct=75.0,
        median_oos_median_days_to_pass=3.0,
        per_fold_oos=(),
    )

    longevity_result = LongevityOptimizationMCResult(
        strategy="test",
        window="holdout",
        min_profit_per_trade=150.0,
        min_profit_factor=1.2,
        weights={"survival_score": 0.4, "drawdown_score": 0.2, "efficiency_score": 0.2, "capital_score": 0.2},
        mc_iterations=500,
        mc_block_size=5,
        bootstrap_iterations=1000,
        optimal_risk_dollars=75.0,
        median_longevity_score=0.85,
        p05_longevity_score=0.70,
        median_components={"survival_score": 0.9, "drawdown_score": 0.8, "efficiency_score": 0.9, "capital_score": 0.8},
        p05_components={"survival_score": 0.7, "drawdown_score": 0.6, "efficiency_score": 0.7, "capital_score": 0.6},
        median_avg_pnl_per_trade=200.0,
        p05_avg_pnl_per_trade=180.0,
        median_accounts_used=1.0,
        median_accounts_blown=0.0,
        per_account_survival_days=(25,),
        per_account_summary=({"survival_days": 25},),
    )

    result = run_sizing_comparison(
        fold_pairs,
        holdout,
        speed_result,
        longevity_result,
        fixed_risk_dollars=50.0,  # Different from optimal 100
        fixed_contracts=None,
    )

    assert result.track_b_fixed_risk is not None
    assert result.track_b_fixed_risk["risk_dollars"] == 50.0
    assert "fixed_risk_vs_optimizer" in result.deltas


def test_run_sizing_comparison_with_fixed_contracts():
    """Test with fixed contracts track."""
    fold_pairs = [
        ([_make_trade(i, 200.0, contracts=2) for i in range(1, 8)], [_make_trade(i, 150.0, contracts=2) for i in range(8, 12)]),
    ]
    holdout = [_make_trade(i, 200.0, contracts=2) for i in range(20, 30)]

    speed_result = SpeedOptimizationAggregateResult(
        strategy="test",
        pass_floor_pct=40.0,
        speed_target_days=5.0,
        attempt_budget=10,
        n_folds=1,
        viable_folds=1,
        optimal_risk_dollars=100.0,
        median_oos_utility=0.75,
        min_oos_utility=0.75,
        median_oos_pass_rate_pct=75.0,
        median_oos_median_days_to_pass=3.0,
        per_fold_oos=(),
    )

    longevity_result = LongevityOptimizationMCResult(
        strategy="test",
        window="holdout",
        min_profit_per_trade=150.0,
        min_profit_factor=1.2,
        weights={"survival_score": 0.4, "drawdown_score": 0.2, "efficiency_score": 0.2, "capital_score": 0.2},
        mc_iterations=500,
        mc_block_size=5,
        bootstrap_iterations=1000,
        optimal_risk_dollars=75.0,
        median_longevity_score=0.85,
        p05_longevity_score=0.70,
        median_components={"survival_score": 0.9, "drawdown_score": 0.8, "efficiency_score": 0.9, "capital_score": 0.8},
        p05_components={"survival_score": 0.7, "drawdown_score": 0.6, "efficiency_score": 0.7, "capital_score": 0.6},
        median_avg_pnl_per_trade=200.0,
        p05_avg_pnl_per_trade=180.0,
        median_accounts_used=1.0,
        median_accounts_blown=0.0,
        per_account_survival_days=(25,),
        per_account_summary=({"survival_days": 25},),
    )

    result = run_sizing_comparison(
        fold_pairs,
        holdout,
        speed_result,
        longevity_result,
        fixed_risk_dollars=None,
        fixed_contracts=1,
    )

    assert result.track_c_fixed_contracts is not None
    assert result.track_c_fixed_contracts["fixed_contracts"] == 1
    assert "fixed_contracts_vs_optimizer" in result.deltas


def test_run_sizing_comparison_sanity_flags_small_sample():
    """Test sanity flag for small holdout sample."""
    fold_pairs = [
        ([_make_trade(i, 200.0) for i in range(1, 5)], [_make_trade(i, 150.0) for i in range(5, 8)]),
    ]
    holdout = [_make_trade(i, 200.0) for i in range(20, 25)]  # Only 5 trades

    speed_result = SpeedOptimizationAggregateResult(
        strategy="test",
        pass_floor_pct=40.0,
        speed_target_days=5.0,
        attempt_budget=10,
        n_folds=1,
        viable_folds=1,
        optimal_risk_dollars=100.0,
        median_oos_utility=0.75,
        min_oos_utility=0.75,
        median_oos_pass_rate_pct=75.0,
        median_oos_median_days_to_pass=3.0,
        per_fold_oos=(),
    )

    longevity_result = LongevityOptimizationMCResult(
        strategy="test",
        window="holdout",
        min_profit_per_trade=150.0,
        min_profit_factor=1.2,
        weights={"survival_score": 0.4, "drawdown_score": 0.2, "efficiency_score": 0.2, "capital_score": 0.2},
        mc_iterations=500,
        mc_block_size=5,
        bootstrap_iterations=1000,
        optimal_risk_dollars=75.0,
        median_longevity_score=0.85,
        p05_longevity_score=0.70,
        median_components={"survival_score": 0.9, "drawdown_score": 0.8, "efficiency_score": 0.9, "capital_score": 0.8},
        p05_components={"survival_score": 0.7, "drawdown_score": 0.6, "efficiency_score": 0.7, "capital_score": 0.6},
        median_avg_pnl_per_trade=200.0,
        p05_avg_pnl_per_trade=180.0,
        median_accounts_used=1.0,
        median_accounts_blown=0.0,
        per_account_survival_days=(25,),
        per_account_summary=({"survival_days": 25},),
    )

    result = run_sizing_comparison(
        fold_pairs,
        holdout,
        speed_result,
        longevity_result,
    )

    # Should have sanity flag about small sample
    assert any("small" in flag.lower() for flag in result.sanity_flags)
