"""Test optimize_longevity_holdout_mc and related functions."""

from __future__ import annotations

import pandas as pd
import pytest

from v3.position_sizing import (
    optimize_longevity_holdout_mc,
    _bootstrap_pnl_p05,
    _compute_longevity_components,
)
from v3.trades import TradeResult
from v3.config import DEFAULT_FUNDED_EXPRESS_SIM
from v3.funded_express_sim import ExpressFundedSimResult


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


def test_bootstrap_pnl_p05_empty():
    """Empty trades."""
    result = _bootstrap_pnl_p05([], n=100)
    assert result == 0.0


def test_bootstrap_pnl_p05_deterministic():
    """Check bootstrap produces reasonable p05."""
    trades = [_make_trade(i, 100.0) for i in range(1, 11)]  # 10 profitable trades
    p05 = _bootstrap_pnl_p05(trades, n=500, seed=42)
    # Should be positive since all trades are profitable
    assert p05 > 0.0
    # p05 should be less than or equal to mean
    assert p05 <= 100.0


def test_compute_longevity_components():
    """Test component scoring."""
    # Create a mock sim result
    sim = ExpressFundedSimResult(
        accrued_pnl_bank=2000.0,
        funded_accounts_failed=0,
        stints_opened=1,
        funded_accounts_used=1,
        max_nominal_peak_balance=52000.0,
        worst_daily_drawdown=500.0,
        worst_stint_peak_to_trough_drawdown_from_peak_balance=1000.0,
        current_account_active=True,
        current_account_pnl=2000.0,
        current_max_drawdown=500.0,
        current_win_rate_pct=60.0,
        current_avg_r_multiple=1.5,
        current_profit_factor=2.0,
        current_sharpe_annualized=1.5,
        total_win_rate_pct=60.0,
        total_avg_r_multiple=1.5,
        best_trade_pnl=300.0,
        worst_trade_pnl=-100.0,
        stints_summary=(
            {
                "stint_index": 0,
                "trades_applied_count": 10,
                "survival_days": 20,
            },
        ),
    )

    components = _compute_longevity_components(sim, DEFAULT_FUNDED_EXPRESS_SIM, 150.0, {"survival": 0.4, "drawdown": 0.2, "efficiency": 0.2, "capital": 0.2})
    assert "survival_score" in components
    assert "drawdown_score" in components
    assert "efficiency_score" in components
    assert "capital_score" in components
    # survival and drawdown are bounded [0, 1]; efficiency and capital can exceed 1
    assert 0.0 <= components["survival_score"] <= 1.0
    assert 0.0 <= components["drawdown_score"] <= 1.0
    assert components["efficiency_score"] >= 0.0  # unbounded above
    # capital_score can be negative (loss) or > 1 (large gain)


def test_optimize_longevity_holdout_mc_empty():
    """Empty trade list."""
    result = optimize_longevity_holdout_mc([], "test_strategy")
    assert result.strategy == "test_strategy"
    assert result.optimal_risk_dollars == 0.0
    assert len(result.per_account_summary) == 0


def test_optimize_longevity_holdout_mc_simple():
    """Simple profitable trades."""
    trades = [_make_trade(i, 200.0) for i in range(1, 11)]  # 10 profitable trades

    result = optimize_longevity_holdout_mc(
        trades,
        "test_strategy",
        risk_levels=[50.0, 100.0],
        min_profit_per_trade=150.0,
        min_profit_factor=1.0,
        mc_iterations=100,  # Small for speed
        bootstrap_iterations=50,
    )

    assert result.strategy == "test_strategy"
    # With clearly profitable trades, should find a winner
    if len(trades) >= 10:
        assert result.optimal_risk_dollars > 0.0
        assert result.median_longevity_score >= 0.0


def test_optimize_longevity_holdout_mc_has_per_account_survival():
    """Verify per_account_survival_days is populated."""
    trades = [_make_trade(i, 200.0) for i in range(1, 16)]  # 15 profitable trades

    result = optimize_longevity_holdout_mc(
        trades,
        "test_strategy",
        risk_levels=[50.0],
        min_profit_per_trade=100.0,
        mc_iterations=50,
    )

    if result.optimal_risk_dollars > 0.0:
        # Should have per-account survival info
        assert len(result.per_account_survival_days) > 0
        assert len(result.per_account_summary) > 0
        # Each account should be a dict with survival_days
        for acct in result.per_account_summary:
            assert "survival_days" in acct


def test_optimize_longevity_holdout_mc_weights():
    """Test custom weights."""
    trades = [_make_trade(i, 150.0) for i in range(1, 12)]

    custom_weights = {
        "survival_score": 0.6,
        "drawdown_score": 0.2,
        "efficiency_score": 0.1,
        "capital_score": 0.1,
    }

    result = optimize_longevity_holdout_mc(
        trades,
        "test_strategy",
        weights=custom_weights,
        mc_iterations=50,
    )

    if result.optimal_risk_dollars > 0.0:
        assert result.weights == custom_weights
