"""Tests for combine_simulator (calendar-day bootstrap Combine pass rate)."""
from __future__ import annotations

import pandas as pd
import pytest

from v3.combine_simulator import (
    CombineSimResult,
    combine_sim_summary_dict,
    run_combine_simulator,
)
from v3.topstep import simulate_topstep
from v3.trades import TradeResult


def _trade(exit_time, pnl, bars_held=5, strategy="test"):
    ts = pd.Timestamp(exit_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    return TradeResult(
        strategy=strategy,
        entry_time=ts - pd.Timedelta(minutes=bars_held),
        exit_time=ts,
        direction="long",
        entry=100.0, stop=99.0, target=102.0,
        exit=100.0 + pnl / 2.0,
        contracts=1,
        gross_pnl=float(pnl), commission=0.0, net_pnl=float(pnl),
        r_multiple=pnl / 100.0,
        exit_reason="target" if pnl > 0 else "stop",
        bars_held=bars_held,
    )


def _winning_trades():
    trades = []
    base = pd.Timestamp("2024-01-02", tz="America/New_York")
    for i in range(20):
        day = base + pd.Timedelta(days=i)
        trades.append(_trade(f"{day.date()} 10:00", 200.0))
    return trades


def _losing_trades():
    trades = []
    base = pd.Timestamp("2024-01-02", tz="America/New_York")
    for i in range(20):
        day = base + pd.Timedelta(days=i)
        trades.append(_trade(f"{day.date()} 10:00", -150.0))
    return trades


def _volatile_trades():
    trades = []
    base = pd.Timestamp("2024-01-02", tz="America/New_York")
    for i in range(30):
        day = base + pd.Timedelta(days=i)
        pnl = 1_200.0 if i % 2 == 0 else -900.0
        trades.append(_trade(f"{day.date()} 10:00", pnl))
    return trades


# --- empty input ---

def test_empty_trades_returns_zero_result():
    result = run_combine_simulator([], n_resamples=100)
    assert result.pass_rate_pct == 0.0
    assert result.n_resamples == 0
    assert result.n_passed == 0
    assert result.n_trades == 0
    assert result.median_days_to_pass is None


# --- determinism ---

def test_same_seed_produces_identical_results():
    trades = _winning_trades()
    r1 = run_combine_simulator(trades, n_resamples=200, seed=7)
    r2 = run_combine_simulator(trades, n_resamples=200, seed=7)
    assert r1.pass_rate_pct == r2.pass_rate_pct
    assert r1.n_passed == r2.n_passed


def test_different_seeds_may_differ():
    trades = _volatile_trades()
    r1 = run_combine_simulator(trades, n_resamples=200, seed=1)
    r2 = run_combine_simulator(trades, n_resamples=200, seed=99)
    assert 0.0 <= r1.pass_rate_pct <= 100.0
    assert 0.0 <= r2.pass_rate_pct <= 100.0


# --- winning strategy ---

def test_consistent_winner_has_high_pass_rate():
    result = run_combine_simulator(_winning_trades(), n_resamples=500, seed=42)
    assert result.pass_rate_pct >= 95.0, (
        f"Expected >=95% pass rate for consistent winner, got {result.pass_rate_pct:.1f}%"
    )


def test_consistent_winner_days_to_pass_makes_sense():
    result = run_combine_simulator(_winning_trades(), n_resamples=200, seed=42)
    assert result.median_days_to_pass is not None
    assert 13 <= result.median_days_to_pass <= 17


# --- losing strategy ---

def test_consistent_loser_has_zero_pass_rate():
    result = run_combine_simulator(_losing_trades(), n_resamples=500, seed=42)
    assert result.pass_rate_pct == 0.0
    assert result.n_passed == 0
    assert result.median_days_to_pass is None


def test_consistent_loser_has_high_bust_rate():
    result = run_combine_simulator(_losing_trades(), n_resamples=200, seed=42)
    bust_or_not_passed = result.n_failed_drawdown + result.n_not_passed
    assert bust_or_not_passed == result.n_resamples


# --- counts ---

def test_resample_counts_sum_to_n_resamples():
    result = run_combine_simulator(_volatile_trades(), n_resamples=300, seed=42)
    total = result.n_passed + result.n_failed_drawdown + result.n_not_passed
    assert total == result.n_resamples == 300


def test_pass_rate_pct_consistent_with_n_passed():
    result = run_combine_simulator(_volatile_trades(), n_resamples=400, seed=42)
    expected = 100.0 * result.n_passed / result.n_resamples
    assert abs(result.pass_rate_pct - expected) < 1e-9


# --- EOD trailing drawdown fix ---

def test_eod_trailing_drawdown_intraday_peak_does_not_ratchet_floor():
    from v3.config import TOPSTEP_50K
    trades = [
        _trade("2024-01-02 10:00", 1_800.0),
        _trade("2024-01-02 11:00", -1_500.0),
        _trade("2024-01-03 10:00", 400.0),
        _trade("2024-01-04 10:00", 400.0),
        _trade("2024-01-05 10:00", 400.0),
        _trade("2024-01-08 10:00", 400.0),
        _trade("2024-01-09 10:00", 400.0),
        _trade("2024-01-10 10:00", 400.0),
        _trade("2024-01-11 10:00", 400.0),
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed, f"Expected pass with EOD trailing, got: {result.reason}"
    assert not result.failed


def test_daily_loss_lock_ignores_later_same_day_trades_without_failing():
    from v3.config import TOPSTEP_50K
    trades = [
        _trade("2024-01-02 10:00", -600.0),
        _trade("2024-01-02 10:30", -500.0),
        _trade("2024-01-02 11:00", 2_000.0),
        _trade("2024-01-03 10:00", 1_000.0),
        _trade("2024-01-04 10:00", 1_000.0),
        _trade("2024-01-05 10:00", 1_000.0),
        _trade("2024-01-08 10:00", 1_000.0),
        _trade("2024-01-09 10:00", 1_000.0),
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed
    assert not result.failed
    assert result.days_to_pass == 6
    assert result.final_balance == pytest.approx(53_900.0)
    assert result.max_daily_loss >= 1_100.0


# --- pass gate does NOT require min_trading_days ---

def test_combine_passes_in_fewer_than_min_trading_days():
    from v3.config import TOPSTEP_50K
    trades = [
        _trade("2024-01-02 10:00", 1_000.0),
        _trade("2024-01-03 10:00", 1_000.0),
        _trade("2024-01-04 10:00", 1_000.0),
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed
    assert result.days_to_pass == 3


# --- same calendar day stays one simulated day ---

def test_calendar_day_grouping_keeps_same_day_trades_together():
    """Both trades on the losing day (-$600, -$500) always share one simulated day.
    If they were split across days, worst_max_drawdown could be $600; together it is $1,100."""
    bad_day_trades = [
        _trade("2024-01-02 10:00", -600.0),
        _trade("2024-01-02 10:30", -500.0),
    ]
    good_day_trades = [
        _trade(f"2024-01-{d:02d} 10:00", 200.0) for d in range(3, 23)
    ]
    trades = bad_day_trades + good_day_trades
    result = run_combine_simulator(trades, n_resamples=500, seed=42)
    assert result.worst_max_drawdown == pytest.approx(1100.0), (
        f"Day grouping broken: worst_max_drawdown={result.worst_max_drawdown} (expected 1100)"
    )
    assert result.pct_daily_limit_hit > 0.0


# --- summary dict ---

def test_summary_dict_contains_all_keys():
    result = run_combine_simulator(_winning_trades(), n_resamples=50, seed=1)
    d = combine_sim_summary_dict(result)
    expected_keys = {
        "combine_pass_rate_pct", "combine_n_resamples", "combine_n_passed",
        "combine_n_failed_drawdown", "combine_n_not_passed",
        "combine_median_days_to_pass", "combine_mean_days_to_pass",
        "combine_min_days_to_pass", "combine_max_days_to_pass",
        "combine_pct_daily_limit_hit", "combine_mean_max_drawdown",
        "combine_worst_max_drawdown", "combine_n_trades", "combine_n_trading_days",
    }
    assert expected_keys == set(d.keys())


def test_summary_dict_pass_rate_matches_result():
    result = run_combine_simulator(_winning_trades(), n_resamples=50, seed=1)
    d = combine_sim_summary_dict(result)
    assert d["combine_pass_rate_pct"] == result.pass_rate_pct


# --- metadata ---

def test_n_resamples_requested_stored_on_result():
    result = run_combine_simulator(_winning_trades(), n_resamples=123, seed=1)
    assert result.n_resamples_requested == 123
    assert result.n_resamples == 123


def test_n_trading_days_reflects_input_day_count():
    trades = _winning_trades()
    result = run_combine_simulator(trades, n_resamples=10, seed=1)
    assert result.n_trading_days == 20
    assert result.n_trades == 20
