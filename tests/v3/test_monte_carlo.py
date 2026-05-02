from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from v3.monte_carlo import (
    MCResult,
    _dd_duration_trades,
    block_bootstrap_permute,
    mc_summary_dict,
    mc_summary_text,
    run_mc,
)
from v3.trades import TradeResult

TZ = ZoneInfo("America/New_York")


def _ts(day: int, hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(2025, 6, day, hour, minute, tz=TZ)


def _trade(exit_ts: pd.Timestamp, net: float, r: float = 0.5) -> TradeResult:
    entry_ts = exit_ts - pd.Timedelta(minutes=5)
    return TradeResult(
        strategy="test",
        entry_time=entry_ts,
        exit_time=exit_ts,
        direction="long",
        entry=0.0,
        stop=0.0,
        target=1.0,
        exit=0.0,
        contracts=1,
        gross_pnl=net,
        commission=0.0,
        net_pnl=net,
        r_multiple=r,
        exit_reason="test",
        bars_held=1,
        params={},
    )


def _make_trades(n: int = 20) -> list[TradeResult]:
    trades = []
    for i in range(n):
        day = (i // 3) + 2
        net = 100.0 if i % 2 == 0 else -50.0
        trades.append(_trade(_ts(day, 10 + (i % 4)), net, r=1.0 if net > 0 else -0.5))
    return trades


def test_block_bootstrap_preserves_trade_count():
    import random
    trades = _make_trades(20)
    rng = random.Random(42)
    result = block_bootstrap_permute(trades, block_size=5, rng=rng)
    assert len(result) == len(trades)


def test_block_bootstrap_returns_same_trades_different_order():
    import random
    trades = _make_trades(20)
    rng = random.Random(1)
    result = block_bootstrap_permute(trades, block_size=5, rng=rng)
    assert set(id(t) for t in result) == set(id(t) for t in trades)


def test_block_bootstrap_empty_returns_empty():
    import random
    rng = random.Random(0)
    assert block_bootstrap_permute([], block_size=5, rng=rng) == []


def test_run_mc_returns_mc_result():
    trades = _make_trades(30)
    result = run_mc(trades, n_perms=50, block_size=5, seed=42)
    assert isinstance(result, MCResult)
    assert result.n_perms == 50
    assert result.block_size == 5


def test_run_mc_empty_trades():
    result = run_mc([], n_perms=50, block_size=5, seed=42)
    assert isinstance(result, MCResult)
    assert result.pnl_p05 == 0.0
    assert result.win_rate_mean == 0.0


def test_run_mc_ci_bands_ordered():
    trades = _make_trades(40)
    result = run_mc(trades, n_perms=200, block_size=5, seed=42)
    assert result.win_rate_ci_lo <= result.win_rate_mean <= result.win_rate_ci_hi
    assert result.expectancy_ci_lo <= result.expectancy_mean <= result.expectancy_ci_hi
    assert result.pnl_p05 <= result.pnl_p50 <= result.pnl_p95


def test_run_mc_equity_curves_stored():
    trades = _make_trades(20)
    result = run_mc(trades, n_perms=10, block_size=3, seed=0)
    assert len(result.equity_curves) == 10
    # Each curve starts at 0
    for curve in result.equity_curves:
        assert curve[0] == pytest.approx(0.0)


def test_run_mc_actual_equity_matches_original():
    trades = _make_trades(20)
    result = run_mc(trades, n_perms=5, block_size=5, seed=0)
    expected_final = sum(t.net_pnl for t in trades)
    assert result.actual_equity[-1] == pytest.approx(expected_final)


def test_mc_summary_dict_has_required_keys():
    trades = _make_trades(20)
    result = run_mc(trades, n_perms=20, block_size=5, seed=42)
    d = mc_summary_dict(result)
    required = {
        "pnl_p05", "pnl_p50", "pnl_p95",
        "win_rate_mean", "win_rate_ci_lo", "win_rate_ci_hi",
        "expectancy_mean", "expectancy_ci_lo", "expectancy_ci_hi",
        "max_daily_loss_mean", "max_daily_loss_worst_p05",
        "dd_duration_mean", "dd_duration_p95",
    }
    assert required <= set(d)


def test_mc_summary_text_contains_key_labels():
    trades = _make_trades(20)
    result = run_mc(trades, n_perms=20, block_size=5, seed=42)
    text = mc_summary_text(result, title="Test MC")
    assert "Test MC" in text
    assert "win_rate" in text
    assert "expectancy" in text
    assert "dd_duration" in text


def test_run_mc_positive_pnl_trades_have_positive_p50():
    trades = [_trade(_ts(2 + i // 4, 10 + i % 4), 100.0, r=1.0) for i in range(40)]
    result = run_mc(trades, n_perms=100, block_size=5, seed=0)
    assert result.pnl_p50 > 0


def test_block_size_larger_than_trades_runs_without_error():
    trades = _make_trades(5)
    result = run_mc(trades, n_perms=10, block_size=20, seed=0)
    assert isinstance(result, MCResult)


def test_dd_duration_exact_recovery():
    # Peak at idx 1, equity drops then recovers to exact peak at idx 3 — duration = 2
    assert _dd_duration_trades([0.0, 10.0, 8.0, 10.0]) == 2


def test_dd_duration_ongoing_drawdown_at_end():
    # Still in drawdown at final bar — duration = 1 (peak idx 1, final idx 2)
    assert _dd_duration_trades([0.0, 10.0, 7.0]) == 1


def test_dd_duration_no_drawdown():
    assert _dd_duration_trades([0.0, 5.0, 10.0, 15.0]) == 0


def test_dd_duration_single_element():
    assert _dd_duration_trades([100.0]) == 0
