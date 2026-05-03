"""Test long/short breakdown in compute_metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from v3.evaluator import compute_metrics
from v3.trades import TradeResult


def _make_trade(direction: str, net_pnl: float, r_multiple: float = 1.0) -> TradeResult:
    """Helper to create a minimal TradeResult."""
    entry_time = pd.Timestamp("2024-01-01 10:00:00")
    exit_time = pd.Timestamp("2024-01-01 11:00:00")
    return TradeResult(
        strategy="test_strategy",
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
        entry=100.0,
        stop=99.0,
        target=101.0,
        exit=100.5 if net_pnl > 0 else 99.5,
        contracts=1,
        gross_pnl=net_pnl + 1.40,  # Add back commission
        commission=1.40,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        exit_reason="target" if net_pnl > 0 else "stop",
        bars_held=1,
    )


def test_compute_metrics_empty():
    """Empty trade list returns zero counts."""
    metrics = compute_metrics([])
    assert metrics["total_trades"] == 0
    assert metrics["long_trades"] == 0
    assert metrics["short_trades"] == 0
    assert metrics["long_win_rate"] == 0.0
    assert metrics["short_win_rate"] == 0.0
    assert metrics["long_net_pnl"] == 0.0
    assert metrics["short_net_pnl"] == 0.0


def test_compute_metrics_long_only():
    """Long-only trades."""
    trades = [
        _make_trade("long", 100.0, 1.0),
        _make_trade("long", -50.0, -0.5),
        _make_trade("long", 200.0, 2.0),
    ]
    metrics = compute_metrics(trades)
    assert metrics["long_trades"] == 3
    assert metrics["short_trades"] == 0
    assert metrics["long_win_rate"] == pytest.approx(2/3)
    assert metrics["short_win_rate"] == 0.0
    assert metrics["long_net_pnl"] == pytest.approx(250.0)
    assert metrics["short_net_pnl"] == 0.0


def test_compute_metrics_short_only():
    """Short-only trades."""
    trades = [
        _make_trade("short", 75.0, 0.75),
        _make_trade("short", -100.0, -1.0),
    ]
    metrics = compute_metrics(trades)
    assert metrics["long_trades"] == 0
    assert metrics["short_trades"] == 2
    assert metrics["long_win_rate"] == 0.0
    assert metrics["short_win_rate"] == pytest.approx(0.5)
    assert metrics["long_net_pnl"] == 0.0
    assert metrics["short_net_pnl"] == pytest.approx(-25.0)


def test_compute_metrics_mixed():
    """Mixed long/short trades."""
    trades = [
        _make_trade("long", 100.0, 1.0),
        _make_trade("long", -50.0, -0.5),
        _make_trade("short", 75.0, 0.75),
        _make_trade("short", -100.0, -1.0),
    ]
    metrics = compute_metrics(trades)
    assert metrics["total_trades"] == 4
    assert metrics["long_trades"] == 2
    assert metrics["short_trades"] == 2
    assert metrics["long_win_rate"] == pytest.approx(0.5)  # 1 win / 2 long
    assert metrics["short_win_rate"] == pytest.approx(0.5)  # 1 win / 2 short
    assert metrics["long_net_pnl"] == pytest.approx(50.0)  # 100 - 50
    assert metrics["short_net_pnl"] == pytest.approx(-25.0)  # 75 - 100
