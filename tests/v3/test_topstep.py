from __future__ import annotations

import pandas as pd

from v3.topstep import simulate_topstep
from v3.trades import TradeResult


def _trade(exit_time: str, pnl: float) -> TradeResult:
    ts = pd.Timestamp(exit_time, tz="America/New_York")
    return TradeResult(
        strategy="test",
        entry_time=ts - pd.Timedelta(minutes=5),
        exit_time=ts,
        direction="long",
        entry=100.0,
        stop=99.0,
        target=102.0,
        exit=100.0,
        contracts=1,
        gross_pnl=pnl,
        commission=0.0,
        net_pnl=pnl,
        r_multiple=pnl / 100.0,
        exit_reason="target" if pnl > 0 else "stop",
        bars_held=1,
        params={},
    )


def test_topstep_groups_trades_by_futures_session_day() -> None:
    trades = [
        _trade("2025-06-02 18:05", 1_000.0),
        _trade("2025-06-03 09:30", 500.0),
        _trade("2025-06-03 18:05", 1_000.0),
        _trade("2025-06-04 09:30", 500.0),
    ]

    result = simulate_topstep(trades)

    assert result.passed is True
    assert result.days_to_pass == 2
