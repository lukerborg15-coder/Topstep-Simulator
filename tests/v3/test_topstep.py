from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from v3.config import TOPSTEP_50K
from v3.topstep import count_sequential_eval_passes, simulate_topstep
from v3.trades import TradeResult


TZ = ZoneInfo("America/New_York")


def _trade(exit_ts: pd.Timestamp, net: float) -> TradeResult:
    return TradeResult(
        strategy="test",
        entry_time=exit_ts - pd.Timedelta(minutes=5),
        exit_time=exit_ts,
        direction="long",
        entry=100.0,
        stop=99.0,
        target=101.0,
        exit=100.0,
        contracts=1,
        gross_pnl=net,
        commission=0.0,
        net_pnl=net,
        r_multiple=net / 100.0,
        exit_reason="test",
        bars_held=1,
        params={},
    )


def test_topstep_daily_loss_groups_overnight_trades_into_same_futures_session() -> None:
    trades = [
        _trade(pd.Timestamp(2024, 1, 2, 18, 5, tz=TZ), -600.0),
        _trade(pd.Timestamp(2024, 1, 3, 10, 0, tz=TZ), -500.0),
        _trade(pd.Timestamp(2024, 1, 4, 10, 0, tz=TZ), 1200.0),
    ]

    result = simulate_topstep(trades, TOPSTEP_50K)

    assert result.passed is False
    assert result.failed is False
    assert result.max_daily_loss == 1100.0


def test_topstep_sequential_eval_drops_full_futures_session_after_termination() -> None:
    trades = [
        _trade(pd.Timestamp(2024, 1, 2, 18, 5, tz=TZ), -600.0),
        _trade(pd.Timestamp(2024, 1, 3, 10, 0, tz=TZ), -1500.0),
        _trade(pd.Timestamp(2024, 1, 3, 18, 5, tz=TZ), 1000.0),
        _trade(pd.Timestamp(2024, 1, 4, 18, 5, tz=TZ), 1000.0),
        _trade(pd.Timestamp(2024, 1, 8, 10, 0, tz=TZ), 1000.0),
    ]

    passes, log = count_sequential_eval_passes(trades, TOPSTEP_50K)

    assert passes == 1
    assert len(log) == 2
