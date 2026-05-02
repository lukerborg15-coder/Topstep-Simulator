from __future__ import annotations

import pandas as pd

from v3.config import FundedExpressSimRules, TopStepRules
from v3.position_sizing import (
    LongevityOptimizationResult,
    SpeedOptimizationResult,
    optimize_for_longevity_holdout,
    optimize_for_speed_wf,
)
from v3.trades import TradeResult


def _ts(day: int) -> pd.Timestamp:
    return pd.Timestamp(2025, 1, day, 10, 0, tz="America/New_York")


def _trade(day: int, net_pnl: float, entry: float = 100.0, stop: float = 99.0) -> TradeResult:
    exit_time = _ts(day)
    return TradeResult(
        strategy="unit_position_sizing",
        entry_time=exit_time - pd.Timedelta(minutes=5),
        exit_time=exit_time,
        direction="long",
        entry=entry,
        stop=stop,
        target=entry + 1.0,
        exit=entry + 1.0 if net_pnl >= 0 else stop,
        contracts=1,
        gross_pnl=net_pnl,
        commission=0.0,
        net_pnl=net_pnl,
        r_multiple=net_pnl / 60.0,
        exit_reason="unit",
        bars_held=1,
        regime="test",
        params={"fixture": "position_sizing"},
    )


def _trades_10(net_pnl: float = 60.0) -> list[TradeResult]:
    return [_trade(day, net_pnl) for day in range(1, 11)]


def _speed_rules() -> TopStepRules:
    return TopStepRules(
        profit_target=100.0,
        max_drawdown=1_000.0,
        daily_loss_limit=1_000.0,
        consistency_pct_of_target=100.0,
    )


def _longevity_rules() -> FundedExpressSimRules:
    return FundedExpressSimRules(
        account_size=10_000.0,
        max_drawdown=5_000.0,
        daily_loss_limit=5_000.0,
        lock_trigger_balance=999_999.0,
        locked_floor_balance=0.0,
    )


def test_optimize_for_speed_wf_basic():
    result = optimize_for_speed_wf(
        _trades_10(),
        strategy="unit",
        window="WF1_test",
        risk_levels=[2.0, 4.0],
        pass_floor_pct=1.0,
        rules=_speed_rules(),
    )

    assert isinstance(result, SpeedOptimizationResult)
    assert result.candidates
    assert result.optimal_risk_dollars == result.candidates[0]["risk_dollars"]
    assert result.optimal_risk_dollars == 4.0
    assert result.mean_days_to_pass == 1.0


def test_optimize_for_speed_wf_filters_zero_contract_risks():
    result = optimize_for_speed_wf(
        _trades_10(),
        strategy="unit",
        window="WF1_test",
        risk_levels=[1.0, 2.0],
        pass_floor_pct=1.0,
        rules=_speed_rules(),
    )

    assert result.candidates
    assert {candidate["risk_dollars"] for candidate in result.candidates} == {2.0}


def test_optimize_for_speed_wf_no_trades():
    result = optimize_for_speed_wf([], strategy="unit", window="WF1_test")

    assert result.optimal_risk_dollars == 0.0
    assert result.pass_rate_pct == 0.0
    assert result.candidates == ()


def test_optimize_for_speed_wf_single_trade_pass():
    result = optimize_for_speed_wf(
        [_trade(1, 60.0)],
        strategy="unit",
        window="WF1_test",
        risk_levels=[4.0],
        pass_floor_pct=1.0,
        rules=_speed_rules(),
    )

    assert result.optimal_risk_dollars == 4.0
    assert result.pass_rate_pct == 100.0
    assert result.mean_days_to_pass == 1.0
    assert len(result.candidates) == 1


def test_optimize_for_speed_wf_respects_pass_floor():
    result = optimize_for_speed_wf(
        [_trade(1, 60.0)],
        strategy="unit",
        window="WF1_test",
        risk_levels=[2.0, 4.0],
        pass_floor_pct=100.0,
        rules=_speed_rules(),
    )

    assert result.optimal_risk_dollars == 4.0
    assert {candidate["risk_dollars"] for candidate in result.candidates} == {4.0}
    assert all(candidate["pass_rate_pct"] >= 100.0 for candidate in result.candidates)


def test_optimize_for_speed_wf_returns_top_5():
    result = optimize_for_speed_wf(
        _trades_10(),
        strategy="unit",
        window="WF1_test",
        risk_levels=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
        pass_floor_pct=1.0,
        rules=_speed_rules(),
    )

    assert len(result.candidates) == 5
    assert [candidate["risk_dollars"] for candidate in result.candidates] == [
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
    ]


def test_optimize_for_longevity_holdout_basic():
    result = optimize_for_longevity_holdout(
        _trades_10(),
        strategy="unit",
        window="holdout",
        risk_levels=[2.0, 4.0],
        min_profit_per_trade=1.0,
        rules=_longevity_rules(),
    )

    assert isinstance(result, LongevityOptimizationResult)
    assert result.candidates
    assert result.optimal_risk_dollars == result.candidates[0]["risk_dollars"]
    assert result.optimal_risk_dollars == 4.0
    assert result.avg_pnl_per_trade == 120.0


def test_optimize_for_longevity_holdout_respects_min_profit():
    result = optimize_for_longevity_holdout(
        _trades_10(),
        strategy="unit",
        window="holdout",
        risk_levels=[2.0, 4.0],
        min_profit_per_trade=100.0,
        rules=_longevity_rules(),
    )

    assert result.optimal_risk_dollars == 4.0
    assert {candidate["risk_dollars"] for candidate in result.candidates} == {4.0}
    assert all(candidate["avg_pnl_per_trade"] >= 100.0 for candidate in result.candidates)


def test_optimize_for_longevity_holdout_no_trades():
    result = optimize_for_longevity_holdout([], strategy="unit", window="holdout")

    assert result.optimal_risk_dollars == 0.0
    assert result.avg_pnl_per_trade == 0.0
    assert result.candidates == ()


def test_optimize_for_longevity_holdout_single_trade():
    result = optimize_for_longevity_holdout(
        [_trade(1, 60.0)],
        strategy="unit",
        window="holdout",
        risk_levels=[2.0],
        min_profit_per_trade=1.0,
        rules=_longevity_rules(),
    )

    assert result.optimal_risk_dollars == 2.0
    assert result.avg_pnl_per_trade == 60.0
    assert result.total_trades_executed == 1
    assert len(result.candidates) == 1


def test_optimize_for_longevity_holdout_all_winners():
    result = optimize_for_longevity_holdout(
        _trades_10(75.0),
        strategy="unit",
        window="holdout",
        risk_levels=[2.0, 4.0, 6.0],
        min_profit_per_trade=1.0,
        rules=_longevity_rules(),
    )

    assert result.optimal_risk_dollars == 6.0
    assert result.avg_pnl_per_trade == 225.0
    assert result.accounts_blown == 0
    assert result.total_trades_executed == 10


def test_optimize_for_longevity_holdout_all_losers():
    result = optimize_for_longevity_holdout(
        _trades_10(-60.0),
        strategy="unit",
        window="holdout",
        risk_levels=[2.0, 4.0],
        min_profit_per_trade=0.0,
        rules=_longevity_rules(),
    )

    assert result.optimal_risk_dollars == 0.0
    assert result.avg_pnl_per_trade == 0.0
    assert result.candidates
    assert all(candidate["avg_pnl_per_trade"] < 0.0 for candidate in result.candidates)
