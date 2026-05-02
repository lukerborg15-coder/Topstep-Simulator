from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from v3.config import FundedExpressSimRules
from v3.funded_express_sim import simulate_express_funded_resets
from v3.trades import TradeResult

TZ = ZoneInfo("America/New_York")


def _ts(day: int, hour: int = 10, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(2025, 7, day, hour, minute, tz=TZ)


def _trade(day: int, net: float, r_multiple: float = 0.0) -> TradeResult:
    exit_ts = _ts(day)
    return TradeResult(
        strategy="test",
        entry_time=exit_ts - pd.Timedelta(minutes=5),
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
        r_multiple=r_multiple,
        exit_reason="test",
        bars_held=1,
        regime="unknown",
        params={},
    )


def test_express_funded_sim_computes_survival_days():
    result = simulate_express_funded_resets(
        [_trade(1, 100.0), _trade(2, 100.0), _trade(4, 100.0)],
        rules=FundedExpressSimRules(),
    )

    assert result.stints_summary[0]["survival_days"] == 3


def test_express_funded_sim_computes_win_rate():
    result = simulate_express_funded_resets(
        [_trade(1, 100.0), _trade(2, -50.0), _trade(3, 25.0)],
        rules=FundedExpressSimRules(),
    )

    assert result.total_win_rate_pct == pytest.approx(100 * 2 / 3)
    assert result.stints_summary[0]["win_rate_pct"] == pytest.approx(100 * 2 / 3)


def test_express_funded_sim_computes_avg_r():
    result = simulate_express_funded_resets(
        [
            _trade(1, 100.0, r_multiple=1.0),
            _trade(2, -50.0, r_multiple=-0.5),
            _trade(3, 25.0, r_multiple=0.25),
        ],
        rules=FundedExpressSimRules(),
    )

    assert result.total_avg_r_multiple == pytest.approx(0.25)
    assert result.stints_summary[0]["avg_r_multiple"] == pytest.approx(0.25)


def test_express_funded_sim_handles_active_account():
    result = simulate_express_funded_resets(
        [_trade(1, 100.0, 1.0), _trade(2, -25.0, -0.25)],
        rules=FundedExpressSimRules(),
    )

    assert result.current_account_active is True
    assert result.current_account_pnl == 75.0
    assert result.current_max_drawdown is not None
    assert result.current_win_rate_pct == 50.0
    assert result.current_avg_r_multiple == pytest.approx(0.375)
    assert result.current_profit_factor == pytest.approx(4.0)
    assert result.current_sharpe_annualized is not None


def test_express_funded_sim_handles_blown_final_stint():
    result = simulate_express_funded_resets(
        [_trade(1, -2500.0, -2.5)],
        rules=FundedExpressSimRules(),
    )

    assert result.current_account_active is False
    assert result.current_account_pnl is None
    assert result.current_max_drawdown is None
    assert result.current_win_rate_pct is None
    assert result.current_avg_r_multiple is None
    assert result.current_profit_factor is None
    assert result.current_sharpe_annualized is None


def test_express_funded_sim_computes_profit_factor():
    mixed = simulate_express_funded_resets(
        [_trade(1, 100.0), _trade(2, 50.0), _trade(3, -60.0)],
        rules=FundedExpressSimRules(),
    )
    winners = simulate_express_funded_resets(
        [_trade(1, 100.0), _trade(2, 50.0)],
        rules=FundedExpressSimRules(),
    )

    assert mixed.current_profit_factor == pytest.approx(2.5)
    assert mixed.stints_summary[0]["profit_factor"] == pytest.approx(2.5)
    assert winners.current_profit_factor == math.inf
    assert winners.stints_summary[0]["profit_factor"] == math.inf


def test_express_funded_sim_single_trade():
    result = simulate_express_funded_resets(
        [_trade(1, 150.0, 1.5)],
        rules=FundedExpressSimRules(),
    )

    assert result.stints_opened == 1
    assert result.funded_accounts_used == 1
    assert result.accrued_pnl_bank == 150.0
    assert result.current_account_pnl == 150.0
    assert result.total_win_rate_pct == 100.0
    assert result.total_avg_r_multiple == 1.5
    assert result.stints_summary[0]["trades_applied_count"] == 1


def test_express_funded_sim_all_winners():
    result = simulate_express_funded_resets(
        [_trade(1, 100.0, 1.0), _trade(2, 200.0, 2.0), _trade(3, 300.0, 3.0)],
        rules=FundedExpressSimRules(),
    )

    assert result.funded_accounts_failed == 0
    assert result.current_account_active is True
    assert result.accrued_pnl_bank == 600.0
    assert result.current_profit_factor == math.inf
    assert result.total_win_rate_pct == 100.0
    assert result.best_trade_pnl == 300.0
    assert result.worst_trade_pnl == 100.0
