from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from v3.config import FundedExpressSimRules
from v3.funded_express_sim import simulate_express_funded_resets
from v3.trades import TradeResult

TZ = ZoneInfo("America/New_York")


def _ts(day: int, hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(2025, 6, day, hour, minute, tz=TZ)


def _trade(exit_ts: pd.Timestamp, net: float, entry_offset_min: int = 1, r_multiple: float = 0.0) -> TradeResult:
    entry_ts = exit_ts - pd.Timedelta(minutes=entry_offset_min)
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
        r_multiple=r_multiple,
        exit_reason="test",
        bars_held=1,
        regime="unknown",
        params={},
    )


def test_express_floor_lock_freezes_floor_after_trigger():
    """Post-lock dips must survive levels that would trip a continued EOD trailing floor."""
    locked_trades = [
        _trade(_ts(10, 10), 52100 - 50000),
        _trade(_ts(11, 10), 800),
        _trade(_ts(12, 15), -2799),  # 52900 -> 50101; floor locked at 50100
    ]
    locked = simulate_express_funded_resets(locked_trades, rules=FundedExpressSimRules())
    assert locked.funded_accounts_failed == 0
    assert locked.max_nominal_peak_balance >= 52900

    unlocked = simulate_express_funded_resets(
        locked_trades,
        rules=FundedExpressSimRules(lock_trigger_balance=999_999.0),
    )
    assert unlocked.funded_accounts_failed >= 1


def test_two_breach_resets_accrues_bank_disjoint_stints():
    """Sequential convention: breaches drop the whole termination calendar day."""
    trades = [
        _trade(_ts(14, 11), +500),
        _trade(_ts(15, 9), +100),
        _trade(_ts(15, 14), -2200),
        _trade(_ts(16, 11), +100),
        _trade(_ts(17, 11), +100),
        _trade(_ts(17, 13), -2820),
    ]
    rules = FundedExpressSimRules()
    r = simulate_express_funded_resets(trades, rules=rules)
    assert r.funded_accounts_failed == 2
    assert r.stints_opened == 2
    assert r.funded_accounts_used == 3
    assert r.accrued_pnl_bank == sum(s["bank_increment"] for s in r.stints_summary)


def test_worst_daily_drawdown_intraday():
    trades = [_trade(_ts(20, 10), -600), _trade(_ts(20, 14), +200)]
    r = simulate_express_funded_resets(trades, rules=FundedExpressSimRules())
    assert r.worst_daily_drawdown >= 600.0


def test_empty_holdout_trade_list_returns_neutral_aggregate():
    r = simulate_express_funded_resets([], rules=FundedExpressSimRules())
    assert r.stints_opened == 0
    assert r.funded_accounts_used == 0
    assert r.accrued_pnl_bank == 0.0
    assert r.funded_accounts_failed == 0


def test_one_clean_stint_funded_accounts_used():
    trades = [_trade(_ts(30, 9), +100)]
    r = simulate_express_funded_resets(trades, rules=FundedExpressSimRules())
    assert r.funded_accounts_failed == 0
    assert r.stints_opened == 1
    assert r.funded_accounts_used == 1


def test_rich_metrics_span_breached_and_active_stints():
    trades = [
        _trade(_ts(1, 10), +500, r_multiple=1.0),
        _trade(_ts(2, 10), -250, r_multiple=-0.5),
        _trade(_ts(3, 10), -2600, r_multiple=-2.0),
        _trade(_ts(4, 10), +100, r_multiple=0.2),
        _trade(_ts(5, 10), +300, r_multiple=0.6),
    ]
    r = simulate_express_funded_resets(trades, rules=FundedExpressSimRules())

    assert r.funded_accounts_failed == 1
    assert r.current_account_active is True
    assert r.current_account_pnl == 400.0
    assert r.current_max_drawdown == 0.0
    assert r.current_win_rate_pct == 100.0
    assert r.current_avg_r_multiple == 0.4
    assert r.current_profit_factor == math.inf
    assert r.current_sharpe_annualized > 0
    assert r.total_win_rate_pct == 60.0
    assert r.total_avg_r_multiple == pytest.approx(-0.14)
    assert r.best_trade_pnl == 500
    assert r.worst_trade_pnl == -2600

    first, second = r.stints_summary
    assert first["survival_days"] == 2
    assert first["win_rate_pct"] == pytest.approx(100 / 3)
    assert first["avg_r_multiple"] == pytest.approx(-0.5)
    assert first["best_trade_pnl"] == 500
    assert first["worst_trade_pnl"] == -2600
    assert first["profit_factor"] == pytest.approx(500 / 2850)

    assert second["survival_days"] == 1
    assert second["win_rate_pct"] == 100.0
    assert second["avg_r_multiple"] == 0.4
    assert second["best_trade_pnl"] == 300
    assert second["worst_trade_pnl"] == 100
    assert second["profit_factor"] == math.inf

    output = r.to_dict()
    for key in (
        "current_account_active",
        "current_account_pnl",
        "current_max_drawdown",
        "current_win_rate_pct",
        "current_avg_r_multiple",
        "current_profit_factor",
        "current_sharpe_annualized",
        "total_win_rate_pct",
        "total_avg_r_multiple",
        "best_trade_pnl",
        "worst_trade_pnl",
    ):
        assert key in output


def test_rich_metrics_all_losers_and_single_trade_edges():
    losing = simulate_express_funded_resets(
        [
            _trade(_ts(6, 10), -100, r_multiple=-0.2),
            _trade(_ts(7, 10), -200, r_multiple=-0.4),
        ],
        rules=FundedExpressSimRules(),
    )
    assert losing.current_account_active is True
    assert losing.current_profit_factor == 0.0
    assert losing.total_win_rate_pct == 0.0
    assert losing.best_trade_pnl == -100
    assert losing.worst_trade_pnl == -200

    single = simulate_express_funded_resets(
        [_trade(_ts(8, 10), +100, r_multiple=0.5)],
        rules=FundedExpressSimRules(),
    )
    assert single.current_profit_factor == math.inf
    assert single.current_sharpe_annualized == 0.0
    assert single.stints_summary[0]["profit_factor"] == math.inf
