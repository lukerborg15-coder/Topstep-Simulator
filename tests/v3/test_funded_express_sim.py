from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from v3.config import FundedExpressSimRules
from v3.funded_express_sim import simulate_express_funded_resets
from v3.trades import TradeResult

TZ = ZoneInfo("America/New_York")


def _ts(day: int, hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(2025, 6, day, hour, minute, tz=TZ)


def _trade(exit_ts: pd.Timestamp, net: float, entry_offset_min: int = 1) -> TradeResult:
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
        r_multiple=0.0,
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
