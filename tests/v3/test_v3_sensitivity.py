"""Tests for parameter sensitivity / cliff detection."""
from __future__ import annotations

import pandas as pd
import pytest

from v3.combine_simulator import run_combine_simulator
from v3.sensitivity import (
    SensitivityReport,
    apply_sensitivity_to_verdict,
    run_sensitivity,
    sensitivity_summary_dict,
)
from v3.trades import TradeResult
from v3.verdict import VerdictResult, compute_verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade(exit_time: str, pnl: float) -> TradeResult:
    ts = pd.Timestamp(exit_time).tz_localize("America/New_York")
    return TradeResult(
        strategy="test",
        entry_time=ts - pd.Timedelta(minutes=5),
        exit_time=ts,
        direction="long",
        entry=100.0, stop=99.0, target=102.0,
        exit=100.0 + pnl / 2.0,
        contracts=1,
        gross_pnl=float(pnl), commission=0.0, net_pnl=float(pnl),
        r_multiple=pnl / 100.0,
        exit_reason="target" if pnl > 0 else "stop",
        bars_held=5,
    )


def _winning_trades() -> list[TradeResult]:
    """20 days × $200 = $4000 — reliably passes Combine."""
    base = pd.Timestamp("2024-01-02", tz="America/New_York")
    return [_trade(f"{(base + pd.Timedelta(days=i)).date()} 10:00", 200.0) for i in range(20)]


def _losing_trades() -> list[TradeResult]:
    """20 days × -$150 — reliably fails."""
    base = pd.Timestamp("2024-01-02", tz="America/New_York")
    return [_trade(f"{(base + pd.Timedelta(days=i)).date()} 10:00", -150.0) for i in range(20)]


def _make_trades_fn(mapping: dict) -> callable:
    """Return a trades_fn that maps param dict -> trade list via a lookup."""
    def trades_fn(params: dict) -> list[TradeResult]:
        key = params.get("width", 1.0)
        return mapping.get(key, _losing_trades())
    return trades_fn


def _clean_verdict(strategy: str = "test_strategy") -> VerdictResult:
    """A COMBINE-READY verdict with no issues."""
    from v3.combine_simulator import CombineSimResult
    sim = CombineSimResult(
        n_resamples=1000, pass_rate_pct=80.0,
        n_passed=800, n_failed_drawdown=50, n_not_passed=150,
        median_days_to_pass=12.0, mean_days_to_pass=12.0,
        min_days_to_pass=8, max_days_to_pass=18,
        pct_daily_limit_hit=20.0,
        mean_max_drawdown=700.0, worst_max_drawdown=1300.0,
        n_trades=100, n_trading_days=50, n_resamples_requested=1000,
    )
    return compute_verdict(strategy, sim)


def _reject_verdict(strategy: str = "test_strategy") -> VerdictResult:
    """A hard-REJECT verdict."""
    from v3.combine_simulator import CombineSimResult
    sim = CombineSimResult(
        n_resamples=1000, pass_rate_pct=20.0,
        n_passed=200, n_failed_drawdown=700, n_not_passed=100,
        median_days_to_pass=None, mean_days_to_pass=None,
        min_days_to_pass=None, max_days_to_pass=None,
        pct_daily_limit_hit=10.0,
        mean_max_drawdown=500.0, worst_max_drawdown=900.0,
        n_trades=100, n_trading_days=50, n_resamples_requested=1000,
    )
    return compute_verdict(strategy, sim)


# ---------------------------------------------------------------------------
# Flat landscape — no cliff
# ---------------------------------------------------------------------------

def test_flat_landscape_returns_is_cliff_false():
    """All param values produce similar (winning) pass rates — no cliff."""
    # All three values for 'width' produce winning trades
    trades_fn = _make_trades_fn({1.0: _winning_trades(), 2.0: _winning_trades(), 3.0: _winning_trades()})
    report = run_sensitivity(
        strategy_name="flat_test",
        default_params={"width": 1.0},
        param_grid={"width": (1.0, 2.0, 3.0)},
        sim_fn=run_combine_simulator,
        trades_fn=trades_fn,
        n_resamples=100,
        seed=42,
    )
    assert not report.is_cliff
    assert report.cliff_params == ()


def test_flat_landscape_min_neighbor_close_to_default():
    trades_fn = _make_trades_fn({1.0: _winning_trades(), 2.0: _winning_trades(), 3.0: _winning_trades()})
    report = run_sensitivity(
        strategy_name="flat_test",
        default_params={"width": 1.0},
        param_grid={"width": (1.0, 2.0, 3.0)},
        sim_fn=run_combine_simulator,
        trades_fn=trades_fn,
        n_resamples=100,
        seed=42,
    )
    # Min neighbour should be close to default (both near 100%)
    drop_pp = report.default_pass_rate - report.min_neighbor_pass_rate
    assert drop_pp <= 25.0, f"Expected flat landscape, drop={drop_pp:.1f}pp"


# ---------------------------------------------------------------------------
# Cliff detected
# ---------------------------------------------------------------------------

def test_cliff_param_detected_when_neighbor_drops_more_than_threshold():
    """Default=winning, one neighbour=losing → cliff on 'width'."""
    trades_fn = _make_trades_fn({1.0: _winning_trades(), 2.0: _losing_trades(), 3.0: _losing_trades()})
    report = run_sensitivity(
        strategy_name="cliff_test",
        default_params={"width": 1.0},
        param_grid={"width": (1.0, 2.0, 3.0)},
        sim_fn=run_combine_simulator,
        trades_fn=trades_fn,
        n_resamples=100,
        seed=42,
    )
    assert report.is_cliff
    assert "width" in report.cliff_params


def test_cliff_correctly_identifies_only_cliffed_param():
    """Two params: 'width' cliffs, 'lookback' is flat. Only 'width' flagged."""
    def trades_fn(params: dict) -> list[TradeResult]:
        if params["width"] == 1.0:
            return _winning_trades()
        return _losing_trades()

    report = run_sensitivity(
        strategy_name="mixed_test",
        default_params={"width": 1.0, "lookback": 5},
        param_grid={"width": (1.0, 2.0, 3.0), "lookback": (3, 5, 7)},
        sim_fn=run_combine_simulator,
        trades_fn=trades_fn,
        n_resamples=100,
        seed=42,
    )
    assert report.is_cliff
    assert "width" in report.cliff_params
    assert "lookback" not in report.cliff_params


# ---------------------------------------------------------------------------
# Drop threshold boundary
# ---------------------------------------------------------------------------

def test_drop_below_threshold_is_not_a_cliff():
    """A drop of exactly drop_threshold should NOT trigger cliff (strictly greater than)."""
    # We need a neighbour that drops exactly at the boundary
    # Use a custom sim_fn that returns controlled pass rates
    call_count = [0]

    def controlled_sim(trades, rules=None, n_resamples=100, seed=42):
        from v3.combine_simulator import CombineSimResult
        # First call = default params (pass_rate=60), subsequent = neighbour (pass_rate=35)
        # Drop = 25pp = exactly drop_threshold → should NOT be a cliff
        call_count[0] += 1
        rate = 60.0 if call_count[0] == 1 else 35.0
        return CombineSimResult(
            n_resamples=n_resamples, pass_rate_pct=rate,
            n_passed=int(rate), n_failed_drawdown=0, n_not_passed=n_resamples - int(rate),
            median_days_to_pass=10.0 if rate > 0 else None,
            mean_days_to_pass=10.0 if rate > 0 else None,
            min_days_to_pass=8 if rate > 0 else None,
            max_days_to_pass=15 if rate > 0 else None,
            pct_daily_limit_hit=10.0, mean_max_drawdown=500.0, worst_max_drawdown=900.0,
            n_trades=20, n_trading_days=20, n_resamples_requested=n_resamples,
        )

    report = run_sensitivity(
        strategy_name="boundary_test",
        default_params={"width": 1.0},
        param_grid={"width": (1.0, 2.0)},
        sim_fn=controlled_sim,
        trades_fn=lambda p: [],
        n_resamples=100,
        seed=42,
        drop_threshold=0.25,
    )
    # 25pp drop at 0.25 threshold = NOT a cliff (must be strictly greater than)
    assert not report.is_cliff


def test_drop_above_threshold_is_a_cliff():
    """A drop of drop_threshold + epsilon triggers cliff."""
    call_count = [0]

    def controlled_sim(trades, rules=None, n_resamples=100, seed=42):
        from v3.combine_simulator import CombineSimResult
        call_count[0] += 1
        rate = 60.0 if call_count[0] == 1 else 33.9  # drop = 26.1pp > 25pp threshold
        return CombineSimResult(
            n_resamples=n_resamples, pass_rate_pct=rate,
            n_passed=int(rate), n_failed_drawdown=0, n_not_passed=n_resamples - int(rate),
            median_days_to_pass=10.0 if rate > 0 else None,
            mean_days_to_pass=10.0 if rate > 0 else None,
            min_days_to_pass=8 if rate > 0 else None,
            max_days_to_pass=15 if rate > 0 else None,
            pct_daily_limit_hit=10.0, mean_max_drawdown=500.0, worst_max_drawdown=900.0,
            n_trades=20, n_trading_days=20, n_resamples_requested=n_resamples,
        )

    report = run_sensitivity(
        strategy_name="cliff_boundary_test",
        default_params={"width": 1.0},
        param_grid={"width": (1.0, 2.0)},
        sim_fn=controlled_sim,
        trades_fn=lambda p: [],
        n_resamples=100,
        seed=42,
        drop_threshold=0.25,
    )
    assert report.is_cliff


# ---------------------------------------------------------------------------
# apply_sensitivity_to_verdict
# ---------------------------------------------------------------------------

def test_apply_sensitivity_downgrades_combine_ready_when_cliff():
    verdict = _clean_verdict()
    assert verdict.verdict == "COMBINE-READY"

    cliff_report = SensitivityReport(
        strategy="test_strategy",
        is_cliff=True,
        cliff_params=("width",),
        param_results={"width": {"1.0": 80.0, "2.0": 10.0}},
        default_pass_rate=80.0,
        min_neighbor_pass_rate=10.0,
        drop_threshold=0.25,
        default_params={"width": 1.0},
    )
    updated = apply_sensitivity_to_verdict(verdict, cliff_report)
    assert updated.verdict == "PROMISING"
    assert updated.sensitivity_flag is True
    assert any("cliff" in r for r in updated.warn_reasons)


def test_apply_sensitivity_leaves_combine_ready_intact_when_no_cliff():
    verdict = _clean_verdict()
    assert verdict.verdict == "COMBINE-READY"

    flat_report = SensitivityReport(
        strategy="test_strategy",
        is_cliff=False,
        cliff_params=(),
        param_results={"width": {"1.0": 80.0, "2.0": 78.0}},
        default_pass_rate=80.0,
        min_neighbor_pass_rate=78.0,
        drop_threshold=0.25,
        default_params={"width": 1.0},
    )
    updated = apply_sensitivity_to_verdict(verdict, flat_report)
    assert updated.verdict == "COMBINE-READY"
    assert updated.sensitivity_flag is False


def test_apply_sensitivity_leaves_reject_unchanged_regardless():
    verdict = _reject_verdict()
    assert verdict.verdict == "REJECT"

    cliff_report = SensitivityReport(
        strategy="test_strategy",
        is_cliff=True,
        cliff_params=("width",),
        param_results={"width": {"1.0": 20.0, "2.0": 0.0}},
        default_pass_rate=20.0,
        min_neighbor_pass_rate=0.0,
        drop_threshold=0.25,
        default_params={"width": 1.0},
    )
    updated = apply_sensitivity_to_verdict(verdict, cliff_report)
    assert updated.verdict == "REJECT"
    assert updated.sensitivity_flag is True


def test_apply_sensitivity_adds_cliff_warning_to_warn_reasons():
    verdict = _clean_verdict()
    cliff_report = SensitivityReport(
        strategy="test_strategy",
        is_cliff=True,
        cliff_params=("lookback", "width"),
        param_results={},
        default_pass_rate=80.0,
        min_neighbor_pass_rate=10.0,
        drop_threshold=0.25,
        default_params={"lookback": 5, "width": 1.0},
    )
    updated = apply_sensitivity_to_verdict(verdict, cliff_report)
    assert any("lookback" in r and "width" in r for r in updated.warn_reasons)


def test_apply_sensitivity_no_duplicate_warn_on_reapply():
    """Calling apply twice should not double-append the cliff warning."""
    verdict = _clean_verdict()
    cliff_report = SensitivityReport(
        strategy="test_strategy",
        is_cliff=True,
        cliff_params=("width",),
        param_results={},
        default_pass_rate=80.0,
        min_neighbor_pass_rate=10.0,
        drop_threshold=0.25,
        default_params={"width": 1.0},
    )
    once = apply_sensitivity_to_verdict(verdict, cliff_report)
    twice = apply_sensitivity_to_verdict(once, cliff_report)
    cliff_warns = [r for r in twice.warn_reasons if "cliff" in r]
    assert len(cliff_warns) == 1


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------

def test_sensitivity_summary_dict_contains_required_keys():
    report = SensitivityReport(
        strategy="test",
        is_cliff=False,
        cliff_params=(),
        param_results={"width": {"1.0": 80.0}},
        default_pass_rate=80.0,
        min_neighbor_pass_rate=80.0,
        drop_threshold=0.25,
        default_params={"width": 1.0},
    )
    d = sensitivity_summary_dict(report)
    expected = {
        "sensitivity_strategy",
        "sensitivity_is_cliff",
        "sensitivity_cliff_params",
        "sensitivity_default_pass_rate",
        "sensitivity_min_neighbor_pass_rate",
        "sensitivity_drop_threshold",
        "sensitivity_param_results",
    }
    assert expected == set(d.keys())


def test_sensitivity_summary_dict_values_match_report():
    report = SensitivityReport(
        strategy="test",
        is_cliff=True,
        cliff_params=("width",),
        param_results={"width": {"1.0": 80.0, "2.0": 10.0}},
        default_pass_rate=80.0,
        min_neighbor_pass_rate=10.0,
        drop_threshold=0.25,
        default_params={"width": 1.0},
    )
    d = sensitivity_summary_dict(report)
    assert d["sensitivity_is_cliff"] is True
    assert d["sensitivity_cliff_params"] == ["width"]
    assert d["sensitivity_default_pass_rate"] == 80.0
