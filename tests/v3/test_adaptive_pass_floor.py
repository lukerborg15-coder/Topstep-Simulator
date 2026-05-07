"""Tests for adaptive pass-rate floor in the speed optimizer."""

from __future__ import annotations

import pandas as pd
import pytest

import v3.position_sizing as position_sizing
from v3.position_sizing import (
    ADAPTIVE_FLOOR_RATIO,
    MIN_FLOOR_FLOOR,
    _compute_adaptive_floor,
    _speed_utility,
    optimize_speed_wf_aggregate,
)
from v3.trades import TradeResult


def _make_trade(day: int, net_pnl: float = 200.0) -> TradeResult:
    entry_time = pd.Timestamp(f"2024-01-{day:02d} 10:00:00")
    exit_time = pd.Timestamp(f"2024-01-{day:02d} 11:00:00")
    return TradeResult(
        strategy="adaptive_test",
        entry_time=entry_time,
        exit_time=exit_time,
        direction="long",
        entry=100.0,
        stop=99.0,
        target=101.0,
        exit=100.5,
        contracts=1,
        gross_pnl=net_pnl + 1.40,
        commission=1.40,
        net_pnl=net_pnl,
        r_multiple=net_pnl / 100.0,
        exit_reason="target",
        bars_held=1,
    )


def _stub_metrics(risk: float, pass_rate: float, median_days: float = 5.0) -> dict:
    return {
        "risk_dollars": risk,
        "pass_rate_pct": pass_rate,
        "median_days_to_pass": median_days,
        "mean_days_to_pass": median_days,
        "iqr_days_to_pass": (median_days, median_days),
        "p90_days_to_pass": median_days,
        "std_days_to_pass": 0.0,
        "passes": 1,
        "attempts": 1,
        "min_contracts": 1,
        "max_contracts": 1,
        "utility": _speed_utility(pass_rate / 100.0, median_days, 10.0),
    }


def _patch_eval(monkeypatch, pass_rate_by_risk):
    monkeypatch.setattr(
        position_sizing,
        "_get_valid_risk_levels",
        lambda trades, risk_levels, *a, **k: list(risk_levels),
    )

    def fake_evaluate(trades, risk, *args, **kwargs):
        pr = pass_rate_by_risk.get(risk, 0.0)
        return _stub_metrics(risk, pr)

    monkeypatch.setattr(position_sizing, "_evaluate_risk_on_trades", fake_evaluate)


# ---------------------------------------------------------------------------
# _compute_adaptive_floor unit tests
# ---------------------------------------------------------------------------


def test_compute_adaptive_floor_user_lower_wins():
    # best=80% → adaptive = 56%, user 40% → user wins (lower), no adaptive applied
    eff, adaptive, applied = _compute_adaptive_floor(80.0, 40.0)
    assert adaptive == pytest.approx(56.0)
    assert eff == pytest.approx(40.0)
    assert applied is False


def test_compute_adaptive_floor_adaptive_lower_wins():
    # best=35% → adaptive = 24.5%, user 40% → adaptive wins (lower), applied
    eff, adaptive, applied = _compute_adaptive_floor(35.0, 40.0)
    assert adaptive == pytest.approx(24.5)
    assert eff == pytest.approx(24.5)
    assert applied is True


def test_compute_adaptive_floor_min_floor_floor_clamps():
    # best=15% → adaptive raw = 10.5% → clamped to MIN_FLOOR_FLOOR=20.0
    eff, adaptive, applied = _compute_adaptive_floor(15.0, 40.0)
    assert adaptive == pytest.approx(MIN_FLOOR_FLOOR)
    assert eff == pytest.approx(MIN_FLOOR_FLOOR)
    assert applied is True


def test_compute_adaptive_floor_constants_match_spec():
    assert ADAPTIVE_FLOOR_RATIO == 0.7
    assert MIN_FLOOR_FLOOR == 20.0


# ---------------------------------------------------------------------------
# Aggregate optimizer behavior
# ---------------------------------------------------------------------------


def test_aggregate_high_pass_rates_use_user_floor(monkeypatch, capsys):
    """best=80% → adaptive=56% → user floor 40% lower → user wins."""
    fold = ([_make_trade(1)], [_make_trade(2)])
    _patch_eval(monkeypatch, {50.0: 80.0, 100.0: 60.0, 150.0: 50.0})

    result = optimize_speed_wf_aggregate(
        [fold],
        "test",
        risk_levels=[50.0, 100.0, 150.0],
        pass_floor_pct=40.0,
    )

    assert result.effective_pass_floor_pct == pytest.approx(40.0)
    assert result.adaptive_floor_applied is False
    captured = capsys.readouterr().out
    assert "Adaptive pass floor" not in captured


def test_aggregate_low_pass_rates_trigger_adaptive(monkeypatch, capsys):
    """best=35% → adaptive=24.5% < user 40% → adaptive wins."""
    fold = ([_make_trade(1)], [_make_trade(2)])
    _patch_eval(monkeypatch, {50.0: 35.0, 100.0: 30.0, 150.0: 25.0})

    result = optimize_speed_wf_aggregate(
        [fold],
        "test",
        risk_levels=[50.0, 100.0, 150.0],
        pass_floor_pct=40.0,
        min_viable_folds=1,
    )

    assert result.adaptive_floor_applied is True
    assert result.effective_pass_floor_pct == pytest.approx(24.5)
    # All three risks should pass the relaxed floor; winner exists
    assert result.optimal_risk_dollars > 0.0
    captured = capsys.readouterr().out
    assert "Adaptive pass floor: 24.5%" in captured
    assert "best train pass rate: 35.0%" in captured


def test_aggregate_catastrophic_clamps_to_min_floor(monkeypatch, capsys):
    """best=15% → adaptive raw=10.5% clamped to 20.0%."""
    fold = ([_make_trade(1)], [_make_trade(2)])
    _patch_eval(monkeypatch, {50.0: 15.0, 100.0: 12.0})

    result = optimize_speed_wf_aggregate(
        [fold],
        "test",
        risk_levels=[50.0, 100.0],
        pass_floor_pct=40.0,
        min_viable_folds=1,
    )

    assert result.adaptive_floor_applied is True
    assert result.effective_pass_floor_pct == pytest.approx(MIN_FLOOR_FLOOR)
    # Both risks below 20% → none viable
    assert result.optimal_risk_dollars == 0.0
    captured = capsys.readouterr().out
    assert "Adaptive pass floor: 20.0%" in captured


def test_aggregate_empty_folds_no_crash():
    result = optimize_speed_wf_aggregate([], "test", pass_floor_pct=40.0)
    assert result.optimal_risk_dollars == 0.0
    assert result.effective_pass_floor_pct == pytest.approx(40.0)
    assert result.adaptive_floor_applied is False
    assert result.per_fold_floors == ()


def test_aggregate_per_fold_floors_recorded(monkeypatch):
    """Each fold's floor data is captured in per_fold_floors."""
    fold1 = ([_make_trade(1)], [_make_trade(2)])
    fold2 = ([_make_trade(3)], [_make_trade(4)])
    _patch_eval(monkeypatch, {50.0: 80.0, 100.0: 70.0})

    result = optimize_speed_wf_aggregate(
        [fold1, fold2],
        "test",
        risk_levels=[50.0, 100.0],
        pass_floor_pct=40.0,
    )

    assert len(result.per_fold_floors) == 2
    for record in result.per_fold_floors:
        assert record["best_train_pass_rate_pct"] == pytest.approx(80.0)
        assert record["adaptive_floor_pct"] == pytest.approx(56.0)
        assert record["effective_floor_pct"] == pytest.approx(40.0)
        assert record["adaptive_applied"] is False


def test_aggregate_mixed_folds_one_triggers_adaptive(monkeypatch):
    """Fold 0: best 80% (no adaptive). Fold 1: best 35% (adaptive). Aggregate flag True."""
    fold0_train = [_make_trade(1)]
    fold0_test = [_make_trade(2)]
    fold1_train = [_make_trade(3)]
    fold1_test = [_make_trade(4)]

    monkeypatch.setattr(
        position_sizing,
        "_get_valid_risk_levels",
        lambda trades, risk_levels, *a, **k: list(risk_levels),
    )

    fold0_rates = {50.0: 80.0, 100.0: 70.0}
    fold1_rates = {50.0: 35.0, 100.0: 30.0}

    def fake_evaluate(trades, risk, *args, **kwargs):
        # Identify fold by trade reference
        if trades is fold0_train or trades is fold0_test:
            pr = fold0_rates.get(risk, 0.0)
        elif trades is fold1_train or trades is fold1_test:
            pr = fold1_rates.get(risk, 0.0)
        else:
            pr = 0.0
        return _stub_metrics(risk, pr)

    monkeypatch.setattr(position_sizing, "_evaluate_risk_on_trades", fake_evaluate)

    result = optimize_speed_wf_aggregate(
        [(fold0_train, fold0_test), (fold1_train, fold1_test)],
        "test",
        risk_levels=[50.0, 100.0],
        pass_floor_pct=40.0,
        min_viable_folds=1,
    )

    assert result.adaptive_floor_applied is True
    assert len(result.per_fold_floors) == 2
    assert result.per_fold_floors[0]["adaptive_applied"] is False
    assert result.per_fold_floors[0]["best_train_pass_rate_pct"] == pytest.approx(80.0)
    assert result.per_fold_floors[0]["effective_floor_pct"] == pytest.approx(40.0)
    assert result.per_fold_floors[1]["adaptive_applied"] is True
    assert result.per_fold_floors[1]["best_train_pass_rate_pct"] == pytest.approx(35.0)
    assert result.per_fold_floors[1]["effective_floor_pct"] == pytest.approx(24.5)
