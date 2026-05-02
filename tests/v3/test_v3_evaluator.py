from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import v3.evaluator as evaluator_module
from v3.config import DEFAULT_RISK_DOLLARS, EASTERN_TZ, MNQ, TOPSTEP_50K, DateWindow, WINDOWS
from v3.evaluator import (
    EvaluationResult,
    aggregate_wf_metrics,
    all_folds_meet_min_seq_pass_rate,
    compute_metrics,
    evaluate_strategy,
    fold_seq_eval_pass_rate,
    run_walk_forward,
    simulate_trades,
    walk_forward_development_window,
)
from v3.strategies import STRATEGIES, StrategySpec, TradeSignal
from v3.trades import TradeResult


REQUIRED_METRIC_KEYS = {
    "total_trades",
    "win_rate",
    "avg_r",
    "total_net_pnl",
    "profit_factor",
    "max_drawdown",
    "sharpe",
    "avg_trade_duration_bars",
}

REQUIRED_WF_KEYS = {
    "wf_folds",
    "wf_passed_folds",
    "wf_avg_score",
    "wf_avg_net_pnl",
    "wf_oos_total_pnl",
    "wf_consistency",
    "wf_seq_eval_passes_by_fold",
    "wf_fold_seq_pass_rates",
}


def _synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range(
        "2022-09-01 10:00",
        "2024-08-31 10:00",
        freq="B",
        tz=EASTERN_TZ,
    )
    base = 12_000.0 + np.linspace(0.0, 75.0, len(index))
    noise = rng.normal(0.0, 0.75, len(index)).cumsum()
    close = base + noise
    open_ = close + rng.normal(0.0, 0.25, len(index))
    high = np.maximum(open_, close) + 1.25
    low = np.minimum(open_, close) - 1.25
    volume = rng.integers(100, 1_000, len(index))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _manual_signals(frame: pd.DataFrame) -> list[TradeSignal]:
    ts = frame.index[3]
    entry = float(frame.loc[ts, "close"])
    return [
        TradeSignal(
            time=ts,
            direction="long",
            entry=entry,
            stop=entry - 10.0,
            target=entry + 0.25,
            strategy="unit_eval",
            params={"target_offset": 0.25},
        )
    ]


def _two_bar_target_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02 10:00", tz=EASTERN_TZ),
            pd.Timestamp("2024-01-02 10:05", tz=EASTERN_TZ),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.5, 102.25],
            "low": [99.5, 99.75],
            "close": [100.0, 102.0],
            "volume": [100, 100],
        },
        index=index,
    )


def _pivot_break_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 10:00", tz=EASTERN_TZ),
            pd.Timestamp("2024-01-01 10:05", tz=EASTERN_TZ),
            pd.Timestamp("2024-01-02 10:00", tz=EASTERN_TZ),
            pd.Timestamp("2024-01-02 10:05", tz=EASTERN_TZ),
            pd.Timestamp("2024-01-02 10:10", tz=EASTERN_TZ),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.9, 101.0, 101.6],
            "high": [101.0, 100.8, 101.05, 101.7, 105.0],
            "low": [99.0, 99.2, 100.7, 100.9, 101.5],
            "close": [100.2, 100.0, 101.0, 101.5, 104.0],
            "volume": [100, 100, 100, 100, 100],
        },
        index=index,
    )


def _trade_result(net_pnl: float, r_multiple: float, bars_held: int) -> TradeResult:
    exit_time = pd.Timestamp("2024-01-02 10:00", tz=EASTERN_TZ) + pd.Timedelta(minutes=bars_held)
    return TradeResult(
        strategy="unit_eval",
        entry_time=pd.Timestamp("2024-01-02 10:00", tz=EASTERN_TZ),
        exit_time=exit_time,
        direction="long",
        entry=100.0,
        stop=99.0,
        target=101.0,
        exit=101.0,
        contracts=1,
        gross_pnl=float(net_pnl),
        commission=0.0,
        net_pnl=float(net_pnl),
        r_multiple=float(r_multiple),
        exit_reason="unit",
        bars_held=bars_held,
    )


def _oos_eval(params: dict[str, Any], seq_passes: int) -> EvaluationResult:
    return EvaluationResult(
        strategy="unit_eval",
        timeframe="1d",
        params=dict(params),
        window="oos",
        metrics={"avg_r": 0.0, "total_net_pnl": 0.0},
        topstep={"topstep_passed": True, "topstep_score": 1.0, "seq_eval_passes": seq_passes},
        trades=[],
    )

def _wf_selection_result(
    params: dict[str, Any],
    *,
    topstep_passed: bool,
    topstep_score: float,
    avg_r: float = 0.0,
) -> EvaluationResult:
    return EvaluationResult(
        strategy="unit_eval",
        timeframe="1d",
        params=dict(params),
        window="WF_train",
        metrics={"avg_r": avg_r, "total_net_pnl": 0.0},
        topstep={"topstep_passed": topstep_passed, "topstep_score": topstep_score},
        trades=[],
    )


def _install_unit_strategy(monkeypatch: Any) -> None:
    def generate(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
        step = int(params["step"])
        target_offset = float(params["target_offset"])
        signals: list[TradeSignal] = []
        for i in range(step, max(len(df) - 1, 0), step):
            ts = df.index[i]
            entry = float(df["close"].iloc[i])
            signals.append(
                TradeSignal(
                    time=ts,
                    direction="long",
                    entry=entry,
                    stop=entry - 10.0,
                    target=entry + target_offset,
                    strategy="unit_eval",
                    params=dict(params),
                )
            )
        return signals

    monkeypatch.setitem(
        STRATEGIES,
        "unit_eval",
        StrategySpec(
            name="unit_eval",
            generate=generate,
            default_params={"step": 20, "target_offset": 0.25},
            param_grid={"step": (15, 20), "target_offset": (0.25, 0.5)},
            max_signals_per_day=1,
        ),
    )


def test_simulate_trades_skips_signal_when_floor_contracts_is_zero() -> None:
    frame = _two_bar_target_frame()
    signal = TradeSignal(
        time=frame.index[0],
        direction="long",
        entry=100.0,
        stop=99.0,
        target=102.0,
        strategy="unit_eval",
        params={},
    )
    trades = simulate_trades(frame, [signal], instrument=MNQ, risk_dollars=1.0)
    assert trades == []


def test_simulate_trades_caps_contracts_at_max_contracts() -> None:
    frame = _two_bar_target_frame()
    signal = TradeSignal(
        time=frame.index[0],
        direction="long",
        entry=100.0,
        stop=99.0,
        target=102.0,
        strategy="unit_eval",
        params={},
    )
    trades = simulate_trades(
        frame, [signal], instrument=MNQ, risk_dollars=1_000_000.0, max_contracts=3
    )
    assert len(trades) == 1
    assert trades[0].contracts == 3


def test_fold_seq_eval_pass_rate_handles_attempts_and_explicit_rate() -> None:
    assert fold_seq_eval_pass_rate({"seq_eval_passes": 2, "seq_eval_attempts": 5}) == pytest.approx(0.4)
    assert fold_seq_eval_pass_rate({"seq_eval_pass_rate": 0.5, "seq_eval_passes": 1, "seq_eval_attempts": 5}) == pytest.approx(0.5)
    assert fold_seq_eval_pass_rate({"seq_eval_passes": 1, "seq_eval_attempts": 0}) == 0.0


def test_all_folds_meet_min_seq_pass_rate_borderline() -> None:
    folds = [
        EvaluationResult(
            "u", "5m", {}, "W1", {}, {"seq_eval_passes": 2, "seq_eval_attempts": 5}, [], None
        ),
        EvaluationResult(
            "u", "5m", {}, "W2", {}, {"seq_eval_passes": 1, "seq_eval_attempts": 5}, [], None
        ),
    ]
    ok, rates = all_folds_meet_min_seq_pass_rate(folds, 40.0)
    assert rates == [0.4, 0.2]
    assert ok is False

    ok2, _ = all_folds_meet_min_seq_pass_rate(folds[:1], 40.0)
    assert ok2 is True


def test_simulate_trades_returns_trade_results_with_expected_fields() -> None:
    frame = _synthetic_ohlcv()
    trades = simulate_trades(frame, _manual_signals(frame))

    assert trades
    assert all(isinstance(trade, TradeResult) for trade in trades)
    trade = trades[0]
    assert trade.strategy == "unit_eval"
    assert trade.direction == "long"
    assert trade.entry_time in frame.index
    assert trade.exit_time in frame.index
    assert trade.contracts > 0
    assert trade.commission > 0.0
    assert trade.exit_reason in {"target", "stop", "session_end", "data_end"}
    assert trade.bars_held >= 1
    assert isinstance(trade.params, dict)


def test_simulate_trades_applies_target_fill_commission_and_slippage() -> None:
    frame = _two_bar_target_frame()
    signal = TradeSignal(
        time=frame.index[0],
        direction="long",
        entry=100.0,
        stop=99.0,
        target=102.0,
        strategy="unit_eval",
        params={"case": "target"},
    )

    trades = simulate_trades(frame, [signal], instrument=MNQ, risk_dollars=DEFAULT_RISK_DOLLARS)

    assert len(trades) == 1
    trade = trades[0]
    expected_contracts = min(
        TOPSTEP_50K.max_micro_contracts,
        int(DEFAULT_RISK_DOLLARS // ((signal.entry - signal.stop) * MNQ.point_value)),
    )
    expected_entry = signal.entry + MNQ.slippage_points_per_side
    expected_exit = signal.target - MNQ.slippage_points_per_side
    expected_gross = (expected_exit - expected_entry) * expected_contracts * MNQ.point_value
    expected_commission = expected_contracts * MNQ.commission_round_turn
    expected_net = expected_gross - expected_commission
    expected_risk = (expected_entry - signal.stop) * expected_contracts * MNQ.point_value

    assert trade.exit_reason == "target"
    assert trade.entry_time == frame.index[0]
    assert trade.exit_time == frame.index[1]
    assert trade.bars_held == 1
    assert trade.contracts == expected_contracts == TOPSTEP_50K.max_micro_contracts
    assert trade.entry == expected_entry
    assert trade.exit == expected_exit
    assert trade.gross_pnl == expected_gross
    assert trade.commission == expected_commission
    assert trade.net_pnl == expected_net
    assert trade.r_multiple == expected_net / expected_risk


def test_compute_metrics_returns_required_keys() -> None:
    frame = _synthetic_ohlcv()
    metrics = compute_metrics(simulate_trades(frame, _manual_signals(frame)))

    assert REQUIRED_METRIC_KEYS <= set(metrics)


def test_compute_metrics_calculates_sharpe_and_average_duration() -> None:
    trades = [
        _trade_result(net_pnl=100.0, r_multiple=1.0, bars_held=3),
        _trade_result(net_pnl=-50.0, r_multiple=-0.5, bars_held=6),
        _trade_result(net_pnl=25.0, r_multiple=0.25, bars_held=9),
    ]

    metrics = compute_metrics(trades)

    r_values = np.array([1.0, -0.5, 0.25])
    expected_sharpe = float(np.mean(r_values) / np.std(r_values, ddof=1) * np.sqrt(252.0))
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)
    assert metrics["avg_trade_duration_bars"] == pytest.approx(6.0)


def test_compute_metrics_includes_starting_equity_in_drawdown() -> None:
    trades = [
        _trade_result(net_pnl=-100.0, r_multiple=-1.0, bars_held=1),
        _trade_result(net_pnl=-50.0, r_multiple=-0.5, bars_held=2),
        _trade_result(net_pnl=25.0, r_multiple=0.25, bars_held=3),
    ]

    metrics = compute_metrics(trades)

    assert metrics["max_drawdown"] == pytest.approx(150.0)


def test_evaluation_result_has_required_fields_and_default_combine_sim() -> None:
    assert [field.name for field in fields(EvaluationResult)] == [
        "strategy",
        "timeframe",
        "params",
        "window",
        "metrics",
        "topstep",
        "trades",
        "combine_sim",
    ]

    result = EvaluationResult(
        strategy="unit_eval",
        timeframe="1d",
        params={},
        window="unit",
        metrics={},
        topstep={},
        trades=[],
    )

    assert result.combine_sim is None


def test_evaluate_strategy_returns_result_for_requested_window(monkeypatch: Any) -> None:
    _install_unit_strategy(monkeypatch)
    frame = _synthetic_ohlcv()

    result = evaluate_strategy(
        frame,
        "unit_eval",
        "1d",
        {"step": 20, "target_offset": 0.25},
        WINDOWS.walk_forward[0].test,
    )

    assert isinstance(result, EvaluationResult)
    assert result.strategy == "unit_eval"
    assert result.timeframe == "1d"
    assert result.window == WINDOWS.walk_forward[0].test.name
    assert result.combine_sim is None
    assert REQUIRED_METRIC_KEYS <= set(result.metrics)
    assert "topstep_score" in result.topstep
    assert all(WINDOWS.walk_forward[0].test.start <= str(trade.entry_time.date()) <= WINDOWS.walk_forward[0].test.end for trade in result.trades)


def test_evaluate_strategy_attaches_pivots_for_builtin_pivot_strategy() -> None:
    frame = _pivot_break_frame()
    window = DateWindow("pivot_unit", "2024-01-02", "2024-01-02")

    assert "camarilla_h4" not in frame.columns
    result = evaluate_strategy(
        frame,
        "session_pivot_break",
        "5min",
        {"atr_period": 1, "stop_atr_mult": 1.0, "target_atr_mult": 1.0},
        window,
    )

    assert result.trades
    assert result.trades[0].strategy == "session_pivot_break"
    assert result.trades[0].entry_time == pd.Timestamp("2024-01-02 10:05", tz=EASTERN_TZ)


def test_walk_forward_development_window_returns_correct_bounds() -> None:
    dev = walk_forward_development_window(WINDOWS)
    # Start = min of all train.start; end = max of all test.end
    expected_start = min(wf.train.start for wf in WINDOWS.walk_forward)
    expected_end = max(wf.test.end for wf in WINDOWS.walk_forward)
    assert dev.start == expected_start
    assert dev.end == expected_end
    assert dev.name == "wf_development"


def test_run_walk_forward_returns_one_oos_result_per_fold(monkeypatch: Any) -> None:
    _install_unit_strategy(monkeypatch)
    frame = _synthetic_ohlcv()

    best_params, folds, wf_ok = run_walk_forward(frame, "unit_eval", "1d", max_grid=3)

    assert best_params
    assert wf_ok in (True, False)
    assert len(folds) == len(WINDOWS.walk_forward) == 2
    assert [fold.window for fold in folds] == [wf.test.name for wf in WINDOWS.walk_forward]
    assert all(isinstance(fold, EvaluationResult) for fold in folds)


def test_run_walk_forward_scores_with_avg_r_bonus_and_returns_param_mode(monkeypatch: Any) -> None:
    def unused_generate(df: pd.DataFrame, params: dict[str, Any]) -> list[TradeSignal]:
        raise AssertionError("run_walk_forward should use monkeypatched evaluate_strategy")

    monkeypatch.setitem(
        STRATEGIES,
        "scoring_probe",
        StrategySpec(
            name="scoring_probe",
            generate=unused_generate,
            default_params={},
            param_grid={"label": ("topstep_only", "mode_candidate", "rare_candidate")},
            max_signals_per_day=None,
        ),
    )

    def fake_evaluate_strategy(
        frame: pd.DataFrame,
        strategy_name: str,
        timeframe: str,
        params: dict[str, Any],
        window: Any,
        **_: Any,
    ) -> EvaluationResult:
        topstep_score = 0.0
        avg_r = 0.0
        if window.name.endswith("_train"):
            if params["label"] == "topstep_only":
                topstep_score = 100.0
            elif params["label"] == "mode_candidate":
                # score = 80*1 + 1*25 = 105 > 100 -> wins both folds
                topstep_score = 80.0
                avg_r = 1.0

        return EvaluationResult(
            strategy=strategy_name,
            timeframe=timeframe,
            params=dict(params),
            window=window.name,
            metrics={"avg_r": avg_r, "total_net_pnl": 0.0},
            topstep={"topstep_passed": False, "topstep_score": topstep_score},
            trades=[],
        )

    monkeypatch.setattr(evaluator_module, "evaluate_strategy", fake_evaluate_strategy)

    best_params, folds, wf_ok = run_walk_forward(pd.DataFrame(), "scoring_probe", "1d")

    assert [fold.params["label"] for fold in folds] == [
        "mode_candidate",
        "mode_candidate",
    ]
    assert wf_ok is False  # empty OOS trades -> seq passes 0 -> fallback


def test_robust_params_marks_met_when_enough_folds_hit_seq_threshold() -> None:
    p = {"atr_period": 14}
    selected_params = [p, p, p, p]
    train_results = [_wf_selection_result(p, topstep_passed=True, topstep_score=5.0) for _ in range(4)]
    oos_results = [_oos_eval(p, 3) for _ in range(4)]
    picked, ok = evaluator_module._robust_params(
        selected_params,
        oos_results,
        train_results,
        min_eval_passes_per_fold=2,
        min_folds_meeting_passes=2,
    )
    assert ok is True
    assert picked == p


def test_robust_params_falls_back_when_not_enough_folds_meet_seq_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    p = {"atr_period": 14}
    selected_params = [p, p, p, p]
    train_results = [_wf_selection_result(p, topstep_passed=True, topstep_score=5.0) for _ in range(4)]
    oos_results = [_oos_eval(p, s) for s in (2, 2, 1, 1)]
    picked, ok = evaluator_module._robust_params(
        selected_params,
        oos_results,
        train_results,
        min_eval_passes_per_fold=2,
        min_folds_meeting_passes=3,
    )
    assert ok is False
    assert picked == p
    assert "sequential eval passes" in capsys.readouterr().out


def test_robust_params_prefers_higher_mean_train_score_among_tied_keys() -> None:
    a = {"k": "a"}
    b = {"k": "b"}
    selected_params = [a, a, b, b]
    train_results = [
        _wf_selection_result(a, topstep_passed=True, topstep_score=10.0),
        _wf_selection_result(a, topstep_passed=True, topstep_score=10.0),
        _wf_selection_result(b, topstep_passed=True, topstep_score=50.0),
        _wf_selection_result(b, topstep_passed=True, topstep_score=50.0),
    ]
    oos_results = [_oos_eval(selected_params[i], 3) for i in range(4)]
    picked, ok = evaluator_module._robust_params(
        selected_params,
        oos_results,
        train_results,
        min_eval_passes_per_fold=2,
        min_folds_meeting_passes=2,
    )
    assert ok is True
    assert picked == b


def test_run_walk_forward_rejects_empty_candidate_grid(monkeypatch: Any) -> None:
    _install_unit_strategy(monkeypatch)

    with pytest.raises(ValueError, match="max_grid"):
        run_walk_forward(_synthetic_ohlcv(), "unit_eval", "1d", max_grid=0)


def test_aggregate_wf_metrics_returns_required_keys(monkeypatch: Any) -> None:
    _install_unit_strategy(monkeypatch)
    frame = _synthetic_ohlcv()
    _, folds, _wf_ok = run_walk_forward(frame, "unit_eval", "1d", max_grid=2)

    metrics = aggregate_wf_metrics(folds)

    assert REQUIRED_WF_KEYS == set(metrics)
    assert metrics["wf_folds"] == 2


def test_aggregate_wf_metrics_calculates_summary_values() -> None:
    folds = [
        EvaluationResult(
            strategy="unit_eval",
            timeframe="1d",
            params={"fold": 1},
            window="WF1_test",
            metrics={"avg_r": 0.4, "total_net_pnl": 100.0},
            topstep={"topstep_passed": True, "topstep_score": 10.0},
            trades=[],
        ),
        EvaluationResult(
            strategy="unit_eval",
            timeframe="1d",
            params={"fold": 2},
            window="WF2_test",
            metrics={"avg_r": -0.2, "total_net_pnl": -20.0},
            topstep={"topstep_passed": False, "topstep_score": -5.0},
            trades=[],
        ),
        EvaluationResult(
            strategy="unit_eval",
            timeframe="1d",
            params={"fold": 3},
            window="WF3_test",
            metrics={"avg_r": 0.0, "total_net_pnl": 50.0},
            topstep={"topstep_passed": True, "topstep_score": 40.0},
            trades=[],
        ),
    ]

    metrics = aggregate_wf_metrics(folds)

    net_pnls = np.array([100.0, -20.0, 50.0])
    expected_scores = np.array([20.0, -10.0, 40.0])
    assert metrics["wf_passed_folds"] == 2
    assert metrics["wf_seq_eval_passes_by_fold"] == [0, 0, 0]
    assert metrics["wf_fold_seq_pass_rates"] == [0.0, 0.0, 0.0]
    assert metrics["wf_avg_score"] == pytest.approx(float(np.mean(expected_scores)))
    assert metrics["wf_avg_net_pnl"] == pytest.approx(float(np.mean(net_pnls)))
    assert metrics["wf_oos_total_pnl"] == pytest.approx(130.0)
    assert metrics["wf_consistency"] == pytest.approx(float(np.std(net_pnls, ddof=1)))


def test_new_evaluator_and_tests_do_not_import_prior_optimizer() -> None:
    forbidden = "v2" + "_optimizer"
    repo = Path(__file__).resolve().parents[2]

    assert forbidden not in (repo / "src" / "v3" / "evaluator.py").read_text(encoding="utf-8")
    assert forbidden not in Path(__file__).read_text(encoding="utf-8")
