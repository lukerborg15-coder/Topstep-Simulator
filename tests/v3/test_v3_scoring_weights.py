from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from v3 import cli
import v3.evaluator as evaluator
from v3.evaluator import EvaluationResult
from v3.strategies import STRATEGIES, StrategySpec


def _result(*, topstep_score: float, avg_r: float) -> EvaluationResult:
    return EvaluationResult(
        strategy="unit_scoring",
        timeframe="5min",
        params={},
        window="unit",
        metrics={
            "total_trades": 24,
            "win_rate": 0.50,
            "avg_r": avg_r,
            "total_net_pnl": 250.0,
            "profit_factor": 1.50,
            "max_drawdown": 100.0,
            "sharpe": 1.0,
            "avg_trade_duration_bars": 3.0,
        },
        topstep={"topstep_passed": True, "topstep_score": topstep_score},
        trades=[],
    )


def test_default_score_uses_topstep_plus_25_times_avg_r() -> None:
    assert evaluator._score_result(_result(topstep_score=100.0, avg_r=0.8)) == pytest.approx(120.0)


def test_custom_scoring_weights_change_score(monkeypatch: pytest.MonkeyPatch) -> None:
    from v3.config import ScoringWeights

    monkeypatch.setattr(evaluator, "SCORING_WEIGHTS", ScoringWeights(topstep_weight=2.0, avg_r_weight=10.0))

    assert evaluator._score_result(_result(topstep_score=100.0, avg_r=0.8)) == pytest.approx(208.0)


def test_parser_exposes_scoring_weight_defaults_and_custom_values() -> None:
    defaults = cli.build_parser().parse_args(["--strategy", "unit_scoring"])
    custom = cli.build_parser().parse_args(
        [
            "--strategy",
            "unit_scoring",
            "--topstep-weight",
            "2.0",
            "--avg-r-weight",
            "10.0",
        ]
    )

    assert defaults.topstep_weight == pytest.approx(1.0)
    assert defaults.avg_r_weight == pytest.approx(25.0)
    assert custom.topstep_weight == pytest.approx(2.0)
    assert custom.avg_r_weight == pytest.approx(10.0)


def test_main_applies_custom_scoring_weights_before_walk_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v3.config import ScoringWeights

    class StopPipeline(Exception):
        pass

    captured: dict[str, ScoringWeights] = {}
    spec = StrategySpec(
        name="unit_scoring",
        generate=lambda df, params: [],
        default_params={"width": 1.0},
        param_grid={"width": (1.0, 2.0, 3.0)},
        max_signals_per_day=None,
    )
    monkeypatch.setitem(STRATEGIES, "unit_scoring", spec)
    monkeypatch.setattr(cli, "load_user_strategies", lambda: None)
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())

    def fake_run_walk_forward(*args: Any, **kwargs: Any) -> Any:
        captured["weights"] = evaluator.SCORING_WEIGHTS
        raise StopPipeline

    monkeypatch.setattr(cli, "run_walk_forward", fake_run_walk_forward)

    with pytest.raises(StopPipeline):
        cli.main(
            [
                "--strategy",
                "unit_scoring",
                "--data-dir",
                ".",
                "--output-dir",
                ".",
                "--topstep-weight",
                "2.0",
                "--avg-r-weight",
                "10.0",
                "--skip-sensitivity",
            ]
        )

    assert captured["weights"] == ScoringWeights(topstep_weight=2.0, avg_r_weight=10.0)


def test_scoring_weights_change_walk_forward_param_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from v3.config import SCORING_WEIGHTS, ScoringWeights

    spec = StrategySpec(
        name="unit_scoring_selection",
        generate=lambda df, params: [],
        default_params={"label": "topstep"},
        param_grid={"label": ("topstep", "avg_r")},
        max_signals_per_day=None,
    )
    monkeypatch.setitem(STRATEGIES, "unit_scoring_selection", spec)

    def fake_evaluate_strategy(
        frame: pd.DataFrame,
        strategy_name: str,
        timeframe: str,
        params: dict[str, Any],
        window: Any,
        **_: Any,
    ) -> EvaluationResult:
        if params["label"] == "topstep":
            result = _result(topstep_score=100.0, avg_r=0.0)
        else:
            result = _result(topstep_score=80.0, avg_r=1.0)
        result.params = dict(params)
        return result

    monkeypatch.setattr(evaluator, "evaluate_strategy", fake_evaluate_strategy)

    def selected_label(weights: ScoringWeights) -> str:
        monkeypatch.setattr(evaluator, "SCORING_WEIGHTS", weights)
        params, _, _ = evaluator.run_walk_forward(
            pd.DataFrame(),
            "unit_scoring_selection",
            "5min",
            min_folds_meeting_passes=1,
            min_eval_passes_per_fold=1,
        )
        return str(params["label"])

    assert selected_label(SCORING_WEIGHTS) == "avg_r"
    assert selected_label(ScoringWeights(topstep_weight=2.0, avg_r_weight=25.0)) == "topstep"
    assert selected_label(ScoringWeights(topstep_weight=1.0, avg_r_weight=10.0)) == "topstep"
    assert selected_label(ScoringWeights(topstep_weight=1.0, avg_r_weight=50.0)) == "avg_r"
