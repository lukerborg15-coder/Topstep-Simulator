from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd
import pytest

from v3 import cli, evaluator, pipeline_config
from v3.combine_simulator import CombineSimResult, run_combine_simulator as run_combine_original
from v3.config import (
    EASTERN_TZ,
    SESSION_END,
    SESSION_START,
    STRATEGY_NAMES,
    DateWindow,
    PipelineWindows,
    WalkForwardWindow,
)
from v3.position_sizing import LongevityOptimizationResult, SpeedOptimizationResult
from v3.strategies import STRATEGIES, StrategySpec, TradeSignal, load_user_strategies, register_strategy
from v3.trades import TradeResult


def _cli_e2e_result_json(output_dir: Path) -> Path:
    return output_dir / "json" / "cli_e2e_mock_5min_result.json"


# Two-fold layout matching new WINDOWS default.
NARROW_PIPELINE_WINDOWS = PipelineWindows(
    walk_forward=(
        WalkForwardWindow(
            "WF1",
            DateWindow("WF1_train", "2024-06-03", "2024-06-07"),
            DateWindow("WF1_test", "2024-06-10", "2024-06-13"),
        ),
        WalkForwardWindow(
            "WF2",
            DateWindow("WF2_train", "2024-06-03", "2024-06-13"),
            DateWindow("WF2_test", "2024-06-14", "2024-06-21"),
        ),
    ),
    holdout=DateWindow("holdout", "2024-06-24", "2024-06-28"),
)


def _session_bar_index(start: str, end: str) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end, tz=EASTERN_TZ)
    intra_parts: list[pd.DatetimeIndex] = []
    for day in days:
        day_start = pd.Timestamp(f"{day.date()} {SESSION_START}", tz=EASTERN_TZ)
        day_end = pd.Timestamp(f"{day.date()} {SESSION_END}", tz=EASTERN_TZ)
        intra_parts.append(pd.date_range(day_start, day_end, freq="5min", inclusive="both"))
    concatenated = np.concatenate([ix.to_numpy(dtype="datetime64[ns]") for ix in intra_parts])
    return pd.DatetimeIndex(concatenated)


def synthetic_narrow_frame() -> pd.DataFrame:
    idx = _session_bar_index("2024-06-03", "2024-06-29")
    rng = np.random.default_rng(123)
    n = len(idx)
    walk = 18_050.0 + np.cumsum(rng.normal(0.0, 2.5, size=n))
    close = walk
    open_ = np.r_[walk[0], walk[:-1]] + rng.normal(0.0, 1.5, size=n)
    high = np.maximum(open_, close) + rng.uniform(2.0, 6.0, size=n)
    low = np.minimum(open_, close) - rng.uniform(2.0, 6.0, size=n)
    volume = rng.integers(800, 3_800, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def cli_e2e_generate(df: pd.DataFrame, params: dict) -> list[TradeSignal]:
    signals: list[TradeSignal] = []
    step = max(20, len(df) // 12)
    start = min(160, len(df) // 4)
    for i in range(start, len(df) - 4, step):
        ts = df.index[i]
        px = float(df["close"].iloc[i])
        signals.append(
            TradeSignal(
                time=ts,
                direction="long",
                entry=px,
                stop=px - 4.0,
                target=px + 10.0,
                strategy="cli_e2e_mock",
                params=dict(params),
            )
        )
    return signals


MOCK_CLI_SPEC = StrategySpec(
    name="cli_e2e_mock",
    generate=cli_e2e_generate,
    default_params={"width": 1.0},
    param_grid={"width": (1.0, 2.0, 3.0)},
    max_signals_per_day=None,
)


def _fake_eval(
    *,
    trades: int = 24,
    trade_results: list[TradeResult] | None = None,
    win_rate: float = 0.50,
    profit_factor: float = 1.50,
    topstep_passed: bool = True,
    window: str = "fake",
    total_net_pnl: float = 250.0,
    seq_eval_passes: int = 10,
    seq_eval_attempts: int = 10,
) -> evaluator.EvaluationResult:
    return evaluator.EvaluationResult(
        strategy="cli_e2e_mock",
        timeframe="5min",
        params=dict(MOCK_CLI_SPEC.default_params),
        window=window,
        metrics={
            "total_trades": trades,
            "win_rate": win_rate,
            "avg_r": 0.10,
            "total_net_pnl": total_net_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": 100.0,
            "sharpe": 1.0,
            "avg_trade_duration_bars": 3.0,
        },
        topstep={
            "topstep_passed": topstep_passed,
            "topstep_score": 1.0,
            "seq_eval_passes": seq_eval_passes,
            "seq_eval_attempts": seq_eval_attempts,
        },
        trades=[] if trade_results is None else trade_results,
    )


def _patch_cli_wf_oos_two_folds(monkeypatch: pytest.MonkeyPatch, fold_factory=_fake_eval) -> None:
    """Avoid running OOS re-eval on empty OHLCV in unit tests (two-fold layout)."""

    monkeypatch.setattr(
        cli,
        "wf_oos_folds_for_selected_params",
        lambda *a, **k: [fold_factory(window=f"WF{i}") for i in range(1, 3)],
    )


# Backward-compat alias for tests that haven't been updated yet.
_patch_cli_wf_oos_four_folds = _patch_cli_wf_oos_two_folds


def _passing_combine() -> CombineSimResult:
    return CombineSimResult(
        n_resamples=10,
        pass_rate_pct=100.0,
        n_passed=10,
        n_failed_drawdown=0,
        n_not_passed=0,
        median_days_to_pass=5.0,
        mean_days_to_pass=5.0,
        min_days_to_pass=5,
        max_days_to_pass=5,
        pct_daily_limit_hit=0.0,
        mean_max_drawdown=100.0,
        worst_max_drawdown=200.0,
        n_trades=24,
        n_trading_days=6,
        n_resamples_requested=10,
    )


def _sizing_trade(net_pnl: float = 250.0) -> TradeResult:
    return TradeResult(
        strategy="cli_e2e_mock",
        entry_time=pd.Timestamp("2024-06-10 09:35", tz=EASTERN_TZ),
        exit_time=pd.Timestamp("2024-06-10 10:05", tz=EASTERN_TZ),
        direction="long",
        entry=18000.0,
        stop=17995.0,
        target=18015.0,
        exit=18012.0,
        contracts=2,
        gross_pnl=net_pnl + 2.8,
        commission=2.8,
        net_pnl=net_pnl,
        r_multiple=1.2,
        exit_reason="target",
        bars_held=6,
        params=dict(MOCK_CLI_SPEC.default_params),
    )


@pytest.fixture
def synth_frame_narrow() -> pd.DataFrame:
    return synthetic_narrow_frame()


@pytest.fixture
def cli_mock_registered() -> None:
    load_user_strategies()
    if "cli_e2e_mock" in STRATEGIES:
        del STRATEGIES["cli_e2e_mock"]
    register_strategy(MOCK_CLI_SPEC)
    yield
    del STRATEGIES["cli_e2e_mock"]


@pytest.fixture
def narrow_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator, "WINDOWS", NARROW_PIPELINE_WINDOWS)
    monkeypatch.setattr(pipeline_config, "WINDOWS", NARROW_PIPELINE_WINDOWS)


@pytest.fixture
def faster_combine(monkeypatch: pytest.MonkeyPatch):
    """Cap resamples so tests remain fast."""

    def _wrapped(
        trades,
        rules=None,
        n_resamples: int = 1_000,
        seed: int | None = 42,
    ):
        from v3.config import TOPSTEP_50K
        capped = max(120, min(n_resamples, 240))
        return run_combine_original(trades, rules if rules is not None else TOPSTEP_50K, n_resamples=capped, seed=seed)

    monkeypatch.setattr(cli, "run_combine_simulator", _wrapped)


def test_list_strategies_prints_all_base_strategies(capsys):
    result = cli.main(["--list-strategies"])

    captured = capsys.readouterr()
    assert result == 0
    for name in STRATEGY_NAMES:
        assert name in captured.out


def test_invalid_strategy_exits_nonzero_and_prints_available_names(capsys):
    load_user_strategies()
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--strategy", "nonexistent_strategy"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.code != 0
    assert "Unknown strategy" in output
    for name in STRATEGY_NAMES:
        assert name in output


def test_missing_strategy_flag_exits(monkeypatch: pytest.MonkeyPatch):
    load_user_strategies()
    stderr_capture = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr_capture)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert exc_info.value.code != 0
    merged = stderr_capture.getvalue()
    assert "--strategy" in merged


def _parse_and_apply_mode(argv: list[str]) -> argparse.Namespace:
    args = cli.build_parser().parse_args(["--strategy", "cli_e2e_mock", *argv])
    cli._apply_mode_defaults(args)
    return args


def test_mode_defaults_to_quick_pipeline() -> None:
    args = _parse_and_apply_mode([])

    assert args.mode == "quick"
    assert args.skip_wf is False
    assert args.skip_sensitivity is True


@pytest.mark.parametrize(
    ("mode", "skip_wf", "skip_sensitivity"),
    [
        ("quick", False, True),
        ("full", False, False),
        ("holdout-only", True, True),
    ],
)
def test_mode_sets_pipeline_shortcuts(mode: str, skip_wf: bool, skip_sensitivity: bool) -> None:
    args = _parse_and_apply_mode(["--mode", mode])

    assert args.skip_wf is skip_wf
    assert args.skip_sensitivity is skip_sensitivity


def test_legacy_skip_flags_still_work_without_mode() -> None:
    skip_sensitivity = _parse_and_apply_mode(["--skip-sensitivity"])
    skip_wf = _parse_and_apply_mode(["--skip-wf"])

    assert skip_sensitivity.skip_wf is False
    assert skip_sensitivity.skip_sensitivity is True
    assert skip_wf.skip_wf is True
    assert skip_wf.skip_sensitivity is True


def test_min_wf_passes_is_passed_to_walk_forward(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(cli, "evaluate_strategy", lambda *args, **kwargs: _fake_eval(window="holdout"))
    monkeypatch.setattr(cli, "run_combine_simulator", lambda *args, **kwargs: _passing_combine())

    def fake_run_walk_forward(*args: Any, **kwargs: Any) -> Any:
        captured["max_grid"] = kwargs.get("max_grid", args[3] if len(args) > 3 else None)
        captured["min_eval_passes_per_fold"] = kwargs.get("min_eval_passes_per_fold")
        captured["min_folds_meeting_passes"] = kwargs.get("min_folds_meeting_passes")
        captured["windows"] = kwargs.get("windows")
        captured["risk_dollars"] = kwargs.get("risk_dollars")
        captured["max_contracts"] = kwargs.get("max_contracts")
        return dict(MOCK_CLI_SPEC.default_params), [_fake_eval(window=f"WF{i}") for i in range(1, 5)], True

    monkeypatch.setattr(cli, "run_walk_forward", fake_run_walk_forward)
    _patch_cli_wf_oos_four_folds(monkeypatch)

    output_dir = tmp_path / "out_min_wf_passes"

    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--max-grid", "7",
            "--min-wf-passes", "3",
            "--skip-sensitivity",
        ]
    )

    assert captured["max_grid"] == 7
    assert captured["min_folds_meeting_passes"] == 3
    assert captured["min_eval_passes_per_fold"] == 2
    from v3.config import DEFAULT_MAX_CONTRACTS, DEFAULT_RISK_DOLLARS

    assert captured["risk_dollars"] == DEFAULT_RISK_DOLLARS
    assert captured["max_contracts"] == DEFAULT_MAX_CONTRACTS


def test_skip_wf_uses_default_params(
    cli_mock_registered: None,
    narrow_pipeline: None,
    synth_frame_narrow: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    faster_combine,
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: synth_frame_narrow.copy())

    output_dir = tmp_path / "out_skip"
    output_dir.mkdir()
    monkeypatch.setattr("sys.stderr", io.StringIO())

    code = cli.main(
        [
            "--strategy",
            "cli_e2e_mock",
            "--timeframe",
            "5min",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--skip-wf",
            "--max-grid",
            "10",
            "--force",
        ]
    )

    assert code == 0
    result_json = _cli_e2e_result_json(output_dir)
    blob = json.loads(result_json.read_text())
    assert blob["skip_walk_forward"] is True
    assert blob["walk_forward"]["best_params"] == dict(MOCK_CLI_SPEC.default_params)
    frozen = blob["holdout"]["params"]
    assert frozen == MOCK_CLI_SPEC.default_params
    verdict = blob["verdict"]["verdict"]
    assert verdict in {"REJECT", "PROMISING", "COMBINE-READY"}


def test_pipeline_end_to_end_mock_strategy(
    cli_mock_registered: None,
    narrow_pipeline: None,
    synth_frame_narrow: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    faster_combine,
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: synth_frame_narrow.copy())

    output_dir = tmp_path / "out_full"
    output_dir.mkdir()
    monkeypatch.setattr("sys.stderr", io.StringIO())

    code = cli.main(
        [
            "--strategy",
            "cli_e2e_mock",
            "--timeframe",
            "5min",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--max-grid",
            "9",
            "--force",
        ]
    )

    assert code == 0
    result_json = _cli_e2e_result_json(output_dir)
    blob = json.loads(result_json.read_text())
    assert blob["skip_walk_forward"] is False
    assert blob["walk_forward"]["aggregate"]["wf_folds"] == 2
    assert len(blob["walk_forward"]["oos_folds"]) == 2

    verdict = blob["verdict"]["verdict"]
    assert verdict in {"REJECT", "PROMISING", "COMBINE-READY"}
    frozen_dir = Path(output_dir) / "frozen_params"
    if verdict == "REJECT":
        assert blob["freeze"] is None
        assert not (frozen_dir / "audit_log.jsonl").exists()
    else:
        assert (frozen_dir / "audit_log.jsonl").exists()


def test_optimize_sizing_for_speed_writes_fold_json_and_console(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_call: dict[str, Any] = {}
    fold_trade = _sizing_trade()

    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        cli,
        "run_walk_forward",
        lambda *args, **kwargs: (dict(MOCK_CLI_SPEC.default_params), [], True),
    )
    monkeypatch.setattr(
        cli,
        "wf_oos_folds_for_selected_params",
        lambda *args, **kwargs: [_fake_eval(window="WF1_test", trade_results=[fold_trade])],
    )
    monkeypatch.setattr(cli, "evaluate_strategy", lambda *args, **kwargs: _fake_eval(window="holdout"))

    def fake_optimize_for_speed_wf(*args: Any, **kwargs: Any) -> SpeedOptimizationResult:
        captured_call["trades"] = args[0]
        captured_call["strategy"] = kwargs["strategy"]
        captured_call["window"] = kwargs["window"]
        captured_call["pass_floor_pct"] = kwargs["pass_floor_pct"]
        captured_call["pass_target_pct"] = kwargs["pass_target_pct"]
        return SpeedOptimizationResult(
            strategy=kwargs["strategy"],
            window=kwargs["window"],
            pass_floor_pct=kwargs["pass_floor_pct"],
            pass_target_pct=kwargs["pass_target_pct"],
            optimal_risk_dollars=150.0,
            pass_rate_pct=68.5,
            mean_days_to_pass=8.2,
            std_days_to_pass=2.1,
            min_contracts_used=2,
            max_contracts_used=5,
            candidates=(
                {
                    "risk_dollars": 150.0,
                    "pass_rate_pct": 68.5,
                    "mean_days_to_pass": 8.2,
                    "std_days_to_pass": 2.1,
                    "min_contracts": 2,
                    "max_contracts": 5,
                },
                {
                    "risk_dollars": 100.0,
                    "pass_rate_pct": 65.0,
                    "mean_days_to_pass": 9.1,
                    "std_days_to_pass": 1.5,
                    "min_contracts": 2,
                    "max_contracts": 8,
                },
            ),
        )

    monkeypatch.setattr(cli, "optimize_for_speed_wf", fake_optimize_for_speed_wf)

    output_dir = tmp_path / "out_sizing_speed"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--skip-sensitivity",
            "--force",
            "--optimize-sizing-for-speed",
            "--pass-floor-pct", "45",
            "--pass-target-pct", "80",
        ]
    )

    captured = capsys.readouterr()
    out_json = output_dir / "cli_e2e_mock_wf1_speed_optimization.json"
    payload = json.loads(out_json.read_text())

    assert code == 0
    assert captured_call == {
        "trades": [fold_trade],
        "strategy": "cli_e2e_mock",
        "window": "WF1_test",
        "pass_floor_pct": 45.0,
        "pass_target_pct": 80.0,
    }
    assert payload["optimal_risk_dollars"] == 150.0
    assert payload["candidates"][0]["risk_dollars"] == 150.0
    assert "=== WF1_test Speed Optimization ===" in captured.out
    assert "Optimal Risk: $150/trade" in captured.out
    assert "1. $150  68.5%   8.2 days   2-5 contracts" in captured.out


def test_optimize_sizing_for_longevity_writes_holdout_json_and_console(
    cli_mock_registered: None,
    synth_frame_narrow: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_call: dict[str, Any] = {}
    holdout_trade = _sizing_trade(net_pnl=300.0)

    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: synth_frame_narrow.copy())
    monkeypatch.setattr(
        cli,
        "evaluate_strategy",
        lambda *args, **kwargs: _fake_eval(window="holdout", trade_results=[holdout_trade]),
    )

    def fake_optimize_for_longevity_holdout(*args: Any, **kwargs: Any) -> LongevityOptimizationResult:
        captured_call["trades"] = args[0]
        captured_call["strategy"] = kwargs["strategy"]
        captured_call["min_profit_per_trade"] = kwargs["min_profit_per_trade"]
        return LongevityOptimizationResult(
            strategy=kwargs["strategy"],
            window=kwargs["window"],
            min_profit_per_trade=kwargs["min_profit_per_trade"],
            optimal_risk_dollars=200.0,
            avg_pnl_per_trade=185.0,
            total_pnl=18500.0,
            funded_accounts_used=1,
            accounts_blown=0,
            total_trades_executed=100,
            longevity_score=1.37,
            candidates=(
                {
                    "risk_dollars": 200.0,
                    "avg_pnl_per_trade": 185.0,
                    "total_pnl": 18500.0,
                    "funded_accounts_used": 1,
                    "longevity_score": 1.37,
                },
            ),
        )

    monkeypatch.setattr(cli, "optimize_for_longevity_holdout", fake_optimize_for_longevity_holdout)

    output_dir = tmp_path / "out_sizing_longevity"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--skip-wf",
            "--skip-sensitivity",
            "--optimize-sizing-for-longevity",
            "--min-profit-per-trade", "175",
        ]
    )

    captured = capsys.readouterr()
    out_json = output_dir / "cli_e2e_mock_holdout_longevity_optimization.json"
    payload = json.loads(out_json.read_text())

    assert code == 0
    assert captured_call == {
        "trades": [holdout_trade],
        "strategy": "cli_e2e_mock",
        "min_profit_per_trade": 175.0,
    }
    assert payload["optimal_risk_dollars"] == 200.0
    assert payload["candidates"][0]["avg_pnl_per_trade"] == 185.0
    assert "=== Holdout Longevity Optimization ===" in captured.out
    assert "Optimal Risk: $200/trade" in captured.out
    assert "1. $200   $185/trade   1 account   score=1.37" in captured.out


# ---------------------------------------------------------------------------
# Sensitivity integration tests
# ---------------------------------------------------------------------------

def test_skip_sensitivity_flag_omits_sensitivity_from_result(
    cli_mock_registered: None,
    narrow_pipeline: None,
    synth_frame_narrow: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    faster_combine,
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: synth_frame_narrow.copy())
    output_dir = tmp_path / "out_skip_sens"
    output_dir.mkdir()
    monkeypatch.setattr("sys.stderr", io.StringIO())

    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--skip-wf",
            "--skip-sensitivity",
            "--force",
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    blob = json.loads(_cli_e2e_result_json(output_dir).read_text())
    assert blob["skip_sensitivity"] is True
    assert blob["sensitivity"] is None
    assert "Skipped: --skip-sensitivity flag" in captured.out


def test_wf_gate_exits_before_holdout_without_force(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """WF gate failure (no --force) should exit before holdout stage."""
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(cli, "run_walk_forward", lambda *args, **kwargs: (
        dict(MOCK_CLI_SPEC.default_params),
        [_fake_eval(topstep_passed=False, window=f"WF{i}") for i in range(1, 3)],
        False,
    ))
    _patch_cli_wf_oos_two_folds(monkeypatch)

    output_dir = tmp_path / "out_wf_gate_early_exit"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "REJECT: walk-forward gates not met" in captured.err
    assert "Stage 4" not in captured.out
    assert not (output_dir / "frozen_params").exists()


def test_wf_robust_fail_exits_before_later_stages_without_force(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())

    monkeypatch.setattr(cli, "run_walk_forward", lambda *args, **kwargs: (
        dict(MOCK_CLI_SPEC.default_params),
        [_fake_eval(topstep_passed=False, window=f"WF{i}") for i in range(1, 5)],
        False,
    ))
    _patch_cli_wf_oos_four_folds(monkeypatch)
    monkeypatch.setattr(cli, "evaluate_strategy", lambda *args, **kwargs: _fake_eval(window="holdout"))
    monkeypatch.setattr(cli, "run_combine_simulator", lambda *args, **kwargs: _passing_combine())

    output_dir = tmp_path / "out_wf_gate"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "REJECT: walk-forward gates not met" in captured.err
    assert "wf_robust_ok=False" in captured.err
    assert "Stage 4" not in captured.out


def test_wf_robust_fail_continues_with_force(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())

    monkeypatch.setattr(cli, "run_walk_forward", lambda *args, **kwargs: (
        dict(MOCK_CLI_SPEC.default_params),
        [_fake_eval(topstep_passed=False, window=f"WF{i}") for i in range(1, 5)],
        False,
    ))
    _patch_cli_wf_oos_four_folds(monkeypatch)
    monkeypatch.setattr(cli, "evaluate_strategy", lambda *args, **kwargs: _fake_eval(window="holdout"))
    monkeypatch.setattr(cli, "run_combine_simulator", lambda *args, **kwargs: _passing_combine())

    output_dir = tmp_path / "out_wf_force"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--force",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "WARN: walk-forward gates not met (--force)" in captured.out
    assert "Stage 6" in captured.out


def test_wf_seq_pass_rate_fail_exits_without_force(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())

    monkeypatch.setattr(
        cli,
        "run_walk_forward",
        lambda *args, **kwargs: (dict(MOCK_CLI_SPEC.default_params), [], True),
    )

    def _low_seq_folds(*a: Any, **k: Any) -> list[evaluator.EvaluationResult]:
        return [
            _fake_eval(window="WF1", seq_eval_passes=2, seq_eval_attempts=5),
            _fake_eval(window="WF2", seq_eval_passes=2, seq_eval_attempts=5),
            _fake_eval(window="WF3", seq_eval_passes=2, seq_eval_attempts=5),
            _fake_eval(window="WF4", seq_eval_passes=1, seq_eval_attempts=5),
        ]

    monkeypatch.setattr(cli, "wf_oos_folds_for_selected_params", _low_seq_folds)

    output_dir = tmp_path / "out_wf_seq"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "wf_all_folds_seq_ok=False" in captured.err or "rate=" in captured.err


def test_strict_flag_does_not_bypass_wf_gate(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())

    monkeypatch.setattr(cli, "run_walk_forward", lambda *args, **kwargs: (
        dict(MOCK_CLI_SPEC.default_params),
        [_fake_eval(topstep_passed=False, window=f"WF{i}") for i in range(1, 5)],
        False,
    ))
    _patch_cli_wf_oos_four_folds(monkeypatch)

    output_dir = tmp_path / "out_zero_wf_strict"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "REJECT: walk-forward gates not met" in captured.err
    assert "Stage 4" not in captured.out


def test_reject_verdict_does_not_freeze(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())

    monkeypatch.setattr(
        cli,
        "run_walk_forward",
        lambda *args, **kwargs: (
            dict(MOCK_CLI_SPEC.default_params),
            [_fake_eval(topstep_passed=True, window=f"WF{i}") for i in range(1, 5)],
            True,
        ),
    )
    _patch_cli_wf_oos_four_folds(monkeypatch)
    monkeypatch.setattr(
        cli,
        "evaluate_strategy",
        lambda *args, **kwargs: _fake_eval(window="holdout", total_net_pnl=-500.0),
    )

    output_dir = tmp_path / "out_reject_no_freeze"
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "VERDICT: REJECT" in captured.out
    assert "Freeze skipped: verdict is REJECT" in captured.out
    assert not (output_dir / "frozen_params").exists()


def test_verdict_threshold_cli_args_persist_in_result_bundle(
    cli_mock_registered: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())

    monkeypatch.setattr(cli, "evaluate_strategy", lambda *args, **kwargs: _fake_eval(window="holdout"))

    output_dir = Path(".pytest_cache") / "out_cli_thresholds"
    output_dir.mkdir(parents=True, exist_ok=True)
    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(output_dir),
            "--output-dir", str(output_dir),
            "--skip-wf",
            "--reject-pass-rate", "90",
        ]
    )

    captured = capsys.readouterr()
    blob = json.loads(_cli_e2e_result_json(output_dir).read_text())
    assert code == 0
    assert blob["verdict_thresholds"]["reject_pass_rate_pct"] == 90.0
    assert "holdout_monte_carlo" in blob
    # Pipeline verdict ignores combine-style pass-rate thresholds; flags stay in JSON for tooling.
    assert "VERDICT:" in captured.out


def test_sensitivity_runs_and_appears_in_result(
    cli_mock_registered: None,
    narrow_pipeline: None,
    synth_frame_narrow: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    faster_combine,
) -> None:
    """When sensitivity is NOT skipped, result JSON contains sensitivity keys."""
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: synth_frame_narrow.copy())
    # Also cap sensitivity resamples so the test stays fast
    from v3 import sensitivity as sens_module
    from v3.combine_simulator import run_combine_simulator as _orig_sim

    def _fast_sim(trades, rules=None, n_resamples=200, seed=42):
        from v3.config import TOPSTEP_50K
        capped = max(40, min(n_resamples, 80))
        return _orig_sim(trades, rules if rules is not None else TOPSTEP_50K, n_resamples=capped, seed=seed)

    monkeypatch.setattr(cli, "run_combine_simulator", _fast_sim)

    output_dir = tmp_path / "out_with_sens"
    output_dir.mkdir()
    monkeypatch.setattr("sys.stderr", io.StringIO())

    code = cli.main(
        [
            "--strategy", "cli_e2e_mock",
            "--timeframe", "5min",
            "--data-dir", str(tmp_path),
            "--output-dir", str(output_dir),
            "--skip-wf",
            "--full",
            "--force",
            "--sensitivity-resamples", "40",
        ]
    )

    assert code == 0
    blob = json.loads(_cli_e2e_result_json(output_dir).read_text())
    assert blob["skip_sensitivity"] is False
    assert blob["sensitivity"] is not None
    sens = blob["sensitivity"]
    assert "sensitivity_is_cliff" in sens
    assert "sensitivity_default_pass_rate" in sens
    assert "sensitivity_param_results" in sens
    assert "width" in sens["sensitivity_param_results"]
    assert blob["verdict"]["sensitivity_flag"] is False
