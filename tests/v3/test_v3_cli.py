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
from v3.strategies import STRATEGIES, StrategySpec, TradeSignal, load_user_strategies, register_strategy


def _cli_e2e_result_json(output_dir: Path) -> Path:
    return output_dir / "json" / "cli_e2e_mock_5min_result.json"


NARROW_PIPELINE_WINDOWS = PipelineWindows(
    in_sample_sanity=DateWindow("in_sample_sanity", "2024-06-03", "2024-06-14"),
    walk_forward=(
        WalkForwardWindow(
            "WF1",
            DateWindow("WF1_train", "2024-06-03", "2024-06-07"),
            DateWindow("WF1_test", "2024-06-10", "2024-06-11"),
        ),
        WalkForwardWindow(
            "WF2",
            DateWindow("WF2_train", "2024-06-03", "2024-06-11"),
            DateWindow("WF2_test", "2024-06-12", "2024-06-13"),
        ),
        WalkForwardWindow(
            "WF3",
            DateWindow("WF3_train", "2024-06-03", "2024-06-13"),
            DateWindow("WF3_test", "2024-06-14", "2024-06-17"),
        ),
        WalkForwardWindow(
            "WF4",
            DateWindow("WF4_train", "2024-06-03", "2024-06-17"),
            DateWindow("WF4_test", "2024-06-18", "2024-06-21"),
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
        trades=[],
    )


def _patch_cli_wf_oos_four_folds(monkeypatch: pytest.MonkeyPatch, fold_factory=_fake_eval) -> None:
    """Avoid running OOS re-eval on empty OHLCV in unit tests."""

    monkeypatch.setattr(
        cli,
        "wf_oos_folds_for_selected_params",
        lambda *a, **k: [fold_factory(window=f"WF{i}") for i in range(1, 5)],
    )


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
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval())
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
    assert blob["walk_forward"]["aggregate"]["wf_folds"] == 4
    assert len(blob["walk_forward"]["oos_folds"]) == 4

    verdict = blob["verdict"]["verdict"]
    assert verdict in {"REJECT", "PROMISING", "COMBINE-READY"}
    frozen_dir = Path(output_dir) / "frozen_params"
    if verdict == "REJECT":
        assert blob["freeze"] is None
        assert not (frozen_dir / "audit_log.jsonl").exists()
    else:
        assert (frozen_dir / "audit_log.jsonl").exists()


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


def test_stage_2_rejects_insufficient_sample_size(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval(trades=5))

    output_dir = tmp_path / "out_stage2_reject"
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
    assert "REJECT: insufficient sample size" in captured.out
    assert "Stage 3" not in captured.out
    assert not (output_dir / "frozen_params").exists()


def test_wf_robust_fail_exits_before_later_stages_without_force(
    cli_mock_registered: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_ohlcv", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval())
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
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval())
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
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval())
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
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval())
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
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval())
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
    monkeypatch.setattr(cli, "run_in_sample_sanity", lambda *args, **kwargs: _fake_eval())
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
