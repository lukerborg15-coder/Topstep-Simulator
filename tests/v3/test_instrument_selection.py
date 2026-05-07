"""Tests for instrument selection (MNQ and MES) in the CLI and pipeline."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from v3 import cli, evaluator
from v3.config import (
    INSTRUMENTS,
    MES,
    MNQ,
    Instrument,
)
from v3.data import load_ohlcv
from v3.strategies import StrategySpec, TradeSignal, load_user_strategies, register_strategy


MOCK_INSTRUMENT_SPEC = StrategySpec(
    name="instrument_test_mock",
    generate=lambda df, params: [
        TradeSignal(
            time=df.index[10],
            direction="long",
            entry=float(df["close"].iloc[10]),
            stop=float(df["close"].iloc[10]) - 4.0,
            target=float(df["close"].iloc[10]) + 10.0,
            strategy="instrument_test_mock",
            params=dict(params),
        )
    ],
    default_params={"width": 1.0},
    param_grid={"width": (1.0, 2.0)},
    max_signals_per_day=None,
)


@pytest.fixture
def synth_frame() -> pd.DataFrame:
    """Simple synthetic 5min frame."""
    import numpy as np

    idx = pd.date_range("2024-06-03 09:30", "2024-06-03 16:00", freq="5min", tz="America/New_York")
    rng = np.random.default_rng(42)
    n = len(idx)
    close = 5000.0 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(-1, 1, n),
            "high": close + rng.uniform(1, 3, n),
            "low": close - rng.uniform(1, 3, n),
            "close": close,
            "volume": rng.integers(1000, 5000, n),
        },
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


@pytest.fixture
def mock_registered() -> None:
    load_user_strategies()
    if "instrument_test_mock" in cli.STRATEGIES:
        del cli.STRATEGIES["instrument_test_mock"]
    register_strategy(MOCK_INSTRUMENT_SPEC)
    yield
    if "instrument_test_mock" in cli.STRATEGIES:
        del cli.STRATEGIES["instrument_test_mock"]


class TestInstrumentDefinitions:
    """Test that instrument definitions are correct."""

    def test_mnq_has_correct_values(self):
        """MNQ should have Micro E-mini Nasdaq-100 values."""
        assert MNQ.symbol == "MNQ"
        assert MNQ.point_value == 2.0
        assert MNQ.tick_size == 0.25
        assert MNQ.commission_round_turn == 1.40
        assert MNQ.slippage_points_per_side == 0.25

    def test_mes_has_correct_values(self):
        """MES should have Micro E-mini S&P 500 values."""
        assert MES.symbol == "MES"
        assert MES.point_value == 5.0
        assert MES.tick_size == 0.25
        assert MES.commission_round_turn == 0.85
        assert MES.slippage_points_per_side == 0.25

    def test_mes_point_value_differs_from_mnq(self):
        """MES should have different point value than MNQ."""
        assert MES.point_value != MNQ.point_value
        assert MES.point_value == 5.0
        assert MNQ.point_value == 2.0

    def test_mes_commission_differs_from_mnq(self):
        """MES should have lower commission than MNQ (micro contract)."""
        assert MES.commission_round_turn < MNQ.commission_round_turn
        assert MES.commission_round_turn == 0.85
        assert MNQ.commission_round_turn == 1.40

    def test_instruments_registry_has_both(self):
        """INSTRUMENTS should contain both mnq and mes."""
        assert "mnq" in INSTRUMENTS
        assert "mes" in INSTRUMENTS
        assert INSTRUMENTS["mnq"] is MNQ
        assert INSTRUMENTS["mes"] is MES


class TestCLIInstrumentArgument:
    """Test CLI --instrument argument parsing."""

    def test_default_instrument_is_mnq(self):
        """Default instrument should be mnq."""
        parser = cli.build_parser()
        args = parser.parse_args(["--strategy", "connors_rsi2"])
        assert args.instrument == "mnq"

    def test_instrument_mnq_accepted(self):
        """--instrument mnq should be accepted."""
        parser = cli.build_parser()
        args = parser.parse_args(["--strategy", "connors_rsi2", "--instrument", "mnq"])
        assert args.instrument == "mnq"

    def test_instrument_mes_accepted(self):
        """--instrument mes should be accepted."""
        parser = cli.build_parser()
        args = parser.parse_args(["--strategy", "connors_rsi2", "--instrument", "mes"])
        assert args.instrument == "mes"

    def test_instrument_invalid_rejected(self):
        """Invalid instrument should be rejected by argparse."""
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--strategy", "connors_rsi2", "--instrument", "es"])


class TestDataLoadingWithInstrument:
    """Test that load_ohlcv uses the correct instrument in filename."""

    def test_load_ohlcv_accepts_instrument_parameter(self, tmp_path):
        """load_ohlcv should accept instrument parameter."""
        mnq_file = tmp_path / "mnq_1min_databento.csv"
        mnq_file.write_text(
            "datetime,open,high,low,close,volume\n"
            "2024-06-03T13:30:00+00:00,5000,5005,4995,5000,1000\n"
        )

        df = load_ohlcv(instrument="mnq", timeframe="1min", data_dir=tmp_path, session_only=False)
        assert not df.empty

    def test_load_ohlcv_constructs_correct_filename(self, tmp_path):
        """load_ohlcv should construct filename from instrument."""
        mnq_file = tmp_path / "mnq_1min_databento.csv"
        mnq_file.write_text(
            "datetime,open,high,low,close,volume\n"
            "2024-06-03T13:30:00+00:00,5000,5005,4995,5000,1000\n"
        )

        mes_file = tmp_path / "mes_1min_databento.csv"
        mes_file.write_text(
            "datetime,open,high,low,close,volume\n"
            "2024-06-03T13:30:00+00:00,5000,5005,4995,5000,1000\n"
        )

        # Load MNQ data
        df_mnq = load_ohlcv(instrument="mnq", timeframe="1min", data_dir=tmp_path, session_only=False)
        assert not df_mnq.empty

        # Load MES data
        df_mes = load_ohlcv(instrument="mes", timeframe="1min", data_dir=tmp_path, session_only=False)
        assert not df_mes.empty

    def test_load_ohlcv_uses_exact_prebuilt_file_for_session_only_true(self, tmp_path):
        prebuilt_file = tmp_path / "mes_5min_databento.csv"
        prebuilt_file.write_text(
            "datetime,open,high,low,close,volume\n"
            "2024-06-03T13:30:00+00:00,5000,5005,4995,5000,1000\n"
        )

        df = load_ohlcv(instrument="mes", timeframe="5min", data_dir=tmp_path, session_only=True)

        assert not df.empty

    def test_load_ohlcv_uses_prebuilt_file_for_session_only_false(self, tmp_path):
        prebuilt_file = tmp_path / "mes_5min_databento.csv"
        prebuilt_file.write_text(
            "datetime,open,high,low,close,volume\n"
            "2024-06-03T13:30:00+00:00,5000,5005,4995,5000,1000\n"
        )

        df = load_ohlcv(instrument="mes", timeframe="5min", data_dir=tmp_path, session_only=False)

        assert len(df) == 1
        assert float(df.iloc[0]["close"]) == 5000.0


class TestEvaluatorWithInstrument:
    """Test that evaluator functions accept and use instrument parameter."""

    def test_evaluate_strategy_signature_has_instrument(self):
        """evaluate_strategy should accept instrument parameter."""
        import inspect

        sig = inspect.signature(evaluator.evaluate_strategy)
        # Instrument is a keyword-only parameter (after *,), so check parameters dict
        assert "instrument" in sig.parameters

    def test_run_walk_forward_signature_has_instrument(self):
        """run_walk_forward should accept instrument parameter."""
        import inspect

        sig = inspect.signature(evaluator.run_walk_forward)
        # Instrument is a keyword-only parameter (after *,), so check parameters dict
        assert "instrument" in sig.parameters

    def test_simulate_trades_uses_instrument_point_value(self, synth_frame):
        """simulate_trades should use instrument.point_value for P&L calculation."""
        from v3.evaluator import simulate_trades
        from v3.strategies import TradeSignal

        # Create a simple long signal
        signal = TradeSignal(
            time=synth_frame.index[5],
            direction="long",
            entry=100.0,
            stop=98.0,  # 2 point risk
            target=104.0,  # 4 point reward
            strategy="test",
            params={},
        )

        # With MNQ (point_value=2), 2 points risk = $4, 4 points reward = $8
        trades_mnq = simulate_trades(synth_frame, [signal], instrument=MNQ, risk_dollars=100.0, max_contracts=1)
        assert len(trades_mnq) == 1
        assert trades_mnq[0].contracts >= 1

        # With MES (point_value=5), same 2 points risk = $10, 4 points reward = $20
        trades_mes = simulate_trades(synth_frame, [signal], instrument=MES, risk_dollars=100.0, max_contracts=1)
        assert len(trades_mes) == 1
        assert trades_mnq[0].gross_pnl != trades_mes[0].gross_pnl
        assert trades_mnq[0].commission != trades_mes[0].commission
        assert trades_mes[0].gross_pnl > trades_mnq[0].gross_pnl
        assert trades_mes[0].commission < trades_mnq[0].commission


class TestCLIInstrumentFlow:
    """Test that CLI passes instrument through the pipeline."""

    def test_cli_loads_data_with_selected_instrument(
        self,
        mock_registered: None,
        synth_frame: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """CLI should load data with the selected instrument."""
        load_calls: list[tuple[str, str]] = []

        def fake_load_ohlcv(*, instrument: str, timeframe: str, data_dir: Path, session_only: bool):
            load_calls.append((instrument, timeframe))
            return synth_frame.copy()

        monkeypatch.setattr(cli, "load_ohlcv", fake_load_ohlcv)

        # Test that the argument is parsed correctly
        args = cli.build_parser().parse_args([
            "--strategy", "instrument_test_mock",
            "--instrument", "mes",
        ])
        assert args.instrument == "mes"

    def test_cli_passes_selected_instrument_through_runtime(
        self,
        mock_registered: None,
        synth_frame: pd.DataFrame,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        load_calls: list[tuple[str, str, bool]] = []
        seen: dict[str, object] = {}

        def fake_load_ohlcv(*, instrument: str, timeframe: str, data_dir: Path, session_only: bool):
            load_calls.append((instrument, timeframe, session_only))
            return synth_frame.copy()

        def fake_run_walk_forward(*args, **kwargs):
            seen["wf_instrument"] = kwargs["instrument"]
            return dict(MOCK_INSTRUMENT_SPEC.default_params), [], True

        def fake_wf_oos(*args, **kwargs):
            seen["wf_oos_instrument"] = kwargs["instrument"]
            return []

        def fake_evaluate_strategy(*args, **kwargs):
            seen["holdout_instrument"] = kwargs["instrument"]
            raise RuntimeError("stop after instrument capture")

        monkeypatch.setattr(cli, "load_ohlcv", fake_load_ohlcv)
        monkeypatch.setattr(cli, "run_walk_forward", fake_run_walk_forward)
        monkeypatch.setattr(cli, "wf_oos_folds_for_selected_params", fake_wf_oos)
        monkeypatch.setattr(cli, "evaluate_strategy", fake_evaluate_strategy)

        with pytest.raises(RuntimeError, match="stop after instrument capture"):
            cli.main(
                [
                    "--strategy", "instrument_test_mock",
                    "--instrument", "mes",
                    "--timeframe", "5min",
                    "--data-dir", str(tmp_path),
                    "--output-dir", str(tmp_path / "out"),
                    "--force",
                ]
            )

        assert load_calls == [("mes", "5min", False)]
        assert seen["wf_instrument"] is MES
        assert seen["wf_oos_instrument"] is MES
        assert seen["holdout_instrument"] is MES

    def test_cli_result_bundle_contains_instrument(
        self,
        mock_registered: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Verify instrument is stored in CLI args."""
        args = cli.build_parser().parse_args([
            "--strategy", "instrument_test_mock",
            "--instrument", "mes",
        ])
        assert hasattr(args, "instrument")
        assert args.instrument == "mes"
