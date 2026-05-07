"""Tests for timeframe derivation and loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_data import resample_to_csv
from src.v3.config import TIMEFRAME_MINUTES
from src.v3.data import _load_ohlcv_from_csv, load_ohlcv, resample_ohlcv


EXPECTED_RTH_BARS_PER_DAY = {
    "1min": 390,
    "2min": 195,
    "3min": 130,
    "5min": 78,
    "15min": 26,
    "30min": 13,
    "1h": 7,
    "4h": 2,
}


def _make_full_day_1min_frame(day: str = "2024-01-15") -> pd.DataFrame:
    times = pd.date_range(
        start=f"{day} 00:00:00",
        end=f"{day} 23:59:00",
        freq="1min",
        tz="America/New_York",
    )
    minute_number = pd.Index(range(len(times)), dtype="int64")
    frame = pd.DataFrame(
        {
            "open": 1000.0 + minute_number.to_numpy(dtype="float64"),
            "high": 1000.5 + minute_number.to_numpy(dtype="float64"),
            "low": 999.5 + minute_number.to_numpy(dtype="float64"),
            "close": 1000.2 + minute_number.to_numpy(dtype="float64"),
            "volume": 10 + minute_number.to_numpy(dtype="int64"),
        },
        index=pd.DatetimeIndex(times, name="datetime"),
    )
    return frame


def _write_1min_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.reset_index().copy()
    payload["datetime"] = payload["datetime"].dt.tz_convert("UTC").map(lambda value: value.isoformat())
    payload.to_csv(path, index=False)


class TestSessionAwareResample:
    def test_resample_1h_anchors_to_session_open(self) -> None:
        frame = _make_full_day_1min_frame()

        result = resample_ohlcv(frame, "1h", session_only=True)

        expected_labels = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
        assert [stamp.strftime("%H:%M") for stamp in result.index] == expected_labels
        assert len(result) == 7

    def test_resample_4h_anchors_to_session_open(self) -> None:
        frame = _make_full_day_1min_frame()

        result = resample_ohlcv(frame, "4h", session_only=True)

        assert [stamp.strftime("%H:%M") for stamp in result.index] == ["09:30", "13:30"]
        assert len(result) == 2

    def test_resample_1h_ohlcv_uses_rth_minutes_only(self) -> None:
        frame = _make_full_day_1min_frame()

        result = resample_ohlcv(frame, "1h", session_only=True)
        first_bar = result.iloc[0]
        source_slice = frame.loc["2024-01-15 09:30:00-05:00":"2024-01-15 10:29:00-05:00"]

        assert first_bar["open"] == source_slice.iloc[0]["open"]
        assert first_bar["high"] == source_slice["high"].max()
        assert first_bar["low"] == source_slice["low"].min()
        assert first_bar["close"] == source_slice.iloc[-1]["close"]
        assert first_bar["volume"] == source_slice["volume"].sum()

    @pytest.mark.parametrize("timeframe,expected_count", EXPECTED_RTH_BARS_PER_DAY.items())
    def test_expected_rth_bar_count_per_day(self, timeframe: str, expected_count: int) -> None:
        frame = _make_full_day_1min_frame()

        result = resample_ohlcv(frame, timeframe, session_only=True)

        assert len(result) == expected_count

    def test_session_only_false_keeps_wall_clock_resampling(self) -> None:
        frame = _make_full_day_1min_frame()

        result = resample_ohlcv(frame, "1h", session_only=False)

        assert len(result) == 24
        assert result.index[0].strftime("%H:%M") == "00:00"
        assert result.index[-1].strftime("%H:%M") == "23:00"

    def test_1min_session_only_filters_to_rth_without_resampling(self) -> None:
        frame = _make_full_day_1min_frame()

        result = resample_ohlcv(frame, "1min", session_only=True)

        assert len(result) == 390
        assert result.index[0].strftime("%H:%M") == "09:30"
        assert result.index[-1].strftime("%H:%M") == "15:59"


class TestLoadOHLCV:
    def test_load_prefers_prebuilt_higher_timeframe_file_when_present(self, tmp_path: Path) -> None:
        source_frame = _make_full_day_1min_frame()
        _write_1min_csv(tmp_path / "mnq_1min_databento.csv", source_frame)

        prebuilt = pd.DataFrame(
            {
                "open": [111.0],
                "high": [112.0],
                "low": [110.0],
                "close": [111.5],
                "volume": [999.0],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-15 00:00:00", tz="America/New_York")], name="datetime"),
        )
        _write_1min_csv(tmp_path / "mnq_1h_databento.csv", prebuilt)

        result = load_ohlcv("mnq", "1h", data_dir=tmp_path, session_only=False)

        pd.testing.assert_frame_equal(result, prebuilt)

    def test_load_1h_respects_rth_session(self, tmp_path: Path) -> None:
        frame = _make_full_day_1min_frame()
        _write_1min_csv(tmp_path / "mnq_1min_databento.csv", frame)

        result = load_ohlcv("mnq", "1h", data_dir=tmp_path, session_only=True)

        assert [stamp.strftime("%H:%M") for stamp in result.index] == [
            "09:30",
            "10:30",
            "11:30",
            "12:30",
            "13:30",
            "14:30",
            "15:30",
        ]

    def test_load_1h_session_only_false_preserves_24h_stream(self, tmp_path: Path) -> None:
        frame = _make_full_day_1min_frame()
        _write_1min_csv(tmp_path / "mes_1min_databento.csv", frame)

        result = load_ohlcv("mes", "1h", data_dir=tmp_path, session_only=False)

        assert len(result) == 24
        assert result.index[0].strftime("%H:%M") == "00:00"
        assert result.index[-1].strftime("%H:%M") == "23:00"

    @pytest.mark.parametrize("instrument", ["mnq", "mes"])
    @pytest.mark.parametrize("timeframe", ["2min", "3min", "5min", "15min", "30min", "1h", "4h"])
    def test_fallback_derives_from_1min_when_prebuilt_file_missing(
        self,
        tmp_path: Path,
        instrument: str,
        timeframe: str,
    ) -> None:
        frame = _make_full_day_1min_frame()
        source_path = tmp_path / f"{instrument}_1min_databento.csv"
        _write_1min_csv(source_path, frame)

        expected_frame = resample_ohlcv(frame, timeframe, session_only=True)

        derived_frame = load_ohlcv(instrument, timeframe, data_dir=tmp_path, session_only=True)

        pd.testing.assert_frame_equal(expected_frame, derived_frame)

    def test_missing_1min_source_still_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_ohlcv("mnq", "4h", data_dir=tmp_path, session_only=True)

    def test_all_supported_timeframes_load_for_both_assets(self, tmp_path: Path) -> None:
        frame = _make_full_day_1min_frame()
        for instrument in ("mnq", "mes"):
            _write_1min_csv(tmp_path / f"{instrument}_1min_databento.csv", frame)

        for instrument in ("mnq", "mes"):
            for timeframe, expected_count in EXPECTED_RTH_BARS_PER_DAY.items():
                result = load_ohlcv(instrument, timeframe, data_dir=tmp_path, session_only=True)
                assert len(result) == expected_count, f"{instrument} {timeframe} count mismatch"
                assert set(result.columns) == {"open", "high", "low", "close", "volume"}

    def test_builder_output_has_expected_4h_labels(self, tmp_path: Path) -> None:
        frame = _make_full_day_1min_frame()
        source_path = tmp_path / "mnq_1min_databento.csv"
        _write_1min_csv(source_path, frame)

        resample_to_csv(source_path=source_path, timeframe="4h", output_dir=tmp_path)
        built = _load_ohlcv_from_csv(tmp_path / "mnq_4h_databento.csv")

        assert [stamp.strftime("%H:%M") for stamp in built.index] == [
            "00:00",
            "04:00",
            "08:00",
            "12:00",
            "16:00",
            "20:00",
        ]

    def test_builder_resamples_full_session_and_preserves_overnight_bars(self, tmp_path: Path) -> None:
        frame = _make_full_day_1min_frame()
        source_path = tmp_path / "mes_1min_databento.csv"
        _write_1min_csv(source_path, frame)

        row_count, first_ts, last_ts = resample_to_csv(source_path=source_path, timeframe="1h", output_dir=tmp_path)
        built = _load_ohlcv_from_csv(tmp_path / "mes_1h_databento.csv")

        assert row_count == 24
        assert first_ts.strftime("%H:%M") == "00:00"
        assert last_ts.strftime("%H:%M") == "23:00"
        assert built.index[0].strftime("%H:%M") == "00:00"
        assert built.index[1].strftime("%H:%M") == "01:00"
        overnight_slice = frame.loc["2024-01-15 00:00:00-05:00":"2024-01-15 00:59:00-05:00"]
        first_bar = built.iloc[0]
        assert first_bar["open"] == overnight_slice.iloc[0]["open"]
        assert first_bar["close"] == overnight_slice.iloc[-1]["close"]


def test_all_supported_timeframes_are_accounted_for() -> None:
    assert set(TIMEFRAME_MINUTES) == set(EXPECTED_RTH_BARS_PER_DAY)
