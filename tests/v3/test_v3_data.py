from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from v3.data import load_ohlcv


def _load_build_data_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "build_data.py"
    spec = importlib.util.spec_from_file_location("build_data_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_full_day_1min_frame(day: str = "2024-01-15") -> pd.DataFrame:
    times = pd.date_range(
        start=f"{day} 00:00:00",
        end=f"{day} 23:59:00",
        freq="1min",
        tz="America/New_York",
    )
    minute_number = pd.Index(range(len(times)), dtype="int64")
    return pd.DataFrame(
        {
            "open": 1000.0 + minute_number.to_numpy(dtype="float64"),
            "high": 1000.5 + minute_number.to_numpy(dtype="float64"),
            "low": 999.5 + minute_number.to_numpy(dtype="float64"),
            "close": 1000.2 + minute_number.to_numpy(dtype="float64"),
            "volume": 10 + minute_number.to_numpy(dtype="int64"),
        },
        index=pd.DatetimeIndex(times, name="datetime"),
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.reset_index().copy()
    payload["datetime"] = payload["datetime"].map(lambda value: value.isoformat())
    path.write_text(payload.to_csv(index=False), encoding="utf-8")


def test_load_ohlcv_prefers_prebuilt_timeframe_file_when_present(tmp_path: Path) -> None:
    source_frame = _make_full_day_1min_frame()
    _write_csv(tmp_path / "mes_1min_databento.csv", source_frame)

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
    _write_csv(tmp_path / "mes_1h_databento.csv", prebuilt)

    result = load_ohlcv("mes", "1h", data_dir=tmp_path, session_only=False)

    pd.testing.assert_frame_equal(result, prebuilt)


def test_load_ohlcv_falls_back_to_1min_when_prebuilt_missing(tmp_path: Path) -> None:
    frame = _make_full_day_1min_frame()
    _write_csv(tmp_path / "mnq_1min_databento.csv", frame)

    result = load_ohlcv("mnq", "5min", data_dir=tmp_path, session_only=False)

    assert len(result) == 288
    assert result.index[0] == pd.Timestamp("2024-01-15 00:00:00", tz="America/New_York")
    assert result.index[-1] == pd.Timestamp("2024-01-15 23:55:00", tz="America/New_York")


def test_build_data_resample_preserves_overnight_full_session_structure(tmp_path: Path) -> None:
    frame = _make_full_day_1min_frame()
    source_path = tmp_path / "mnq_1min_databento.csv"
    _write_csv(source_path, frame)

    build_data = _load_build_data_module()
    row_count, first_ts, last_ts = build_data.resample_to_csv(source_path=source_path, timeframe="1h", output_dir=tmp_path)

    assert row_count == 24
    assert first_ts.strftime("%H:%M") == "00:00"
    assert last_ts.strftime("%H:%M") == "23:00"

    built = load_ohlcv("mnq", "1h", data_dir=tmp_path, session_only=False)
    assert built.index[0].strftime("%H:%M") == "00:00"
    assert built.index[1].strftime("%H:%M") == "01:00"
