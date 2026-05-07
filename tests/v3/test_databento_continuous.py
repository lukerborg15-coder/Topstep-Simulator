from __future__ import annotations

import pandas as pd
import pytest

from v3.databento import (
    build_continuous_ohlcv,
    determine_continuous_contracts,
    validate_continuous_ohlcv,
)


def _sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_event": "2024-06-14T13:30:00Z",
                "symbol": "MESM4",
                "open": 5300.0,
                "high": 5301.0,
                "low": 5299.5,
                "close": 5300.5,
                "volume": 120,
            },
            {
                "ts_event": "2024-06-14T13:30:00Z",
                "symbol": "MESU4",
                "open": 5305.0,
                "high": 5306.0,
                "low": 5304.5,
                "close": 5305.5,
                "volume": 80,
            },
            {
                "ts_event": "2024-06-14T13:30:00Z",
                "symbol": "MESM4-MESU4",
                "open": 5.0,
                "high": 6.0,
                "low": 4.0,
                "close": 5.0,
                "volume": 9999,
            },
            {
                "ts_event": "2024-06-17T13:30:00Z",
                "symbol": "MESM4",
                "open": 5310.0,
                "high": 5311.0,
                "low": 5309.0,
                "close": 5310.5,
                "volume": 100,
            },
            {
                "ts_event": "2024-06-17T13:30:00Z",
                "symbol": "MESU4",
                "open": 5311.0,
                "high": 5312.0,
                "low": 5310.0,
                "close": 5311.5,
                "volume": 400,
            },
        ]
    )


def test_determine_continuous_contracts_rolls_forward_only() -> None:
    daily_volume = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-06-13").date(), "symbol": "MESM4", "volume": 900},
            {"date": pd.Timestamp("2024-06-13").date(), "symbol": "MESU4", "volume": 700},
            {"date": pd.Timestamp("2024-06-14").date(), "symbol": "MESM4", "volume": 850},
            {"date": pd.Timestamp("2024-06-14").date(), "symbol": "MESU4", "volume": 800},
            {"date": pd.Timestamp("2024-06-17").date(), "symbol": "MESM4", "volume": 500},
            {"date": pd.Timestamp("2024-06-17").date(), "symbol": "MESU4", "volume": 1400},
            {"date": pd.Timestamp("2024-06-18").date(), "symbol": "MESM4", "volume": 1600},
            {"date": pd.Timestamp("2024-06-18").date(), "symbol": "MESU4", "volume": 1200},
        ]
    )

    selected = determine_continuous_contracts(daily_volume)

    assert selected == {
        pd.Timestamp("2024-06-13").date(): "MESM4",
        pd.Timestamp("2024-06-14").date(): "MESM4",
        pd.Timestamp("2024-06-17").date(): "MESU4",
        pd.Timestamp("2024-06-18").date(): "MESU4",
    }


def test_build_continuous_ohlcv_excludes_spreads_and_uses_selected_contract() -> None:
    result = build_continuous_ohlcv(_sample_raw_frame(), root_symbol="MES")

    assert list(result["symbol"]) == ["MESM4", "MESU4"]
    assert list(result["volume"]) == [120, 400]
    assert result["datetime"].is_monotonic_increasing


def test_validate_continuous_ohlcv_rejects_spread_symbol() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-06-14 09:30:00-04:00"]),
            "symbol": ["MESM4-MESU4"],
            "open": [5.0],
            "high": [6.0],
            "low": [4.0],
            "close": [5.5],
            "volume": [10],
        }
    )

    with pytest.raises(ValueError, match="non-outright"):
        validate_continuous_ohlcv(frame, root_symbol="MES")


def test_validate_continuous_ohlcv_rejects_duplicate_timestamps() -> None:
    dt = pd.Timestamp("2024-06-14 09:30:00-04:00")
    frame = pd.DataFrame(
        {
            "datetime": [dt, dt],
            "symbol": ["MESM4", "MESM4"],
            "open": [5300.0, 5300.5],
            "high": [5301.0, 5301.5],
            "low": [5299.0, 5300.0],
            "close": [5300.5, 5301.0],
            "volume": [10, 12],
        }
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_continuous_ohlcv(frame, root_symbol="MES")


def test_validate_continuous_ohlcv_rejects_roll_gap() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-06-14 09:30:00-04:00",
                    "2024-06-17 09:30:00-04:00",
                ]
            ),
            "symbol": ["MESM4", "MESU4"],
            "open": [5300.0, 5600.0],
            "high": [5301.0, 5601.0],
            "low": [5299.0, 5599.0],
            "close": [5300.5, 5600.5],
            "volume": [10, 12],
        }
    )

    with pytest.raises(ValueError, match="rollover discontinuity"):
        validate_continuous_ohlcv(frame, root_symbol="MES", max_roll_gap_pct=0.03)
