from __future__ import annotations

import numpy as np
import pandas as pd

from v3.volume_profile import attach_volume_profile, compute_naked_pocs, compute_session_vp

EASTERN_TZ = "America/New_York"


def _rth_1min(date_str: str, base_price: float = 100.0, volume: int = 1) -> pd.DataFrame:
    start = pd.Timestamp(f"{date_str} 09:30", tz=EASTERN_TZ)
    end = pd.Timestamp(f"{date_str} 15:59", tz=EASTERN_TZ)
    idx = pd.date_range(start, end, freq="1min")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.full(n, base_price),
            "high": np.full(n, base_price + 0.5),
            "low": np.full(n, base_price - 0.5),
            "close": np.full(n, base_price),
            "volume": np.full(n, float(volume)),
        },
        index=idx,
    )


def _overnight_1min(
    from_date: str, to_date: str, base_price: float = 100.0, volume: int = 1
) -> pd.DataFrame:
    start = pd.Timestamp(f"{from_date} 16:01", tz=EASTERN_TZ)
    end = pd.Timestamp(f"{to_date} 09:29", tz=EASTERN_TZ)
    if start >= end:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    idx = pd.date_range(start, end, freq="1min")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.full(n, base_price),
            "high": np.full(n, base_price + 0.5),
            "low": np.full(n, base_price - 0.5),
            "close": np.full(n, base_price),
            "volume": np.full(n, float(volume)),
        },
        index=idx,
    )


def _rth_5min(date_str: str, base_price: float = 100.0) -> pd.DataFrame:
    start = pd.Timestamp(f"{date_str} 09:30", tz=EASTERN_TZ)
    end = pd.Timestamp(f"{date_str} 15:55", tz=EASTERN_TZ)
    idx = pd.date_range(start, end, freq="5min")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.full(n, base_price),
            "high": np.full(n, base_price + 0.5),
            "low": np.full(n, base_price - 0.5),
            "close": np.full(n, base_price),
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )


def _full_session_1min(session_date: str, base_price: float = 100.0, volume: int = 1) -> pd.DataFrame:
    session_day = pd.Timestamp(session_date)
    evening_start = (session_day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    evening = pd.date_range(
        pd.Timestamp(f"{evening_start} 18:00", tz=EASTERN_TZ),
        pd.Timestamp(f"{session_date} 16:59", tz=EASTERN_TZ),
        freq="1min",
    )
    n = len(evening)
    return pd.DataFrame(
        {
            "open": np.full(n, base_price),
            "high": np.full(n, base_price + 0.5),
            "low": np.full(n, base_price - 0.5),
            "close": np.full(n, base_price),
            "volume": np.full(n, float(volume)),
        },
        index=evening,
    )


def _session_key(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str)


def test_rth_mode_known_vp() -> None:
    idx_d1_low = pd.date_range(
        pd.Timestamp("2024-01-02 09:30", tz=EASTERN_TZ), periods=370, freq="1min"
    )
    idx_d1_high = pd.date_range(
        pd.Timestamp("2024-01-02 15:48", tz=EASTERN_TZ), periods=10, freq="1min"
    )
    d1_low = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0},
        index=idx_d1_low,
    )
    d1_high = pd.DataFrame(
        {"open": 110.0, "high": 110.5, "low": 109.5, "close": 110.0, "volume": 1000.0},
        index=idx_d1_high,
    )
    d2 = _rth_1min("2024-01-03", base_price=105.0)
    raw_1min = pd.concat([d1_low, d1_high, d2]).sort_index()

    session_vp = compute_session_vp(raw_1min, session_mode="rth")

    row = session_vp.loc[_session_key("2024-01-03")]
    assert abs(row["pdVPOC"] - 110.0) < 0.5
    assert row["pdVAH"] >= row["pdVPOC"]
    assert row["pdVAL"] <= row["pdVPOC"]
    assert pd.isna(session_vp.loc[_session_key("2024-01-02"), "pdVPOC"])


def test_rth_mode_value_area_pct() -> None:
    idx_d1_low = pd.date_range(
        pd.Timestamp("2024-01-02 09:30", tz=EASTERN_TZ), periods=370, freq="1min"
    )
    idx_d1_high = pd.date_range(
        pd.Timestamp("2024-01-02 15:48", tz=EASTERN_TZ), periods=10, freq="1min"
    )
    d1_low = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0},
        index=idx_d1_low,
    )
    d1_high = pd.DataFrame(
        {"open": 110.0, "high": 110.5, "low": 109.5, "close": 110.0, "volume": 1000.0},
        index=idx_d1_high,
    )
    d2 = _rth_1min("2024-01-03", base_price=105.0)
    raw_1min = pd.concat([d1_low, d1_high, d2]).sort_index()

    session_vp = compute_session_vp(raw_1min, session_mode="rth")
    row = session_vp.loc[_session_key("2024-01-03")]

    d1_bars = raw_1min[raw_1min.index.normalize() == pd.Timestamp("2024-01-02", tz=EASTERN_TZ)]
    total_vol = float(d1_bars["volume"].sum())
    va_vol = float(d1_bars[d1_bars["close"].between(row["pdVAL"], row["pdVAH"])]["volume"].sum())
    assert va_vol / total_vol >= 0.70


def test_rth_mode_monday_aggregation_excludes_monday_rth() -> None:
    fri = _rth_1min("2024-01-05", base_price=200.0, volume=1000)
    overnight = _overnight_1min("2024-01-05", "2024-01-08", base_price=200.0, volume=1)
    mon = _rth_1min("2024-01-08", base_price=300.0, volume=1000)

    raw_1min = pd.concat([fri, overnight, mon]).sort_index()
    session_vp = compute_session_vp(raw_1min, session_mode="rth")

    row = session_vp.loc[_session_key("2024-01-08")]
    assert abs(row["pdVPOC"] - 200.0) < 1.0
    assert abs(row["pdVPOC"] - 300.0) > 50.0


def test_rth_mode_no_lookahead() -> None:
    d1 = _rth_1min("2024-01-02", base_price=100.0, volume=100)
    d2_1min = _rth_1min("2024-01-03", base_price=150.0, volume=100)
    raw_1min = pd.concat([d1, d2_1min]).sort_index()

    d2_5min = _rth_5min("2024-01-03", base_price=150.0)
    result = attach_volume_profile(d2_5min, raw_1min, session_mode="rth")

    for _, row in result.iterrows():
        assert abs(row["pdVPOC"] - 100.0) < 1.0


def test_rth_mode_naked_poc_fill() -> None:
    day_labels = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    prices = [20.0, 30.0, 40.0, 50.0, 60.0]
    days_1min = pd.concat(
        [_rth_1min(d, base_price=p, volume=1000) for d, p in zip(day_labels, prices)]
    ).sort_index()

    session_vp = compute_session_vp(days_1min, session_mode="rth")
    d3_vpoc = session_vp.loc[_session_key("2024-01-04"), "pdVPOC"]

    d5_start = pd.Timestamp("2024-01-08 09:30", tz=EASTERN_TZ)
    d5_idx = pd.date_range(d5_start, periods=10, freq="5min")
    opens = [d3_vpoc - 1.0] + [100.0] * 9
    closes = [d3_vpoc + 1.0] + [100.0] * 9
    exec_5min = pd.DataFrame(
        {
            "open": opens,
            "high": [c + 0.5 for c in closes],
            "low": [o - 0.5 for o in opens],
            "close": closes,
            "volume": [100.0] * 10,
        },
        index=d5_idx,
    )

    naked = compute_naked_pocs(days_1min, exec_5min, session_vp, lookback_sessions=3, session_mode="rth")
    assert d3_vpoc not in naked.iloc[0]
    for i in range(1, 10):
        assert d3_vpoc not in naked.iloc[i]


def test_rth_mode_naked_poc_excludes_n_minus_1() -> None:
    day_labels = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    prices = [20.0, 30.0, 40.0, 50.0, 60.0]
    days_1min = pd.concat(
        [_rth_1min(d, base_price=p, volume=1000) for d, p in zip(day_labels, prices)]
    ).sort_index()

    session_vp = compute_session_vp(days_1min, session_mode="rth")
    d4_vpoc = session_vp.loc[_session_key("2024-01-05"), "pdVPOC"]

    d5_start = pd.Timestamp("2024-01-08 09:30", tz=EASTERN_TZ)
    d5_idx = pd.date_range(d5_start, periods=5, freq="5min")
    exec_5min = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [100.5] * 5,
            "low": [99.5] * 5,
            "close": [100.0] * 5,
            "volume": [100.0] * 5,
        },
        index=d5_idx,
    )

    naked = compute_naked_pocs(days_1min, exec_5min, session_vp, lookback_sessions=3, session_mode="rth")
    for pocs in naked:
        assert d4_vpoc not in pocs


def test_attachment_shape_rth_mode() -> None:
    from v3.volume_profile import VP_LEVEL_COLUMNS

    d1 = _rth_1min("2024-01-02", base_price=100.0, volume=100)
    d2_1min = _rth_1min("2024-01-03", base_price=105.0, volume=100)
    raw_1min = pd.concat([d1, d2_1min]).sort_index()

    exec_5min = _rth_5min("2024-01-03", base_price=105.0)
    result = attach_volume_profile(exec_5min, raw_1min, session_mode="rth")

    assert list(result.index) == list(exec_5min.index)
    assert len(result) == len(exec_5min)
    for col in VP_LEVEL_COLUMNS:
        assert col in result.columns
    assert all(isinstance(v, list) for v in result["naked_pocs"])


def test_full_mode_uses_prior_full_session_without_lookahead() -> None:
    prev_session = _full_session_1min("2024-01-08", base_price=100.0, volume=100)
    current_session_overnight = _full_session_1min("2024-01-09", base_price=150.0, volume=100)
    current_session_overnight = current_session_overnight.loc[
        current_session_overnight.index < pd.Timestamp("2024-01-09 09:30", tz=EASTERN_TZ)
    ]
    raw_1min = pd.concat([prev_session, current_session_overnight]).sort_index()

    exec_idx = pd.date_range(
        pd.Timestamp("2024-01-09 09:30", tz=EASTERN_TZ),
        periods=5,
        freq="5min",
    )
    exec_5min = pd.DataFrame(
        {
            "open": [150.0] * 5,
            "high": [150.5] * 5,
            "low": [149.5] * 5,
            "close": [150.0] * 5,
            "volume": [100.0] * 5,
        },
        index=exec_idx,
    )

    result = attach_volume_profile(exec_5min, raw_1min)
    assert (result["pdVPOC"] - 100.0).abs().max() < 1.0


def test_rth_mode_does_not_mark_normal_monday_as_post_holiday() -> None:
    fri = _rth_1min("2024-01-05", base_price=200.0, volume=1000)
    overnight = _overnight_1min("2024-01-05", "2024-01-08", base_price=200.0, volume=1)
    mon = _rth_1min("2024-01-08", base_price=300.0, volume=1000)

    raw_1min = pd.concat([fri, overnight, mon]).sort_index()
    session_vp = compute_session_vp(raw_1min, session_mode="rth")

    assert bool(session_vp.loc[_session_key("2024-01-08"), "is_post_holiday_session"]) is False
