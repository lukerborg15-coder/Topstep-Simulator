from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from v3.config import DateWindow, DEFAULT_DATA_DIR
from v3.regime_classifier import (
    RegimeFitResult,
    attach_day_regimes,
    assert_regime_mutual_exclusivity,
    audit_regime_distribution,
    classify_day_regimes,
    classify_regime_fit,
    regime_summary_dict,
    regime_summary_text,
)
from v3.trades import TradeResult

TZ = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Legacy test helpers
# ---------------------------------------------------------------------------


def _ts(day: int, hour: int = 10, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(2024, 6, day, hour, minute, tz=TZ)


def _make_frame_with_vol_patches(n_calm: int = 60, n_vol: int = 60) -> pd.DataFrame:
    """Build intraday OHLCV: first half = calm (low stdev), second half = volatile."""
    rng = np.random.default_rng(42)
    n_total = n_calm + n_vol
    idx = pd.date_range("2024-06-03 09:30", periods=n_total, freq="5min", tz=TZ)

    # Calm: small moves; volatile: large moves
    calm_returns = rng.normal(0, 0.0005, n_calm)
    vol_returns = rng.normal(0, 0.005, n_vol)
    log_returns = np.concatenate([calm_returns, vol_returns])

    close = 18_000.0 * np.exp(np.cumsum(log_returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(1.0, 3.0, n_total)
    low = np.minimum(open_, close) - rng.uniform(1.0, 3.0, n_total)
    volume = rng.integers(500, 2000, n_total)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_trade(entry_ts: pd.Timestamp, net: float, r: float = 0.5) -> TradeResult:
    exit_ts = entry_ts + pd.Timedelta(minutes=5)
    return TradeResult(
        strategy="test",
        entry_time=entry_ts,
        exit_time=exit_ts,
        direction="long",
        entry=0.0,
        stop=0.0,
        target=1.0,
        exit=0.0,
        contracts=1,
        gross_pnl=net,
        commission=0.0,
        net_pnl=net,
        r_multiple=r,
        exit_reason="test",
        bars_held=1,
        params={},
    )


# ---------------------------------------------------------------------------
# Legacy tests — preserved exactly
# ---------------------------------------------------------------------------


def test_classify_returns_regime_fit_result():
    frame = _make_frame_with_vol_patches()
    window = DateWindow("test", "2024-06-03", "2024-06-30")
    trades = [_make_trade(frame.index[i], 50.0) for i in range(10, 60, 2)]
    result = classify_regime_fit(frame, trades, window, vol_window=20)
    assert isinstance(result, RegimeFitResult)
    assert result.verdict in {"prefers_calm", "prefers_volatile", "mixed", "insufficient_data"}


def test_empty_trades_returns_insufficient_data():
    frame = _make_frame_with_vol_patches()
    window = DateWindow("test", "2024-06-03", "2024-06-30")
    result = classify_regime_fit(frame, [], window)
    assert result.verdict == "insufficient_data"
    assert result.total_trades == 0


def test_too_few_trades_per_bucket_returns_insufficient_data():
    frame = _make_frame_with_vol_patches()
    window = DateWindow("test", "2024-06-03", "2024-06-30")
    # Only 5 trades total — below n_min=10 per bucket
    trades = [_make_trade(frame.index[i * 5 + 25], 100.0) for i in range(5)]
    result = classify_regime_fit(frame, trades, window, vol_window=20, n_min=10)
    assert result.verdict == "insufficient_data"


def test_calm_preferring_strategy():
    """Trades in calm bars should yield prefers_calm when calm expectancy much higher."""
    frame = _make_frame_with_vol_patches(n_calm=120, n_vol=120)
    window = DateWindow("test", "2024-06-03", "2024-06-30")

    # Force trades in calm zone (first 120 bars) to win big, volatile zone to lose
    calm_trades = [_make_trade(frame.index[i], 500.0, r=5.0) for i in range(25, 100, 3)]
    vol_trades = [_make_trade(frame.index[i], -500.0, r=-5.0) for i in range(125, 220, 3)]
    trades = calm_trades + vol_trades

    result = classify_regime_fit(frame, trades, window, vol_window=10, n_min=5, expectancy_epsilon=1.0)
    # With extreme pnl difference, should prefer calm
    assert result.verdict in {"prefers_calm", "mixed"}


def test_regime_summary_dict_has_required_keys():
    frame = _make_frame_with_vol_patches()
    window = DateWindow("test", "2024-06-03", "2024-06-30")
    trades = [_make_trade(frame.index[i], 50.0) for i in range(10, 100, 2)]
    result = classify_regime_fit(frame, trades, window, vol_window=20)
    d = regime_summary_dict(result)
    assert "regime_verdict" in d
    assert "calm" in d
    assert "volatile" in d
    assert "total_trades" in d


def test_regime_summary_text_contains_verdict():
    frame = _make_frame_with_vol_patches()
    window = DateWindow("test", "2024-06-03", "2024-06-30")
    trades = [_make_trade(frame.index[i], 50.0) for i in range(10, 100, 2)]
    result = classify_regime_fit(frame, trades, window, vol_window=20)
    text = regime_summary_text(result)
    assert result.verdict.replace("_", " ").upper() in text.upper()
    assert "Calm" in text
    assert "Volatile" in text


def test_window_name_propagated():
    frame = _make_frame_with_vol_patches()
    window = DateWindow("holdout", "2024-06-03", "2024-06-30")
    result = classify_regime_fit(frame, [], window)
    assert result.window_name == "holdout"


# ---------------------------------------------------------------------------
# New per-day labeler helpers
# ---------------------------------------------------------------------------

_DE_LOOKBACK = 3  # small lookback so tests don't need many prior bars


def _make_two_day_frames(
    day2_940_close: float,
    or_high_val: float,
    or_low_val: float,
    day1_close: float = 100.0,
    day2_930_close: float = 100.0,
    day2_935_close: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal exec_5min (2 days) + raw_1min OR bars for regime labeler tests.

    With _DE_LOOKBACK=3, signed DE at day2 9:40 depends on:
        close values at: day1 bar[-1], day2 bars[0,1,2].

    day2 bar[2] == day2_940_close is the classification bar.
    All day1 bars are flat at day1_close (default 100).
    """
    # Day 1: 79 bars 09:30–16:00, all closes at day1_close
    d1_idx = pd.date_range("2024-06-03 09:30", periods=79, freq="5min", tz=TZ)
    d1_closes = np.full(79, day1_close)

    # Day 2: 79 bars 09:30–16:00; control first three
    d2_idx = pd.date_range("2024-06-04 09:30", periods=79, freq="5min", tz=TZ)
    d2_closes = np.full(79, day2_940_close)
    d2_closes[0] = day2_930_close
    d2_closes[1] = day2_935_close
    # d2_closes[2] stays at day2_940_close

    full_idx = d1_idx.append(d2_idx)
    full_closes = np.concatenate([d1_closes, d2_closes])

    exec_5min = pd.DataFrame(
        {
            "open": full_closes,
            "high": full_closes + 0.5,
            "low": full_closes - 0.5,
            "close": full_closes,
            "volume": np.full(len(full_idx), 1000),
        },
        index=full_idx,
    )

    # raw_1min: 10 bars 09:30–09:39 on day 2 (the opening range)
    or_idx = pd.date_range("2024-06-04 09:30", periods=10, freq="1min", tz=TZ)
    or_mid = (or_high_val + or_low_val) / 2
    raw_1min = pd.DataFrame(
        {
            "open": np.full(10, or_mid),
            "high": np.full(10, or_high_val),
            "low": np.full(10, or_low_val),
            "close": np.full(10, or_mid),
            "volume": np.full(10, 500),
        },
        index=or_idx,
    )

    return exec_5min, raw_1min


# ---------------------------------------------------------------------------
# New labeler tests
# ---------------------------------------------------------------------------


def test_trend_up_label():
    """Price above OR high, DE strongly positive → trend_up."""
    # day1_close=100, day2 9:40 close=115. or_high=101, or_low=99.
    # DE at 9:40 with lookback=3: direction=|115-100|=15, path=|0|+|0|+|15|=15 → DE=1.0 signed +1
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=115.0, or_high_val=101.0, or_low_val=99.0
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)
    ts_940 = pd.Timestamp("2024-06-04 09:40", tz=TZ)
    assert regimes.loc[ts_940] == "trend_up"


def test_trend_down_label():
    """Price below OR low, DE strongly negative → trend_down."""
    # day2 9:40 close=85. or_high=101, or_low=99.
    # DE: direction=|85-100|=15, path=15 → -1.0
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=85.0, or_high_val=101.0, or_low_val=99.0
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)
    ts_940 = pd.Timestamp("2024-06-04 09:40", tz=TZ)
    assert regimes.loc[ts_940] == "trend_down"


def test_balance_label():
    """Price inside OR, near-zero DE → balance."""
    # day2: 9:30=100, 9:35=101, 9:40=100. day1_close=100.
    # DE: direction=|100-100|=0, path=|0|+|1|+|1|=2 → 0.0; sign=0 → signed_de=0.0
    # or: high=101.5, low=98.5 → price 100 is inside
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=100.0,
        or_high_val=101.5,
        or_low_val=98.5,
        day2_935_close=101.0,
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)
    ts_940 = pd.Timestamp("2024-06-04 09:40", tz=TZ)
    assert regimes.loc[ts_940] == "balance"


def test_unclear_broke_range_no_de_confirm():
    """Price above OR high but DE below threshold → unclear."""
    # day2: 9:30=100, 9:35=104, 9:40=102. or_high=101.
    # DE: direction=|102-100|=2, path=|0|+|4|+|2|=6 → 0.333; signed +0.333 < 0.35 threshold
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=102.0,
        or_high_val=101.0,
        or_low_val=98.0,
        day2_935_close=104.0,
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)
    ts_940 = pd.Timestamp("2024-06-04 09:40", tz=TZ)
    assert regimes.loc[ts_940] == "unclear"


def test_forward_fill_after_940():
    """All bars from 9:40 ET onward on the same day carry the same regime label."""
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=115.0, or_high_val=101.0, or_low_val=99.0
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)

    d2_mask = regimes.index.normalize() == pd.Timestamp("2024-06-04", tz=TZ)
    d2_regimes = regimes[d2_mask]
    from_940 = d2_regimes.between_time("09:40", "16:00", inclusive="both")

    assert (from_940 == "trend_up").all(), f"Not all bars forward-filled: {from_940.unique()}"


def test_pre_940_bars_are_unclear():
    """Bars before 9:40 ET get 'unclear' regardless of the day's regime."""
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=115.0, or_high_val=101.0, or_low_val=99.0
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)

    d2_mask = regimes.index.normalize() == pd.Timestamp("2024-06-04", tz=TZ)
    d2_regimes = regimes[d2_mask]
    before_940 = d2_regimes.between_time("09:30", "09:35", inclusive="both")

    assert (before_940 == "unclear").all(), f"Pre-9:40 bars not unclear: {before_940.unique()}"


def test_mutual_exclusivity_valid_series():
    """assert_regime_mutual_exclusivity does not raise on a correctly forward-filled series."""
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=115.0, or_high_val=101.0, or_low_val=99.0
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)
    assert_regime_mutual_exclusivity(regimes)  # should not raise


def test_mutual_exclusivity_raises_on_broken_series():
    """assert_regime_mutual_exclusivity raises ValueError when a day has mixed labels post-9:40."""
    exec_5min, raw_1min = _make_two_day_frames(
        day2_940_close=115.0, or_high_val=101.0, or_low_val=99.0
    )
    regimes = classify_day_regimes(exec_5min, raw_1min, de_lookback=_DE_LOOKBACK)

    # Deliberately corrupt: inject a different label at some bar after 9:40 on day 2
    ts_break = pd.Timestamp("2024-06-04 10:00", tz=TZ)
    regimes.loc[ts_break] = "balance"

    with pytest.raises(ValueError, match="2024-06-04"):
        assert_regime_mutual_exclusivity(regimes)


_DATA_5MIN = DEFAULT_DATA_DIR / "mnq_5min_databento.csv"
_DATA_1MIN = DEFAULT_DATA_DIR / "mnq_1min_databento.csv"


@pytest.mark.skipif(
    not (_DATA_5MIN.exists() and _DATA_1MIN.exists()),
    reason="real MNQ data not available",
)
def test_full_history_smoke():
    """Load real MNQ data, run attach_day_regimes, verify no NaN and valid labels."""
    from v3.data import load_ohlcv

    exec_5min = load_ohlcv("mnq", "5min")
    raw_1min = load_ohlcv("mnq", "1min")

    result = attach_day_regimes(exec_5min, raw_1min)

    assert "day_regime" in result.columns
    assert result["day_regime"].isna().sum() == 0, "NaN labels found"
    assert set(result["day_regime"].unique()).issubset(
        {"trend_up", "trend_down", "balance", "unclear"}
    ), f"Unexpected labels: {result['day_regime'].unique()}"
