from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from v3.config import DateWindow
from v3.regime_classifier import (
    RegimeFitResult,
    classify_regime_fit,
    regime_summary_dict,
    regime_summary_text,
)
from v3.trades import TradeResult

TZ = ZoneInfo("America/New_York")


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
