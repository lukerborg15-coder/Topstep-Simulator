"""Tests for VP Dual-Mode bug fixes: A4, A6, B1, B2, B3."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from v3.strategies import STRATEGIES, load_user_strategies

STRATEGY_KEY = "vp_dual_mode"
_DATE = "2024-01-02"
_TZ = "America/New_York"

# Minimal params: atr_period=3 so ATR is valid after 3 bars; cooldown=1 so A6 test
# exercises the cap rather than the cooldown.
_FAST_PARAMS = {
    "atr_period": 3,
    "level_tolerance_atr_mult": 0.5,
    "rejection_volume_mult": 0.8,
    "continuation_volume_mult": 1.5,
    "stop_buffer_atr_mult": 0.2,
    "per_level_cooldown_bars": 1,
    "target_r_multiple": 1.5,
    "retest_window_bars": 10,
    "atr_cap_points": 100.0,
    "naked_poc_lookback_sessions": 3,
    "vp_rows": 400,
    "va_pct": 0.70,
    "de_threshold": 0.35,
}


@pytest.fixture(autouse=True)
def load_strats():
    load_user_strategies()


def _ts(time_str: str) -> pd.Timestamp:
    return pd.Timestamp(f"{_DATE} {time_str}", tz=_TZ)


def _frame(rows: list[dict]) -> pd.DataFrame:
    data = {k: [r[k] for r in rows] for k in rows[0] if k != "ts"}
    index = pd.DatetimeIndex([r["ts"] for r in rows])
    return pd.DataFrame(data, index=index)


def _generate(frame: pd.DataFrame, **overrides) -> list:
    spec = STRATEGIES[STRATEGY_KEY]
    params = {**spec.default_params, **_FAST_PARAMS, **overrides}
    return spec.generate(frame, params)


def _filler(time_str: str, regime: str = "balance") -> dict:
    """Flat bar that won't trigger any setup on its own."""
    return {
        "ts": _ts(time_str),
        "open": 98.0, "high": 100.0, "low": 96.0, "close": 98.0, "volume": 1000,
        "pdVAH": 100.0, "pdVAL": 90.0, "pdVPOC": 95.0,
        "naked_pocs": [],
        "day_regime": regime,
        "is_post_holiday_session": False,
    }


# ---------------------------------------------------------------------------
# A4: Zero-volume bars skipped
# ---------------------------------------------------------------------------

def test_a4_zero_volume_bar_produces_no_signal():
    """A zero-volume bar that otherwise meets rejection criteria is skipped (A4)."""
    rows = [
        _filler("09:40"),
        _filler("09:45"),
        _filler("09:50"),
        # probe above pdVAH=100, snap back below — but volume=0
        {**_filler("09:55"), "high": 100.5, "low": 95.0, "close": 99.5, "open": 100.2, "volume": 0},
    ]
    assert _generate(_frame(rows)) == []


def test_a4_non_zero_volume_bar_can_produce_signal():
    """Same bar with low but non-zero volume does produce a signal (A4 gate not triggered)."""
    rows = [
        _filler("09:40"),
        _filler("09:45"),
        _filler("09:50"),
        # volume=400 < volume_median(1000) * rejection_volume_mult(0.8) = 800 → vol_ok
        {**_filler("09:55"), "high": 100.5, "low": 95.0, "close": 99.5, "open": 100.2, "volume": 400},
    ]
    assert len(_generate(_frame(rows))) == 1


# ---------------------------------------------------------------------------
# B3: Minimum stop distance gate
# ---------------------------------------------------------------------------

def test_b3_too_tight_stop_distance_rejected():
    """A rejection bar where dist(entry, stop) < 1.0 pt is silently dropped (B3)."""
    rows = [
        _filler("09:40"),
        _filler("09:45"),
        _filler("09:50"),
        # high=100.01 barely pokes pdVAH=100; stop ≈ 100.01 + 0.2*ATR; dist < 1.0
        {**_filler("09:55"), "high": 100.01, "low": 95.0, "close": 99.99, "open": 100.05, "volume": 400},
    ]
    assert _generate(_frame(rows)) == []


def test_b3_sufficient_stop_distance_passes():
    """Same setup with a wider probe produces a signal (stop dist > 1.0 pt)."""
    rows = [
        _filler("09:40"),
        _filler("09:45"),
        _filler("09:50"),
        # high=100.5; stop ≈ 100.5 + 0.2*ATR ≈ 101.4; dist ≈ 1.9 > 1.0
        {**_filler("09:55"), "high": 100.5, "low": 95.0, "close": 99.5, "open": 100.2, "volume": 400},
    ]
    assert len(_generate(_frame(rows))) == 1


# ---------------------------------------------------------------------------
# A6: Per-day continuation-signal cap (max 2 per level per day)
# ---------------------------------------------------------------------------

def _breakout_up(time_str: str) -> dict:
    """Bar that crosses pdVAH=100 with volume > 1.5x median — registers as continuation breakout."""
    return {
        "ts": _ts(time_str),
        "open": 99.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 2000,
        "pdVAH": 100.0, "pdVAL": 90.0, "pdVPOC": 95.0,
        "naked_pocs": [],
        "day_regime": "trend_up",
        "is_post_holiday_session": False,
    }


def _retest_up(time_str: str) -> dict:
    """Bar that touches pdVAH=100 (low=99.5) and snaps back above — triggers continuation entry."""
    return {
        "ts": _ts(time_str),
        "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000,
        "pdVAH": 100.0, "pdVAL": 90.0, "pdVPOC": 95.0,
        "naked_pocs": [],
        "day_regime": "trend_up",
        "is_post_holiday_session": False,
    }


def _dip_below(time_str: str) -> dict:
    """Bar that retreats below pdVAH so the next breakout can be detected."""
    return {
        "ts": _ts(time_str),
        "open": 100.0, "high": 100.5, "low": 97.0, "close": 99.5, "volume": 1000,
        "pdVAH": 100.0, "pdVAL": 90.0, "pdVPOC": 95.0,
        "naked_pocs": [],
        "day_regime": "trend_up",
        "is_post_holiday_session": False,
    }


def test_a6_third_continuation_signal_blocked():
    """Only 2 continuation signals fire despite 3 valid breakout+retest pairs (A6 cap)."""
    rows = [
        _filler("09:40", regime="trend_up"),
        _filler("09:45", regime="trend_up"),
        _filler("09:50", regime="trend_up"),
        _breakout_up("09:55"),    # breakout 1
        _retest_up("10:00"),      # → signal 1 (count=1)
        _dip_below("10:05"),
        _breakout_up("10:10"),    # breakout 2
        _retest_up("10:15"),      # → signal 2 (count=2)
        _dip_below("10:20"),
        _breakout_up("10:25"),    # breakout 3 — blocked by A6 (count not < 2)
        _retest_up("10:30"),      # no signal (no active breakout)
    ]
    signals = _generate(_frame(rows))
    cont = [s for s in signals if s.metadata.get("mode") == "continuation"]
    assert len(cont) == 2, f"Expected 2 continuation signals; got {len(cont)}"


# ---------------------------------------------------------------------------
# B2: Retest must physically touch the broken level
# ---------------------------------------------------------------------------

def test_b2_retest_without_level_touch_fires_no_signal():
    """A retest bar whose low stays above pdVAH (no touch) does not produce a signal (B2)."""
    rows = [
        _filler("09:40", regime="trend_up"),
        _filler("09:45", regime="trend_up"),
        _filler("09:50", regime="trend_up"),
        _breakout_up("09:55"),
        # low=100.1 > breakout_level=100.0 → no touch
        {
            "ts": _ts("10:00"),
            "open": 100.3, "high": 101.0, "low": 100.1, "close": 100.5, "volume": 1000,
            "pdVAH": 100.0, "pdVAL": 90.0, "pdVPOC": 95.0,
            "naked_pocs": [], "day_regime": "trend_up", "is_post_holiday_session": False,
        },
    ]
    cont = [s for s in _generate(_frame(rows)) if s.metadata.get("mode") == "continuation"]
    assert cont == [], "No-touch retest should not produce a continuation signal (B2)"


def test_b2_retest_with_level_touch_fires_signal():
    """A retest that does touch pdVAH (low ≤ 100.0) and snaps back fires exactly one signal."""
    rows = [
        _filler("09:40", regime="trend_up"),
        _filler("09:45", regime="trend_up"),
        _filler("09:50", regime="trend_up"),
        _breakout_up("09:55"),
        _retest_up("10:00"),   # low=99.5 ≤ 100 ✓
    ]
    cont = [s for s in _generate(_frame(rows)) if s.metadata.get("mode") == "continuation"]
    assert len(cont) == 1


# ---------------------------------------------------------------------------
# B1: Continuation stop must be on the correct side of entry
# ---------------------------------------------------------------------------

def test_b1_long_continuation_stop_below_entry():
    """Every long continuation signal has stop strictly below entry (B1)."""
    rows = [
        _filler("09:40", regime="trend_up"),
        _filler("09:45", regime="trend_up"),
        _filler("09:50", regime="trend_up"),
        _breakout_up("09:55"),
        _retest_up("10:00"),
    ]
    for s in _generate(_frame(rows)):
        if s.metadata.get("mode") == "continuation" and s.direction == "long":
            assert s.stop < s.entry, f"Long stop {s.stop:.4f} not below entry {s.entry:.4f} (B1)"


def test_b1_short_continuation_stop_above_entry():
    """Every short continuation signal has stop strictly above entry (B1)."""
    rows = [
        _filler("09:40", regime="trend_down"),
        _filler("09:45", regime="trend_down"),
        _filler("09:50", regime="trend_down"),
        # Breakdown: prior_close=98 ≥ pdVAL=90; close=89 < 90; vol=2000 > 1500
        {
            "ts": _ts("09:55"),
            "open": 91.0, "high": 93.0, "low": 88.0, "close": 89.0, "volume": 2000,
            "pdVAH": 100.0, "pdVAL": 90.0, "pdVPOC": 95.0,
            "naked_pocs": [], "day_regime": "trend_down", "is_post_holiday_session": False,
        },
        # Short retest: high=90.5 ≥ pdVAL=90; close=89.5 < 90; close < open
        {
            "ts": _ts("10:00"),
            "open": 90.5, "high": 90.5, "low": 88.5, "close": 89.5, "volume": 1000,
            "pdVAH": 100.0, "pdVAL": 90.0, "pdVPOC": 95.0,
            "naked_pocs": [], "day_regime": "trend_down", "is_post_holiday_session": False,
        },
    ]
    for s in _generate(_frame(rows)):
        if s.metadata.get("mode") == "continuation" and s.direction == "short":
            assert s.stop > s.entry, f"Short stop {s.stop:.4f} not above entry {s.entry:.4f} (B1)"
