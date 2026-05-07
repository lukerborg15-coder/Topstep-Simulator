"""Unit tests for VP Dual-Mode strategy."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from v3.user_strategies.vp_dual_mode import vp_dual_mode_generate
from v3.strategies import TradeSignal
from v3.config import EASTERN_TZ

@pytest.fixture
def base_df():
    """Create a base DataFrame with required columns."""
    index = pd.date_range("2024-01-02 09:30", "2024-01-02 16:00", freq="5min", tz=EASTERN_TZ)
    df = pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0,
        "pdVAH": 105.0,
        "pdVAL": 95.0,
        "pdVPOC": 100.0,
        "naked_pocs": [[] for _ in range(len(index))],
        "day_regime": "balance"
    }, index=index)
    return df

@pytest.fixture
def default_params():
    return {
        "level_tolerance_atr_mult": 0.3,
        "rejection_volume_mult": 0.8,
        "continuation_volume_mult": 1.5,
        "stop_buffer_atr_mult": 0.2,
        "naked_poc_lookback_sessions": 3,
        "per_level_cooldown_bars": 20,
        "target_r_multiple": 1.5,
        "retest_window_bars": 10,
        "atr_cap_points": 100.0,
        "atr_period": 14,
    }

def test_universal_gates_time(base_df, default_params):
    """Test that signals only fire between 09:40 and 15:30."""
    # Force a rejection long setup at 09:35
    base_df.loc["2024-01-02 09:35", "low"] = 94.9
    base_df.loc["2024-01-02 09:35", "close"] = 95.1
    base_df.loc["2024-01-02 09:35", "open"] = 95.0
    base_df.loc["2024-01-02 09:35", "volume"] = 100.0 # small vol to pass rejection gate
    
    signals = vp_dual_mode_generate(base_df, default_params)
    
    # 09:35 is before 09:40 gate
    assert not any(s.time.time() == pd.Timestamp("09:35").time() for s in signals)

def test_universal_gates_nan_vp_levels(base_df, default_params):
    """Test that bars with NaN pdVAH or pdVAL are skipped."""
    ts = "2024-01-02 11:00"
    # Set up a valid rejection long at pdVAL, then corrupt pdVAH
    base_df.loc[ts, "pdVAH"] = float("nan")
    base_df.loc[ts, "low"] = 94.8
    base_df.loc[ts, "open"] = 95.1
    base_df.loc[ts, "close"] = 95.3
    base_df.loc[ts, "volume"] = 10.0

    signals = vp_dual_mode_generate(base_df, default_params)
    assert not any(s.time.strftime("%H:%M") == "11:00" for s in signals)


def test_universal_gates_atr_cap(base_df, default_params):
    """Test ATR cap gate."""
    # High volatility bar at 10:00
    ts = "2024-01-02 10:00"
    base_df.loc[ts, "low"] = 94.9
    base_df.loc[ts, "close"] = 95.1
    base_df.loc[ts, "high"] = 250.0 # huge high to blow ATR
    base_df.loc[ts, "volume"] = 100.0
    
    # ATR cap is 100
    signals = vp_dual_mode_generate(base_df, default_params)
    assert not any(s.time.strftime("%H:%M") == "10:00" for s in signals)

def test_rejection_signals_long(base_df, default_params):
    """Test valid long rejection at pdVAL."""
    ts = "2024-01-02 11:00"
    # pdVAL is 95.0.
    # Long setup: bar low < level, bar close > level, bar close > bar open, low vol.
    # bar_low=94.5 ensures stop distance ≥ 1.0 pt after B3 gate (was 94.8 which gave dist≈0.95).
    base_df.loc[ts, "low"] = 94.5
    base_df.loc[ts, "open"] = 95.1
    base_df.loc[ts, "close"] = 95.3
    base_df.loc[ts, "high"] = 95.5
    base_df.loc[ts, "volume"] = 10.0  # small
    
    signals = vp_dual_mode_generate(base_df, default_params)
    
    relevant = [s for s in signals if s.time.strftime("%H:%M") == "11:00"]
    assert len(relevant) == 1
    assert relevant[0].direction == "long"
    assert relevant[0].metadata["mode"] == "rejection"
    assert relevant[0].metadata["level_name"] == "pdVAL"

def test_naked_poc_rejection(base_df, default_params):
    """Test rejection at a naked POC."""
    ts = "2024-01-02 11:30"
    base_df.at[ts, "naked_pocs"] = [102.0]
    
    # Short rejection at 102.0: bar high > 102.0, bar close < 102.0, bar close < bar open, low vol.
    # bar_high=102.5 ensures stop distance ≥ 1.0 pt after B3 gate (was 102.2 which gave dist≈0.9).
    base_df.loc[ts, "high"] = 102.5
    base_df.loc[ts, "open"] = 101.9
    base_df.loc[ts, "close"] = 101.7
    base_df.loc[ts, "low"] = 101.5
    base_df.loc[ts, "volume"] = 10.0
    
    signals = vp_dual_mode_generate(base_df, default_params)
    relevant = [s for s in signals if s.time.strftime("%H:%M") == "11:30"]
    assert len(relevant) == 1
    assert relevant[0].direction == "short"
    assert relevant[0].metadata["level_name"] == "naked_poc_102.0"

def test_continuation_flow(base_df, default_params):
    """Test Step 1 (breakout) and Step 2 (retest) for continuation."""
    base_df["day_regime"] = "trend_up"
    # pdVAH is 105.0. 
    
    # Step 1: Breakout at 11:00
    # Prior close <= 105, current close > 105, high vol, green candle.
    ts_break = "2024-01-02 11:00"
    base_df.loc["2024-01-02 10:55", "close"] = 104.9
    base_df.loc[ts_break, "open"] = 105.1
    base_df.loc[ts_break, "close"] = 105.5
    base_df.loc[ts_break, "high"] = 105.6
    base_df.loc[ts_break, "volume"] = 5000.0 # High vol
    
    # Ensure price stays above breakout level at 11:05 to avoid invalidation
    base_df.loc["2024-01-02 11:05", "open"] = 105.4
    base_df.loc["2024-01-02 11:05", "close"] = 105.3
    base_df.loc["2024-01-02 11:05", "low"] = 105.1
    
    # Step 2: Retest at 11:10 (within 10 bar window)
    # bar low must reach breakout_level=105.0 (B2 touch requirement).
    # bar_close=105.7 gives stop dist ≥ 1.0 pt after B3 gate (was 105.4 with low=105.05).
    ts_retest = "2024-01-02 11:10"
    base_df.loc[ts_retest, "open"] = 105.2
    base_df.loc[ts_retest, "low"] = 104.9   # touches 105.0 level
    base_df.loc[ts_retest, "high"] = 105.8
    base_df.loc[ts_retest, "close"] = 105.7
    base_df.loc[ts_retest, "volume"] = 100.0
    
    signals = vp_dual_mode_generate(base_df, default_params)
    
    relevant = [s for s in signals if s.time.strftime("%H:%M") == "11:10"]
    assert len(relevant) == 1
    assert relevant[0].direction == "long"
    assert relevant[0].metadata["mode"] == "continuation"
    assert relevant[0].metadata["level_name"] == "pdVAH"

def test_continuation_invalidation(base_df, default_params):
    """Test that breakout resets if price closes on wrong side."""
    base_df["day_regime"] = "trend_up"
    
    # Step 1: Breakout at 11:00
    ts_break = "2024-01-02 11:00"
    base_df.loc["2024-01-02 10:55", "close"] = 104.9
    base_df.loc[ts_break, "open"] = 105.1
    base_df.loc[ts_break, "close"] = 105.5
    base_df.loc[ts_break, "volume"] = 5000.0
    
    # Invalidation at 11:05: close < 105.0
    base_df.loc["2024-01-02 11:05", "close"] = 104.8
    
    # Attempted retest at 11:10 should fail
    ts_retest = "2024-01-02 11:10"
    base_df.loc[ts_retest, "open"] = 105.2
    base_df.loc[ts_retest, "low"] = 105.05
    base_df.loc[ts_retest, "close"] = 105.4
    
    signals = vp_dual_mode_generate(base_df, default_params)
    assert not any(s.time.strftime("%H:%M") == "11:10" for s in signals)

def test_cooldown(base_df, default_params):
    """Test that level is skipped within cooldown period."""
    # bar_low=94.5 ensures stop distance ≥ 1.0 pt (B3) so ts1 actually fires and registers cooldown.
    ts1 = "2024-01-02 11:00"
    base_df.loc[ts1, "low"] = 94.5
    base_df.loc[ts1, "open"] = 95.1
    base_df.loc[ts1, "close"] = 95.3
    base_df.loc[ts1, "volume"] = 10.0

    # Identical setup at 11:10 (only 2 bars later) — blocked by per_level_cooldown_bars=20
    ts2 = "2024-01-02 11:10"
    base_df.loc[ts2, "low"] = 94.5
    base_df.loc[ts2, "open"] = 95.1
    base_df.loc[ts2, "close"] = 95.3
    base_df.loc[ts2, "volume"] = 10.0
    
    signals = vp_dual_mode_generate(base_df, default_params)
    
    # Should only have the first signal
    assert len(signals) == 1
    assert signals[0].time.strftime("%H:%M") == "11:00"

def test_volume_median_logic(base_df, default_params):
    """Verify 10-bar rolling median logic for same-day bars only."""
    # Day 1: High volumes
    # Day 2: Rejection setup should be compared against Day 2 volumes only.
    
    idx2 = pd.date_range("2024-01-03 09:30", "2024-01-03 16:00", freq="5min", tz=EASTERN_TZ)
    df2 = pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0, # Very low volume on Day 2
        "pdVAH": 105.0,
        "pdVAL": 95.0,
        "pdVPOC": 100.0,
        "naked_pocs": [[] for _ in range(len(idx2))],
        "day_regime": "balance"
    }, index=idx2)
    
    combined_df = pd.concat([base_df, df2])
    
    # On Day 2 at 11:00, setup with volume 15.0
    # Day 2 median of prior 09:30-10:55 is 10.0.
    # 15.0 > 10.0 * 0.8 (rejection_volume_mult) -> SHOULD FAIL if median works correctly.
    ts = "2024-01-03 11:00"
    combined_df.loc[ts, "low"] = 94.8
    combined_df.loc[ts, "open"] = 95.1
    combined_df.loc[ts, "close"] = 95.3
    combined_df.loc[ts, "volume"] = 15.0 
    
    signals = vp_dual_mode_generate(combined_df, default_params)
    
    # Check if Day 2 11:00 signal exists
    day2_signals = [s for s in signals if s.time.date() == pd.Timestamp("2024-01-03").date()]
    assert not any(s.time.strftime("%H:%M") == "11:00" for s in day2_signals)
