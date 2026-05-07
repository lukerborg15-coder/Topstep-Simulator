"""VP Dual-Mode strategy — Rejection (Balance) and Continuation (Trend).

Integrates Volume Profile levels (pdVAH, pdVAL, pdVPOC, naked_pocs) with
intraday regime classification.

VP attachment can be RTH-based or full-session-based. This strategy keeps its
own entry window restrictions (09:40-15:30 ET) regardless of which VP session
mode produced the levels.

Sanity Check Stats (WF1 Train):
[Sanity check output will be pasted here after running scripts/vp_sanity_check.py]
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from v3.config import SESSION_END
from v3.indicators import atr
from v3.strategies import StrategySpec, TradeSignal, _append_signal, register_strategy

STRATEGY_KEY = "vp_dual_mode"
MIN_STOP_DISTANCE_POINTS = 1.0  # B3: below this, wins are net losers after costs


def vp_dual_mode_generate(df: pd.DataFrame, params: dict) -> list:
    atr_period = int(params.get("atr_period", 14))
    level_tolerance_atr_mult = float(params.get("level_tolerance_atr_mult", 0.3))
    rejection_volume_mult = float(params.get("rejection_volume_mult", 0.8))
    continuation_volume_mult = float(params.get("continuation_volume_mult", 1.5))
    stop_buffer_atr_mult = float(params.get("stop_buffer_atr_mult", 0.2))
    per_level_cooldown_bars = int(params.get("per_level_cooldown_bars", 20))
    target_r_multiple = float(params.get("target_r_multiple", 1.5))
    retest_window_bars = int(params.get("retest_window_bars", 10))
    atr_cap_points = float(params.get("atr_cap_points", 100.0))

    atr_series = atr(df, atr_period)

    # State tracking
    level_last_trade_idx: dict[str, int] = {}
    breakout_count_today: dict[str, int] = {}  # A6: per-day cap (max 2 signals per level)

    # Continuation state (reset per day)
    breakout_active = False
    breakout_bar_idx = -1
    breakout_level = 0.0
    breakout_direction = ""

    # Diagnostic counters (B1, B3)
    discarded_stop_side_count = 0
    too_tight_count = 0

    signals: list = []

    session_end_time = pd.Timestamp(SESSION_END).time()

    df_dates = df.index.normalize()

    for i in range(1, len(df)):
        ts = df.index[i]
        row = df.iloc[i]

        # Day reset
        if df_dates[i] != df_dates[i - 1]:
            breakout_active = False
            level_last_trade_idx.clear()   # A1: VP levels change each day; cross-day cooldown on a changed level is spurious
            breakout_count_today.clear()    # A6: reset per-day cap
            _regime_at_day_start = str(row.get("day_regime", "unclear"))
            _is_rejection_day = _regime_at_day_start == "balance"
            _is_continuation_day = _regime_at_day_start in ("trend_up", "trend_down")
            assert not (_is_rejection_day and _is_continuation_day), (
                f"Regime mutual exclusivity violated on {df_dates[i].date()}"
            )

        # --- Universal hard gates ---
        if not (ts.time() >= pd.Timestamp("09:40").time() and ts.time() <= pd.Timestamp("15:30").time()):
            continue

        atr_val = float(atr_series.iloc[i])
        if not np.isfinite(atr_val) or atr_val > atr_cap_points:
            continue

        pdVAH = float(row.get("pdVAH", np.nan))
        pdVAL = float(row.get("pdVAL", np.nan))
        if not np.isfinite(pdVAH) or not np.isfinite(pdVAL):
            continue

        day_regime = str(row.get("day_regime", "unclear"))
        if day_regime not in ("trend_up", "trend_down", "balance"):
            continue

        # A4: skip zero-volume bars entirely (data anomaly, not a valid signal candidate)
        bar_volume = float(row["volume"])
        if bar_volume == 0:
            continue

        bar_open = float(row["open"])
        bar_close = float(row["close"])
        bar_high = float(row["high"])
        bar_low = float(row["low"])

        # Volume median: same-day RTH bars, last 10 or fewer
        day_start_idx = i
        while day_start_idx > 0 and df_dates[day_start_idx - 1] == df_dates[i]:
            day_start_idx -= 1

        same_day_prior_volumes = df["volume"].iloc[day_start_idx:i]
        if len(same_day_prior_volumes) < 3:
            volume_median = np.nan
        else:
            lookback = min(10, len(same_day_prior_volumes))
            volume_median = float(same_day_prior_volumes.iloc[-lookback:].median())

        is_post_holiday = bool(row.get("is_post_holiday_session", False))

        # --- Rejection Mode ---
        if day_regime == "balance":
            naked_pocs = row.get("naked_pocs", [])
            eligible_levels = [("pdVAH", pdVAH), ("pdVAL", pdVAL)]
            if naked_pocs:
                sorted_naked = sorted(naked_pocs, key=lambda p: abs(p - bar_close))
                for p in sorted_naked:
                    eligible_levels.append((f"naked_poc_{p}", p))

            signal_this_bar_long = False
            signal_this_bar_short = False

            for level_id, level in eligible_levels:
                if signal_this_bar_long and signal_this_bar_short:
                    break

                if i - level_last_trade_idx.get(level_id, -1000) < per_level_cooldown_bars:
                    continue

                vol_ok = True
                if not np.isnan(volume_median):
                    if bar_volume >= volume_median * rejection_volume_mult:
                        vol_ok = False

                # Short setup (probe above level, snap back below)
                if not signal_this_bar_short and level >= bar_close:
                    if (bar_high > level
                            and (bar_high - level) <= level_tolerance_atr_mult * atr_val
                            and bar_close < level
                            and vol_ok
                            and bar_close < bar_open):
                        entry = bar_close
                        stop = bar_high + (stop_buffer_atr_mult * atr_val)
                        dist = abs(entry - stop)
                        if dist < MIN_STOP_DISTANCE_POINTS:  # B3
                            too_tight_count += 1
                        else:
                            target = entry - (dist * target_r_multiple)
                            metadata = {
                                "mode": "rejection",
                                "level_name": level_id,
                                "level_price": float(level),
                                "regime": day_regime,
                                "is_post_holiday_session": is_post_holiday,
                            }
                            old_len = len(signals)
                            _append_signal(signals, ts, -1, entry, stop, target, STRATEGY_KEY, params, metadata)
                            if len(signals) > old_len:
                                level_last_trade_idx[level_id] = i
                                signal_this_bar_short = True

                # Long setup (probe below level, snap back above)
                if not signal_this_bar_long and level <= bar_close:
                    if (bar_low < level
                            and (level - bar_low) <= level_tolerance_atr_mult * atr_val
                            and bar_close > level
                            and vol_ok
                            and bar_close > bar_open):
                        entry = bar_close
                        stop = bar_low - (stop_buffer_atr_mult * atr_val)
                        dist = abs(entry - stop)
                        if dist < MIN_STOP_DISTANCE_POINTS:  # B3
                            too_tight_count += 1
                        else:
                            target = entry + (dist * target_r_multiple)
                            metadata = {
                                "mode": "rejection",
                                "level_name": level_id,
                                "level_price": float(level),
                                "regime": day_regime,
                                "is_post_holiday_session": is_post_holiday,
                            }
                            old_len = len(signals)
                            _append_signal(signals, ts, 1, entry, stop, target, STRATEGY_KEY, params, metadata)
                            if len(signals) > old_len:
                                level_last_trade_idx[level_id] = i
                                signal_this_bar_long = True

        # --- Continuation Mode ---
        elif day_regime in ("trend_up", "trend_down"):
            if not breakout_active:
                target_level_id = "pdVAH" if day_regime == "trend_up" else "pdVAL"

                # A6: per-day cap — at most 2 continuation signals per level per day
                if breakout_count_today.get(target_level_id, 0) < 2:
                    target_level = pdVAH if day_regime == "trend_up" else pdVAL
                    prior_close = float(df.iloc[i - 1]["close"])

                    breakout_detected = False
                    if day_regime == "trend_up":
                        if (prior_close <= target_level
                                and bar_close > target_level
                                and not np.isnan(volume_median)
                                and bar_volume > volume_median * continuation_volume_mult
                                and bar_close > bar_open
                                and (i - level_last_trade_idx.get(target_level_id, -1000) >= per_level_cooldown_bars)):
                            breakout_direction = "long"
                            breakout_detected = True
                    else:
                        if (prior_close >= target_level
                                and bar_close < target_level
                                and not np.isnan(volume_median)
                                and bar_volume > volume_median * continuation_volume_mult
                                and bar_close < bar_open
                                and (i - level_last_trade_idx.get(target_level_id, -1000) >= per_level_cooldown_bars)):
                            breakout_direction = "short"
                            breakout_detected = True

                    if breakout_detected:
                        breakout_active = True
                        breakout_bar_idx = i
                        breakout_level = target_level

            else:
                # Retest monitoring — invalidation
                if breakout_direction == "long":
                    if bar_close < breakout_level:
                        breakout_active = False
                else:
                    if bar_close > breakout_level:
                        breakout_active = False

                if not breakout_active:
                    continue

                # Expiry
                if (i - breakout_bar_idx) >= retest_window_bars:
                    breakout_active = False
                    continue

                # B2: retest must actually touch the broken level and snap back
                retest_triggered = False
                if breakout_direction == "long":
                    if (bar_low <= breakout_level
                            and bar_close > breakout_level
                            and bar_close > bar_open):
                        retest_triggered = True
                else:
                    if (bar_high >= breakout_level
                            and bar_close < breakout_level
                            and bar_close < bar_open):
                        retest_triggered = True

                if retest_triggered:
                    entry = bar_close
                    target_level_id = "pdVAH" if breakout_direction == "long" else "pdVAL"

                    if breakout_direction == "long":
                        stop_a = bar_low - (stop_buffer_atr_mult * atr_val)
                        stop_b = breakout_level - (0.2 * atr_val)
                        # B1: filter to candidates strictly below entry
                        candidates = [s for s in (stop_a, stop_b) if s < entry]
                        if not candidates:
                            discarded_stop_side_count += 1
                            breakout_active = False
                            continue
                        stop = max(candidates)
                        dist = abs(entry - stop)
                        if dist < MIN_STOP_DISTANCE_POINTS:  # B3
                            too_tight_count += 1
                            breakout_active = False
                            continue
                        target = entry + (dist * target_r_multiple)
                        metadata = {
                            "mode": "continuation",
                            "level_name": "pdVAH",
                            "level_price": float(breakout_level),
                            "regime": day_regime,
                            "breakout_bar_idx": breakout_bar_idx,
                            "discarded_stop_side_count": discarded_stop_side_count,
                            "is_post_holiday_session": is_post_holiday,
                        }
                        old_len = len(signals)
                        _append_signal(signals, ts, 1, entry, stop, target, STRATEGY_KEY, params, metadata)
                        if len(signals) > old_len:
                            level_last_trade_idx["pdVAH"] = i
                            breakout_count_today[target_level_id] = breakout_count_today.get(target_level_id, 0) + 1
                            breakout_active = False

                    else:
                        stop_a = bar_high + (stop_buffer_atr_mult * atr_val)
                        stop_b = breakout_level + (0.2 * atr_val)
                        # B1: filter to candidates strictly above entry
                        candidates = [s for s in (stop_a, stop_b) if s > entry]
                        if not candidates:
                            discarded_stop_side_count += 1
                            breakout_active = False
                            continue
                        stop = min(candidates)
                        dist = abs(entry - stop)
                        if dist < MIN_STOP_DISTANCE_POINTS:  # B3
                            too_tight_count += 1
                            breakout_active = False
                            continue
                        target = entry - (dist * target_r_multiple)
                        metadata = {
                            "mode": "continuation",
                            "level_name": "pdVAL",
                            "level_price": float(breakout_level),
                            "regime": day_regime,
                            "breakout_bar_idx": breakout_bar_idx,
                            "discarded_stop_side_count": discarded_stop_side_count,
                            "is_post_holiday_session": is_post_holiday,
                        }
                        old_len = len(signals)
                        _append_signal(signals, ts, -1, entry, stop, target, STRATEGY_KEY, params, metadata)
                        if len(signals) > old_len:
                            level_last_trade_idx["pdVAL"] = i
                            breakout_count_today[target_level_id] = breakout_count_today.get(target_level_id, 0) + 1
                            breakout_active = False

    # B7: surface per-run diagnostic counters on the last signal for vp_sanity_check.py
    if signals:
        last = signals[-1]
        new_meta = {
            **last.metadata,
            "too_tight_count": too_tight_count,
            "discarded_stop_side_count": discarded_stop_side_count,
        }
        signals[-1] = dataclasses.replace(last, metadata=new_meta)

    return signals


register_strategy(
    StrategySpec(
        name=STRATEGY_KEY,
        generate=vp_dual_mode_generate,
        default_params={
            "vp_session_mode": "full",
            "level_tolerance_atr_mult": 0.3,
            "rejection_volume_mult": 0.8,
            "continuation_volume_mult": 1.5,
            "stop_buffer_atr_mult": 0.2,
            "naked_poc_lookback_sessions": 3,
            "per_level_cooldown_bars": 20,
            "target_r_multiple": 1.5,
            "vp_rows": 400,
            "va_pct": 0.70,
            "retest_window_bars": 10,
            "de_threshold": 0.35,
            "atr_cap_points": 100.0,
            "atr_period": 14,
        },
        param_grid={
            "level_tolerance_atr_mult": (0.2, 0.3, 0.5),
            "rejection_volume_mult": (0.6, 0.8, 1.0),
            "continuation_volume_mult": (1.3, 1.5, 2.0),
            "stop_buffer_atr_mult": (0.1, 0.2, 0.3),
            "naked_poc_lookback_sessions": (1, 3, 5),
            "per_level_cooldown_bars": (10, 20, 40),
            "target_r_multiple": (0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
            # Fixed params — single value only, not searched
            "vp_session_mode": ("full",),
            "vp_rows": (400,),
            "va_pct": (0.70,),
            "retest_window_bars": (10,),
            "de_threshold": (0.35,),
            "atr_cap_points": (100.0,),
            "atr_period": (14,),
        },
        max_signals_per_day=None,
        requires=("volume_profile", "regime_classifier"),
    )
)
