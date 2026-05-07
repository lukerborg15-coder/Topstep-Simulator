"""Sanity check for the VP Dual-Mode strategy.

Runs on MNQ 5min data with the WF1 train window.
Prints signal distribution and key performance stats.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v3.config import WINDOWS  # noqa: E402
from v3.data import load_ohlcv, slice_window  # noqa: E402
from v3.evaluator import evaluate_strategy  # noqa: E402
from v3.regime_classifier import attach_day_regimes  # noqa: E402
from v3.strategies import STRATEGIES, load_user_strategies  # noqa: E402
from v3.volume_profile import attach_volume_profile  # noqa: E402


def main() -> None:
    load_user_strategies()
    print("Loading data...")
    raw_1min = load_ohlcv("MNQ", "1min", session_only=False)
    exec_5min = load_ohlcv("MNQ", "5min")

    wf1 = WINDOWS.walk_forward[0]
    train_window = wf1.train

    print(
        f"Running sanity check on {train_window.name} ({train_window.start} to {train_window.end})..."
    )

    strategy_name = "vp_dual_mode"
    params = {
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
    }

    result = evaluate_strategy(
        exec_5min,
        strategy_name,
        "5min",
        params,
        train_window,
        raw_frame=raw_1min,
    )

    sliced = slice_window(exec_5min, train_window)
    signal_frame = attach_volume_profile(sliced, raw_1min)
    signal_frame = attach_day_regimes(signal_frame, raw_1min)

    print("\n--- DEBUG ---")
    print(f"Signal frame rows: {len(signal_frame)}")
    print(f"NaN pdVAH: {signal_frame['pdVAH'].isna().sum()}")
    print(f"NaN pdVAL: {signal_frame['pdVAL'].isna().sum()}")
    print(f"Regime distribution:\n{signal_frame['day_regime'].value_counts()}")

    spec = STRATEGIES[strategy_name]
    signals = spec.generate(signal_frame, params)

    metrics = result.metrics
    trades = result.trades

    print("\n--- RESULTS ---")
    print(f"Total signals generated: {len(signals)}")
    print(f"Total trades simulated: {len(trades)}")

    if not signals:
        print("FAIL: No signals generated.")
        return

    modes = [signal.metadata.get("mode", "unknown") for signal in signals]
    directions = [signal.direction for signal in signals]
    levels = [signal.metadata.get("level_name", "unknown") for signal in signals]

    mode_counts = pd.Series(modes).value_counts()
    dir_counts = pd.Series(directions).value_counts()
    level_counts = pd.Series(levels).value_counts()

    print("\nModes:")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} ({count / len(signals) * 100:.1f}%)")

    print("\nDirections:")
    for direction, count in dir_counts.items():
        print(f"  {direction}: {count}")

    print("\nTop 5 Levels:")
    for level_name, count in level_counts.head(5).items():
        print(f"  {level_name}: {count}")

    print(f"\nWin Rate: {metrics['win_rate'] * 100:.1f}%")
    print(f"Avg R: {metrics['avg_r']:.2f}")
    print(f"Net PnL: ${metrics['total_net_pnl']:.2f}")
    print(f"Max DD: ${metrics['max_drawdown']:.2f}")

    if trades:
        exit_counts = pd.Series([trade.exit_reason for trade in trades]).value_counts()
        print("\nExit reasons:")
        for reason, count in exit_counts.items():
            print(f"  {reason}: {count} ({count / len(trades) * 100:.1f}%)")

    signal_dates = [signal.time.normalize() for signal in signals]
    unique_days = len(set(signal_dates))
    all_trading_dates = signal_frame.index.normalize().unique()
    zero_signal_days = len(all_trading_dates) - unique_days
    print(f"\nTrading days with signals: {unique_days}")
    print(f"Days with zero signals: {zero_signal_days}")
    print(f"Avg signals/day: {len(signals) / unique_days:.2f}" if unique_days else "N/A")

    if len(signals) < 100:
        print("\nWARNING: Total signals < 100 (likely architecture bug).")
    if any(count / len(signals) > 0.9 for count in mode_counts):
        print("\nWARNING: One mode accounts for > 90% of signals (regime classifier likely broken).")
    if unique_days and len(signals) / unique_days > 10:
        print("\nWARNING: Over-trading (Avg signals/day > 10).")

    print("\n--- B7 DIAGNOSTICS ---")

    max_cap = 50
    capped = sum(1 for trade in trades if trade.contracts >= max_cap)
    if trades:
        pct_capped = capped / len(trades) * 100
        flag = " [FLAG: > 5%]" if pct_capped > 5.0 else ""
        print(f"pct_trades_at_max_contract_cap: {pct_capped:.1f}%  ({capped}/{len(trades)}){flag}")
    else:
        print("pct_trades_at_max_contract_cap: N/A (no trades)")

    last_meta = signals[-1].metadata if signals else {}
    too_tight = int(last_meta.get("too_tight_count", 0))
    discarded_side = int(last_meta.get("discarded_stop_side_count", 0))

    total_pre_gate = len(signals) + too_tight
    if total_pre_gate > 0:
        pct_tight = too_tight / total_pre_gate * 100
        flag = " [FLAG: > 30%]" if pct_tight > 30.0 else ""
        print(f"pct_signals_below_min_stop_distance: {pct_tight:.1f}%  ({too_tight}/{total_pre_gate}){flag}")
    else:
        print("pct_signals_below_min_stop_distance: N/A")

    cont_signals = [signal for signal in signals if signal.metadata.get("mode") == "continuation"]
    cont_attempted = len(cont_signals) + discarded_side
    if cont_attempted > 0:
        pct_discarded = discarded_side / cont_attempted * 100
        flag = " [FLAG: > 10%]" if pct_discarded > 10.0 else ""
        print(
            "continuation_discarded_stop_side_count: "
            f"{discarded_side}  ({pct_discarded:.1f}% of {cont_attempted} attempted){flag}"
        )
    else:
        print("continuation_discarded_stop_side_count: N/A (no continuation breakouts)")

    retest_dists = []
    for signal in cont_signals:
        level_price = signal.metadata.get("level_price")
        if level_price is None:
            continue
        bar_ts = signal.time
        if bar_ts not in signal_frame.index:
            continue
        bar = signal_frame.loc[bar_ts]
        if signal.direction == "long":
            retest_dists.append(float(bar["low"]) - float(level_price))
        else:
            retest_dists.append(float(level_price) - float(bar["high"]))

    if retest_dists:
        arr = np.array(retest_dists)
        pct_positive = (arr > 0).sum() / len(arr) * 100
        flag = " [FLAG: some > 0 - B2 retest gap]" if pct_positive > 0 else ""
        print(
            f"retest_distance_distribution (n={len(arr)}): "
            f"min={arr.min():.2f} p25={np.percentile(arr, 25):.2f} "
            f"p50={np.median(arr):.2f} p75={np.percentile(arr, 75):.2f} "
            f"max={arr.max():.2f}{flag}"
        )
    else:
        print("retest_distance_distribution: N/A (no continuation signals)")

    if trades:
        post_holiday_trades = [
            trade
            for trade in trades
            if any(
                signal.time == trade.entry_time
                and signal.metadata.get("is_post_holiday_session", False)
                for signal in signals
            )
        ]
        normal_trades = [trade for trade in trades if trade not in post_holiday_trades]
        pct_post_holiday = len(post_holiday_trades) / len(trades) * 100

        def _stats(trades_subset: list) -> str:
            if not trades_subset:
                return "N/A"
            wins = sum(1 for trade in trades_subset if trade.r_multiple > 0)
            win_rate = wins / len(trades_subset) * 100
            avg_r = np.mean([trade.r_multiple for trade in trades_subset])
            return f"wr={win_rate:.1f}%  avg_r={avg_r:.2f}  n={len(trades_subset)}"

        print(f"pct_trades_post_holiday: {pct_post_holiday:.1f}%")
        print(f"  post_holiday  -> {_stats(post_holiday_trades)}")
        print(f"  normal        -> {_stats(normal_trades)}")
    else:
        print("pct_trades_post_holiday: N/A (no trades)")


if __name__ == "__main__":
    main()
