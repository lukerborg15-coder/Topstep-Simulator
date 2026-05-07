"""Audit the intraday regime distribution across the full MNQ dataset.

Usage:
    python scripts/diagnostics/audit_regimes.py
    python scripts/diagnostics/audit_regimes.py --de-threshold 0.40
    python scripts/diagnostics/audit_regimes.py --de-lookback 30

Flags if unclear > 20% or any single regime (trend_up/trend_down/balance) < 15%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from v3.data import load_ohlcv  # noqa: E402
from v3.regime_classifier import audit_regime_distribution, classify_day_regimes  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MNQ regime distribution")
    parser.add_argument("--de-threshold", type=float, default=0.35)
    parser.add_argument("--de-lookback", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    de_threshold = args.de_threshold
    de_lookback = args.de_lookback

    print("Loading 5-minute MNQ data...")
    exec_5min = load_ohlcv("mnq", "5min")
    print("Loading 1-minute MNQ data...")
    raw_1min = load_ohlcv("mnq", "1min")

    print(f"\nRunning audit (de_threshold={de_threshold}, de_lookback={de_lookback})...")

    dist = audit_regime_distribution(
        exec_5min, raw_1min, de_threshold=de_threshold, de_lookback=de_lookback
    )

    print("\n" + "=" * 50)
    print("Regime Distribution (full dataset)")
    print("=" * 50)
    print(f"Total days:   {dist['total_days']}")
    print(f"  trend_up:   {dist['trend_up']:4d}  ({dist['trend_up_pct']:5.1f}%)")
    print(f"  trend_down: {dist['trend_down']:4d}  ({dist['trend_down_pct']:5.1f}%)")
    print(f"  balance:    {dist['balance']:4d}  ({dist['balance_pct']:5.1f}%)")
    print(f"  unclear:    {dist['unclear']:4d}  ({dist['unclear_pct']:5.1f}%)")
    print(f"  de_threshold: {dist['de_threshold_used']}")

    print("\n" + "=" * 50)
    print("Per-Year Breakdown")
    print("=" * 50)

    regimes = classify_day_regimes(
        exec_5min, raw_1min, de_threshold=de_threshold, de_lookback=de_lookback
    )

    yearly: dict[int, dict[str, int]] = {}
    labels_order = ("trend_up", "trend_down", "balance", "unclear")

    for day_date, day_5min in exec_5min.groupby(exec_5min.index.normalize()):
        bars_940 = day_5min.between_time("09:40", "09:40", inclusive="both")
        if bars_940.empty:
            continue
        ts_940 = bars_940.index[0]
        label = str(regimes.loc[ts_940])
        year = day_date.year
        if year not in yearly:
            yearly[year] = {lbl: 0 for lbl in labels_order}
        yearly[year][label] += 1

    print(f"{'Year':>6}  {'up':>5}  {'down':>5}  {'bal':>5}  {'unc':>5}  {'total':>5}")
    print("-" * 42)
    for year in sorted(yearly):
        yd = yearly[year]
        total_yr = sum(yd.values())
        print(
            f"  {year}  "
            f"{yd['trend_up']:3d}({yd['trend_up'] / total_yr * 100:4.0f}%)  "
            f"{yd['trend_down']:3d}({yd['trend_down'] / total_yr * 100:4.0f}%)  "
            f"{yd['balance']:3d}({yd['balance'] / total_yr * 100:4.0f}%)  "
            f"{yd['unclear']:3d}({yd['unclear'] / total_yr * 100:4.0f}%)  "
            f"{total_yr}"
        )

    print("\n" + "=" * 50)
    print("Flags")
    print("=" * 50)
    issues = 0

    if dist["unclear_pct"] > 20.0:
        print(
            f"  WARNING: unclear={dist['unclear_pct']:.1f}% > 20% - threshold may be too loose"
        )
        issues += 1
    else:
        print(f"  OK: unclear={dist['unclear_pct']:.1f}% <= 20%")

    for lbl in ("trend_up", "trend_down", "balance"):
        pct = dist[f"{lbl}_pct"]
        if pct < 15.0:
            print(f"  WARNING: {lbl}={pct:.1f}% < 15% - regime starved of signals")
            issues += 1
        else:
            print(f"  OK: {lbl}={pct:.1f}% >= 15%")

    if issues == 0:
        print("\n  All checks passed.")
    else:
        print(f"\n  {issues} warning(s). Consider adjusting de_threshold or de_lookback.")


if __name__ == "__main__":
    main()
