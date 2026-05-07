r"""
Data builder utility for normalized instrument/timeframe CSV artifacts.

Usage:
    # Build MNQ 1min (if raw source available)
    python scripts/build_data.py --instrument mnq --timeframes 1min

    # Build MES from raw Databento source
    python scripts/build_data.py --instrument mes --timeframes 1min --source C:\path\to\source.csv

    # Build 5min and 15min from existing 1min source
    python scripts/build_data.py --instrument mnq --timeframes 5min 15min

    # Full pipeline: normalize to 1min then generate 5min, 15min, 1h
    python scripts/build_data.py --instrument mes --timeframes 5min 15min 1h --source C:\path\to\source.csv

Output:
    Data/{instrument}_{timeframe}_databento.csv

Requires:
    pandas
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v3.data import format_ohlcv_csv, resample_ohlcv
from v3.databento import build_continuous_from_databento, format_databento_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized instrument/timeframe CSVs")
    parser.add_argument(
        "--instrument",
        required=True,
        choices=["mnq", "mes"],
        help="Instrument symbol",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1min"],
        help="Target timeframes to build (default: 1min)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Source Databento ZIP or CSV (must contain ts_event, open, high, low, close, volume, symbol)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"Output directory (default: {DEFAULT_DATA_DIR})")
    return parser.parse_args()


def normalize_from_databento(
    source_path: Path,
    instrument: str,
    output_dir: Path,
):
    """Convert raw Databento OHLCV ZIP/CSV to a clean continuous 1-minute dataset.

    Returns:
        (row_count, first_timestamp, last_timestamp, selected_contracts)
    """
    print(f"Reading {source_path}...")
    root_symbol = instrument.upper()
    result, stats = build_continuous_from_databento(source_path=source_path, root_symbol=root_symbol)
    formatted = format_databento_output(result)

    # Save
    output_path = output_dir / f"{instrument}_1min_databento.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(output_path, index=False)

    return stats


def resample_to_csv(
    source_path: Path,
    timeframe: str,
    output_dir: Path,
) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """Resample 1min CSV to target timeframe and save.

    Returns:
        (row_count, first_timestamp, last_timestamp)
    """
    print(f"Loading {source_path} for {timeframe} resampling...")

    frame = pd.read_csv(source_path)
    index = pd.to_datetime(frame.pop("datetime"), utc=True).dt.tz_convert("America/New_York")
    frame.index = pd.DatetimeIndex(index, name="datetime")
    frame = frame.sort_index()

    # Convert numeric columns
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])

    period_minutes = _timeframe_to_minutes(timeframe)
    print(f"Resampling to {timeframe} ({period_minutes}min) from full-session 1-minute data...")
    resampled = resample_ohlcv(frame, timeframe, session_only=False)
    result = format_ohlcv_csv(resampled)

    # Save
    output_path = output_dir / f"{source_path.stem.replace('_1min_databento', '')}_{timeframe}_databento.csv"
    result.to_csv(output_path, index=False)

    first_ts = pd.Timestamp(result["datetime"].iloc[0])
    last_ts = pd.Timestamp(result["datetime"].iloc[-1])

    return len(result), first_ts, last_ts


def _timeframe_to_minutes(timeframe: str) -> int:
    mapping = {
        "1min": 1, "2min": 2, "3min": 3, "5min": 5,
        "15min": 15, "30min": 30, "1h": 60, "4h": 240,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {list(mapping.keys())}")
    return mapping[timeframe]


def main() -> None:
    args = parse_args()
    instrument = args.instrument.lower()

    print(f"\n=== Data Builder for {instrument.upper()} ===\n")

    # Step 1: Normalize source to 1min if provided
    if args.source:
        stats = normalize_from_databento(
            source_path=args.source,
            instrument=instrument,
            output_dir=args.output_dir,
        )
        print(f"Normalized {stats.row_count:,} rows")
        print(f"  {stats.first_timestamp.date()} to {stats.last_timestamp.date()}")
        print("  Contract rule: strict outright symbols only, then hold the current contract until a later expiry beats it on daily volume")
        print(f"  Rollovers: {len(stats.contract_switches) - 1}")
        print(f"  Max roll gap: {stats.max_roll_gap_pct:.2%}")
        print(f"  Saved to: {args.output_dir / f'{instrument}_1min_databento.csv'}\n")

    # Step 2: Generate higher timeframes
    base_1min = args.output_dir / f"{instrument}_1min_databento.csv"
    for tf in args.timeframes:
        if tf == "1min":
            print(f"  {tf}: already available at {base_1min}")
            continue

        target_path = args.output_dir / f"{instrument}_{tf}_databento.csv"
        if target_path.exists():
            print(f"  {tf}: already exists at {target_path}")
            continue

        try:
            row_count, first_ts, last_ts = resample_to_csv(
                source_path=base_1min,
                timeframe=tf,
                output_dir=args.output_dir,
            )
            print(f"  {tf}: {row_count:,} rows ({first_ts.date()} to {last_ts.date()})")
        except FileNotFoundError:
            print(f"  {tf}: SKIPPED - 1min source not found at {base_1min}")
        except Exception as e:
            print(f"  {tf}: ERROR - {e}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
