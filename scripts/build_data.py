from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v3.data import format_ohlcv_csv, resample_ohlcv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build higher-timeframe CSVs from full-session 1-minute data.")
    parser.add_argument("--instrument", required=True, choices=["mnq", "mes"])
    parser.add_argument("--timeframes", nargs="+", default=["1min"])
    parser.add_argument("--source", type=Path, default=None, help="Optional path to an existing 1-minute CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def resample_to_csv(
    source_path: Path,
    timeframe: str,
    output_dir: Path,
) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    frame = pd.read_csv(source_path)
    index = pd.to_datetime(frame.pop("datetime"), utc=True).dt.tz_convert("America/New_York")
    frame.index = pd.DatetimeIndex(index, name="datetime")
    frame = frame.sort_index()

    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])

    resampled = resample_ohlcv(frame, timeframe, session_only=False)
    result = format_ohlcv_csv(resampled)

    output_path = output_dir / f"{source_path.stem.replace('_1min_databento', '')}_{timeframe}_databento.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    first_ts = pd.Timestamp(result["datetime"].iloc[0])
    last_ts = pd.Timestamp(result["datetime"].iloc[-1])
    return len(result), first_ts, last_ts


def main() -> None:
    args = parse_args()
    instrument = args.instrument.lower()
    base_1min = args.source if args.source is not None else args.output_dir / f"{instrument}_1min_databento.csv"

    if not base_1min.exists():
        raise FileNotFoundError(f"Missing 1-minute source file: {base_1min}")

    for timeframe in args.timeframes:
        if timeframe == "1min":
            continue
        row_count, first_ts, last_ts = resample_to_csv(base_1min, timeframe, args.output_dir)
        print(f"{timeframe}: {row_count} rows {first_ts} -> {last_ts}")


if __name__ == "__main__":
    main()
