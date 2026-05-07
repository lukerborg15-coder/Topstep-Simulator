"""Quick verification that MES data loads correctly with the pipeline loader."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v3.data import load_ohlcv  # noqa: E402


def main() -> None:
    print("Testing MES data loading with the pipeline loader...")

    try:
        df = load_ohlcv(
            instrument="mes",
            timeframe="1min",
            data_dir=PROJECT_ROOT / "Data",
            session_only=False,
        )
        print("\nOK: Successfully loaded MES data")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index TZ: {df.index.tz}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nFirst 3 rows:\n{df.head(3)}")
        print(f"\nLast 3 rows:\n{df.tail(3)}")

        unique_dates = df.index.date
        print(f"\nUnique trading dates: {len(set(unique_dates))}")

        sample_date = df.index[0].date()
        day_data = df[df.index.date == sample_date]
        print(f"\nSample day {sample_date}: {len(day_data)} bars")
        print(f"  Start: {day_data.index.min()}")
        print(f"  End: {day_data.index.max()}")
        print(
            f"  Duration: {(day_data.index.max() - day_data.index.min()).total_seconds() / 3600:.1f} hours"
        )
    except Exception as exc:
        print(f"\nERROR: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
