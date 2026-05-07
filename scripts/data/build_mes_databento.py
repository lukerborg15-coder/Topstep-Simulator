"""Rebuild the repo-local MES 1-minute base file from a Databento export.

This keeps the old helper entrypoint but removes any machine-specific source
path. Callers must provide the raw Databento ZIP or CSV file explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from v3.databento import build_continuous_from_databento, format_databento_output  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "Data" / "mes_1min_databento.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Data/mes_1min_databento.csv from a Databento export")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the raw Databento ZIP or CSV export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, stats = build_continuous_from_databento(args.source, "MES")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    format_databento_output(frame).to_csv(args.output, index=False)

    print(f"Saved to: {args.output}")
    print(f"Rows: {stats.row_count:,}")
    print(f"Range: {stats.first_timestamp} -> {stats.last_timestamp}")
    print(f"Rollovers: {len(stats.contract_switches) - 1}")
    print(f"Max roll gap: {stats.max_roll_gap_pct:.2%}")


if __name__ == "__main__":
    main()
