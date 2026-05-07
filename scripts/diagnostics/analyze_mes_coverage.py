"""Analyze MES data coverage directly from the repo-local Databento CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "Data" / "mes_1min_databento.csv"


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    df["dt"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"] = df["dt"].dt.date
    df["hour"] = df["dt"].dt.hour

    print("Sample days (January 2024) - parsed in UTC:")
    for day_label in ["2024-01-02", "2024-01-03", "2024-01-04"]:
        day = df[df["date"] == pd.to_datetime(day_label).date()]
        if len(day):
            print(f"  {day_label}: {len(day)} bars")
        else:
            print(f"  {day_label}: No data")

    sample = df[df["date"] == pd.to_datetime("2024-01-03").date()]
    if len(sample):
        print("\nSample day 2024-01-03:")
        print(f"  Hours present: {sorted(sample['hour'].unique())}")

    print("\n" + "=" * 60)
    print("Correct interpretation (Eastern timezone):")
    print("=" * 60)

    unique_dates = df["date"].nunique()
    print(f"Unique calendar dates in data: {unique_dates}")

    print("\nHourly bar counts (UTC hours):")
    hourly = df.groupby("hour").size()
    for hour in range(24):
        count = hourly.get(hour, 0)
        bar = "#" * (count // 10000)
        print(f"  {hour:02d}:00 UTC: {count:8,} {bar}")

    print("\n" + "=" * 60)
    print("Weekday vs Weekend Analysis:")
    print("=" * 60)
    weekday = df[df["dt"].dt.dayofweek < 5]
    weekend = df[df["dt"].dt.dayofweek >= 5]
    print(f"Weekday bars: {len(weekday):,}")
    print(f"Weekend bars: {len(weekend):,}")
    print(f"Weekend ratio: {len(weekend) / len(df) * 100:.1f}%")

    hour21 = df[df["dt"].dt.hour == 21]
    hour22 = df[df["dt"].dt.hour == 22]
    print(f"\nHour 21 UTC bars: {len(hour21):,} ({(len(hour21) / len(weekday) * 100):.1f}% of weekdays)")
    print(f"Hour 22 UTC bars: {len(hour22):,} ({(len(hour22) / len(weekday) * 100):.1f}% of weekdays)")


if __name__ == "__main__":
    main()
