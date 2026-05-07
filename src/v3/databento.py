from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import zipfile

import pandas as pd

from .config import EASTERN_TZ


_MONTH_CODES = "FGHJKMNQUVXZ"
_OUTRIGHT_PATTERN_TEMPLATE = r"^{root}[" + _MONTH_CODES + r"][0-9]$"


@dataclass(frozen=True)
class ContinuousBuildStats:
    root_symbol: str
    row_count: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    contract_switches: tuple[tuple[pd.Timestamp, str], ...]
    max_roll_gap_pct: float


def outright_symbol_pattern(root_symbol: str) -> re.Pattern[str]:
    root = root_symbol.upper()
    return re.compile(_OUTRIGHT_PATTERN_TEMPLATE.format(root=re.escape(root)))


def is_valid_outright_symbol(symbol: str, root_symbol: str) -> bool:
    return bool(outright_symbol_pattern(root_symbol).match(symbol))


def contract_expiry_key(symbol: str) -> tuple[int, int]:
    month_code = symbol[-2]
    year_digit = int(symbol[-1])
    return 2020 + year_digit, _MONTH_CODES.index(month_code)


def determine_continuous_contracts(daily_volume: pd.DataFrame) -> dict[object, str]:
    """Build a monotonic daily contract map from daily volume totals.

    Rule:
    - start with the highest-volume outright on the first available date
    - stay on the current contract until a later expiry beats it in daily volume
    - once rolled, never roll backward to an older expiry
    """
    required = {"date", "symbol", "volume"}
    missing = required.difference(daily_volume.columns)
    if missing:
        raise ValueError(f"daily_volume missing required columns: {sorted(missing)}")

    ordered = daily_volume.copy()
    ordered["expiry_key"] = ordered["symbol"].map(contract_expiry_key)
    ordered = ordered.sort_values(["date", "expiry_key"])

    selection: dict[object, str] = {}
    current_symbol: str | None = None

    for date, group in ordered.groupby("date", sort=True):
        volumes = {str(row.symbol): float(row.volume) for row in group.itertuples(index=False)}
        available = sorted(volumes, key=contract_expiry_key)
        if not available:
            continue

        if current_symbol is None:
            current_symbol = max(available, key=lambda symbol: (volumes[symbol], -contract_expiry_key(symbol)[0], -contract_expiry_key(symbol)[1]))
            selection[date] = current_symbol
            continue

        current_key = contract_expiry_key(current_symbol)
        current_volume = volumes.get(current_symbol, float("-inf"))
        better_later = [
            symbol
            for symbol in available
            if contract_expiry_key(symbol) > current_key and volumes[symbol] > current_volume
        ]

        if better_later:
            current_symbol = min(better_later, key=contract_expiry_key)
        elif current_symbol not in volumes:
            later_available = [symbol for symbol in available if contract_expiry_key(symbol) > current_key]
            pool = later_available or available
            current_symbol = max(pool, key=lambda symbol: (volumes[symbol], -contract_expiry_key(symbol)[0], -contract_expiry_key(symbol)[1]))

        selection[date] = current_symbol

    return selection


def build_continuous_ohlcv(raw_frame: pd.DataFrame, root_symbol: str) -> pd.DataFrame:
    required = {"ts_event", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(raw_frame.columns)
    if missing:
        raise ValueError(f"raw_frame missing required columns: {sorted(missing)}")

    pattern = outright_symbol_pattern(root_symbol)
    frame = raw_frame.loc[raw_frame["symbol"].astype(str).str.match(pattern)].copy()
    if frame.empty:
        raise ValueError(f"No outright {root_symbol.upper()} rows found in raw source")

    frame["datetime"] = pd.to_datetime(frame["ts_event"], utc=True).dt.tz_convert(EASTERN_TZ)
    frame["date"] = frame["datetime"].dt.date

    daily_volume = frame.groupby(["date", "symbol"], as_index=False)["volume"].sum()
    selected = determine_continuous_contracts(daily_volume)

    frame["selected_symbol"] = frame["date"].map(selected)
    frame = frame.loc[frame["symbol"] == frame["selected_symbol"]].copy()
    frame = frame.sort_values("datetime").reset_index(drop=True)

    result = frame[["datetime", "symbol", "open", "high", "low", "close", "volume"]].copy()
    validate_continuous_ohlcv(result, root_symbol=root_symbol)
    return result


def validate_continuous_ohlcv(
    frame: pd.DataFrame,
    root_symbol: str,
    *,
    max_roll_gap_pct: float = 0.03,
) -> None:
    required = {"datetime", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame missing required columns: {sorted(missing)}")

    pattern = outright_symbol_pattern(root_symbol)
    bad_symbols = sorted(set(frame.loc[~frame["symbol"].astype(str).str.match(pattern), "symbol"].astype(str)))
    if bad_symbols:
        raise ValueError(f"Found non-outright {root_symbol.upper()} symbols: {bad_symbols[:5]}")

    ordered = frame.sort_values("datetime").reset_index(drop=True)
    duplicate_count = int(ordered["datetime"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Found duplicate timestamps in continuous {root_symbol.upper()} data: {duplicate_count}")

    price_columns = ["open", "high", "low", "close"]
    numeric = ordered[price_columns].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(ordered["volume"], errors="coerce")
    if numeric.isna().any().any() or volume.isna().any():
        raise ValueError("Found non-numeric OHLCV values")
    if (numeric <= 0).any().any():
        raise ValueError("Found impossible prices: OHLC must be positive")
    if (volume < 0).any():
        raise ValueError("Found impossible volume: volume must be non-negative")
    if ((numeric["high"] < numeric[["open", "close"]].max(axis=1)) | (numeric["low"] > numeric[["open", "close"]].min(axis=1)) | (numeric["high"] < numeric["low"])).any():
        raise ValueError("Found impossible prices: OHLC relationship check failed")

    switched = ordered["symbol"].ne(ordered["symbol"].shift())
    if switched.sum() <= 1:
        return

    boundary = ordered.loc[switched].copy()
    boundary = boundary.iloc[1:].copy()
    boundary["prev_close"] = ordered["close"].shift().loc[boundary.index]
    boundary["gap_pct"] = (boundary["open"] - boundary["prev_close"]).abs() / boundary["prev_close"]
    bad_gap = boundary.loc[boundary["gap_pct"] > max_roll_gap_pct]
    if not bad_gap.empty:
        row = bad_gap.sort_values("gap_pct", ascending=False).iloc[0]
        raise ValueError(
            "Found rollover discontinuity in "
            f"{root_symbol.upper()} continuous data at {row['datetime']}: "
            f"{row['symbol']} gap {row['gap_pct']:.2%}"
        )


def build_continuous_from_databento(source_path: Path, root_symbol: str) -> tuple[pd.DataFrame, ContinuousBuildStats]:
    daily_volume_totals: defaultdict[tuple[object, str], float] = defaultdict(float)

    for chunk in iter_databento_ohlcv_chunks(source_path, usecols=["ts_event", "symbol", "volume"]):
        filtered = filter_outright_rows(chunk, root_symbol)
        if filtered.empty:
            continue
        datetimes = pd.to_datetime(filtered["ts_event"], utc=True).dt.tz_convert(EASTERN_TZ)
        grouped = (
            filtered.assign(date=datetimes.dt.date)
            .groupby(["date", "symbol"], as_index=False)["volume"]
            .sum()
        )
        for row in grouped.itertuples(index=False):
            daily_volume_totals[(row.date, str(row.symbol))] += float(row.volume)

    daily_volume = pd.DataFrame(
        [
            {"date": date, "symbol": symbol, "volume": volume}
            for (date, symbol), volume in daily_volume_totals.items()
        ]
    )
    if daily_volume.empty:
        raise ValueError(f"No outright {root_symbol.upper()} rows found in {source_path}")

    selected = determine_continuous_contracts(daily_volume)
    result_chunks: list[pd.DataFrame] = []

    for chunk in iter_databento_ohlcv_chunks(
        source_path,
        usecols=["ts_event", "symbol", "open", "high", "low", "close", "volume"],
    ):
        filtered = filter_outright_rows(chunk, root_symbol)
        if filtered.empty:
            continue
        filtered["datetime"] = pd.to_datetime(filtered["ts_event"], utc=True).dt.tz_convert(EASTERN_TZ)
        filtered["date"] = filtered["datetime"].dt.date
        filtered["selected_symbol"] = filtered["date"].map(selected)
        chosen = filtered.loc[filtered["symbol"] == filtered["selected_symbol"], ["datetime", "symbol", "open", "high", "low", "close", "volume"]]
        if not chosen.empty:
            result_chunks.append(chosen.copy())

    if not result_chunks:
        raise ValueError(f"No continuous {root_symbol.upper()} rows selected from {source_path}")

    result = pd.concat(result_chunks, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    validate_continuous_ohlcv(result, root_symbol=root_symbol)

    switched = result["symbol"].ne(result["symbol"].shift())
    boundary = result.loc[switched].copy()
    boundary["prev_close"] = result["close"].shift().loc[boundary.index]
    boundary["gap_pct"] = (boundary["open"] - boundary["prev_close"]).abs() / boundary["prev_close"]
    max_gap_pct = float(boundary["gap_pct"].iloc[1:].max()) if len(boundary) > 1 else 0.0
    stats = ContinuousBuildStats(
        root_symbol=root_symbol.upper(),
        row_count=len(result),
        first_timestamp=pd.Timestamp(result["datetime"].iloc[0]),
        last_timestamp=pd.Timestamp(result["datetime"].iloc[-1]),
        contract_switches=tuple((pd.Timestamp(row.datetime), str(row.symbol)) for row in boundary.itertuples(index=False)),
        max_roll_gap_pct=max_gap_pct,
    )
    return result, stats


def filter_outright_rows(frame: pd.DataFrame, root_symbol: str) -> pd.DataFrame:
    pattern = outright_symbol_pattern(root_symbol)
    return frame.loc[frame["symbol"].astype(str).str.match(pattern)].copy()


def iter_databento_ohlcv_chunks(
    source_path: Path,
    *,
    usecols: list[str],
    chunksize: int = 500_000,
):
    source = Path(source_path)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected exactly one CSV inside {source}, found {csv_names}")
            with archive.open(csv_names[0]) as handle:
                yield from pd.read_csv(handle, usecols=usecols, chunksize=chunksize)
        return

    yield from pd.read_csv(source, usecols=usecols, chunksize=chunksize)


def format_databento_output(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    formatted["datetime"] = formatted["datetime"].map(_format_datetime)
    return formatted[["datetime", "open", "high", "low", "close", "volume"]]


def _format_datetime(value: pd.Timestamp) -> str:
    offset = value.strftime("%z")
    if len(offset) == 5:
        offset = offset[:3] + ":" + offset[3:]
    return value.strftime("%Y-%m-%d %H:%M:%S") + offset
