# Session Resampling Spec

## Scope

- Canonical source for derivation is `1min` OHLCV.
- Session-aware derived timeframes apply to both `MNQ` and `MES`.
- Execution session is `09:30-16:00 America/New_York`.

## Session policy

- `1min` data remains the canonical full 24-hour continuous stream.
- When `session_only=True`, the loader filters minute data to the regular session before any higher-timeframe aggregation.
- Session filtering is left-closed and right-open: `[09:30, 16:00)`.
- That means the last in-session `1min` bar is `15:59`.

## Resampling policy

- Derived bars are left-labeled by their start timestamp.
- Session-aware higher timeframes are anchored to each trading day's `09:30` session open, not midnight or wall-clock hour boundaries.
- Aggregation is:
  - `open = first`
  - `high = max`
  - `low = min`
  - `close = last`
  - `volume = sum`

## Partial-bar policy

- Opening bars always start at `09:30`.
- Trailing partial bars are kept if they contain at least one in-session minute.
- Expected session-aware bar counts per full RTH day are:
  - `1min = 390`
  - `2min = 195`
  - `3min = 130`
  - `5min = 78`
  - `15min = 26`
  - `30min = 13`
  - `1h = 7`
  - `4h = 2`

## Determinism contract

- Runtime loader prefers prebuilt higher-timeframe CSVs when present, then falls back to deriving them from `1min`.
- Data builder uses the same shared resampling code path as the loader.
- Prebuilt higher-timeframe CSVs and loader-derived higher timeframes must therefore match exactly for session-aware loads.

## Examples

- `1h` RTH labels: `09:30`, `10:30`, `11:30`, `12:30`, `13:30`, `14:30`, `15:30`
- `4h` RTH labels: `09:30`, `13:30`
