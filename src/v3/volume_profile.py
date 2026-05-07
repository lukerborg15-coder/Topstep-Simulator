from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .config import EASTERN_TZ, SESSION_END, SESSION_START
from .futures_session import ensure_eastern_timestamp, futures_session_date
from .indicators import atr

VolumeProfileSessionMode = Literal["rth", "full"]

# A9: partial prior sessions produce unreliable VP levels.
_MIN_PRIOR_RTH_BARS = 200
_MIN_PRIOR_FULL_SESSION_BARS = 1000  # Full session is ~1380 1-min bars.

VP_LEVEL_COLUMNS = (
    "pdVAH",
    "pdVAL",
    "pdVPOC",
    "naked_pocs",
    "is_post_holiday_session",
    "dist_to_pdVAH_atr",
    "dist_to_pdVAL_atr",
    "dist_to_pdVPOC_atr",
    "dist_to_nearest_naked_poc_atr",
)


def compute_session_vp(
    raw_1min: pd.DataFrame,
    vp_rows: int = 400,
    va_pct: float = 0.70,
    *,
    session_mode: VolumeProfileSessionMode = "full",
) -> pd.DataFrame:
    """Compute prior-session volume profile for each execution session.

    Session modes:
    - ``"full"``: previous full CME equity-index futures session, keyed by
      futures session date. This is the default for full-session execution.
    - ``"rth"``: legacy RTH-oriented behavior using previous RTH plus the
      overnight window leading into today's 09:30 ET open.

    Returns one row per session key (tz-naive midnight Timestamp) with columns
    ``pdVAH``, ``pdVAL``, ``pdVPOC``, and ``is_post_holiday_session``.
    """
    mode = _normalize_session_mode(session_mode)
    session_keys = _session_keys_from_index(raw_1min.index, mode)
    if session_keys.empty:
        return pd.DataFrame(columns=["pdVAH", "pdVAL", "pdVPOC", "is_post_holiday_session"])

    records: dict[pd.Timestamp, dict[str, float | bool]] = {}
    for session_key in sorted(session_keys.unique()):
        prior_bars, is_post_holiday = _get_prior_cycle_bars(raw_1min, session_key, mode)
        if prior_bars.empty:
            records[session_key] = {
                "pdVAH": np.nan,
                "pdVAL": np.nan,
                "pdVPOC": np.nan,
                "is_post_holiday_session": False,
            }
            continue

        if _count_core_session_bars(prior_bars, mode) < _minimum_prior_bars(mode):
            records[session_key] = {
                "pdVAH": np.nan,
                "pdVAL": np.nan,
                "pdVPOC": np.nan,
                "is_post_holiday_session": is_post_holiday,
            }
            continue

        vah, val, vpoc = _compute_vp(prior_bars, vp_rows, va_pct)
        records[session_key] = {
            "pdVAH": vah,
            "pdVAL": val,
            "pdVPOC": vpoc,
            "is_post_holiday_session": is_post_holiday,
        }

    result = pd.DataFrame.from_dict(records, orient="index")
    result.index.name = "session_date"
    return result


def compute_naked_pocs(
    raw_1min: pd.DataFrame,
    exec_5min: pd.DataFrame,
    session_vp: pd.DataFrame,
    lookback_sessions: int = 3,
    *,
    session_mode: VolumeProfileSessionMode = "full",
) -> pd.Series:
    """Track naked prior-session POCs for each execution bar.

    Fill detection uses ``exec_5min`` only. With full-session execution bars,
    overnight price action can fill naked POCs; with RTH-only execution bars,
    it cannot.
    """
    del raw_1min  # kept for API compatibility

    mode = _normalize_session_mode(session_mode)
    all_session_keys = sorted(session_vp.index)
    key_to_idx = {d: i for i, d in enumerate(all_session_keys)}

    vpoc_by_key: dict[pd.Timestamp, float | None] = {}
    for key in all_session_keys:
        v = session_vp.loc[key, "pdVPOC"]
        vpoc_by_key[key] = float(v) if pd.notna(v) else None

    filled: set[pd.Timestamp] = set()
    result: dict[pd.Timestamp, list[float]] = {}
    exec_session_keys = _session_keys_from_index(exec_5min.index, mode)

    for ts in exec_5min.index:
        current_key = exec_session_keys.loc[ts]
        cur_idx = key_to_idx.get(current_key, -1)

        if cur_idx < 0:
            result[ts] = []
            continue

        eligible: list[pd.Timestamp] = []
        for k in range(2, lookback_sessions + 2):
            prior_idx = cur_idx - k
            if prior_idx >= 0:
                eligible.append(all_session_keys[prior_idx])

        bar_open = float(exec_5min.at[ts, "open"])
        bar_close = float(exec_5min.at[ts, "close"])
        body_low = min(bar_open, bar_close)
        body_high = max(bar_open, bar_close)

        for key in eligible:
            if key in filled:
                continue
            poc = vpoc_by_key.get(key)
            if poc is None:
                continue
            if body_low <= poc <= body_high:
                filled.add(key)

        result[ts] = [
            vpoc_by_key[key]
            for key in eligible
            if key not in filled and vpoc_by_key.get(key) is not None
        ]

    return pd.Series(result, index=exec_5min.index, dtype=object)


def attach_volume_profile(
    exec_5min: pd.DataFrame,
    raw_1min: pd.DataFrame,
    *,
    vp_rows: int = 400,
    va_pct: float = 0.70,
    lookback_sessions: int = 3,
    session_mode: VolumeProfileSessionMode = "full",
) -> pd.DataFrame:
    """Attach VP levels and distances to the execution frame.

    ``session_mode="full"`` is the default because the pipeline is moving to
    full futures-session execution. Strategy-level entry timing still belongs
    in the strategies themselves.
    """
    mode = _normalize_session_mode(session_mode)
    result = exec_5min.copy()

    session_vp = compute_session_vp(raw_1min, vp_rows=vp_rows, va_pct=va_pct, session_mode=mode)
    vp_dict: dict[str, dict] = {
        col: session_vp[col].to_dict()
        for col in ("pdVAH", "pdVAL", "pdVPOC", "is_post_holiday_session")
    }

    session_series = _session_keys_from_index(result.index, mode)
    for col in ("pdVAH", "pdVAL", "pdVPOC"):
        result[col] = session_series.map(vp_dict[col])

    result["is_post_holiday_session"] = (
        session_series.map(vp_dict["is_post_holiday_session"]).fillna(False).astype(bool)
    )

    naked_series = compute_naked_pocs(
        raw_1min,
        exec_5min,
        session_vp,
        lookback_sessions,
        session_mode=mode,
    )
    result["naked_pocs"] = naked_series.values

    bar_atr = atr(exec_5min, period=14)
    close = result["close"]

    result["dist_to_pdVAH_atr"] = (result["pdVAH"] - close).abs() / bar_atr
    result["dist_to_pdVAL_atr"] = (result["pdVAL"] - close).abs() / bar_atr
    result["dist_to_pdVPOC_atr"] = (result["pdVPOC"] - close).abs() / bar_atr

    nearest_raw = pd.array(
        [
            float(min(abs(p - row_close) for p in pocs)) if pocs else np.nan
            for pocs, row_close in zip(result["naked_pocs"], close)
        ],
        dtype="Float64",
    )
    result["dist_to_nearest_naked_poc_atr"] = pd.array(nearest_raw, dtype="Float64") / bar_atr.values

    return result


def _get_prior_cycle_bars(
    raw_1min: pd.DataFrame,
    session_key: pd.Timestamp,
    session_mode: VolumeProfileSessionMode,
) -> tuple[pd.DataFrame, bool]:
    """Return prior-session bars and post-holiday flag for ``session_key``."""
    if session_mode == "full":
        today_open = _full_session_open(session_key)
    else:
        today_open = _rth_session_open(session_key)

    bars_before = raw_1min[raw_1min.index < today_open]
    if bars_before.empty:
        return pd.DataFrame(), False

    prior_session_keys = _session_keys_from_index(bars_before.index, session_mode)
    if prior_session_keys.empty:
        return pd.DataFrame(), False

    prev_session_key = max(prior_session_keys.unique())
    is_post_holiday = _is_post_holiday_session(prev_session_key, session_key)

    if session_mode == "full":
        raw_keys = _session_keys_from_index(raw_1min.index, session_mode)
        return raw_1min.loc[raw_keys == prev_session_key].copy(), is_post_holiday

    prev_session_start = _rth_session_open(prev_session_key)
    prev_session_end = _rth_session_close(prev_session_key)
    overnight_start = today_open - pd.Timedelta(hours=24)

    mask_prev_rth = (raw_1min.index >= prev_session_start) & (raw_1min.index < prev_session_end)
    mask_overnight = (raw_1min.index >= overnight_start) & (raw_1min.index < today_open)
    combined = raw_1min.loc[mask_prev_rth | mask_overnight].copy()
    return combined, is_post_holiday


def _normalize_session_mode(session_mode: str) -> VolumeProfileSessionMode:
    mode = str(session_mode).strip().lower()
    if mode not in {"rth", "full"}:
        raise ValueError(f"Unsupported VP session_mode: {session_mode!r}")
    return mode  # type: ignore[return-value]


def _session_keys_from_index(
    index: pd.DatetimeIndex,
    session_mode: VolumeProfileSessionMode,
) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype="datetime64[ns]")

    if session_mode == "full":
        keys = [futures_session_date(ts) for ts in index]
        return pd.Series(keys, index=index)

    positions = index.indexer_between_time(SESSION_START, SESSION_END, include_start=True, include_end=False)
    if len(positions) == 0:
        return pd.Series(dtype="datetime64[ns]")
    filtered_index = index.take(positions)
    keys = [_naive_calendar_date(ts) for ts in filtered_index]
    return pd.Series(keys, index=filtered_index)


def _naive_calendar_date(value: pd.Timestamp) -> pd.Timestamp:
    return ensure_eastern_timestamp(value).normalize().tz_localize(None)


def _minimum_prior_bars(session_mode: VolumeProfileSessionMode) -> int:
    return _MIN_PRIOR_FULL_SESSION_BARS if session_mode == "full" else _MIN_PRIOR_RTH_BARS


def _count_core_session_bars(bars: pd.DataFrame, session_mode: VolumeProfileSessionMode) -> int:
    if bars.empty:
        return 0
    if session_mode == "full":
        return len(bars)
    return len(bars.between_time(SESSION_START, SESSION_END, inclusive="left"))


def _rth_session_open(session_key: pd.Timestamp) -> pd.Timestamp:
    session_day = pd.Timestamp(session_key).normalize()
    return pd.Timestamp(session_day, tz=EASTERN_TZ) + pd.Timedelta(hours=9, minutes=30)


def _rth_session_close(session_key: pd.Timestamp) -> pd.Timestamp:
    session_day = pd.Timestamp(session_key).normalize()
    return pd.Timestamp(session_day, tz=EASTERN_TZ) + pd.Timedelta(hours=16)


def _full_session_open(session_key: pd.Timestamp) -> pd.Timestamp:
    session_day = pd.Timestamp(session_key).normalize()
    return pd.Timestamp(session_day, tz=EASTERN_TZ) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)


def _is_post_holiday_session(prev_session_key: pd.Timestamp, current_session_key: pd.Timestamp) -> bool:
    prev_day = pd.Timestamp(prev_session_key).normalize().tz_localize(None)
    current_day = pd.Timestamp(current_session_key).normalize().tz_localize(None)
    expected_gap_days = 3 if prev_day.weekday() == 4 else 1
    expected_next = prev_day + pd.Timedelta(days=expected_gap_days)
    return current_day > expected_next


def _compute_vp(
    bars: pd.DataFrame,
    vp_rows: int,
    va_pct: float,
) -> tuple[float, float, float]:
    """Return (VAH, VAL, VPOC) from a pool of bars."""
    price_low = float(bars["low"].min())
    price_high = float(bars["high"].max())

    if price_low >= price_high:
        mid = (price_low + price_high) / 2.0
        return mid, mid, mid

    bin_edges = np.linspace(price_low, price_high, vp_rows + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    volumes = np.zeros(vp_rows, dtype=float)
    closes = bars["close"].values.astype(float)
    bar_vols = bars["volume"].values.astype(float)

    raw_idx = np.floor((closes - price_low) / bin_width).astype(int)
    raw_idx = np.clip(raw_idx, 0, vp_rows - 1)
    np.add.at(volumes, raw_idx, bar_vols)

    vpoc_idx = int(np.argmax(volumes))
    vpoc = float((bin_edges[vpoc_idx] + bin_edges[vpoc_idx + 1]) / 2.0)

    total = float(volumes.sum())
    target = va_pct * total
    accumulated = float(volumes[vpoc_idx])
    lo_idx = vpoc_idx
    hi_idx = vpoc_idx

    while accumulated < target:
        can_lo = lo_idx > 0
        can_hi = hi_idx < vp_rows - 1
        if not can_lo and not can_hi:
            break
        lo_vol = float(volumes[lo_idx - 1]) if can_lo else -1.0
        hi_vol = float(volumes[hi_idx + 1]) if can_hi else -1.0
        if hi_vol >= lo_vol:
            hi_idx += 1
            accumulated += float(volumes[hi_idx])
        else:
            lo_idx -= 1
            accumulated += float(volumes[lo_idx])

    vah = float(bin_edges[hi_idx + 1])
    val = float(bin_edges[lo_idx])
    return vah, val, vpoc


__all__ = [
    "VP_LEVEL_COLUMNS",
    "VolumeProfileSessionMode",
    "attach_volume_profile",
    "compute_naked_pocs",
    "compute_session_vp",
]
