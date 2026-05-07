# Builder Task — Bollinger Band Mean Reversion Strategy
# (Self-Contained — All Source Files Embedded)

---

## Your Job

You are a PATCH-ONLY coding agent. You will create **two things**:

1. Append two new functions (`adx` and `supertrend`) to `src/v3/indicators.py`
2. Create a new file: `src/v3/user_strategies/bollinger_band_mean_reversion.py`

That is the **entire scope**. Touch nothing else.

---

## Hard Rules

- Do NOT modify any existing function in any file.
- Do NOT touch `strategies.py`, `evaluator.py`, `topstep.py`, `config.py`, `data.py`, or any test file.
- Do NOT load data from disk inside the strategy — the input `df` is already the LTF DataFrame.
- Do NOT use future bar data anywhere.
- Do NOT run the pipeline or any tests. Just produce the two code outputs.
- If you are unsure about scope, do less, not more.

---

## Existing Source Files (Read These — Do Not Modify)

### `src/v3/config.py` (relevant excerpt)

```python
SESSION_END = "16:00"
SESSION_START = "09:30"

TIMEFRAME_MINUTES: dict[str, int] = {
    "1min": 1,
    "2min": 2,
    "3min": 3,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "4h": 240,
}
```

### `src/v3/indicators.py` (current full file — APPEND TO THIS, DO NOT MODIFY EXISTING FUNCTIONS)

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def linreg_value(series: pd.Series, period: int) -> pd.Series:
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_mean = values.mean()
        slope = ((x - x_mean) * (values - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return float(slope * (period - 1) + intercept)

    return series.rolling(period, min_periods=period).apply(calc, raw=True)


def directional_efficiency(close: pd.Series, lookback: int = 50) -> pd.Series:
    direction = (close - close.shift(lookback)).abs()
    path = close.diff().abs().rolling(lookback, min_periods=lookback).sum()
    return direction / path.replace(0, np.nan)


def rolling_slope(close: pd.Series, lookback: int = 50) -> pd.Series:
    x = np.arange(lookback, dtype=float)
    x = x - x.mean()
    denom = (x**2).sum()

    def calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values - values.mean()
        return float((x * y).sum() / denom)

    return close.rolling(lookback, min_periods=lookback).apply(calc, raw=True)

# ← APPEND YOUR NEW FUNCTIONS AFTER THIS LINE. DO NOT EDIT ANYTHING ABOVE.
```

### `src/v3/strategies.py` (relevant excerpt — understand _append_signal and register_strategy)

```python
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TradeSignal:
    time: pd.Timestamp
    direction: str       # "long" or "short"
    entry: float
    stop: float
    target: float
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class StrategySpec:
    name: str
    generate: Callable
    default_params: dict[str, Any]
    param_grid: dict[str, tuple[Any, ...]]
    max_signals_per_day: int | None
    session_start: str = "09:30"
    session_end: str = "16:00"
    requires: tuple[str, ...] = ()
    filter_of: str | None = None

def _append_signal(signals, ts, direction, entry, stop, target, strategy, params, metadata=None):
    """Validates geometry then appends. For longs: stop < entry < target required.
    For shorts: target < entry < stop required. Silently drops invalid signals."""
    if not np.isfinite([entry, stop, target]).all():
        return
    if direction > 0 and not (stop < entry < target):
        return
    if direction < 0 and not (target < entry < stop):
        return
    signals.append(TradeSignal(
        time=ts,
        direction="long" if direction > 0 else "short",
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        strategy=strategy,
        params=dict(params),
        metadata=metadata or {},
    ))

def register_strategy(spec: StrategySpec) -> None:
    """Register a strategy — validates it and adds to global STRATEGIES dict."""
    # (validator and STRATEGIES dict are already wired — just call this at module bottom)
    ...
```

### `src/v3/data.py` (relevant excerpt — understand resample_ohlcv)

```python
def resample_ohlcv(frame: pd.DataFrame, timeframe: str, *, session_only: bool = False) -> pd.DataFrame:
    """Resample a DataFrame to a higher timeframe.
    - frame: tz-aware datetime-indexed OHLCV DataFrame
    - timeframe: one of the keys in TIMEFRAME_MINUTES (e.g. '15min', '30min', '1h')
    - session_only: if True, filters to 09:30-16:00 ET and anchors bars to session open
    Returns a resampled OHLCV DataFrame. Raises ValueError for unsupported timeframes.
    """
    ...
```

### `src/v3/user_strategies/hl2_sma_retrace_atr.py` (CANONICAL PATTERN — follow this exactly)

```python
"""HL2 SMA retrace with ATR stop/target — user strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from v3.config import SESSION_END
from v3.indicators import atr
from v3.strategies import StrategySpec, _append_signal, register_strategy

STRATEGY_KEY = "hl2_sma_retrace_atr"


def _exit_bar_index(df, entry_idx, *, stop, target, is_long):
    """Scan forward from entry_idx+1. Stop checked before target on same bar."""
    session_end_time = pd.Timestamp(SESSION_END).time()
    for j in range(entry_idx + 1, len(df)):
        row = df.iloc[j]
        ts = df.index[j]
        if is_long:
            if row["low"] <= stop:
                return j
            if row["high"] >= target:
                return j
        else:
            if row["high"] >= stop:
                return j
            if row["low"] <= target:
                return j
        if ts.time() >= session_end_time:
            return j
    return len(df) - 1


def hl2_sma_retrace_generate(df: pd.DataFrame, params: dict) -> list:
    # ... (indicator computation, signal loop, etc.)
    signals: list = []
    # ... loop that calls _append_signal and sets next_allowed
    return signals


register_strategy(
    StrategySpec(
        name=STRATEGY_KEY,
        generate=hl2_sma_retrace_generate,
        default_params={
            "ma_length": 21,
            "atr_period": 14,
            "untouched_lookback": 8,
            "stop_atr_mult": 1.0,
            "target_atr_mult": 2.0,
        },
        param_grid={
            "ma_length": tuple(range(15, 31)),
            "atr_period": (10, 14, 20),
            "untouched_lookback": tuple(range(5, 16)),
            "stop_atr_mult": (0.5, 1.0, 1.5, 2.0),
            "target_atr_mult": (0.5, 1.0, 1.5, 2.0),
        },
        max_signals_per_day=None,
    )
)
```

---

## Output 1 — Append to `src/v3/indicators.py`

Append these two functions **after the last existing function** in `indicators.py`. Do not modify anything above.

### Function: `adx(df, period=14) -> pd.Series`

Standard Wilder ADX. Implementation requirements:

```
+DM[i] = max(high[i] - high[i-1], 0) if that value > max(low[i-1] - low[i], 0) else 0
-DM[i] = max(low[i-1] - low[i], 0)   if that value > max(high[i] - high[i-1], 0) else 0
TR = true_range(df)   ← use the existing true_range() function

Smooth +DM, -DM, TR using Wilder: series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

+DI = 100 * smoothed_plus_dm / smoothed_tr
-DI = 100 * smoothed_minus_dm / smoothed_tr
DX  = 100 * abs(+DI - -DI) / (+DI + -DI)      ← replace 0-denominator with NaN
ADX = DX.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
```

Return the ADX Series (same index as input df).

### Function: `supertrend(df, period=10, multiplier=3.0) -> tuple[pd.Series, pd.Series]`

Standard Supertrend. Implementation requirements:

```
hl2 = (df["high"] + df["low"]) / 2
atr_vals = atr(df, period)          ← use the existing atr() function
basic_upper = hl2 + multiplier * atr_vals
basic_lower = hl2 - multiplier * atr_vals

Initialize final_upper, final_lower, trend as numpy arrays of NaN/0.

Loop i from 1 to len(df)-1 (explicit Python for-loop — no vectorized shift):
    # Upper band
    if basic_upper[i] < final_upper[i-1] or df["close"][i-1] > final_upper[i-1]:
        final_upper[i] = basic_upper[i]
    else:
        final_upper[i] = final_upper[i-1]

    # Lower band
    if basic_lower[i] > final_lower[i-1] or df["close"][i-1] < final_lower[i-1]:
        final_lower[i] = basic_lower[i]
    else:
        final_lower[i] = final_lower[i-1]

    # Trend direction
    prev_trend = trend[i-1]
    close_i = df["close"].iloc[i]
    if prev_trend == -1 and close_i > final_upper[i]:
        trend[i] = 1
    elif prev_trend == 1 and close_i < final_lower[i]:
        trend[i] = -1
    else:
        trend[i] = prev_trend if prev_trend != 0 else 1  # default bullish on warmup end

supertrend_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
```

Return `(pd.Series(supertrend_line, index=df.index), pd.Series(trend, index=df.index))`.
Warm-up bars (where ATR is NaN) should have NaN in supertrend_line and 0 in trend.

---

## Output 2 — Create `src/v3/user_strategies/bollinger_band_mean_reversion.py`

Follow the exact module structure of `hl2_sma_retrace_atr.py`. Here is the complete specification:

### Imports

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from v3.config import SESSION_END, TIMEFRAME_MINUTES
from v3.data import resample_ohlcv
from v3.indicators import atr, rsi, adx, supertrend
from v3.strategies import StrategySpec, _append_signal, register_strategy

STRATEGY_KEY = "bollinger_band_mean_reversion"
```

### `_exit_bar_index` function

Two exit modes, controlled by `exit_mode` parameter:

**Mode A `"bb_band_tp"`**: Stop is fixed (set at entry). TP is the dynamic BB band evaluated at each forward bar `j`.
- Long: check `low[j] <= stop` (stop hit), then check `high[j] >= bb_upper[j]` (TP hit)
- Short: check `high[j] >= stop` (stop hit), then check `low[j] <= bb_lower[j]` (TP hit)
- Session end: exit if `ts.time() >= session_end_time`

**Mode B `"fixed_rr"`**: Both stop and target are fixed at entry. Same scan logic as `hl2_sma_retrace_atr._exit_bar_index`.

Signature:
```python
def _exit_bar_index(
    df: pd.DataFrame,
    entry_idx: int,
    *,
    stop: float,
    target: float,
    is_long: bool,
    exit_mode: str,
    bb_upper: np.ndarray,   # only used in bb_band_tp mode
    bb_lower: np.ndarray,   # only used in bb_band_tp mode
) -> int:
```

### `bollinger_band_mean_reversion_generate(df, params)` function

Step-by-step:

**Step 1 — Validate timeframe combo**
```python
htf_tf = params["htf_timeframe"]
ltf_tf = params["ltf_timeframe"]
if TIMEFRAME_MINUTES.get(htf_tf, 0) <= TIMEFRAME_MINUTES.get(ltf_tf, 0):
    return []   # invalid combo — silent return, not a crash
```

**Step 2 — Resample to HTF**
```python
htf_df = resample_ohlcv(df, htf_tf, session_only=False)
```

**Step 3 — Compute HTF indicators on HTF DataFrame**
```python
htf_rsi   = rsi(htf_df["close"], int(params["htf_rsi_period"]))
htf_adx   = adx(htf_df, int(params["htf_adx_period"]))
_, htf_st = supertrend(htf_df, int(params["supertrend_period"]), float(params["supertrend_mult"]))
```

**Step 4 — Forward-fill HTF indicators onto LTF index**
```python
ltf_index = df.index
htf_rsi_ltf = htf_rsi.reindex(ltf_index, method="ffill")
htf_adx_ltf = htf_adx.reindex(ltf_index, method="ffill")
htf_st_ltf  = htf_st.reindex(ltf_index, method="ffill")
```

**Step 5 — Compute LTF indicators on LTF DataFrame**
```python
atr_period = int(params["atr_period"])
bb_period  = int(params["bb_period"])
bb_std     = float(params["bb_std"])

atr_vals  = atr(df, atr_period)
bb_mid    = df["close"].rolling(bb_period, min_periods=bb_period).mean()
bb_std_s  = df["close"].rolling(bb_period, min_periods=bb_period).std()
bb_upper  = (bb_mid + bb_std * bb_std_s).to_numpy(dtype=float)
bb_lower  = (bb_mid - bb_std * bb_std_s).to_numpy(dtype=float)
ltf_adx   = adx(df, int(params["ltf_adx_period"]))
```

**Step 6 — Convert to numpy arrays for fast iteration**
```python
closes    = df["close"].to_numpy(dtype=float)
highs     = df["high"].to_numpy(dtype=float)
lows      = df["low"].to_numpy(dtype=float)
atr_arr   = atr_vals.to_numpy(dtype=float)
ltf_adx_a = ltf_adx.to_numpy(dtype=float)
htf_rsi_a = htf_rsi_ltf.to_numpy(dtype=float)
htf_adx_a = htf_adx_ltf.to_numpy(dtype=float)
htf_st_a  = htf_st_ltf.to_numpy(dtype=float)
```

**Step 7 — Signal loop**
```python
signals: list = []
cooldown_bars = int(params["cooldown_bars"])
htf_rsi_long  = float(params["htf_rsi_long"])
htf_rsi_short = float(params["htf_rsi_short"])
htf_adx_thresh = float(params["htf_adx_thresh"])
ltf_adx_thresh = float(params["ltf_adx_thresh"])
stop_atr_mult  = float(params["stop_atr_mult"])
exit_mode      = str(params["exit_mode"])
rr_mult        = float(params["rr_mult"])

warmup = max(bb_period, atr_period, int(params["htf_adx_period"]),
             int(params["ltf_adx_period"]), int(params["supertrend_period"]),
             int(params["htf_rsi_period"])) + 1

next_allowed = warmup   # bars before this index are skipped (cooldown + warmup)

for i in range(warmup, len(df)):
    if i < next_allowed:
        continue

    # Check all values are finite before using them
    vals = [htf_rsi_a[i], htf_adx_a[i], htf_st_a[i], ltf_adx_a[i],
            atr_arr[i], bb_upper[i], bb_lower[i], closes[i]]
    if not all(np.isfinite(v) for v in vals):
        continue

    a    = atr_arr[i]
    ts   = df.index[i]

    # LONG entry check
    long_cond = (
        htf_rsi_a[i]  > htf_rsi_long
        and htf_st_a[i] == 1
        and htf_adx_a[i] > htf_adx_thresh
        and ltf_adx_a[i] > ltf_adx_thresh
        and lows[i] <= bb_lower[i]
    )

    # SHORT entry check
    short_cond = (
        htf_rsi_a[i]  < htf_rsi_short
        and htf_st_a[i] == -1
        and htf_adx_a[i] > htf_adx_thresh
        and ltf_adx_a[i] > ltf_adx_thresh
        and highs[i] >= bb_upper[i]
    )

    if long_cond:
        entry = closes[i]
        stop  = entry - stop_atr_mult * a
        if exit_mode == "fixed_rr":
            target = entry + stop_atr_mult * rr_mult * a
        else:
            target = entry + 1.0 * a   # placeholder — dynamic TP handled in exit scanner
        _append_signal(signals, ts, 1, entry, stop, target, STRATEGY_KEY, params)
        exit_idx = _exit_bar_index(
            df, i, stop=stop, target=target, is_long=True,
            exit_mode=exit_mode, bb_upper=bb_upper, bb_lower=bb_lower,
        )
        next_allowed = exit_idx + 1 + cooldown_bars

    elif short_cond:
        entry = closes[i]
        stop  = entry + stop_atr_mult * a
        if exit_mode == "fixed_rr":
            target = entry - stop_atr_mult * rr_mult * a
        else:
            target = entry - 1.0 * a   # placeholder — dynamic TP handled in exit scanner
        _append_signal(signals, ts, -1, entry, stop, target, STRATEGY_KEY, params)
        exit_idx = _exit_bar_index(
            df, i, stop=stop, target=target, is_long=False,
            exit_mode=exit_mode, bb_upper=bb_upper, bb_lower=bb_lower,
        )
        next_allowed = exit_idx + 1 + cooldown_bars

return signals
```

**Important note on `bb_band_tp` placeholder target**: In `bb_band_tp` mode the placeholder target (`entry ± 1 ATR`) is only used to pass `_append_signal`'s geometry guard. The actual exit is determined by `_exit_bar_index` scanning for when price touches the dynamic BB band. The `target` stored on the signal is intentionally approximate — the evaluator uses the actual exit bar price, not the stored target field.

### `register_strategy` call

```python
register_strategy(
    StrategySpec(
        name=STRATEGY_KEY,
        generate=bollinger_band_mean_reversion_generate,
        default_params={
            "htf_timeframe":      "15min",
            "ltf_timeframe":      "5min",
            "bb_period":          20,
            "bb_std":             2.0,
            "atr_period":         14,
            "supertrend_period":  10,
            "supertrend_mult":    3.0,
            "htf_rsi_period":     14,
            "htf_adx_period":     14,
            "ltf_adx_period":     14,
            "htf_rsi_long":       60.0,
            "htf_rsi_short":      40.0,
            "htf_adx_thresh":     35.0,
            "ltf_adx_thresh":     20.0,
            "stop_atr_mult":      2.0,
            "exit_mode":          "bb_band_tp",
            "rr_mult":            1.5,
            "cooldown_bars":      10,
        },
        param_grid={
            "htf_timeframe":  ("15min", "30min", "1h"),
            "ltf_timeframe":  ("5min", "15min"),
            "exit_mode":      ("bb_band_tp", "fixed_rr"),
            "stop_atr_mult":  (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0),
            "rr_mult":        (1.5,),
        },
        max_signals_per_day=None,
    )
)
```

---

## Acceptance Checklist — Verify Before Submitting

- [ ] `adx()` appended to `indicators.py` — no existing code modified
- [ ] `supertrend()` appended to `indicators.py` — uses explicit Python loop, no vectorized lookahead
- [ ] `bollinger_band_mean_reversion.py` created in `user_strategies/`
- [ ] HTF indicators computed on HTF DataFrame, forward-filled with `.reindex(method="ffill")`
- [ ] Invalid TF combos return `[]` without raising
- [ ] Long entry: LTF low touches or crosses BB lower
- [ ] Short entry: LTF high touches or crosses BB upper
- [ ] `bb_band_tp` exit scanner uses `bb_upper[j]`/`bb_lower[j]` at bar `j` (dynamic)
- [ ] `fixed_rr` exit scanner uses fixed stop and target set at entry bar
- [ ] Session end exit applied in both modes
- [ ] Cooldown is in LTF bars
- [ ] `register_strategy()` called at module bottom
- [ ] No other files modified

---

## What to Deliver

1. Full content of the two new functions to append to `indicators.py`
2. Full content of `src/v3/user_strategies/bollinger_band_mean_reversion.py`
3. Confirmation of each checkbox above

Stop here. Do not run anything. Do not modify anything else.
