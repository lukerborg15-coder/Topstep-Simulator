# Builder Prompt — Bollinger Band Mean Reversion Strategy

Use this prompt when sending this implementation task to Codex, Cursor, or another coding agent.

---

## Builder Role

You are the BUILDER for this task.

Your job is to complete **one narrowly scoped task** safely. You are not here to make the repo cleaner, redesign the system, expand the feature, or improve unrelated code.

You must preserve existing behavior outside the assigned scope.

---

## Task Mode

**PATCH ONLY**

Implement the approved plan below. No new design expansion.

Do not proceed into testing or documentation automatically. Stop after the implementation is complete and report what you did.

---

## Cost-Control Rules

- Do not inspect the entire repo.
- Only inspect and edit the files explicitly listed in this task.
- Do not read the same file repeatedly unless necessary.
- Do not include full file contents in your answer.
- Do not continue into another phase without explicit approval.
- Keep the response under **2000 words** unless explicitly asked otherwise.
- Stop after completing PATCH ONLY.
- If more work is needed, report the next recommended prompt instead of continuing.

---

## Project Rules

- Never use future bars for indicators, entries, exits, regimes, sizing, or validation.
- Never optimize on holdout data.
- Never change thresholds to make a strategy pass.
- Never change trading logic without tests.
- Preserve timestamp alignment.
- Preserve train/test/holdout separation.
- Do not silently ignore bad data.
- Do not weaken, delete, or bypass tests to make the task pass.
- Do not change behavior outside the assigned task.
- Do not change slippage, commissions, Topstep constants, execution assumptions, or date windows unless explicitly allowed.
- Do not swallow errors unless the task explicitly asks for graceful degradation and tests prove it.

---

## Task Risk Level

**HIGH** — new strategy logic, new indicator functions, multi-timeframe resampling, dual exit mode parameter switching.

Do not proceed unless all acceptance criteria below are explicitly met.

---

## Files You Are Allowed to Read and Edit

**Read-only (understand structure, do not edit):**
- `src/v3/strategies.py` — understand `StrategySpec`, `_append_signal`, `register_strategy`
- `src/v3/config.py` — understand `SESSION_START`, `SESSION_END`, `TIMEFRAME_MINUTES`
- `src/v3/data.py` — understand `resample_ohlcv`, `load_ohlcv`
- `src/v3/user_strategies/hl2_sma_retrace_atr.py` — use as the canonical pattern for a user strategy file

**Create (new files only, do not modify existing):**
- `src/v3/user_strategies/bollinger_band_mean_reversion.py` — the new strategy file

**Edit (append only, do not modify existing functions):**
- `src/v3/indicators.py` — append `adx()` and `supertrend()` functions only. Do not modify `atr()`, `rsi()`, or any existing function.

**Do not touch under any circumstances:**
- `src/v3/strategies.py`
- `src/v3/evaluator.py`
- `src/v3/topstep.py`
- `src/v3/monte_carlo.py`
- `src/v3/funded_express_sim.py`
- `src/v3/config.py`
- `src/v3/data.py`
- Any existing test files
- Any existing strategy files

---

## What You Are Building

### Strategy Name
`bollinger_band_mean_reversion`

### Concept
Multi-timeframe mean reversion. A higher timeframe (HTF) confirms the trend is strong and directional. A lower timeframe (LTF) triggers the entry when price touches or crosses the Bollinger Band in the direction of the HTF trend (i.e., a pullback entry). This is NOT a reversal strategy — you are fading a short-term extension in the direction of the bigger trend.

---

## Indicator Prerequisites

Before writing the strategy, append the following two functions to `src/v3/indicators.py`. Do not modify any existing code in that file.

### `adx(df, period=14) -> pd.Series`

Standard Wilder ADX:
1. Compute `+DM` = max(high - prev_high, 0) if > max(prev_low - low, 0) else 0
2. Compute `-DM` = max(prev_low - low, 0) if > max(high - prev_high, 0) else 0
3. Smooth `+DM`, `-DM`, and `TR` using Wilder smoothing: `EWM(alpha=1/period, adjust=False, min_periods=period)`
4. `+DI = 100 * smoothed_plus_dm / smoothed_tr`
5. `-DI = 100 * smoothed_minus_dm / smoothed_tr`
6. `DX = 100 * abs(+DI - -DI) / (+DI + -DI)`
7. `ADX = DX.ewm(alpha=1/period, adjust=False, min_periods=period).mean()`

Return the ADX series. Return `NaN` for bars where smoothed TR is zero or not finite.

### `supertrend(df, period=10, multiplier=3.0) -> tuple[pd.Series, pd.Series]`

Standard Supertrend:
1. `hl2 = (high + low) / 2`
2. `basic_upper = hl2 + multiplier * atr(df, period)`
3. `basic_lower = hl2 - multiplier * atr(df, period)`
4. Final bands are computed iteratively (no future data):
   - `final_upper[i] = basic_upper[i] if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1] else final_upper[i-1]`
   - `final_lower[i] = basic_lower[i] if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1] else final_lower[i-1]`
5. `trend[i] = 1 (bullish) if close[i] > final_upper[i-1] else -1 (bearish) if close[i] < final_lower[i-1] else trend[i-1]`

Return `(supertrend_line, trend_direction)`:
- `supertrend_line`: the active band (final_lower when bullish, final_upper when bearish)
- `trend_direction`: Series of +1 (bullish) or -1 (bearish)

Use explicit Python loops for the iterative band computation. Do not use `shift()` in a vectorized way that would introduce look-ahead.

---

## Entry Logic

### Inputs
The strategy `generate` function receives a **single DataFrame** of LTF bars (already resampled). The HTF bars are computed internally by resampling the LTF data upward using `resample_ohlcv` from `v3.data`. Do not load data from disk inside the strategy.

### HTF Indicators (computed on HTF bars, then forward-filled onto LTF index)
- RSI(14) on HTF close
- Supertrend(10, 3.0) on HTF bars → direction (+1/-1)
- ADX(14) on HTF bars

After computing HTF indicators, forward-fill them onto the LTF bar index using `.reindex(ltf_index, method='ffill')`. This ensures no future HTF data leaks into LTF bars.

### LTF Indicators (computed on LTF bars only)
- Bollinger Bands: 20-period SMA of close, ±2.0 std
- ADX(14) on LTF bars

### LONG Entry — all conditions must be true on bar `i`:
1. HTF RSI > 60
2. HTF Supertrend direction == +1 (bullish)
3. HTF ADX > 35
4. LTF ADX > 20
5. LTF candle low touches or crosses the lower Bollinger Band (`low[i] <= bb_lower[i]`)
6. No trade is currently open
7. Cooldown is not active (at least `cooldown_bars` LTF bars have passed since last exit)

### SHORT Entry — all conditions must be true on bar `i`:
1. HTF RSI < 40
2. HTF Supertrend direction == -1 (bearish)
3. HTF ADX > 35
4. LTF ADX > 20
5. LTF candle high touches or crosses the upper Bollinger Band (`high[i] >= bb_upper[i]`)
6. No trade is currently open
7. Cooldown is not active

Entry price = `close[i]` (bar close, consistent with existing strategies).

---

## Exit Logic

Two exit modes are controlled by a single parameter `exit_mode`:

### Exit Mode A: `"bb_band_tp"`
- **Stop Loss**: `entry ± stop_atr_mult * ATR(14, LTF)` at entry bar
- **Take Profit**: Upper BB (`bb_upper[i]`) for longs, Lower BB (`bb_lower[i]`) for shorts, evaluated at the exit bar dynamically (the BB moves bar to bar)
- At each post-entry bar, TP target is `bb_upper[j]` (long) or `bb_lower[j]` (short), recalculated each bar

### Exit Mode B: `"fixed_rr"`
- **Stop Loss**: `entry ± stop_atr_mult * ATR(14, LTF)` at entry bar
- **Take Profit**: `entry ± (stop_atr_mult * rr_mult) * ATR(14, LTF)` — fixed at entry, does not move

Session end exit applies to both modes (exit at session end bar regardless).

Use the same `_exit_bar_index` pattern as `hl2_sma_retrace_atr.py` but adapted:
- For `bb_band_tp`: loop forward, check stop first, then check if `high[j] >= bb_upper[j]` (long) or `low[j] <= bb_lower[j]` (short), then session end
- For `fixed_rr`: loop forward, check stop first, then fixed target, then session end

---

## Timeframe Parameterization

The strategy handles HTF resampling internally. Parameters control which timeframes to use:

- `htf_timeframe`: string, one of `"15min"`, `"30min"`, `"1h"` — the HTF timeframe
- `ltf_timeframe`: string, one of `"5min"`, `"15min"` — the LTF timeframe (this is the timeframe of the input `df`)

**Important constraint**: HTF must always be coarser than LTF. The builder must add a guard:
```python
assert TIMEFRAME_MINUTES[htf_timeframe] > TIMEFRAME_MINUTES[ltf_timeframe], \
    f"HTF must be coarser than LTF: got {htf_timeframe} vs {ltf_timeframe}"
```

When the strategy is called, the input `df` is already at the LTF timeframe. The strategy resamples internally to HTF.

---

## Parameter Grid

The strategy must register with the following `param_grid` for full sweep testing:

```python
default_params = {
    "htf_timeframe": "15min",       # default HTF
    "ltf_timeframe": "5min",        # default LTF (matches input df timeframe)
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "supertrend_period": 10,
    "supertrend_mult": 3.0,
    "htf_rsi_period": 14,
    "htf_adx_period": 14,
    "ltf_adx_period": 14,
    "htf_rsi_long": 60,
    "htf_rsi_short": 40,
    "htf_adx_thresh": 35,
    "ltf_adx_thresh": 20,
    "stop_atr_mult": 2.0,
    "exit_mode": "bb_band_tp",      # "bb_band_tp" or "fixed_rr"
    "rr_mult": 1.5,                 # only used in fixed_rr mode
    "cooldown_bars": 10,
}

param_grid = {
    # Timeframe combos
    "htf_timeframe": ("15min", "30min", "1h"),
    "ltf_timeframe": ("5min", "15min"),           # caller must pass correct df resolution

    # Exit Set A — BB band TP with ATR stops
    # Exit Set B — fixed RR
    "exit_mode": ("bb_band_tp", "fixed_rr"),

    # ATR stop multipliers (shared across both exit modes)
    "stop_atr_mult": (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0),

    # RR multiplier (only used in fixed_rr mode, ignored in bb_band_tp)
    "rr_mult": (1.5,),
}
```

**Note on TF combos**: The grid will produce combinations like `htf_timeframe="15min", ltf_timeframe="15min"` which are invalid (HTF not coarser than LTF). The generate function must detect this and return an empty signal list (do not raise — return `[]`). This avoids crashing the sweep.

---

## Implementation Pattern

Follow the exact same module structure as `src/v3/user_strategies/hl2_sma_retrace_atr.py`:

1. Module-level `STRATEGY_KEY = "bollinger_band_mean_reversion"`
2. A private `_exit_bar_index(...)` function
3. A `bollinger_band_mean_reversion_generate(df, params)` function
4. A `register_strategy(StrategySpec(...))` call at module bottom

Import from:
```python
from v3.config import SESSION_END, TIMEFRAME_MINUTES
from v3.data import resample_ohlcv
from v3.indicators import atr, rsi, adx, supertrend
from v3.strategies import StrategySpec, _append_signal, register_strategy
```

---

## Acceptance Criteria

Before submitting, verify each of the following manually:

- [ ] `adx()` appended to `indicators.py` — no existing functions modified
- [ ] `supertrend()` appended to `indicators.py` — no existing functions modified
- [ ] `bollinger_band_mean_reversion.py` created in `user_strategies/`
- [ ] HTF indicators are forward-filled onto LTF index with `.reindex(method='ffill')` — confirmed no `.shift()` lookahead
- [ ] Entry uses `close[i]` as entry price
- [ ] Stop and TP are computed at the entry bar for `fixed_rr`, and stop only at entry bar for `bb_band_tp`
- [ ] `bb_band_tp` TP target uses the dynamic BB band at each exit-scan bar `j`, not fixed at entry
- [ ] Invalid TF combos (HTF <= LTF) return `[]` without raising
- [ ] Cooldown is counted in LTF bars (not calendar time)
- [ ] Session end exit is applied in both exit modes
- [ ] `register_strategy()` is called and the strategy is importable
- [ ] No changes to any file outside the allowed list

---

## What to Report When Done

1. Exact lines appended to `indicators.py` (show the full new functions only)
2. Full content of `bollinger_band_mean_reversion.py`
3. Confirmation of each acceptance criterion above (check or flag)
4. Any decisions you made that deviated from this spec (there should be none — if you deviated, explain why)

Do not run the pipeline. Do not run tests. Stop here and wait for auditor review.
