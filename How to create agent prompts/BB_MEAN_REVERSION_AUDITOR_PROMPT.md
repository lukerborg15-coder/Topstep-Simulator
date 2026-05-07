# Auditor Prompt — Bollinger Band Mean Reversion Strategy

Use this prompt after the builder has completed the implementation. Send to a strong reviewer (GPT-4o, Claude, etc.).

---

## Auditor Role

You are the AUDITOR for this completed implementation.

Your job is NOT to praise the work. Your job is to find ways this implementation could be wrong, incomplete, over-broad, or falsely optimistic.

Assume the implementation may be wrong even if it looks clean.

You are not a second builder. Do not rewrite code. Protect the project from false confidence.

---

## Audit Mode

**PATCH AUDIT** — Review the code the builder submitted.

Do not run the pipeline. Do not re-implement anything. Report findings only.

---

## Cost-Control Rules

- Do not inspect the entire repo.
- Only inspect the files explicitly listed below.
- Keep the response under **2500 words**.
- Stop after completing PATCH AUDIT.

---

## Files to Review

You must read and review all of the following:

1. `src/v3/indicators.py` — confirm only `adx()` and `supertrend()` were appended; no existing functions modified
2. `src/v3/user_strategies/bollinger_band_mean_reversion.py` — full strategy implementation
3. Diff or explicit builder report of what changed (provided below or attached)

Do NOT read or audit any other files.

---

## Boundary Check — Run This First

Before reviewing any logic, answer each question explicitly (YES / NO / UNCLEAR):

1. Did the builder edit only the allowed files (`indicators.py` and the new strategy file)?
2. Did the builder modify any existing function in `indicators.py`?
3. Did the builder touch `strategies.py`, `evaluator.py`, `topstep.py`, `config.py`, `data.py`, or any test file?
4. Did the builder create any file other than `bollinger_band_mean_reversion.py`?
5. Did the builder change any slippage, commission, Topstep constant, or date window?
6. Did the builder weaken, delete, or bypass any existing test?

If any answer is YES to questions 2–6, **flag as BOUNDARY VIOLATION** and stop.

---

## Indicator Correctness Check

### `adx()` in `indicators.py`

Verify each of the following:

- [ ] `+DM` is computed as `max(high - prev_high, 0)` only when it exceeds `max(prev_low - low, 0)`, else 0
- [ ] `-DM` is computed as `max(prev_low - low, 0)` only when it exceeds `max(high - prev_high, 0)`, else 0
- [ ] `+DM`, `-DM`, and TR are smoothed with Wilder smoothing (`ewm(alpha=1/period, adjust=False, min_periods=period)`)
- [ ] ADX is computed as the EWM of DX, not DX itself
- [ ] Returns `NaN` where smoothed TR is zero or non-finite (no division by zero crash)
- [ ] No future bar data is used at any point

### `supertrend()` in `indicators.py`

Verify each of the following:

- [ ] `hl2 = (high + low) / 2` is correct
- [ ] Basic bands use ATR from the same file's `atr()` function
- [ ] Final band computation uses an **explicit Python loop** (not vectorized `.shift()`) — this is critical for no-lookahead compliance
- [ ] At bar `i`, final bands reference `final_upper[i-1]` and `close[i-1]` only — no bar `i` or later data
- [ ] Trend direction is `+1` when bullish, `-1` when bearish
- [ ] Returns a tuple of two Series: `(supertrend_line, trend_direction)`
- [ ] First `period` bars return `NaN` / neutral values correctly (no index error on warm-up)

---

## Strategy Logic Check

### HTF Resampling and Forward-Fill

This is the most dangerous part of a multi-timeframe strategy. Check carefully:

- [ ] HTF bars are computed by calling `resample_ohlcv(df, htf_timeframe, ...)` — data comes from the input `df`, NOT loaded from disk
- [ ] HTF indicators (`rsi`, `supertrend`, `adx`) are computed **on the HTF DataFrame**, not the LTF DataFrame
- [ ] HTF indicators are aligned to the LTF index using `.reindex(ltf_index, method='ffill')` — confirm this exact method is used
- [ ] Forward-fill means HTF bar `N` only influences LTF bars **after** HTF bar `N` closes — no same-bar contamination
- [ ] The builder does NOT use `.shift()` on HTF indicators to align them, which would cause a one-bar offset bug

**Critical check**: On the bar where a new HTF candle closes, does the strategy correctly use the *just-closed* HTF bar's values or the *previous* HTF bar's values? Either can be defended, but the builder must be consistent and not accidentally use *future* HTF values. Flag if ambiguous.

### Entry Conditions

- [ ] LONG entry: HTF RSI > 60, HTF Supertrend == +1, HTF ADX > 35, LTF ADX > 20, LTF low <= BB lower
- [ ] SHORT entry: HTF RSI < 40, HTF Supertrend == -1, HTF ADX > 35, LTF ADX > 20, LTF high >= BB upper
- [ ] Entry price is `close[i]` (not `open[i+1]` or any other bar — flag if different)
- [ ] No trade is entered if `i < next_allowed` (cooldown enforced)
- [ ] Cooldown is counted in **LTF bars**, not HTF bars and not calendar time

### Exit Logic — Mode A (`bb_band_tp`)

- [ ] Stop is computed from ATR at the **entry bar** and does not move
- [ ] TP target is the BB band value at **each exit-scan bar `j`** (dynamic), not fixed at entry
- [ ] For longs: stop is below entry, TP is `bb_upper[j]` — check the sign is correct
- [ ] For shorts: stop is above entry, TP is `bb_lower[j]` — check the sign is correct
- [ ] Session end exit is applied

### Exit Logic — Mode B (`fixed_rr`)

- [ ] Stop and TP are both computed at the **entry bar** and fixed
- [ ] TP = `entry ± stop_atr_mult * rr_mult * ATR` — confirm the formula
- [ ] For a `stop_atr_mult=4.0` and `rr_mult=1.5`, TP should be `6.0 ATR` from entry — spot-check one example
- [ ] Session end exit is applied

### Invalid Timeframe Combos

- [ ] When `htf_timeframe` is NOT coarser than `ltf_timeframe` (e.g., `htf="15min", ltf="15min"`), the function returns `[]` without raising
- [ ] This is a silent return, not a crash — important for sweep runs

### `_append_signal` Guard

- [ ] All signals pass through `_append_signal` from `strategies.py`
- [ ] For longs: `stop < entry < target` — if this is violated, the signal is silently dropped (correct behavior)
- [ ] For shorts: `target < entry < stop` — same
- [ ] In `bb_band_tp` mode, if at entry the BB upper is already below the entry price for a long, this will correctly fail the guard and drop the signal — confirm this is handled

---

## Parameter Grid Check

- [ ] `stop_atr_mult` grid covers all 8 values: `(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)`
- [ ] `exit_mode` grid covers both: `("bb_band_tp", "fixed_rr")`
- [ ] `htf_timeframe` grid covers all 3: `("15min", "30min", "1h")`
- [ ] `ltf_timeframe` grid covers: `("5min", "15min")`
- [ ] `rr_mult` is `(1.5,)` — single value is intentional
- [ ] `register_strategy(StrategySpec(...))` is called at module bottom
- [ ] `STRATEGY_KEY = "bollinger_band_mean_reversion"` matches the registered name

---

## Lookahead Audit — Most Critical Section

For each of the following, confirm there is **zero future data exposure**:

1. `supertrend()` iterative loop — does bar `i` use only bars `0..i`?
2. `adx()` EWM — does EWM with `adjust=False, min_periods=period` use only past data? (Yes, it does — but confirm the builder didn't accidentally use `adjust=True` which changes the weighting but is still causal.)
3. HTF reindex with `ffill` — does the forward-fill correctly propagate the *last known* HTF value, not a future HTF value?
4. BB bands — rolling mean and std with `min_periods=period` — causal? Yes, but confirm.
5. `_exit_bar_index` — does it scan bars *after* entry (`range(entry_idx + 1, ...)`)? Confirm it does not exit on the entry bar itself.

---

## What to Report

Structure your audit output as follows:

### 1. Boundary Check Results
List each question and answer.

### 2. Confirmed Issues
Things that are definitely wrong. Describe the bug and where it is.

### 3. Risks Requiring Testing
Things that might be wrong but need a test run to confirm.

### 4. Warnings
Things that are technically correct but fragile, potentially misleading, or worth monitoring.

### 5. Verdict
Choose one:
- **PASS** — Safe to run tests
- **CONDITIONAL PASS** — Safe to run tests if specific minor items are fixed first (list them)
- **FAIL** — Must be patched before running tests (list blocking issues)

Do not give a PASS if any confirmed issue exists. Do not give a PASS if lookahead compliance is UNCLEAR on any of the five points above.
