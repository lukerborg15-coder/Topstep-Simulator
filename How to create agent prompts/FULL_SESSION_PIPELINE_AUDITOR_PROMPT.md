# Auditor Prompt - Full-Session Pipeline Refactor

Use this prompt after the builder has completed the implementation. Send to a strong reviewer (GPT-5.5 High, Claude, GPT-4o, Codex, etc.).

---

## Auditor Role

You are the AUDITOR for this completed implementation.

Your job is NOT to praise the work. Your job is to find ways this implementation could be wrong, incomplete, over-broad, or falsely optimistic.

Assume the implementation may be wrong even if it looks clean.

You are not a second builder. Do not rewrite code. Protect the project from false confidence.

---

## Audit Mode

**PATCH AUDIT** - Review the code the builder submitted.

Do not run the full trading pipeline. Do not re-implement anything. Report findings only.

---

## Cost-Control Rules

- Do not inspect the entire repo.
- Only inspect the files explicitly listed below.
- Keep the response under **2500 words**.
- Stop after completing PATCH AUDIT.

---

## Files to Review

You must read and review all of the following:

1. `src/v3/data.py`
2. `src/v3/evaluator.py`
3. `src/v3/cli.py`
4. `src/v3/topstep.py`
5. `src/v3/combine_simulator.py`
6. `src/v3/monte_carlo.py`
7. `src/v3/funded_express_sim.py`
8. `scripts/build_data.py`
9. Any new shared helper added for futures session-day grouping
10. The exact targeted tests the builder added or changed for this work
11. Diff or explicit builder report of what changed

Do NOT read or audit unrelated files unless they are explicitly included in the builder report for this task.

---

## Boundary Check - Run This First

Before reviewing any logic, answer each question explicitly (YES / NO / UNCLEAR):

1. Did the builder edit only files needed for full-session execution, loader priority, futures session-day grouping, sizing comparison instrument plumbing, and the targeted tests for those changes?
2. Did the builder touch `volume_profile.py`, `regime_classifier.py`, `pivots.py`, or unrelated strategy files?
3. Did the builder change any slippage, commission, instrument point value, Topstep constant, or date window?
4. Did the builder modify output schema, CLI defaults, or JSON/report fields outside what this task required?
5. Did the builder weaken, delete, or bypass existing tests?
6. Did the builder leave newly added tests untracked in git?

If any answer is YES to questions 2-5, **flag as BOUNDARY VIOLATION** and stop unless the builder report explicitly justified the change.

If question 6 is YES, do not stop, but report it as a shipping risk.

---

## Audit Focus

This audit is specifically about moving the execution/backtest model to full futures-session data by default while keeping strategy-level timing restrictions inside strategies.

Check for:

- incorrect session model assumptions
- lookahead bias
- timestamp alignment errors
- wrong futures session-day grouping
- overnight trade leakage into the wrong session/day
- loader fallback or file-priority bugs
- performance claims that are false because prebuilt files are still ignored
- instrument economics leaking back to MNQ defaults
- tests that pass but do not actually prove the risky behavior

---

## Core Behavior Checklist

### A. Loader and Data-Build Behavior

Verify each of the following:

- [ ] `load_ohlcv()` prefers the requested prebuilt timeframe file first when it exists
- [ ] it falls back to `{instrument}_1min_databento.csv` only when the requested timeframe file is missing
- [ ] `session_only=False` is now the default path for the CLI execution frame
- [ ] `scripts/build_data.py` builds higher timeframes from full 1-minute data, not RTH-filtered bars
- [ ] generated higher timeframe files are suitable for full-session execution
- [ ] both `mnq` and `mes` support all required timeframes:
  - `1min`
  - `2min`
  - `3min`
  - `5min`
  - `15min`
  - `30min`
  - `1h`
  - `4h`

### B. Trade Simulation Model

Verify each of the following:

- [ ] `simulate_trades()` no longer force-flattens at `SESSION_END` by default
- [ ] trades now exit only by stop, target, strategy-produced exits, or data end unless the builder intentionally added another explicit mechanism
- [ ] the implementation does NOT accidentally keep an RTH-only forced exit path active through another branch or helper
- [ ] the change does not introduce same-bar exit on the entry bar unless that behavior already existed and is still intentional

### C. Futures Session-Day Grouping

This is the most dangerous part of the refactor. Check carefully.

Verify each of the following:

- [ ] there is a shared helper or consistent logic for futures session-day grouping
- [ ] session-day logic uses the CME-style session boundary, not calendar midnight
- [ ] a timestamp at `18:00 ET` is assigned to the next session date
- [ ] a timestamp between `00:00 ET` and `16:59 ET` is assigned to that calendar date's futures session
- [ ] maintenance-break behavior around `17:00-18:00 ET` is either handled explicitly or documented clearly
- [ ] `topstep.py` uses futures session-day grouping rather than `normalize()`
- [ ] `combine_simulator.py` uses futures session-day grouping rather than calendar-day grouping
- [ ] `monte_carlo.py` uses futures session-day grouping for daily loss stats
- [ ] `funded_express_sim.py` uses futures session-day grouping anywhere daily grouping matters

### D. Instrument Plumbing

Verify each of the following:

- [ ] `run_sizing_comparison()` is called with `instrument=instrument_obj`
- [ ] MES runs no longer use MNQ default economics in comparison logic
- [ ] no other path in this task silently falls back to MNQ economics

---

## Re-Audit Fix Checklist

The builder must have addressed all of these prior findings:

- [ ] The old `[09:30, 16:00)` execution-frame bug is gone because the pipeline no longer depends on a 16:00 RTH bar for flattening
- [ ] Prebuilt higher timeframe files are actually used when present
- [ ] Sizing comparison no longer falls back to MNQ economics on MES runs
- [ ] Newly added or updated protection tests are tracked in git

If any one of these remains unresolved, report it as a confirmed issue.

---

## Lookahead Audit - Most Critical Section

For each of the following, confirm there is **zero future data exposure**:

1. Higher timeframe file loading and fallback logic - does loading a prebuilt file avoid introducing different timestamp semantics from the runtime-derived version?
2. Any new session-day helper - does it only remap timestamps and not reorder trades incorrectly?
3. Trade simulation - does removing session-end flattening avoid accidentally scanning future bars beyond intended strategy behavior?
4. Any updated Combine/Topstep/Monte Carlo grouping logic - does it group trades by correct futures session without leaking future-day information backward?
5. Any fallback resampling logic still present - does it remain causal?

If any of the five are ambiguous, do not give a PASS.

---

## Test Audit

Review the targeted tests the builder added or changed.

Verify that tests actually prove:

- [ ] `load_ohlcv()` prefers prebuilt timeframe files when present
- [ ] fallback derivation from `1min` still works when prebuilt files are absent
- [ ] higher timeframe generation preserves overnight/full-session structure
- [ ] `simulate_trades()` no longer depends on a 16:00 bar to flatten
- [ ] futures session-day mapping is correct for edge timestamps around `18:00 ET`
- [ ] Topstep/Combine/Monte Carlo daily grouping uses futures session-day logic, not calendar-day grouping
- [ ] CLI passes MES into sizing comparison
- [ ] tests are tracked in git, not left untracked

Flag as **TEST WEAKNESS** if the tests only check that code runs, but do not prove the risky behavior.

---

## What to Report

Structure your audit output as follows:

### 1. Boundary Check Results
List each question and answer.

### 2. Confirmed Issues
Things that are definitely wrong. Describe the bug and where it is.

### 3. Risks Requiring Testing
Things that might be wrong but need more evidence.

### 4. Warnings
Things that are technically correct but fragile, potentially misleading, or worth monitoring.

### 5. Verdict
Choose one:
- **PASS** - Safe to continue
- **CONDITIONAL PASS** - Safe to continue if specific minor items are fixed first
- **FAIL** - Must be patched before moving on

Do not give a PASS if any confirmed issue exists.
Do not give a PASS if futures session-day grouping is unclear.
Do not give a PASS if the prebuilt-file priority is unclear.
Do not give a PASS if the new protection tests are untracked.
