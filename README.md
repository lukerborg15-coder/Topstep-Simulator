# Topstep Pipeline

Backtesting and evaluation pipeline for Topstep-style futures strategy testing. The current runtime supports both `mnq` and `mes`, uses full-session futures data by default, prefers prebuilt timeframe CSVs when they exist, and falls back to 1-minute resampling only when needed.

## Quick Start

```powershell
pip install -e .
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:PYTHONIOENCODING = "utf-8"

py -3.13 -m v3.cli --list-strategies
py -3.13 -m v3.cli --strategy ttm_squeeze --instrument mnq --timeframe 5min --mode quick
```

Console entry point after install:

```powershell
topstep-pipeline --strategy ttm_squeeze --instrument mnq --timeframe 5min --mode quick
```

## Requirements

- Python 3.13 recommended on this machine
- `pandas`, `numpy`, `scipy`, `matplotlib`

Install:

```powershell
pip install -e .
```

## Data Layout

Expected data folder: `Data/`

Current naming convention:

```text
Data/
  mnq_1min_databento.csv
  mnq_2min_databento.csv
  mnq_3min_databento.csv
  mnq_5min_databento.csv
  mnq_15min_databento.csv
  mnq_30min_databento.csv
  mnq_1h_databento.csv
  mnq_4h_databento.csv
  mes_1min_databento.csv
  mes_2min_databento.csv
  mes_3min_databento.csv
  mes_5min_databento.csv
  mes_15min_databento.csv
  mes_30min_databento.csv
  mes_1h_databento.csv
  mes_4h_databento.csv
```

Supported timeframes:

- `1min`
- `2min`
- `3min`
- `5min`
- `15min`
- `30min`
- `1h`
- `4h`

Loader behavior in [src/v3/data.py](C:/Users/Luker/projects/Topstep pipeline/src/v3/data.py:57):

- Tries `{instrument}_{timeframe}_databento.csv` first
- Falls back to `{instrument}_1min_databento.csv` only if the requested timeframe file is missing
- Main execution frame runs full-session by default with `session_only=False`

Data CSVs in `Data/` are ignored by git on purpose.

## Building Prebuilt Timeframes

Generate higher timeframes once instead of resampling on every run:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:PYTHONIOENCODING = "utf-8"

py -3.13 .\scripts\build_data.py --instrument mnq --timeframes 2min 3min 5min 15min 30min 1h 4h --output-dir Data
py -3.13 .\scripts\build_data.py --instrument mes --timeframes 2min 3min 5min 15min 30min 1h 4h --output-dir Data
```

The builder resamples from full-session 1-minute data and preserves overnight structure.

## Pipeline Modes

- `quick`
  Runs validation, walk-forward, holdout, MC, regime, verdict. Skips sensitivity.
- `full`
  Runs the full pipeline including sensitivity.
- `holdout-only`
  Skips walk-forward and sensitivity. Evaluates holdout with default params.

Examples:

```powershell
py -3.13 -m v3.cli --strategy ttm_squeeze --instrument mnq --timeframe 5min --mode quick
py -3.13 -m v3.cli --strategy ttm_squeeze --instrument mes --timeframe 15min --mode full
py -3.13 -m v3.cli --strategy orb_wick_rejection --instrument mnq --timeframe 3min --mode holdout-only
```

## Built-in Strategies

- `connors_rsi2`
- `ttm_squeeze`
- `orb_ib`
- `orb_volatility_filtered`
- `orb_wick_rejection`
- `session_pivot_rejection`
- `session_pivot_break`
- `hl2_sma_retrace_atr`

List them live:

```powershell
py -3.13 -m v3.cli --list-strategies
```

## Important CLI Flags

- `--strategy`
  Registered strategy key
- `--instrument`
  `mnq` or `mes`
- `--timeframe`
  One of `1min 2min 3min 5min 15min 30min 1h 4h`
- `--mode`
  `quick`, `full`, or `holdout-only`
- `--data-dir`
  Defaults to `Data/`
- `--output-dir`
  Defaults to the configured output root
- `--eval-risk`
  Default `$500`
- `--max-contracts`
  Default `50`
- `--min-wf-passes`
  Default `2`
- `--min-eval-passes-per-fold`
  Default `2`
- `--min-fold-seq-pass-rate-pct`
  Default `40`
- `--force`
  Continue even if walk-forward gates fail
- `--optimize-sizing-for-speed`
- `--optimize-sizing-for-longevity`
- `--compare-fixed-risk`
- `--compare-fixed-contracts`

## What the Pipeline Does

1. Validates the strategy spec.
2. Runs walk-forward selection on expanding folds.
3. Optionally runs sensitivity in `full` mode.
4. Evaluates holdout.
5. Runs holdout Monte Carlo.
6. Runs regime classification.
7. Produces a verdict.
8. Freezes params and writes audit artifacts if the verdict is not `REJECT`.

Daily grouping for Topstep, Combine simulation, Monte Carlo daily loss stats, and funded-express resets uses futures session-day logic from [src/v3/futures_session.py](C:/Users/Luker/projects/Topstep pipeline/src/v3/futures_session.py:1), not calendar-midnight grouping.

## Output Layout

```text
output/
  json/
  txt summaries/
  graphs/
  frozen_params/
```

Typical outputs:

- result bundle JSON
- readable text summary
- sensitivity and Monte Carlo graphs
- frozen params and audit log when not rejected

## Instrument Defaults

MNQ defaults from [src/v3/config.py](C:/Users/Luker/projects/Topstep pipeline/src/v3/config.py:85):

- point value: `$2.00`
- tick size: `0.25`
- commission: `$1.40` round turn
- slippage: `0.25` points per side

MES defaults from [src/v3/cli.py](C:/Users/Luker/projects/Topstep pipeline/src/v3/cli.py:70):

- point value: `$5.00`
- tick size: `0.25`
- commission: `$0.85` round turn
- slippage: `0.25` points per side

## Topstep 50K Defaults

From [src/v3/config.py](C:/Users/Luker/projects/Topstep pipeline/src/v3/config.py:96):

- account size: `$50,000`
- profit target: `$3,000`
- max drawdown: `$2,000`
- daily loss limit: `$1,000`
- min trading days: `5`
- max micro contracts: `50`
- consistency rule: `50%` of target max in one day

## Custom Strategies

Drop custom strategy modules into `src/v3/user_strategies/` and register them with `StrategySpec` plus `register_strategy(spec)`.

Reference implementation:

- [hl2_sma_retrace_atr.py](C:/Users/Luker/projects/Topstep pipeline/src/v3/user_strategies/hl2_sma_retrace_atr.py:1)

## Tests

Run the full v3 suite:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:PYTHONIOENCODING = "utf-8"
py -3.13 -m pytest tests/v3 --basetemp .pytest_tmp
```

Run a narrow check:

```powershell
py -3.13 -m pytest tests/v3/test_v3_evaluator.py tests/v3/test_v3_cli.py --basetemp .pytest_tmp_smoke
```

## Repository Layout

- `src/v3/`
  Main package: loader, evaluator, CLI, Topstep rules, simulations, verdicting
- `src/v3/user_strategies/`
  Auto-loaded custom strategies
- `tests/v3/`
  Pytest coverage for the pipeline
- `scripts/build_data.py`
  Prebuild higher-timeframe CSVs from 1-minute data
- `Data/`
  Local market data, ignored by git

## Notes

- The pipeline now runs against full-session futures bars by default.
- Strategy-level timing restrictions should live inside strategy logic, not in the loader.
- Prebuilt timeframe files are the preferred runtime path. Repeated on-the-fly resampling is just the fallback.
