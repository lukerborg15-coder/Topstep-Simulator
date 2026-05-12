from __future__ import annotations

import importlib
import ast
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v3.strategies import STRATEGIES, load_user_strategies
from v3.validator import StrategyValidationError


USER_STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "src" / "v3" / "user_strategies"
TEST_STRATEGY_NAME = "__test_validator_strategy__"


def _cleanup_generated_test_files(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
    pycache = USER_STRATEGIES_DIR / "__pycache__"
    for stem in ("tmp_test_strategy", "tmp_broken_strategy"):
        for path in pycache.glob(f"{stem}*.pyc"):
            path.unlink()


@pytest.fixture(autouse=True)
def cleanup_user_strategy_files():
    paths = [
        USER_STRATEGIES_DIR / "_tmp_test_strategy.py",
        USER_STRATEGIES_DIR / "_tmp_broken_strategy.py",
        USER_STRATEGIES_DIR / "tmp_test_strategy.py",
        USER_STRATEGIES_DIR / "tmp_broken_strategy.py",
    ]
    _cleanup_generated_test_files(paths)
    STRATEGIES.pop(TEST_STRATEGY_NAME, None)
    for module_name in (
        "v3.user_strategies._tmp_test_strategy",
        "v3.user_strategies._tmp_broken_strategy",
        "v3.user_strategies.tmp_test_strategy",
        "v3.user_strategies.tmp_broken_strategy",
    ):
        sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        _cleanup_generated_test_files(paths)
        STRATEGIES.pop(TEST_STRATEGY_NAME, None)
        for module_name in (
            "v3.user_strategies._tmp_test_strategy",
            "v3.user_strategies._tmp_broken_strategy",
            "v3.user_strategies.tmp_test_strategy",
            "v3.user_strategies.tmp_broken_strategy",
        ):
            sys.modules.pop(module_name, None)
        importlib.invalidate_caches()


def test_load_user_strategies_second_call_is_idempotent():
    """Permanent user strategy modules may register on first load; rerun must not break."""
    load_user_strategies()
    snapshot = dict(STRATEGIES)
    load_user_strategies()
    assert STRATEGIES == snapshot


def test_load_user_strategies_registers_valid_strategy_file():
    strategy_file = USER_STRATEGIES_DIR / "tmp_test_strategy.py"
    strategy_file.write_text(
        """
from __future__ import annotations

from v3.strategies import StrategySpec, register_strategy


def generate(df, params):
    return []


register_strategy(
    StrategySpec(
        name="__test_validator_strategy__",
        generate=generate,
        default_params={"threshold": 1},
        param_grid={"threshold": (1, 2, 3)},
        max_signals_per_day=None,
    )
)
""".lstrip(),
        encoding="utf-8",
    )
    importlib.invalidate_caches()

    load_user_strategies()

    assert TEST_STRATEGY_NAME in STRATEGIES


def test_load_user_strategies_raises_for_broken_strategy_file():
    strategy_file = USER_STRATEGIES_DIR / "tmp_broken_strategy.py"
    strategy_file.write_text(
        """
from __future__ import annotations

from v3.strategies import StrategySpec, register_strategy


def generate(df, params):
    return []


register_strategy(
    StrategySpec(
        name="__test_validator_strategy__",
        generate=generate,
        default_params={},
        param_grid={"missing": (1,)},
        max_signals_per_day=None,
    )
)
""".lstrip(),
        encoding="utf-8",
    )
    importlib.invalidate_caches()

    with pytest.raises(StrategyValidationError):
        load_user_strategies()


def _fingerprint_signals(signals: list) -> list[tuple]:
    fingerprints = []
    for signal in signals:
        if dataclasses.is_dataclass(signal):
            data = dataclasses.asdict(signal)
        elif hasattr(signal, "_asdict"):
            data = signal._asdict()
        elif hasattr(signal, "__dict__"):
            data = vars(signal)
        else:
            data = {
                name: getattr(signal, name)
                for name in ("timestamp", "direction", "entry_price", "stop_price", "target_price", "strategy")
                if hasattr(signal, name)
            }
        fingerprints.append(
            tuple(
                (key, repr(value))
                for key, value in sorted(data.items())
                if key != "params"
            )
        )
    return fingerprints


def _synthetic_ohlcv(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 09:30", periods=rows, freq="5min")
    phase = np.linspace(0.0, 10.0, rows)
    close = pd.Series(100.0 + np.sin(phase) * 1.5 + np.linspace(0.0, 6.0, rows), index=index)
    open_ = close.shift(1).fillna(close.iloc[0] - 0.1)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.75
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.75
    volume = pd.Series(1000 + (np.arange(rows) % 20) * 10, index=index)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _baseline_bollinger_generate(module, df: pd.DataFrame, params: dict) -> list:
    """Pre-optimization behavior copy used to prove signal fingerprints stay fixed."""
    htf_timeframe = str(params["htf_timeframe"])

    bb_period = int(params["bb_period"])
    bb_std = float(params["bb_std"])
    atr_period = int(params["atr_period"])

    supertrend_period = int(params["supertrend_period"])
    supertrend_mult = float(params["supertrend_mult"])

    htf_rsi_period = int(params["htf_rsi_period"])
    htf_adx_period = int(params["htf_adx_period"])
    ltf_adx_period = int(params["ltf_adx_period"])

    htf_rsi_long = float(params["htf_rsi_long"])
    htf_rsi_short = float(params["htf_rsi_short"])
    htf_adx_thresh = float(params["htf_adx_thresh"])
    ltf_adx_thresh = float(params["ltf_adx_thresh"])

    stop_atr_mult = float(params["stop_atr_mult"])
    exit_mode = str(params["exit_mode"])
    rr_mult = float(params["rr_mult"])
    cooldown_bars = int(params["cooldown_bars"])

    htf_df = module._load_htf_data(htf_timeframe)
    if htf_df is None:
        htf_df = module.resample_ohlcv(df, htf_timeframe, session_only=True)

    ltf_df = df.copy()

    if len(htf_df) < max(htf_rsi_period, htf_adx_period, bb_period) + 1:
        return []

    htf_rsi_series = module.rsi(htf_df["close"], htf_rsi_period)
    htf_adx_series = module.adx(htf_df, htf_adx_period)
    ltf_adx_series = module.adx(ltf_df, ltf_adx_period)
    ltf_atr_series = module.atr(ltf_df, atr_period)
    st_data = module.supertrend(ltf_df, supertrend_period, supertrend_mult)
    ltf_st = st_data["st"]
    ltf_st_dir = st_data["dir"]
    bb_mid = ltf_df["close"].rolling(bb_period, min_periods=bb_period).mean()
    bb_std_val = ltf_df["close"].rolling(bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val
    htf_rsi_reindexed = htf_rsi_series.reindex(ltf_df.index, method="ffill")
    htf_adx_reindexed = htf_adx_series.reindex(ltf_df.index, method="ffill")

    signals: list = []
    next_allowed = 0
    warmup = max(htf_rsi_period, htf_adx_period, ltf_adx_period, bb_period, supertrend_period, atr_period) + 1

    def baseline_exit_bar_index(entry_idx: int, *, stop: float, target: float, is_long: bool) -> int:
        session_end_time = pd.Timestamp(module.SESSION_END).time()
        for j in range(entry_idx + 1, len(ltf_df)):
            row = ltf_df.iloc[j]
            ts = ltf_df.index[j]
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
        return len(ltf_df) - 1

    for i in range(warmup, len(ltf_df)):
        if i < next_allowed:
            continue

        ts = ltf_df.index[i]
        if ts.time() >= pd.Timestamp(module.SESSION_END).time():
            continue

        htf_rsi = htf_rsi_reindexed.iloc[i]
        htf_adx = htf_adx_reindexed.iloc[i]
        ltf_adx = ltf_adx_series.iloc[i]
        ltf_st_val = ltf_st.iloc[i]
        st_dir = int(ltf_st_dir.iloc[i])
        a = ltf_atr_series.iloc[i]

        if not all(np.isfinite(x) for x in [htf_rsi, htf_adx, ltf_adx, ltf_st_val, a]):
            continue

        if a <= 0:
            continue

        entry_price = float(ltf_df["close"].iloc[i])
        bb_u = bb_upper.iloc[i]
        bb_l = bb_lower.iloc[i]

        if not np.isfinite(bb_u) or not np.isfinite(bb_l):
            continue

        direction = 0
        stop_price = 0.0
        target_price = 0.0

        if (
            htf_rsi >= htf_rsi_long
            and htf_adx >= htf_adx_thresh
            and ltf_adx >= ltf_adx_thresh
            and st_dir == 1
        ):
            direction = 1
            stop_price = entry_price - stop_atr_mult * a
            if exit_mode == "bb_band_tp":
                target_price = bb_u
            else:
                target_price = entry_price + rr_mult * (entry_price - stop_price)
        elif (
            htf_rsi <= htf_rsi_short
            and htf_adx >= htf_adx_thresh
            and ltf_adx >= ltf_adx_thresh
            and st_dir == -1
        ):
            direction = -1
            stop_price = entry_price + stop_atr_mult * a
            if exit_mode == "bb_band_tp":
                target_price = bb_l
            else:
                target_price = entry_price - rr_mult * (stop_price - entry_price)

        if direction != 0:
            module._append_signal(signals, ts, direction, entry_price, stop_price, target_price, module.STRATEGY_KEY, params)
            next_allowed = baseline_exit_bar_index(
                i,
                stop=stop_price,
                target=target_price,
                is_long=(direction == 1),
            ) + cooldown_bars

    return signals


def test_bollinger_hot_loop_matches_preoptimization_signal_fingerprint(monkeypatch):
    module = importlib.import_module("v3.user_strategies.bollinger_band_mean_reversion")
    df = _synthetic_ohlcv()
    params = dict(module.default_params)
    params.update(
        {
            "htf_rsi_long": 101.0,
            "htf_rsi_short": 101.0,
            "htf_adx_thresh": 0.0,
            "ltf_adx_thresh": 0.0,
            "cooldown_bars": 3,
        }
    )
    monkeypatch.setattr(module, "_load_htf_data", lambda _timeframe, instrument="mnq": df)

    baseline = _baseline_bollinger_generate(module, df, params)
    optimized = module.bollinger_band_mean_reversion_generate(df, params)

    assert len(baseline) > 0
    assert len(optimized) > 0
    assert _fingerprint_signals(optimized) == _fingerprint_signals(baseline)


def test_bollinger_session_end_parse_is_not_inside_per_bar_loops():
    module_path = USER_STRATEGIES_DIR / "bollinger_band_mean_reversion.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    offenders = []
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Timestamp"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pd"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "SESSION_END"
            ):
                offenders.append(node.lineno)

    assert offenders == []
