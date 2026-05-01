from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from v3.strategies import STRATEGIES, StrategySpec, TradeSignal
from v3.validator import StrategyValidationError, validate_filter_references, validate_strategy_spec


def _empty_generate(df: pd.DataFrame, params: dict) -> list[TradeSignal]:
    return []


class _SignalLike:
    direction = "long"
    entry = 100.0
    stop = 99.0
    target = 101.0
    strategy = "cap_test"

    def __init__(self, time):
        self.time = time


class _SignalLikeMissingTime:
    direction = "long"
    entry = 100.0
    stop = 99.0
    target = 101.0
    strategy = "cap_test"


def _valid_spec(
    *,
    default_params: dict | None = None,
    param_grid: dict | None = None,
    generate=_empty_generate,
    max_signals_per_day: int | None = None,
) -> StrategySpec:
    default_params = {"risk_reward": 1.5} if default_params is None else default_params
    param_grid = {"risk_reward": (1.0, 1.5, 2.0)} if param_grid is None else param_grid
    return StrategySpec(
        name="valid_strategy",
        generate=generate,
        default_params=default_params,
        param_grid=param_grid,
        max_signals_per_day=max_signals_per_day,
    )


def test_base_strategy_filter_references_are_valid():
    # Built-in specs predate the stricter user-facing grid contract; keep this
    # test focused on registry consistency without changing strategy behavior.
    validate_filter_references(STRATEGIES)


def test_extra_param_grid_key_rejects():
    spec = _valid_spec(
        default_params={"risk_reward": 1.5},
        param_grid={"risk_reward": (1.0, 1.5, 2.0), "missing": (1, 2, 3)},
    )

    with pytest.raises(StrategyValidationError, match="param_grid keys"):
        validate_strategy_spec(spec)


def test_missing_default_params_key_from_param_grid_rejects():
    spec = _valid_spec(
        default_params={"risk_reward": 1.5, "lookback": 20},
        param_grid={"risk_reward": (1.0, 1.5, 2.0)},
    )

    with pytest.raises(StrategyValidationError, match="param_grid keys"):
        validate_strategy_spec(spec)


def test_grid_entries_with_fewer_than_three_values_reject():
    spec = _valid_spec(
        default_params={"risk_reward": 1.5},
        param_grid={"risk_reward": (1.0, 1.5)},
    )

    with pytest.raises(StrategyValidationError, match="at least 3 values"):
        validate_strategy_spec(spec)


def test_grid_entry_must_be_tuple():
    spec = _valid_spec(
        default_params={"risk_reward": 1.5},
        param_grid={"risk_reward": [1.0, 1.5, 2.0]},
    )

    with pytest.raises(StrategyValidationError, match="must be a tuple"):
        validate_strategy_spec(spec)


def test_default_value_absent_from_grid_rejects():
    spec = _valid_spec(
        default_params={"risk_reward": 1.5},
        param_grid={"risk_reward": (1.0, 2.0, 2.5)},
    )

    with pytest.raises(StrategyValidationError, match="default value"):
        validate_strategy_spec(spec)


def test_unsorted_numeric_grid_rejects():
    spec = _valid_spec(
        default_params={"risk_reward": 1.5},
        param_grid={"risk_reward": (1.0, 2.0, 1.5)},
    )

    with pytest.raises(StrategyValidationError, match="smallest to largest"):
        validate_strategy_spec(spec)


def test_bool_grid_does_not_fail_numeric_ordering():
    spec = _valid_spec(
        default_params={"enabled": False},
        param_grid={"enabled": (False, True, False)},
    )

    validate_strategy_spec(spec)


@pytest.mark.parametrize("max_signals_per_day", [-1, 1.5, True])
def test_max_signals_per_day_must_be_non_negative_int_or_none(max_signals_per_day):
    spec = _valid_spec(max_signals_per_day=max_signals_per_day)

    with pytest.raises(StrategyValidationError, match="non-negative int or None"):
        validate_strategy_spec(spec)


def test_max_signals_per_day_rejects_smoke_run_signals_exceeding_cap():
    def generate(df: pd.DataFrame, params: dict) -> list[TradeSignal]:
        signal_time = pd.Timestamp("2024-01-02 10:00", tz="America/New_York")
        return [
            TradeSignal(signal_time, "long", 100.0, 99.0, 101.0, "cap_test"),
            TradeSignal(signal_time + pd.Timedelta(minutes=5), "long", 101.0, 100.0, 102.0, "cap_test"),
        ]

    spec = _valid_spec(default_params={}, param_grid={}, generate=generate, max_signals_per_day=1)

    with pytest.raises(StrategyValidationError, match="max_signals_per_day"):
        validate_strategy_spec(spec)


def test_capped_smoke_run_rejects_signal_missing_time_attribute():
    def generate(df: pd.DataFrame, params: dict) -> list[object]:
        return [_SignalLikeMissingTime()]

    spec = _valid_spec(default_params={}, param_grid={}, generate=generate, max_signals_per_day=1)

    with pytest.raises(StrategyValidationError, match="timezone-aware pd.Timestamp"):
        validate_strategy_spec(spec)


@pytest.mark.parametrize("time_value", [None, pd.NaT])
def test_capped_smoke_run_rejects_missing_time_values(time_value):
    def generate(df: pd.DataFrame, params: dict) -> list[_SignalLike]:
        return [_SignalLike(time_value)]

    spec = _valid_spec(default_params={}, param_grid={}, generate=generate, max_signals_per_day=1)

    with pytest.raises(StrategyValidationError, match="timezone-aware pd.Timestamp"):
        validate_strategy_spec(spec)


def test_capped_smoke_run_rejects_naive_timestamp():
    def generate(df: pd.DataFrame, params: dict) -> list[_SignalLike]:
        return [_SignalLike(pd.Timestamp("2024-01-02 10:00"))]

    spec = _valid_spec(default_params={}, param_grid={}, generate=generate, max_signals_per_day=1)

    with pytest.raises(StrategyValidationError, match="timezone-aware pd.Timestamp"):
        validate_strategy_spec(spec)


def test_capped_smoke_run_rejects_non_timestamp_time():
    def generate(df: pd.DataFrame, params: dict) -> list[_SignalLike]:
        return [_SignalLike("2024-01-02 10:00")]

    spec = _valid_spec(default_params={}, param_grid={}, generate=generate, max_signals_per_day=1)

    with pytest.raises(StrategyValidationError, match="timezone-aware pd.Timestamp"):
        validate_strategy_spec(spec)


def test_capped_smoke_run_accepts_timezone_aware_timestamp_within_cap():
    def generate(df: pd.DataFrame, params: dict) -> list[_SignalLike]:
        return [_SignalLike(pd.Timestamp("2024-01-02 10:00", tz="America/New_York"))]

    spec = _valid_spec(default_params={}, param_grid={}, generate=generate, max_signals_per_day=1)

    validate_strategy_spec(spec)


def test_filter_reference_must_exist():
    spec = StrategySpec(
        name="dangling_filter",
        generate=_empty_generate,
        default_params={},
        param_grid={},
        max_signals_per_day=None,
        filter_of="missing_strategy",
    )

    with pytest.raises(StrategyValidationError, match="missing_strategy"):
        validate_filter_references({"dangling_filter": spec})


def test_smoke_run_failure_is_wrapped():
    def generate(df: pd.DataFrame, params: dict) -> list[TradeSignal]:
        params["missing"]
        return []

    spec = StrategySpec(
        name="smoke_failure",
        generate=generate,
        default_params={},
        param_grid={},
        max_signals_per_day=None,
    )

    with pytest.raises(StrategyValidationError, match="smoke run failed"):
        validate_strategy_spec(spec)


def test_unknown_requirement_is_rejected():
    spec = StrategySpec(
        name="bad_requirement",
        generate=_empty_generate,
        default_params={},
        param_grid={},
        max_signals_per_day=None,
        requires=("nonexistent_capability",),
    )

    with pytest.raises(StrategyValidationError, match="nonexistent_capability"):
        validate_strategy_spec(spec)


def test_validation_does_not_leave_numpy_rng_reseeded():
    spec = StrategySpec(
        name="rng_preservation",
        generate=_empty_generate,
        default_params={},
        param_grid={},
        max_signals_per_day=None,
    )
    np.random.seed(123)
    expected = np.random.random(5)
    np.random.seed(123)

    validate_strategy_spec(spec)

    assert np.allclose(np.random.random(5), expected)
