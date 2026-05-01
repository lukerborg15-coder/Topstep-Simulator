from __future__ import annotations

from dataclasses import fields

from v3 import combine_simulator, data, indicators, pivots, topstep
from v3.config import WINDOWS, STRATEGY_NAMES, TOPSTEP_50K
from v3.strategies import STRATEGIES, StrategySpec


def test_v3_config_exports_and_windows():
    assert WINDOWS.in_sample_sanity.start == "2022-09-01"
    assert WINDOWS.holdout.start == "2024-09-01"
    assert STRATEGY_NAMES == (
        "connors_rsi2",
        "ttm_squeeze",
        "orb_ib",
        "orb_volatility_filtered",
        "orb_wick_rejection",
        "session_pivot_rejection",
        "session_pivot_break",
    )
    assert TOPSTEP_50K.account_size == 50_000.0


def test_v3_windows_do_not_overlap():
    assert all(window.train.end <= "2024-08-31" for window in WINDOWS.walk_forward)
    holdout_start = WINDOWS.holdout.start
    assert WINDOWS.in_sample_sanity.end < holdout_start
    assert holdout_start <= WINDOWS.holdout.end
    for window in WINDOWS.walk_forward:
        assert window.test.end < holdout_start


def test_v3_strategies_have_expected_metadata():
    # Exclude test-sentinel strategies (names wrapped in __) that may be
    # left behind by test_v3_user_strategies when running in environments
    # where the sandbox cannot delete temp files.
    production_strategies = {k: v for k, v in STRATEGIES.items() if not k.startswith("__")}
    assert set(production_strategies) == set(STRATEGY_NAMES)
    field_names = {field.name for field in fields(StrategySpec)}
    assert "requires" in field_names
    assert "filter_of" in field_names
    for spec in production_strategies.values():
        assert hasattr(spec, "requires")
        assert hasattr(spec, "filter_of")
    assert STRATEGIES["session_pivot_rejection"].requires == ("pivot_levels",)
    assert STRATEGIES["session_pivot_break"].requires == ("pivot_levels",)
    assert STRATEGIES["orb_volatility_filtered"].filter_of == "orb_ib"
    assert STRATEGIES["orb_wick_rejection"].filter_of == "orb_ib"


def test_v3_modules_import_without_error():
    assert data is not None
    assert indicators is not None
    assert pivots is not None
    assert topstep is not None
    assert combine_simulator is not None
