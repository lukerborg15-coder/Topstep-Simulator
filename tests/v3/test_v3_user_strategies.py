from __future__ import annotations

import importlib
import sys
from pathlib import Path

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
