from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pandas as pd

from v3.config import EASTERN_TZ
from v3.strategies import STRATEGIES, StrategySpec, TradeSignal


def _load_script_module() -> types.ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "diagnostics"
        / "vp_sanity_check.py"
    )
    spec = importlib.util.spec_from_file_location("vp_sanity_check_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signal_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02 09:40", tz=EASTERN_TZ),
            pd.Timestamp("2024-01-02 09:45", tz=EASTERN_TZ),
            pd.Timestamp("2024-01-02 09:50", tz=EASTERN_TZ),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 101.5, 102.0],
            "low": [99.5, 100.0, 100.5],
            "close": [100.5, 101.0, 101.5],
            "volume": [1000, 1100, 1200],
        },
        index=index,
    )


def test_vp_sanity_check_uses_raw_1min_for_eval_and_debug_attachment(monkeypatch) -> None:
    module = _load_script_module()
    exec_5min = _signal_frame()
    raw_1min = exec_5min.resample("1min").ffill()
    raw_1min["volume"] = 100
    seen: dict[str, object] = {}
    load_calls: list[tuple[str, bool | None]] = []

    class _Result:
        metrics = {"win_rate": 0.5, "avg_r": 0.2, "total_net_pnl": 100.0, "max_drawdown": 50.0}
        trades = []

    def fake_load_ohlcv(instrument: str, timeframe: str, session_only: bool | None = None):
        load_calls.append((timeframe, session_only))
        return raw_1min if timeframe == "1min" else exec_5min

    def fake_evaluate_strategy(frame, strategy_name, timeframe, params, window, **kwargs):
        seen["eval_raw_frame"] = kwargs.get("raw_frame")
        return _Result()

    def fake_attach_volume_profile(frame, raw_frame, **kwargs):
        seen["vp_raw_frame"] = raw_frame
        out = frame.copy()
        out["pdVAH"] = 101.0
        out["pdVAL"] = 99.0
        out["pdVPOC"] = 100.0
        out["naked_pocs"] = [[] for _ in range(len(out))]
        out["is_post_holiday_session"] = False
        return out

    def fake_attach_day_regimes(frame, raw_frame, **kwargs):
        seen["regime_raw_frame"] = raw_frame
        out = frame.copy()
        out["day_regime"] = "balance"
        return out

    def fake_generate(frame: pd.DataFrame, params: dict) -> list[TradeSignal]:
        ts = frame.index[0]
        px = float(frame["close"].iloc[0])
        return [
            TradeSignal(
                time=ts,
                direction="long",
                entry=px,
                stop=px - 1.0,
                target=px + 1.0,
                strategy="vp_dual_mode",
                params=dict(params),
                metadata={"mode": "rejection", "level_name": "pdVAH", "level_price": 101.0},
            )
        ]

    monkeypatch.setattr(module, "load_user_strategies", lambda: None)
    monkeypatch.setattr(module, "load_ohlcv", fake_load_ohlcv)
    monkeypatch.setattr(module, "evaluate_strategy", fake_evaluate_strategy)
    monkeypatch.setattr(module, "attach_volume_profile", fake_attach_volume_profile)
    monkeypatch.setattr(module, "attach_day_regimes", fake_attach_day_regimes)
    monkeypatch.setattr(module, "slice_window", lambda frame, window: frame)
    monkeypatch.setattr(module, "WINDOWS", types.SimpleNamespace(walk_forward=[types.SimpleNamespace(train=types.SimpleNamespace(name="WF1_train", start="2024-01-02", end="2024-01-02"))]))

    original = STRATEGIES.get("vp_dual_mode")
    STRATEGIES["vp_dual_mode"] = StrategySpec(
        name="vp_dual_mode",
        generate=fake_generate,
        default_params={},
        param_grid={},
        max_signals_per_day=None,
        requires=("volume_profile", "regime_classifier"),
    )
    try:
        module.main()
    finally:
        if original is None:
            del STRATEGIES["vp_dual_mode"]
        else:
            STRATEGIES["vp_dual_mode"] = original

    assert seen["eval_raw_frame"] is raw_1min
    assert seen["vp_raw_frame"] is raw_1min
    assert seen["regime_raw_frame"] is raw_1min
    assert load_calls == [("1min", False), ("5min", None)]
