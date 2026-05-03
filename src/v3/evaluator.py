from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .combine_simulator import CombineSimResult
from .config import (
    DEFAULT_MAX_CONTRACTS,
    DEFAULT_RISK_DOLLARS,
    MNQ,
    SCORING_WEIGHTS,
    SESSION_END,
    TOPSTEP_50K,
    DateWindow,
    Instrument,
    PipelineWindows,
    WINDOWS,
)
from .data import slice_window
from .pivots import attach_pivot_levels
from .strategies import STRATEGIES, TradeSignal
from .topstep import (
    count_sequential_eval_passes,
    simulate_topstep,
    topstep_summary_dict,
)
from .trades import TradeResult


def _validate_eval_risk(risk_dollars: float) -> float:
    risk = float(risk_dollars)
    if risk <= 0.0:
        raise ValueError("Eval risk per trade must be positive")
    return risk


def _contracts_for_fixed_risk(
    signal: TradeSignal,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
) -> int:
    """Floor contracts so dollar risk at stop (signal entry) does not exceed risk_dollars."""
    risk_points = abs(signal.entry - signal.stop)
    if risk_points <= 0:
        return 0
    dollar_per = risk_points * instrument.point_value
    n = int(risk_dollars // dollar_per)
    if n <= 0:
        return 0
    return min(max_contracts, n)


def simulate_trades(
    frame: pd.DataFrame,
    signals: list[TradeSignal],
    instrument: Instrument = MNQ,
    risk_dollars: float = DEFAULT_RISK_DOLLARS,
    max_contracts: int | None = None,
) -> list[TradeResult]:
    risk_dollars = _validate_eval_risk(risk_dollars)
    cap = int(max_contracts) if max_contracts is not None else int(DEFAULT_MAX_CONTRACTS)
    if cap < 1:
        raise ValueError("max_contracts must be at least 1")
    if frame.empty or not signals:
        return []

    session_end_time = pd.Timestamp(SESSION_END).time()
    trades: list[TradeResult] = []
    for signal in sorted(signals, key=lambda item: item.time):
        if signal.time not in frame.index:
            continue

        contracts = _contracts_for_fixed_risk(signal, instrument, risk_dollars, cap)
        if contracts <= 0:
            continue

        is_long = signal.direction == "long"
        entry_idx = frame.index.get_loc(signal.time)
        exit_price = signal.entry
        exit_time = signal.time
        exit_reason = "data_end"
        bars_held = 0

        for i in range(int(entry_idx) + 1, len(frame)):
            ts = frame.index[i]
            row = frame.iloc[i]
            bars_held += 1

            if is_long:
                stop_hit = row["low"] <= signal.stop
                target_hit = row["high"] >= signal.target
            else:
                stop_hit = row["high"] >= signal.stop
                target_hit = row["low"] <= signal.target

            if stop_hit:
                exit_price = signal.stop
                exit_time = ts
                exit_reason = "stop"
                break
            if target_hit:
                exit_price = signal.target
                exit_time = ts
                exit_reason = "target"
                break
            if ts.time() >= session_end_time:
                exit_price = float(row["close"])
                exit_time = ts
                exit_reason = "session_end"
                break
        else:
            exit_price = float(frame["close"].iloc[-1])
            exit_time = frame.index[-1]

        slippage = instrument.slippage_points_per_side
        entry_fill = signal.entry + slippage if is_long else signal.entry - slippage
        exit_fill = exit_price - slippage if is_long else exit_price + slippage
        pnl_points = (exit_fill - entry_fill) if is_long else (entry_fill - exit_fill)
        gross_pnl = pnl_points * contracts * instrument.point_value
        commission = contracts * instrument.commission_round_turn
        net_pnl = gross_pnl - commission
        risk_points = abs(entry_fill - signal.stop)
        risk_dollars_actual = max(risk_points * contracts * instrument.point_value, 1e-9)

        trades.append(
            TradeResult(
                strategy=signal.strategy,
                entry_time=signal.time,
                exit_time=exit_time,
                direction=signal.direction,
                entry=entry_fill,
                stop=signal.stop,
                target=signal.target,
                exit=exit_fill,
                contracts=contracts,
                gross_pnl=float(gross_pnl),
                commission=float(commission),
                net_pnl=float(net_pnl),
                r_multiple=float(net_pnl / risk_dollars_actual),
                exit_reason=exit_reason,
                bars_held=bars_held,
                params=signal.params,
            )
        )

    return trades


def compute_metrics(trades: list[TradeResult]) -> dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_net_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_trade_duration_bars": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "long_net_pnl": 0.0,
            "short_net_pnl": 0.0,
        }

    pnls = np.array([trade.net_pnl for trade in trades], dtype=float)
    rs = np.array([trade.r_multiple for trade in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    equity = np.concatenate(([0.0], np.cumsum(pnls)))
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    r_std = float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0

    # Long/short breakdown
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]

    long_pnls = np.array([t.net_pnl for t in long_trades], dtype=float)
    short_pnls = np.array([t.net_pnl for t in short_trades], dtype=float)

    long_win_rate = float((long_pnls > 0).mean()) if len(long_pnls) > 0 else 0.0
    short_win_rate = float((short_pnls > 0).mean()) if len(short_pnls) > 0 else 0.0
    long_net_pnl = float(long_pnls.sum())
    short_net_pnl = float(short_pnls.sum())

    return {
        "total_trades": int(len(trades)),
        "win_rate": float((pnls > 0).mean()),
        "avg_r": float(np.mean(rs)),
        "total_net_pnl": float(pnls.sum()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() < 0 else float("inf"),
        "max_drawdown": float(drawdown.max(initial=0.0)),
        "sharpe": float(np.mean(rs) / r_std * np.sqrt(252.0)) if r_std > 0 else 0.0,
        "avg_trade_duration_bars": float(np.mean([trade.bars_held for trade in trades])),
        "long_trades": int(len(long_trades)),
        "short_trades": int(len(short_trades)),
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "long_net_pnl": long_net_pnl,
        "short_net_pnl": short_net_pnl,
    }


@dataclass
class EvaluationResult:
    strategy: str
    timeframe: str
    params: dict[str, Any]
    window: str
    metrics: dict[str, Any]
    topstep: dict[str, Any]
    trades: list[TradeResult]
    combine_sim: CombineSimResult | None = None

    @property
    def score(self) -> float:
        return _score_result(self)


def evaluate_strategy(
    frame: pd.DataFrame,
    strategy_name: str,
    timeframe: str,
    params: dict[str, Any],
    window: DateWindow,
    *,
    risk_dollars: float = DEFAULT_RISK_DOLLARS,
    max_contracts: int | None = None,
) -> EvaluationResult:
    sliced = slice_window(frame, window)
    spec = STRATEGIES[strategy_name]
    active_params = dict(spec.default_params)
    active_params.update(params)
    signal_frame = sliced
    if "pivot_levels" in spec.requires:
        signal_frame = attach_pivot_levels(sliced, raw_frame=frame)
    signals = spec.generate(signal_frame, active_params)
    trades = simulate_trades(sliced, signals, risk_dollars=risk_dollars, max_contracts=max_contracts)
    metrics = compute_metrics(trades)
    topstep = topstep_summary_dict(simulate_topstep(trades))
    return EvaluationResult(
        strategy=strategy_name,
        timeframe=timeframe,
        params=active_params,
        window=window.name,
        metrics=metrics,
        topstep=topstep,
        trades=trades,
    )


def attach_sequential_topstep_oos(result: EvaluationResult) -> None:
    """Mutate evaluation topstep dict with OOS sequential Combine eval chain stats."""
    seq_passes, seq_log = count_sequential_eval_passes(result.trades)
    td = dict(result.topstep)
    td["seq_eval_passes"] = seq_passes
    td["seq_eval_attempts"] = len(seq_log)
    attempts = int(len(seq_log))
    td["seq_eval_pass_rate"] = float(seq_passes) / float(attempts) if attempts > 0 else 0.0
    result.topstep = td


def run_walk_forward(
    frame: pd.DataFrame,
    strategy_name: str,
    timeframe: str,
    max_grid: int | None = None,
    min_eval_passes_per_fold: int = 2,
    min_folds_meeting_passes: int | None = None,
    windows: PipelineWindows | None = None,
    *,
    risk_dollars: float = DEFAULT_RISK_DOLLARS,
    max_contracts: int | None = None,
) -> tuple[dict[str, Any], list[EvaluationResult], bool]:
    spec = STRATEGIES[strategy_name]
    candidates = spec.grid()
    if max_grid is not None and max_grid <= 0:
        raise ValueError("max_grid must be positive when provided")
    if min_eval_passes_per_fold < 1:
        raise ValueError("min_eval_passes_per_fold must be at least 1")
    if max_grid is not None:
        candidates = candidates[:max_grid]
    if not candidates:
        raise ValueError(f"No parameter candidates available for strategy {strategy_name!r}")

    w = windows or WINDOWS
    wf_list = w.walk_forward
    n_folds = len(wf_list)
    if min_folds_meeting_passes is None:
        min_folds_meeting_passes = n_folds
    # Clamp to available folds to prevent deadlock on small configs.
    if min_folds_meeting_passes > n_folds:
        import sys
        print(
            f"WARNING: --min-wf-passes {min_folds_meeting_passes} > folds available {n_folds}; "
            f"clamping to {n_folds}.",
            file=sys.stderr,
        )
        min_folds_meeting_passes = n_folds
    if min_folds_meeting_passes < 1:
        raise ValueError("min_folds_meeting_passes must be at least 1")

    selected_params: list[dict[str, Any]] = []
    oos_folds: list[EvaluationResult] = []
    train_best_results: list[EvaluationResult] = []

    for wf in wf_list:
        train_results = [
            evaluate_strategy(frame, strategy_name, timeframe, params, wf.train, risk_dollars=risk_dollars, max_contracts=max_contracts)
            for params in candidates
        ]
        train_best = max(train_results, key=_score_result)
        train_best_results.append(train_best)
        selected_params.append(train_best.params)
        oos = evaluate_strategy(
            frame, strategy_name, timeframe, train_best.params, wf.test, risk_dollars=risk_dollars, max_contracts=max_contracts
        )
        attach_sequential_topstep_oos(oos)
        oos_folds.append(oos)

    selected, wf_robust_ok = _robust_params(
        selected_params,
        oos_folds,
        train_best_results,
        min_eval_passes_per_fold=min_eval_passes_per_fold,
        min_folds_meeting_passes=min_folds_meeting_passes,
    )
    return selected, oos_folds, wf_robust_ok


def fold_seq_eval_pass_rate(topstep: dict[str, Any]) -> float:
    if "seq_eval_pass_rate" in topstep:
        return float(topstep["seq_eval_pass_rate"])
    passes = int(topstep.get("seq_eval_passes", 0))
    attempts = int(topstep.get("seq_eval_attempts", 0))
    if attempts <= 0:
        return 0.0
    return float(passes) / float(attempts)


def all_folds_meet_min_seq_pass_rate(folds: list[EvaluationResult], min_rate_pct: float) -> tuple[bool, list[float]]:
    p = min_rate_pct / 100.0
    rates = [fold_seq_eval_pass_rate(fold.topstep) for fold in folds]
    return all(r >= p for r in rates), rates


def wf_oos_folds_for_selected_params(
    frame: pd.DataFrame,
    strategy_name: str,
    timeframe: str,
    selected_params: dict[str, Any],
    windows: PipelineWindows,
    *,
    risk_dollars: float = DEFAULT_RISK_DOLLARS,
    max_contracts: int | None = None,
) -> list[EvaluationResult]:
    out: list[EvaluationResult] = []
    for wf in windows.walk_forward:
        oos = evaluate_strategy(
            frame,
            strategy_name,
            timeframe,
            selected_params,
            wf.test,
            risk_dollars=risk_dollars,
            max_contracts=max_contracts,
        )
        attach_sequential_topstep_oos(oos)
        out.append(oos)
    return out


def wf_train_test_trades_for_selected_params(
    frame: pd.DataFrame,
    strategy_name: str,
    timeframe: str,
    selected_params: dict[str, Any],
    windows: PipelineWindows,
    *,
    risk_dollars: float = DEFAULT_RISK_DOLLARS,
    max_contracts: int | None = None,
) -> list[tuple[list[TradeResult], list[TradeResult]]]:
    """Extract (train_trades, test_trades) tuples for each WF fold.

    Mirrors wf_oos_folds_for_selected_params but returns trade pairs instead of EvaluationResults.
    Used by position sizing optimizers that need both train and test trade sequences.
    """
    out: list[tuple[list[TradeResult], list[TradeResult]]] = []
    for wf in windows.walk_forward:
        train = evaluate_strategy(
            frame,
            strategy_name,
            timeframe,
            selected_params,
            wf.train,
            risk_dollars=risk_dollars,
            max_contracts=max_contracts,
        )
        test = evaluate_strategy(
            frame,
            strategy_name,
            timeframe,
            selected_params,
            wf.test,
            risk_dollars=risk_dollars,
            max_contracts=max_contracts,
        )
        out.append((train.trades, test.trades))
    return out


def walk_forward_development_window(windows: PipelineWindows) -> DateWindow:
    """Derive development window: min(all train.start) → max(all test.end).

    Covers the full WF calendar (train + OOS) in one contiguous DateWindow.
    Used by sensitivity stage as the evaluation slice.
    """
    wf_list = windows.walk_forward
    if not wf_list:
        raise ValueError("PipelineWindows has no walk_forward folds")
    start = min(wf.train.start for wf in wf_list)
    end = max(wf.test.end for wf in wf_list)
    return DateWindow("wf_development", start, end)


def aggregate_wf_metrics(folds: list[EvaluationResult]) -> dict[str, Any]:
    if not folds:
        return {
            "wf_folds": 0,
            "wf_passed_folds": 0,
            "wf_avg_score": 0.0,
            "wf_avg_net_pnl": 0.0,
            "wf_oos_total_pnl": 0.0,
            "wf_consistency": 0.0,
            "wf_seq_eval_passes_by_fold": [],
            "wf_fold_seq_pass_rates": [],
        }

    net_pnls = [float(fold.metrics.get("total_net_pnl", 0.0)) for fold in folds]
    scores = [_score_result(fold) for fold in folds]
    seq_by_fold = [int(fold.topstep.get("seq_eval_passes", 0)) for fold in folds]
    rates = [fold_seq_eval_pass_rate(fold.topstep) for fold in folds]
    return {
        "wf_folds": len(folds),
        "wf_passed_folds": int(sum(bool(fold.topstep.get("topstep_passed")) for fold in folds)),
        "wf_avg_score": float(np.mean(scores)),
        "wf_avg_net_pnl": float(np.mean(net_pnls)),
        "wf_oos_total_pnl": float(sum(net_pnls)),
        "wf_consistency": float(np.std(net_pnls, ddof=1)) if len(net_pnls) > 1 else 0.0,
        "wf_seq_eval_passes_by_fold": seq_by_fold,
        "wf_fold_seq_pass_rates": rates,
    }


def _score_result(result: EvaluationResult) -> float:
    topstep_component = float(result.topstep.get("topstep_score", 0.0)) * SCORING_WEIGHTS.topstep_weight
    avg_r_component = float(result.metrics.get("avg_r", 0.0)) * SCORING_WEIGHTS.avg_r_weight
    return topstep_component + avg_r_component


def _params_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in params.items()))


def _mode_params(
    selected_params: list[dict[str, Any]],
    train_results: list[EvaluationResult],
) -> dict[str, Any]:
    if not selected_params:
        return {}

    counts_by_key: dict[tuple[tuple[str, str], ...], int] = {}
    score_by_key: dict[tuple[tuple[str, str], ...], float] = {}
    params_by_key: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for params, result in zip(selected_params, train_results, strict=True):
        key = _params_key(params)
        counts_by_key[key] = counts_by_key.get(key, 0) + 1
        score_by_key[key] = score_by_key.get(key, 0.0) + _score_result(result)
        params_by_key[key] = params

    best_key = max(counts_by_key, key=lambda key: (counts_by_key[key], score_by_key[key]))
    return dict(params_by_key[best_key])


def _robust_params(
    selected_params: list[dict[str, Any]],
    oos_results: list[EvaluationResult],
    train_results: list[EvaluationResult],
    min_eval_passes_per_fold: int = 2,
    min_folds_meeting_passes: int = 4,
) -> tuple[dict[str, Any], bool]:
    """
    Pick params that:
    1. Hit sequential OOS eval_passes >= min_eval_passes_per_fold on at least
       min_folds_meeting_passes walk-forward folds (same param key).
    2. Have highest average train score among those candidates.

    If none qualify, fall back to mode selection across folds.

    Returns:
        (params, robust_criteria_met) — robust_criteria_met is False when falling back.
    """
    if min_eval_passes_per_fold < 1 or min_folds_meeting_passes < 1:
        return _mode_params(selected_params, train_results), False

    candidates: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for params, oos, train_r in zip(selected_params, oos_results, train_results, strict=True):
        key = _params_key(params)
        if key not in candidates:
            candidates[key] = {
                "params": params,
                "folds_meeting_eval_passes": 0,
                "total_score": 0.0,
                "fold_count": 0,
            }

        seq_passes = int(oos.topstep.get("seq_eval_passes", 0))
        if seq_passes >= min_eval_passes_per_fold:
            candidates[key]["folds_meeting_eval_passes"] += 1

        candidates[key]["total_score"] += _score_result(train_r)
        candidates[key]["fold_count"] += 1

    robust = {
        key: candidate
        for key, candidate in candidates.items()
        if candidate["folds_meeting_eval_passes"] >= min_folds_meeting_passes
    }

    if not robust:
        print(
            f"WARNING: No params with {min_folds_meeting_passes}+ folds meeting "
            f"{min_eval_passes_per_fold}+ sequential eval passes. "
            "Falling back to mode selection."
        )
        return _mode_params(selected_params, train_results), False

    best_key = max(
        robust,
        key=lambda key: robust[key]["total_score"] / robust[key]["fold_count"],
    )
    return dict(candidates[best_key]["params"]), True


__all__ = [
    "EvaluationResult",
    "aggregate_wf_metrics",
    "all_folds_meet_min_seq_pass_rate",
    "attach_sequential_topstep_oos",
    "compute_metrics",
    "evaluate_strategy",
    "fold_seq_eval_pass_rate",
    "run_walk_forward",
    "simulate_trades",
    "walk_forward_development_window",
    "wf_oos_folds_for_selected_params",
]
