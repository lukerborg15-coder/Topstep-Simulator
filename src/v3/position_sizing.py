"""Position sizing optimization for walk-forward and holdout periods.

Two optimization objectives:
1. Speed: Find risk that minimizes days_to_pass while maintaining pass_rate
2. Longevity: Find risk that maximizes funded account survival while hitting min profit/trade

V2 functions (preferred):
- optimize_speed_wf_aggregate: trains on WF train, evaluates on WF test, aggregates across folds
- optimize_longevity_holdout_mc: block-bootstrap MC over holdout, weighted multi-component score
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .config import (
    DEFAULT_MAX_CONTRACTS,
    MNQ,
    TOPSTEP_50K,
    Instrument,
    TopStepRules,
    FundedExpressSimRules,
    DEFAULT_FUNDED_EXPRESS_SIM,
)
from .evaluator import _contracts_for_fixed_risk
from .funded_express_sim import simulate_express_funded_resets
from .topstep import count_sequential_eval_passes
from .trades import TradeResult


DEFAULT_RISK_GRID: tuple[float, ...] = tuple(float(v) for v in range(50, 1000, 50))


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class SpeedOptimizationResult:
    """LEGACY: per-fold speed optimization result. Use SpeedOptimizationAggregateResult."""

    strategy: str
    window: str
    pass_floor_pct: float
    pass_target_pct: float
    optimal_risk_dollars: float
    pass_rate_pct: float
    mean_days_to_pass: float
    std_days_to_pass: float
    min_contracts_used: int
    max_contracts_used: int
    candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class LongevityOptimizationResult:
    """LEGACY: deterministic longevity result. Use LongevityOptimizationMCResult."""

    strategy: str
    window: str
    min_profit_per_trade: float
    optimal_risk_dollars: float
    avg_pnl_per_trade: float
    total_pnl: float
    funded_accounts_used: int
    accounts_blown: int
    total_trades_executed: int
    longevity_score: float
    candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SpeedOptimizationAggregateResult:
    """Aggregate speed optimization across walk-forward folds (train-fit, OOS-evaluate)."""

    strategy: str
    pass_floor_pct: float
    speed_target_days: float
    attempt_budget: int
    n_folds: int
    optimal_risk_dollars: float
    median_oos_utility: float
    min_oos_utility: float
    median_oos_pass_rate_pct: float
    median_oos_median_days_to_pass: float
    viable_folds: int
    per_fold_oos: tuple[dict[str, Any], ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class LongevityOptimizationMCResult:
    """MC-based longevity optimization with weighted multi-component scoring."""

    strategy: str
    window: str
    min_profit_per_trade: float
    min_profit_factor: float
    weights: dict[str, float]
    mc_iterations: int
    mc_block_size: int
    bootstrap_iterations: int
    optimal_risk_dollars: float
    median_longevity_score: float
    p05_longevity_score: float
    median_components: dict[str, float]
    p05_components: dict[str, float]
    median_avg_pnl_per_trade: float
    p05_avg_pnl_per_trade: float
    median_accounts_used: float
    median_accounts_blown: float
    per_account_survival_days: tuple[int, ...] = ()
    per_account_summary: tuple[dict[str, Any], ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()


# =============================================================================
# Helpers
# =============================================================================


def _count_sequential_eval_passes_capped(
    trades: list[TradeResult],
    rules: TopStepRules,
    max_attempts: int,
) -> tuple[int, list, bool]:
    """Wrap count_sequential_eval_passes; truncate the attempt log to max_attempts."""
    passes_full, log_full = count_sequential_eval_passes(trades, rules)
    truncated = len(log_full) > max_attempts
    log = log_full[:max_attempts]
    passes = sum(1 for r in log if r.passed)
    return passes, log, truncated


def _bootstrap_pnl_p05(
    trades: list[TradeResult],
    n: int = 1000,
    seed: int = 42,
    percentile: float = 5.0,
) -> float:
    """Bootstrap the percentile of mean(net_pnl) across n resamples with replacement."""
    if not trades:
        return 0.0
    pnls = np.array([float(t.net_pnl) for t in trades], dtype=float)
    rng = np.random.default_rng(seed)
    size = len(pnls)
    means = np.empty(n, dtype=float)
    for i in range(n):
        sample = rng.choice(pnls, size=size, replace=True)
        means[i] = float(sample.mean())
    return float(np.percentile(means, percentile))


def _block_bootstrap_trade_sequences(
    trades: list[TradeResult],
    n: int = 500,
    block_size: int = 5,
    seed: int = 42,
):
    """Yield n permuted trade sequences using contiguous blocks."""
    if not trades:
        return
    rng = np.random.default_rng(seed)
    size = len(trades)
    n_blocks = max(1, math.ceil(size / block_size))
    for _ in range(n):
        out: list[TradeResult] = []
        for _b in range(n_blocks):
            start = int(rng.integers(0, max(1, size - block_size + 1)))
            out.extend(trades[start:start + block_size])
        yield out[:size]


def _resize_trades_for_risk(
    trades: list[TradeResult],
    risk_dollars: float,
    instrument: Instrument = MNQ,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> list[TradeResult]:
    """Resize trades by per-trade fixed risk; drop trades that produce zero contracts."""
    out: list[TradeResult] = []
    for trade in trades:
        contracts = _contracts_for_fixed_risk(trade, instrument, risk_dollars, max_contracts)
        if contracts <= 0:
            continue
        original = trade.contracts
        scale = contracts / original if original > 0 else 1.0
        out.append(replace(
            trade,
            gross_pnl=trade.gross_pnl * scale,
            commission=trade.commission * scale,
            net_pnl=trade.net_pnl * scale,
            contracts=contracts,
        ))
    return out


def _resize_trades_for_fixed_contracts(
    trades: list[TradeResult],
    n_contracts: int,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> list[TradeResult]:
    """Force every trade to use n_contracts (capped)."""
    target = max(1, min(n_contracts, max_contracts))
    out: list[TradeResult] = []
    for trade in trades:
        original = trade.contracts
        if original <= 0:
            continue
        scale = target / original
        out.append(replace(
            trade,
            gross_pnl=trade.gross_pnl * scale,
            commission=trade.commission * scale,
            net_pnl=trade.net_pnl * scale,
            contracts=target,
        ))
    return out


def _profit_factor(trades: list[TradeResult]) -> float:
    if not trades:
        return 0.0
    wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    losses = -sum(t.net_pnl for t in trades if t.net_pnl < 0)
    if losses <= 0:
        return math.inf if wins > 0 else 0.0
    return wins / losses


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _true_longevity_score(
    bank_return: float,
    survival_factor: float,
    drawdown_factor: float,
    consistency_factor: float,
    account_efficiency_factor: float,
) -> float:
    return (
        max(0.0, float(bank_return))
        * _clamp01(survival_factor)
        * _clamp01(drawdown_factor)
        * _clamp01(consistency_factor)
        * max(0.0, float(account_efficiency_factor))
    )


def _true_longevity_components_from_sim(
    sim_result,
    rules: FundedExpressSimRules,
    *,
    consistency_factor: float = 1.0,
) -> dict[str, float]:
    accounts_used = max(1, int(sim_result.funded_accounts_used))
    accounts_blown = int(sim_result.funded_accounts_failed)
    bank_return = max(0.0, 1.0 + float(sim_result.accrued_pnl_bank) / float(rules.account_size))
    survival = _clamp01(1.0 - (accounts_blown / accounts_used))
    worst_dd = float(sim_result.worst_stint_peak_to_trough_drawdown_from_peak_balance)
    drawdown = _clamp01(1.0 - (worst_dd / rules.max_drawdown) if rules.max_drawdown > 0 else 0.0)
    account_efficiency = 1.0 / math.sqrt(accounts_used)
    consistency = _clamp01(consistency_factor)
    longevity = _true_longevity_score(bank_return, survival, drawdown, consistency, account_efficiency)
    return {
        "bank_return_score": bank_return,
        "survival_score": survival,
        "drawdown_score": drawdown,
        "consistency_score": consistency,
        "account_efficiency_score": account_efficiency,
        "longevity_score": longevity,
        "accounts_used": float(accounts_used),
        "accounts_blown": float(accounts_blown),
    }


def _compute_longevity_components(
    sim_result,
    rules: FundedExpressSimRules,
    target_pnl_per_trade: float,
    weights: dict[str, float],
) -> dict[str, float]:
    """Compute true longevity components plus legacy diagnostics from a funded sim result."""
    components = _true_longevity_components_from_sim(sim_result, rules)
    total_trades = sum(int(s.get("trades_applied_count", 0)) for s in sim_result.stints_summary)
    total_pnl = sim_result.accrued_pnl_bank
    avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0.0
    efficiency = (avg_pnl / target_pnl_per_trade) if target_pnl_per_trade > 0 else 0.0
    capital = total_pnl / rules.account_size if rules.account_size > 0 else 0.0
    components.update({
        "efficiency_score": efficiency,
        "capital_score": capital,
        "avg_pnl_per_trade": avg_pnl,
    })
    return components


def _get_valid_risk_levels(
    trades: list[TradeResult],
    risk_levels: list[float],
    instrument: Instrument = MNQ,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
    coverage_threshold: float = 0.5,
) -> list[float]:
    """Risk is valid if at least coverage_threshold fraction of trades produce >=1 contract."""
    if not trades:
        return []
    valid: list[float] = []
    n = len(trades)
    for risk in risk_levels:
        hits = 0
        for trade in trades:
            if _contracts_for_fixed_risk(trade, instrument, risk, max_contracts) > 0:
                hits += 1
        if hits / n >= coverage_threshold:
            valid.append(risk)
    return valid


def _measure_contracts_range(
    trades: list[TradeResult],
    risk_dollars: float,
    instrument: Instrument = MNQ,
    max_contracts: int = DEFAULT_MAX_CONTRACTS,
) -> tuple[int, int]:
    if not trades:
        return (0, 0)
    counts: list[int] = []
    for trade in trades:
        c = _contracts_for_fixed_risk(trade, instrument, risk_dollars, max_contracts)
        if c > 0:
            counts.append(c)
    if not counts:
        return (0, 0)
    return (min(counts), max(counts))


def _speed_utility(pass_rate: float, median_days: float, target_days: float) -> float:
    """utility = pass_rate * exp(-max(0, median_days - target) / 5)."""
    if pass_rate <= 0 or not math.isfinite(median_days):
        return 0.0
    decay = math.exp(-max(0.0, median_days - target_days) / 5.0)
    return pass_rate * decay


def _evaluate_risk_on_trades(
    trades: list[TradeResult],
    risk: float,
    rules: TopStepRules,
    attempt_budget: int,
    instrument: Instrument,
) -> dict[str, Any]:
    """Run capped sequential eval, return metrics dict."""
    resized = _resize_trades_for_risk(trades, risk, instrument, DEFAULT_MAX_CONTRACTS)
    if not resized:
        return {
            "risk_dollars": risk,
            "pass_rate_pct": 0.0,
            "passes": 0,
            "attempts": 0,
            "median_days_to_pass": float("inf"),
            "mean_days_to_pass": float("inf"),
            "iqr_days_to_pass": (0.0, 0.0),
            "p90_days_to_pass": float("inf"),
            "std_days_to_pass": 0.0,
            "utility": 0.0,
            "min_contracts": 0,
            "max_contracts": 0,
        }

    passes, log, truncated = _count_sequential_eval_passes_capped(resized, rules, attempt_budget)
    attempts = len(log)
    days = [r.days_to_pass for r in log if r.passed and r.days_to_pass is not None]
    pass_rate = (passes / min(attempt_budget, attempts)) * 100.0 if attempts > 0 else 0.0

    if days:
        arr = np.array(days, dtype=float)
        median_days = float(np.median(arr))
        mean_days = float(arr.mean())
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        p90 = float(np.percentile(arr, 90))
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    else:
        median_days = float("inf")
        mean_days = float("inf")
        p25 = p75 = 0.0
        p90 = float("inf")
        std = 0.0

    min_c, max_c = _measure_contracts_range(resized, risk, instrument)

    return {
        "risk_dollars": risk,
        "pass_rate_pct": pass_rate,
        "passes": passes,
        "attempts": attempts,
        "median_days_to_pass": median_days,
        "mean_days_to_pass": mean_days,
        "iqr_days_to_pass": (p25, p75),
        "p90_days_to_pass": p90,
        "std_days_to_pass": std,
        "utility": _speed_utility(pass_rate / 100.0, median_days, 10.0),
        "min_contracts": min_c,
        "max_contracts": max_c,
        "attempts_truncated": truncated,
    }


# =============================================================================
# LEGACY optimizers (kept for backward compat; CLI no longer calls these)
# =============================================================================


def optimize_for_speed_wf(
    trades: list[TradeResult],
    strategy: str,
    window: str,
    risk_levels: list[float] | None = None,
    pass_floor_pct: float = 40.0,
    pass_target_pct: float = 75.0,
    instrument: Instrument = MNQ,
    rules: TopStepRules = TOPSTEP_50K,
) -> SpeedOptimizationResult:
    """LEGACY single-fold speed optimizer."""
    if risk_levels is None:
        risk_levels = list(DEFAULT_RISK_GRID)

    empty = SpeedOptimizationResult(
        strategy=strategy, window=window,
        pass_floor_pct=pass_floor_pct, pass_target_pct=pass_target_pct,
        optimal_risk_dollars=0.0, pass_rate_pct=0.0,
        mean_days_to_pass=0.0, std_days_to_pass=0.0,
        min_contracts_used=0, max_contracts_used=0, candidates=(),
    )
    if not trades:
        return empty

    valid = _get_valid_risk_levels(trades, risk_levels, instrument, coverage_threshold=0.0)
    if not valid:
        return empty

    results: list[dict[str, Any]] = []
    for risk in valid:
        m = _evaluate_risk_on_trades(trades, risk, rules, attempt_budget=1000, instrument=instrument)
        results.append(m)

    viable = [r for r in results if r["pass_rate_pct"] >= pass_floor_pct]
    if not viable:
        return replace(empty, candidates=tuple(results))

    viable_sorted = sorted(viable, key=lambda r: (r["mean_days_to_pass"], -r["pass_rate_pct"]))
    top_5 = viable_sorted[:5]
    best = top_5[0]
    return SpeedOptimizationResult(
        strategy=strategy, window=window,
        pass_floor_pct=pass_floor_pct, pass_target_pct=pass_target_pct,
        optimal_risk_dollars=best["risk_dollars"],
        pass_rate_pct=best["pass_rate_pct"],
        mean_days_to_pass=best["mean_days_to_pass"],
        std_days_to_pass=best["std_days_to_pass"],
        min_contracts_used=best["min_contracts"],
        max_contracts_used=best["max_contracts"],
        candidates=tuple(top_5),
    )


def optimize_for_longevity_holdout(
    trades: list[TradeResult],
    strategy: str,
    window: str = "holdout",
    risk_levels: list[float] | None = None,
    min_profit_per_trade: float = 150.0,
    instrument: Instrument = MNQ,
    rules: FundedExpressSimRules = DEFAULT_FUNDED_EXPRESS_SIM,
) -> LongevityOptimizationResult:
    """LEGACY deterministic longevity optimizer."""
    if risk_levels is None:
        risk_levels = list(DEFAULT_RISK_GRID)

    empty = LongevityOptimizationResult(
        strategy=strategy, window=window,
        min_profit_per_trade=min_profit_per_trade,
        optimal_risk_dollars=0.0, avg_pnl_per_trade=0.0, total_pnl=0.0,
        funded_accounts_used=0, accounts_blown=0,
        total_trades_executed=0, longevity_score=0.0, candidates=(),
    )
    if not trades:
        return empty

    valid = _get_valid_risk_levels(trades, risk_levels, instrument, coverage_threshold=0.0)
    if not valid:
        return empty

    results: list[dict[str, Any]] = []
    for risk in valid:
        resized = _resize_trades_for_risk(trades, risk, instrument)
        if not resized:
            continue
        sim = simulate_express_funded_resets(resized, rules)
        total_pnl = sim.accrued_pnl_bank
        total_trades = sum(s.get("trades_applied_count", 0) for s in sim.stints_summary)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        longevity = sim.funded_accounts_used + (total_pnl / rules.account_size)
        results.append({
            "risk_dollars": risk,
            "avg_pnl_per_trade": avg_pnl,
            "total_pnl": total_pnl,
            "funded_accounts_used": sim.funded_accounts_used,
            "accounts_blown": sim.funded_accounts_failed,
            "total_trades_executed": total_trades,
            "longevity_score": longevity,
        })

    viable = [r for r in results if r["avg_pnl_per_trade"] >= min_profit_per_trade]
    if not viable:
        return replace(empty, candidates=tuple(results))
    viable_sorted = sorted(viable, key=lambda r: (-r["longevity_score"], -r["avg_pnl_per_trade"]))
    top_5 = viable_sorted[:5]
    best = top_5[0]
    return LongevityOptimizationResult(
        strategy=strategy, window=window,
        min_profit_per_trade=min_profit_per_trade,
        optimal_risk_dollars=best["risk_dollars"],
        avg_pnl_per_trade=best["avg_pnl_per_trade"],
        total_pnl=best["total_pnl"],
        funded_accounts_used=best["funded_accounts_used"],
        accounts_blown=best["accounts_blown"],
        total_trades_executed=best["total_trades_executed"],
        longevity_score=best["longevity_score"],
        candidates=tuple(top_5),
    )


# =============================================================================
# V2 OPTIMIZERS (CLI uses these)
# =============================================================================


def optimize_speed_wf_aggregate(
    fold_trade_pairs: list[tuple[list[TradeResult], list[TradeResult]]],
    strategy: str,
    risk_levels: list[float] | None = None,
    pass_floor_pct: float = 40.0,
    speed_target_days: float = 10.0,
    attempt_budget: int = 10,
    coverage_threshold: float = 0.5,
    instrument: Instrument = MNQ,
    rules: TopStepRules = TOPSTEP_50K,
    min_viable_folds: int | None = None,
) -> SpeedOptimizationAggregateResult:
    """Train-fit per fold, OOS evaluate, aggregate across folds.

    Returns the risk that wins on median OOS utility (with worst-fold tiebreaker)
    among risks that were viable in >= min_viable_folds folds (default ceil(n_folds/2)).
    """
    if risk_levels is None:
        risk_levels = list(DEFAULT_RISK_GRID)

    n_folds = len(fold_trade_pairs)
    if min_viable_folds is None:
        min_viable_folds = math.ceil(n_folds / 2) if n_folds > 0 else 0

    empty = SpeedOptimizationAggregateResult(
        strategy=strategy,
        pass_floor_pct=pass_floor_pct,
        speed_target_days=speed_target_days,
        attempt_budget=attempt_budget,
        n_folds=n_folds,
        optimal_risk_dollars=0.0,
        median_oos_utility=0.0,
        min_oos_utility=0.0,
        median_oos_pass_rate_pct=0.0,
        median_oos_median_days_to_pass=0.0,
        viable_folds=0,
    )
    if n_folds == 0:
        return empty

    # For each risk level, evaluate train+test on every fold.
    per_risk: dict[float, dict[str, Any]] = {}
    for risk in risk_levels:
        train_utilities: list[float] = []
        oos_metrics: list[dict[str, Any]] = []
        viable_train = 0
        for fold_idx, (train_trades, test_trades) in enumerate(fold_trade_pairs):
            valid_for_fold = _get_valid_risk_levels(
                train_trades, [risk], instrument, coverage_threshold=coverage_threshold
            )
            if not valid_for_fold:
                oos_metrics.append({"fold_index": fold_idx, "risk_dollars": risk, "viable": False})
                continue
            train_m = _evaluate_risk_on_trades(train_trades, risk, rules, attempt_budget, instrument)
            train_m["utility"] = _speed_utility(
                train_m["pass_rate_pct"] / 100.0, train_m["median_days_to_pass"], speed_target_days
            )
            if train_m["utility"] <= 0 or train_m["pass_rate_pct"] < pass_floor_pct:
                test_m = _evaluate_risk_on_trades(test_trades, risk, rules, attempt_budget, instrument)
                test_m["utility"] = _speed_utility(
                    test_m["pass_rate_pct"] / 100.0, test_m["median_days_to_pass"], speed_target_days
                )
                test_m["viable"] = False
                test_m["fold_index"] = fold_idx
                test_m["train_utility"] = train_m["utility"]
                test_m["train_pass_rate_pct"] = train_m["pass_rate_pct"]
                test_m["reject_reason"] = "train_pass_floor_or_utility"
                oos_metrics.append(test_m)
                continue
            viable_train += 1
            train_utilities.append(train_m["utility"])
            test_m = _evaluate_risk_on_trades(test_trades, risk, rules, attempt_budget, instrument)
            test_m["utility"] = _speed_utility(
                test_m["pass_rate_pct"] / 100.0, test_m["median_days_to_pass"], speed_target_days
            )
            test_m["viable"] = True
            test_m["fold_index"] = fold_idx
            test_m["train_utility"] = train_m["utility"]
            oos_metrics.append(test_m)

        measured_oos = [m for m in oos_metrics if "utility" in m]
        oos_utilities = [m["utility"] for m in measured_oos]
        oos_pass_rates = [m["pass_rate_pct"] for m in measured_oos]
        oos_medians = [
            m["median_days_to_pass"]
            for m in measured_oos
            if math.isfinite(m["median_days_to_pass"])
        ]
        per_risk[risk] = {
            "risk_dollars": risk,
            "viable_folds": viable_train,
            "median_oos_utility": float(np.median(oos_utilities)) if oos_utilities else 0.0,
            "min_oos_utility": float(min(oos_utilities)) if oos_utilities else 0.0,
            "median_oos_pass_rate_pct": float(np.median(oos_pass_rates)) if oos_pass_rates else 0.0,
            "median_oos_median_days_to_pass": float(np.median(oos_medians)) if oos_medians else float("inf"),
            "per_fold": tuple(oos_metrics),
        }

    survivors = [c for c in per_risk.values() if c["viable_folds"] >= min_viable_folds]
    if not survivors:
        # No candidate met the hard viability floor. Still surface the best measured
        # risk so fixed-risk comparisons have a real optimizer track instead of $0.
        all_candidates = tuple(
            sorted(
                per_risk.values(),
                key=lambda c: (
                    -c["median_oos_pass_rate_pct"],
                    c["median_oos_median_days_to_pass"] if math.isfinite(c["median_oos_median_days_to_pass"]) else float("inf"),
                    -c["median_oos_utility"],
                    -c["min_oos_utility"],
                ),
            )
        )
        if not all_candidates:
            return empty
        best = all_candidates[0]
        return replace(
            empty,
            optimal_risk_dollars=best["risk_dollars"],
            median_oos_utility=best["median_oos_utility"],
            min_oos_utility=best["min_oos_utility"],
            median_oos_pass_rate_pct=best["median_oos_pass_rate_pct"],
            median_oos_median_days_to_pass=best["median_oos_median_days_to_pass"],
            viable_folds=best["viable_folds"],
            per_fold_oos=best["per_fold"],
            candidates=all_candidates[:5],
        )

    survivors_sorted = sorted(
        survivors,
        key=lambda c: (-c["median_oos_utility"], -c["min_oos_utility"]),
    )
    top_5 = survivors_sorted[:5]
    best = top_5[0]

    return SpeedOptimizationAggregateResult(
        strategy=strategy,
        pass_floor_pct=pass_floor_pct,
        speed_target_days=speed_target_days,
        attempt_budget=attempt_budget,
        n_folds=n_folds,
        optimal_risk_dollars=best["risk_dollars"],
        median_oos_utility=best["median_oos_utility"],
        min_oos_utility=best["min_oos_utility"],
        median_oos_pass_rate_pct=best["median_oos_pass_rate_pct"],
        median_oos_median_days_to_pass=best["median_oos_median_days_to_pass"],
        viable_folds=best["viable_folds"],
        per_fold_oos=best["per_fold"],
        candidates=tuple(top_5),
    )


def optimize_longevity_holdout_mc(
    trades: list[TradeResult],
    strategy: str,
    window: str = "holdout",
    risk_levels: list[float] | None = None,
    min_profit_per_trade: float = 150.0,
    min_profit_factor: float = 1.2,
    weights: dict[str, float] | None = None,
    mc_iterations: int = 500,
    mc_block_size: int = 5,
    bootstrap_iterations: int = 1000,
    confidence_level: float = 0.05,
    coverage_threshold: float = 0.5,
    instrument: Instrument = MNQ,
    rules: FundedExpressSimRules = DEFAULT_FUNDED_EXPRESS_SIM,
) -> LongevityOptimizationMCResult:
    """Block-bootstrap MC longevity optimization.

    Per risk: compute bootstrap diagnostics and MC funded-account longevity. Rank by
    measured median longevity score. The thresholds are reported as diagnostics only;
    they do not gate candidates out of the optimizer.
    """
    if risk_levels is None:
        risk_levels = list(DEFAULT_RISK_GRID)
    if weights is None:
        weights = {"survival": 0.4, "drawdown": 0.2, "efficiency": 0.2, "capital": 0.2}

    p_pct = confidence_level * 100.0  # e.g. 0.05 -> 5

    empty = LongevityOptimizationMCResult(
        strategy=strategy, window=window,
        min_profit_per_trade=min_profit_per_trade,
        min_profit_factor=min_profit_factor,
        weights=weights,
        mc_iterations=mc_iterations,
        mc_block_size=mc_block_size,
        bootstrap_iterations=bootstrap_iterations,
        optimal_risk_dollars=0.0,
        median_longevity_score=0.0, p05_longevity_score=0.0,
        median_components={}, p05_components={},
        median_avg_pnl_per_trade=0.0, p05_avg_pnl_per_trade=0.0,
        median_accounts_used=0.0, median_accounts_blown=0.0,
    )
    if not trades:
        return empty

    valid = _get_valid_risk_levels(trades, risk_levels, instrument, coverage_threshold=coverage_threshold)
    if not valid:
        return empty

    results: list[dict[str, Any]] = []

    for risk in valid:
        resized = _resize_trades_for_risk(trades, risk, instrument)
        if not resized:
            continue

        pf = _profit_factor(resized)
        p05_avg = _bootstrap_pnl_p05(resized, n=bootstrap_iterations, seed=42, percentile=p_pct)

        # Baseline (deterministic) sim for surfaced per-account survival data
        baseline_sim = simulate_express_funded_resets(resized, rules)
        comparison_longevity_score = 1.0 + baseline_sim.accrued_pnl_bank / rules.account_size
        baseline_components = _compute_longevity_components(
            baseline_sim, rules, min_profit_per_trade, weights
        )

        # MC loop
        comp_lists: dict[str, list[float]] = {
            "survival_score": [], "drawdown_score": [], "efficiency_score": [],
            "capital_score": [], "bank_return_score": [], "account_efficiency_score": [],
            "consistency_score": [],
            "longevity_score": [], "avg_pnl_per_trade": [], "accounts_used": [], "accounts_blown": [],
        }
        for permuted in _block_bootstrap_trade_sequences(
            resized, n=mc_iterations, block_size=mc_block_size, seed=42
        ):
            sim = simulate_express_funded_resets(permuted, rules)
            comps = _compute_longevity_components(sim, rules, min_profit_per_trade, weights)
            for k, v in comps.items():
                comp_lists[k].append(float(v))

        median_comps = {k: float(np.median(v)) if v else 0.0 for k, v in comp_lists.items()}
        p05_comps = {k: float(np.percentile(v, p_pct)) if v else 0.0 for k, v in comp_lists.items()}
        consistency_factor = (
            _clamp01(p05_comps["bank_return_score"] / median_comps["bank_return_score"])
            if median_comps["bank_return_score"] > 0.0
            else 0.0
        )
        true_longevity_score = _true_longevity_score(
            median_comps["bank_return_score"],
            median_comps["survival_score"],
            median_comps["drawdown_score"],
            consistency_factor,
            median_comps["account_efficiency_score"],
        )
        p05_longevity_score = _true_longevity_score(
            p05_comps["bank_return_score"],
            p05_comps["survival_score"],
            p05_comps["drawdown_score"],
            consistency_factor,
            p05_comps["account_efficiency_score"],
        )

        results.append({
            "risk_dollars": risk,
            "rejected": False,
            "profit_factor": pf,
            "profit_factor_floor": min_profit_factor,
            "p05_avg_pnl_floor": min_profit_per_trade,
            "comparison_longevity_score": comparison_longevity_score,
            "mc_median_longevity_score": median_comps["longevity_score"],
            "median_longevity_score": true_longevity_score,
            "p05_longevity_score": p05_longevity_score,
            "median_components": {
                k: median_comps[k]
                for k in (
                    "bank_return_score",
                    "survival_score",
                    "drawdown_score",
                    "efficiency_score",
                    "capital_score",
                    "account_efficiency_score",
                )
            } | {"consistency_score": consistency_factor},
            "p05_components": {
                k: p05_comps[k]
                for k in (
                    "bank_return_score",
                    "survival_score",
                    "drawdown_score",
                    "efficiency_score",
                    "capital_score",
                    "account_efficiency_score",
                )
            } | {"consistency_score": consistency_factor},
            "median_avg_pnl_per_trade": median_comps["avg_pnl_per_trade"],
            "p05_avg_pnl_per_trade": p05_comps["avg_pnl_per_trade"],
            "median_accounts_used": median_comps["accounts_used"],
            "median_accounts_blown": median_comps["accounts_blown"],
            "baseline_components": baseline_components,
            "baseline_per_account": tuple(dict(s) for s in baseline_sim.stints_summary),
            "baseline_survival_days": tuple(int(s.get("survival_days", 0)) for s in baseline_sim.stints_summary),
        })

    survivors = [r for r in results if not r.get("rejected")]
    if not survivors:
        # No survivor passed every hard longevity filter. Still return the best
        # measured risk so reports/comparisons do not collapse to a fake $0 track.
        all_candidates = tuple(
            sorted(
                results,
                key=lambda r: (-float(r.get("median_longevity_score", 0.0)), -float(r.get("p05_longevity_score", 0.0))),
            )
        )
        if not all_candidates:
            return empty
        best = all_candidates[0]
        return replace(
            empty,
            optimal_risk_dollars=best["risk_dollars"],
            median_longevity_score=float(best.get("median_longevity_score", 0.0)),
            p05_longevity_score=float(best.get("p05_longevity_score", 0.0)),
            median_components=dict(best.get("median_components", {})),
            p05_components=dict(best.get("p05_components", {})),
            median_avg_pnl_per_trade=float(best.get("median_avg_pnl_per_trade", 0.0)),
            p05_avg_pnl_per_trade=float(best.get("p05_avg_pnl_per_trade", 0.0)),
            median_accounts_used=float(best.get("median_accounts_used", 0.0)),
            median_accounts_blown=float(best.get("median_accounts_blown", 0.0)),
            per_account_survival_days=tuple(best.get("baseline_survival_days", ())),
            per_account_summary=tuple(best.get("baseline_per_account", ())),
            candidates=all_candidates[:8],
        )

    survivors_sorted = sorted(
        survivors,
        key=lambda r: (-r["median_longevity_score"], -r["p05_longevity_score"]),
    )
    top_5 = survivors_sorted[:5]
    best = top_5[0]

    return LongevityOptimizationMCResult(
        strategy=strategy, window=window,
        min_profit_per_trade=min_profit_per_trade,
        min_profit_factor=min_profit_factor,
        weights=weights,
        mc_iterations=mc_iterations,
        mc_block_size=mc_block_size,
        bootstrap_iterations=bootstrap_iterations,
        optimal_risk_dollars=best["risk_dollars"],
        median_longevity_score=best["median_longevity_score"],
        p05_longevity_score=best["p05_longevity_score"],
        median_components=best["median_components"],
        p05_components=best["p05_components"],
        median_avg_pnl_per_trade=best["median_avg_pnl_per_trade"],
        p05_avg_pnl_per_trade=best["p05_avg_pnl_per_trade"],
        median_accounts_used=best["median_accounts_used"],
        median_accounts_blown=best["median_accounts_blown"],
        per_account_survival_days=best["baseline_survival_days"],
        per_account_summary=best["baseline_per_account"],
        candidates=tuple(top_5),
    )


__all__ = [
    "SpeedOptimizationResult",
    "LongevityOptimizationResult",
    "SpeedOptimizationAggregateResult",
    "LongevityOptimizationMCResult",
    "DEFAULT_RISK_GRID",
    "optimize_for_speed_wf",
    "optimize_for_longevity_holdout",
    "optimize_speed_wf_aggregate",
    "optimize_longevity_holdout_mc",
    "_resize_trades_for_risk",
    "_resize_trades_for_fixed_contracts",
    "_count_sequential_eval_passes_capped",
    "_evaluate_risk_on_trades",
    "_compute_longevity_components",
    "_speed_utility",
    "_get_valid_risk_levels",
    "_bootstrap_pnl_p05",
    "_block_bootstrap_trade_sequences",
]
