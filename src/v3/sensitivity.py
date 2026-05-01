"""Parameter sensitivity sweep: vary each `param_grid` value and compare Combine pass rates.

Flags a cliff when a one-step neighbor drops more than `drop_threshold` (fraction
of pass-rate points) below the baseline — suggests an over-fit peak at defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .combine_simulator import CombineSimResult, run_combine_simulator
from .trades import TradeResult
from .verdict import VerdictResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensitivityReport:
    """Output of a parameter sensitivity sweep."""

    strategy: str
    is_cliff: bool                          # True = cannot reach COMBINE-READY
    cliff_params: tuple[str, ...]           # which params showed cliff behaviour
    param_results: dict[str, dict]          # param_name -> {str(value): pass_rate_pct}
    default_pass_rate: float                # pass_rate_pct at default_params
    min_neighbor_pass_rate: float           # lowest pass_rate among all 1-step neighbours
    drop_threshold: float                   # cliff threshold used (default 0.25 = 25pp)


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_sensitivity(
    strategy_name: str,
    default_params: dict[str, Any],
    param_grid: dict[str, tuple[Any, ...]],
    sim_fn: Callable[..., CombineSimResult],
    trades_fn: Callable[[dict[str, Any]], list[TradeResult]],
    n_resamples: int = 200,
    seed: int = 42,
    drop_threshold: float = 0.25,
) -> SensitivityReport:
    """Run a sensitivity sweep across every value in param_grid.

    For each parameter, every value in its grid is evaluated independently
    (all other params held at their defaults).  A parameter is flagged as a
    cliff param if ANY neighbour value produces a pass_rate_pct more than
    drop_threshold (expressed as a fraction, so 0.25 = 25 percentage points)
    below the default pass_rate.

    Parameters
    ----------
    strategy_name:
        Display name — used only for the report label.
    default_params:
        The strategy's default parameter dict.
    param_grid:
        Dict of param_name -> tuple of candidate values (must include the
        default value for each param).
    sim_fn:
        Callable with the signature of run_combine_simulator.
    trades_fn:
        Callable(params: dict) -> list[TradeResult].  Runs the backtest for a
        given param set and returns the resulting trades.
    n_resamples:
        Bootstrap iterations per sweep point.  Lower than full-pipeline (200
        vs 1000) to keep sweep time reasonable.
    seed:
        RNG seed for reproducibility.
    drop_threshold:
        Fraction of pass_rate_pct drop that constitutes a cliff.
        0.25 means a neighbour more than 25pp below default is a cliff.
    """
    # Get baseline pass rate at default params
    default_trades = trades_fn(default_params)
    default_sim = sim_fn(default_trades, n_resamples=n_resamples, seed=seed)
    default_pass_rate = default_sim.pass_rate_pct

    param_results: dict[str, dict] = {}
    cliff_params: list[str] = []
    all_neighbor_rates: list[float] = []

    for param_name, grid_values in param_grid.items():
        param_results[param_name] = {}
        param_has_cliff = False

        for value in grid_values:
            # Build params with this single param varied
            test_params = dict(default_params)
            test_params[param_name] = value

            trades = trades_fn(test_params)
            sim = sim_fn(trades, n_resamples=n_resamples, seed=seed)
            rate = sim.pass_rate_pct
            param_results[param_name][str(value)] = rate

            # Only check neighbours (skip the default value itself)
            if value != default_params.get(param_name):
                all_neighbor_rates.append(rate)
                drop = (default_pass_rate - rate) / 100.0  # convert pp to fraction
                if drop > drop_threshold:
                    param_has_cliff = True

        if param_has_cliff:
            cliff_params.append(param_name)

    min_neighbor = min(all_neighbor_rates) if all_neighbor_rates else default_pass_rate

    return SensitivityReport(
        strategy=strategy_name,
        is_cliff=len(cliff_params) > 0,
        cliff_params=tuple(cliff_params),
        param_results=param_results,
        default_pass_rate=default_pass_rate,
        min_neighbor_pass_rate=min_neighbor,
        drop_threshold=drop_threshold,
    )


# ---------------------------------------------------------------------------
# Verdict integration
# ---------------------------------------------------------------------------

def apply_sensitivity_to_verdict(
    verdict: VerdictResult,
    report: SensitivityReport,
) -> VerdictResult:
    """Return a new VerdictResult with sensitivity_flag set from the report.

    If is_cliff is True and the verdict was COMBINE-READY, it is downgraded
    to PROMISING and the cliff warning is added to warn_reasons.

    REJECT verdicts are never changed — a strategy that already fails hard
    gates is not helped by good sensitivity.
    """
    new_sensitivity_flag = report.is_cliff

    new_warn_reasons = list(verdict.warn_reasons)
    # Remove any prior sensitivity warn so we don't double-append on re-runs
    new_warn_reasons = [r for r in new_warn_reasons if "sensitivity_flag" not in r and "cliff" not in r]

    # Only append cliff narrative for non-REJECT verdicts — REJECT bundles stay clean
    if report.is_cliff and verdict.verdict != "REJECT":
        cliff_label = ", ".join(report.cliff_params) if report.cliff_params else "unknown"
        new_warn_reasons.append(f"cliff detected on params: {cliff_label}")

    # Downgrade COMBINE-READY → PROMISING when cliff detected
    if verdict.verdict == "COMBINE-READY" and report.is_cliff:
        new_verdict = "PROMISING"
    else:
        new_verdict = verdict.verdict

    return VerdictResult(
        strategy=verdict.strategy,
        verdict=new_verdict,
        reject_reasons=verdict.reject_reasons,
        warn_reasons=tuple(new_warn_reasons),
        pass_rate_pct=verdict.pass_rate_pct,
        worst_max_drawdown=verdict.worst_max_drawdown,
        pct_daily_limit_hit=verdict.pct_daily_limit_hit,
        mean_max_drawdown=verdict.mean_max_drawdown,
        sensitivity_flag=new_sensitivity_flag,
        wf_robust_ok=verdict.wf_robust_ok,
        holdout_net_pnl=verdict.holdout_net_pnl,
        holdout_max_drawdown=verdict.holdout_max_drawdown,
        holdout_mc_pnl_p05=verdict.holdout_mc_pnl_p05,
    )


def sensitivity_summary_dict(report: SensitivityReport) -> dict[str, Any]:
    """Flat dict suitable for logging and audit stamps."""
    return {
        "sensitivity_strategy": report.strategy,
        "sensitivity_is_cliff": report.is_cliff,
        "sensitivity_cliff_params": list(report.cliff_params),
        "sensitivity_default_pass_rate": report.default_pass_rate,
        "sensitivity_min_neighbor_pass_rate": report.min_neighbor_pass_rate,
        "sensitivity_drop_threshold": report.drop_threshold,
        "sensitivity_param_results": report.param_results,
    }


__all__ = [
    "SensitivityReport",
    "run_sensitivity",
    "apply_sensitivity_to_verdict",
    "sensitivity_summary_dict",
]
