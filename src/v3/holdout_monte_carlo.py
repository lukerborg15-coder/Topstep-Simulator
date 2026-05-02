"""Holdout Monte Carlo: block-bootstrap trade sequence stress-test.

Wraps monte_carlo.run_mc for the holdout stage. Exposes MCResult directly.
Legacy HoldoutMonteCarloResult is an alias for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .monte_carlo import (
    MCResult,
    mc_summary_dict,
    mc_summary_text,
    plot_mc_paths,
    run_mc,
)
from .trades import TradeResult

HoldoutMonteCarloResult = MCResult


def run_holdout_trade_monte_carlo(
    trades: list[TradeResult],
    *,
    n: int = 1000,
    seed: int = 42,
    block_size: int = 5,
    ci_pct: float = 95.0,
) -> MCResult:
    """Block-bootstrap MC on holdout trades. Returns MCResult."""
    return run_mc(trades, n_perms=n, block_size=block_size, seed=seed, ci_pct=ci_pct)


def holdout_monte_carlo_summary_dict(result: MCResult) -> dict[str, Any]:
    d = mc_summary_dict(result)
    # Backward-compatible key aliases used by existing verdict / result bundle code.
    d["holdout_mc_n"] = result.n_perms
    d["holdout_mc_seed"] = result.seed
    d["holdout_mc_pnl_mean"] = result.pnl_mean
    d["holdout_mc_pnl_p05"] = result.pnl_p05
    d["holdout_mc_pnl_p50"] = result.pnl_p50
    d["holdout_mc_pnl_p95"] = result.pnl_p95
    d["holdout_mc_max_dd_mean"] = result.max_dd_mean
    d["holdout_mc_max_dd_p05"] = result.max_dd_p05
    d["holdout_mc_max_dd_p95"] = result.max_dd_p95
    return d


def plot_holdout_mc_paths(result: MCResult, output_path: Path | str) -> Path:
    return plot_mc_paths(result, output_path, title="Holdout Monte Carlo Equity Paths")


__all__ = [
    "HoldoutMonteCarloResult",
    "holdout_monte_carlo_summary_dict",
    "plot_holdout_mc_paths",
    "run_holdout_trade_monte_carlo",
]
