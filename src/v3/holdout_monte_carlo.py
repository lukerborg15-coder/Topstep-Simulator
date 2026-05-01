"""Holdout Monte Carlo: random permutations of the holdout trade list (sequence stress-test)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np

from .evaluator import compute_metrics
from .trades import TradeResult


@dataclass(frozen=True)
class HoldoutMonteCarloResult:
    n: int
    seed: int
    pnl_mean: float
    pnl_p05: float
    pnl_p50: float
    pnl_p95: float
    max_dd_mean: float
    max_dd_p05: float
    max_dd_p95: float


def run_holdout_trade_monte_carlo(
    trades: list[TradeResult],
    *,
    n: int = 1000,
    seed: int = 42,
) -> HoldoutMonteCarloResult:
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    base = list(trades)
    pnls: list[float] = []
    dds: list[float] = []
    for _ in range(n):
        t = base[:]
        rng.shuffle(t)
        m = compute_metrics(t)
        pnls.append(float(m["total_net_pnl"]))
        dds.append(float(m["max_drawdown"]))
    arr_p = np.array(pnls, dtype=float)
    arr_d = np.array(dds, dtype=float)
    return HoldoutMonteCarloResult(
        n=n,
        seed=seed,
        pnl_mean=float(mean(pnls)),
        pnl_p05=float(np.percentile(arr_p, 5)),
        pnl_p50=float(np.percentile(arr_p, 50)),
        pnl_p95=float(np.percentile(arr_p, 95)),
        max_dd_mean=float(mean(dds)),
        max_dd_p05=float(np.percentile(arr_d, 5)),
        max_dd_p95=float(np.percentile(arr_d, 95)),
    )


def holdout_monte_carlo_summary_dict(result: HoldoutMonteCarloResult) -> dict[str, Any]:
    return {
        "holdout_mc_n": result.n,
        "holdout_mc_seed": result.seed,
        "holdout_mc_pnl_mean": result.pnl_mean,
        "holdout_mc_pnl_p05": result.pnl_p05,
        "holdout_mc_pnl_p50": result.pnl_p50,
        "holdout_mc_pnl_p95": result.pnl_p95,
        "holdout_mc_max_dd_mean": result.max_dd_mean,
        "holdout_mc_max_dd_p05": result.max_dd_p05,
        "holdout_mc_max_dd_p95": result.max_dd_p95,
    }


__all__ = [
    "HoldoutMonteCarloResult",
    "holdout_monte_carlo_summary_dict",
    "run_holdout_trade_monte_carlo",
]
