"""Block-bootstrap Monte Carlo for trade sequences.

Two pipeline uses:
  - Sensitivity MC (MC1): permute sensitivity-window trades around selected params
  - Holdout MC (MC2): permute holdout trades

Each stage: N block-bootstrap permutations → equity paths → 4 key metrics with CI bands.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .trades import TradeResult


@dataclass(frozen=True)
class MCResult:
    n_perms: int
    block_size: int
    seed: int
    ci_pct: float
    # PnL distribution
    pnl_mean: float
    pnl_p05: float
    pnl_p50: float
    pnl_p95: float
    # Win rate with CI
    win_rate_mean: float
    win_rate_ci_lo: float
    win_rate_ci_hi: float
    # Expectancy (mean R-multiple) with CI
    expectancy_mean: float
    expectancy_ci_lo: float
    expectancy_ci_hi: float
    # Max daily loss (worst single day)
    max_daily_loss_mean: float
    max_daily_loss_worst_p05: float
    # Drawdown duration (trades from peak to recovery)
    dd_duration_mean: float
    dd_duration_p95: float
    # Max drawdown distribution
    max_dd_mean: float
    max_dd_p05: float
    max_dd_p95: float
    # Equity curves for plotting
    equity_curves: tuple[tuple[float, ...], ...]
    actual_equity: tuple[float, ...]


def block_bootstrap_permute(
    trades: list[TradeResult],
    block_size: int,
    rng: random.Random,
) -> list[TradeResult]:
    """Return block-shuffled permutation. Blocks are shuffled; intra-block order preserved."""
    if not trades:
        return []
    blocks: list[list[TradeResult]] = [
        trades[i : i + block_size] for i in range(0, len(trades), block_size)
    ]
    rng.shuffle(blocks)
    result: list[TradeResult] = []
    for block in blocks:
        result.extend(block)
    return result


def _equity_curve(trades: list[TradeResult]) -> list[float]:
    curve = [0.0]
    equity = 0.0
    for t in trades:
        equity += t.net_pnl
        curve.append(equity)
    return curve


def _max_daily_loss(trades: list[TradeResult]) -> float:
    """Min (most negative) single-day net PnL. Returns 0.0 if no losses."""
    if not trades:
        return 0.0
    daily: dict[Any, float] = {}
    for t in trades:
        day = t.exit_time.normalize()
        daily[day] = daily.get(day, 0.0) + t.net_pnl
    return min(daily.values()) if daily else 0.0


def _dd_duration_trades(equity: list[float]) -> int:
    """Max number of trades from equity peak to recovery (0 if never in drawdown)."""
    n = len(equity)
    if n <= 1:
        return 0
    peak_val = equity[0]
    peak_idx = 0
    max_dur = 0
    in_dd = False
    for i in range(1, n):
        if equity[i] < peak_val:
            in_dd = True
        elif equity[i] >= peak_val:
            if in_dd:
                max_dur = max(max_dur, i - peak_idx)
                in_dd = False
            if equity[i] > peak_val:
                peak_val = equity[i]
            peak_idx = i
    if in_dd:
        max_dur = max(max_dur, (n - 1) - peak_idx)
    return max_dur


def run_mc(
    trades: list[TradeResult],
    *,
    n_perms: int = 1000,
    block_size: int = 5,
    seed: int = 42,
    ci_pct: float = 95.0,
) -> MCResult:
    """Run block-bootstrap MC and compute metrics across all permutations."""
    rng = random.Random(seed)
    base = list(trades)

    win_rates: list[float] = []
    expectancies: list[float] = []
    daily_losses: list[float] = []
    dd_durations: list[float] = []
    pnls: list[float] = []
    max_dds: list[float] = []
    equity_curves: list[tuple[float, ...]] = []

    ci_lo = (100.0 - ci_pct) / 2.0
    ci_hi = 100.0 - ci_lo

    for _ in range(n_perms):
        perm = block_bootstrap_permute(base, block_size, rng)
        if not perm:
            continue

        eq = _equity_curve(perm)
        equity_curves.append(tuple(eq))
        pnls.append(eq[-1])

        arr_eq = np.array(eq)
        peak = np.maximum.accumulate(arr_eq)
        max_dds.append(float(np.max(peak - arr_eq, initial=0.0)))

        win_rates.append(float(np.mean([t.net_pnl > 0 for t in perm])))
        expectancies.append(float(np.mean([t.r_multiple for t in perm])))
        daily_losses.append(_max_daily_loss(perm))
        dd_durations.append(float(_dd_duration_trades(eq)))

    actual_equity = tuple(_equity_curve(base))

    _empty = MCResult(
        n_perms=n_perms,
        block_size=block_size,
        seed=seed,
        ci_pct=ci_pct,
        pnl_mean=0.0,
        pnl_p05=0.0,
        pnl_p50=0.0,
        pnl_p95=0.0,
        win_rate_mean=0.0,
        win_rate_ci_lo=0.0,
        win_rate_ci_hi=0.0,
        expectancy_mean=0.0,
        expectancy_ci_lo=0.0,
        expectancy_ci_hi=0.0,
        max_daily_loss_mean=0.0,
        max_daily_loss_worst_p05=0.0,
        dd_duration_mean=0.0,
        dd_duration_p95=0.0,
        max_dd_mean=0.0,
        max_dd_p05=0.0,
        max_dd_p95=0.0,
        equity_curves=(),
        actual_equity=actual_equity,
    )

    if not win_rates:
        return _empty

    wr = np.array(win_rates)
    ex = np.array(expectancies)
    dl = np.array(daily_losses)
    dd = np.array(dd_durations)
    pnl_arr = np.array(pnls)
    mdd_arr = np.array(max_dds)

    return MCResult(
        n_perms=n_perms,
        block_size=block_size,
        seed=seed,
        ci_pct=ci_pct,
        pnl_mean=float(np.mean(pnl_arr)),
        pnl_p05=float(np.percentile(pnl_arr, 5.0)),
        pnl_p50=float(np.percentile(pnl_arr, 50.0)),
        pnl_p95=float(np.percentile(pnl_arr, 95.0)),
        win_rate_mean=float(np.mean(wr)),
        win_rate_ci_lo=float(np.percentile(wr, ci_lo)),
        win_rate_ci_hi=float(np.percentile(wr, ci_hi)),
        expectancy_mean=float(np.mean(ex)),
        expectancy_ci_lo=float(np.percentile(ex, ci_lo)),
        expectancy_ci_hi=float(np.percentile(ex, ci_hi)),
        max_daily_loss_mean=float(np.mean(dl)),
        max_daily_loss_worst_p05=float(np.percentile(dl, 5.0)),
        dd_duration_mean=float(np.mean(dd)),
        dd_duration_p95=float(np.percentile(dd, 95.0)),
        max_dd_mean=float(np.mean(mdd_arr)),
        max_dd_p05=float(np.percentile(mdd_arr, 5.0)),
        max_dd_p95=float(np.percentile(mdd_arr, 95.0)),
        equity_curves=tuple(equity_curves),
        actual_equity=actual_equity,
    )


def plot_mc_paths(
    result: MCResult,
    output_path: Path | str,
    *,
    title: str = "Monte Carlo Equity Paths",
) -> Path:
    """Save equity path chart: all permutations (sampled) + CI band + actual path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    if result.equity_curves:
        n_steps = max(len(c) for c in result.equity_curves)
        padded = []
        for curve in result.equity_curves:
            arr = np.array(curve, dtype=float)
            if len(arr) < n_steps:
                arr = np.pad(arr, (0, n_steps - len(arr)), mode="edge")
            padded.append(arr)
        matrix = np.array(padded)
        x = np.arange(n_steps)

        ci_lo = (100.0 - result.ci_pct) / 2.0
        ci_hi = 100.0 - ci_lo
        lo = np.percentile(matrix, ci_lo, axis=0)
        hi = np.percentile(matrix, ci_hi, axis=0)
        med = np.percentile(matrix, 50.0, axis=0)

        n_show = min(300, len(result.equity_curves))
        step = max(1, len(result.equity_curves) // n_show)
        for i in range(0, len(result.equity_curves), step):
            ax.plot(x, matrix[i], color="#90b8d8", alpha=0.12, linewidth=0.4)

        ax.fill_between(x, lo, hi, alpha=0.22, color="steelblue", label=f"{result.ci_pct:.0f}% CI band")
        ax.plot(x, med, color="steelblue", linewidth=1.5, linestyle="--", alpha=0.8, label="Median perm")

    actual = np.array(result.actual_equity, dtype=float)
    ax.plot(np.arange(len(actual)), actual, color="darkorange", linewidth=2.5, label="Actual", zorder=5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative Net P&L ($)")
    ax.set_title(
        f"{title}\n"
        f"n={result.n_perms} block_size={result.block_size}  "
        f"{result.ci_pct:.0f}% CI  "
        f"pnl p05={result.pnl_p05:+.0f} p95={result.pnl_p95:+.0f}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def mc_summary_text(result: MCResult, title: str = "Monte Carlo") -> str:
    lines = [
        f"{title}",
        f"  n_perms={result.n_perms}  block_size={result.block_size}  CI={result.ci_pct:.0f}%",
        f"  pnl  p05={result.pnl_p05:+.2f}  p50={result.pnl_p50:+.2f}  p95={result.pnl_p95:+.2f}",
        f"  win_rate  mean={result.win_rate_mean:.3f}  CI=[{result.win_rate_ci_lo:.3f}, {result.win_rate_ci_hi:.3f}]",
        f"  expectancy  mean={result.expectancy_mean:.4f}R  CI=[{result.expectancy_ci_lo:.4f}R, {result.expectancy_ci_hi:.4f}R]",
        f"  max_daily_loss  mean={result.max_daily_loss_mean:.2f}  worst_p95={result.max_daily_loss_worst_p05:.2f}",
        f"  dd_duration  mean={result.dd_duration_mean:.1f} trades  p95={result.dd_duration_p95:.1f} trades",
        f"  max_dd  p05={result.max_dd_p05:.2f}  p95={result.max_dd_p95:.2f}",
    ]
    return "\n".join(lines)


def mc_summary_dict(result: MCResult) -> dict[str, Any]:
    return {
        "n_perms": result.n_perms,
        "block_size": result.block_size,
        "seed": result.seed,
        "ci_pct": result.ci_pct,
        "pnl_mean": result.pnl_mean,
        "pnl_p05": result.pnl_p05,
        "pnl_p50": result.pnl_p50,
        "pnl_p95": result.pnl_p95,
        "win_rate_mean": result.win_rate_mean,
        "win_rate_ci_lo": result.win_rate_ci_lo,
        "win_rate_ci_hi": result.win_rate_ci_hi,
        "expectancy_mean": result.expectancy_mean,
        "expectancy_ci_lo": result.expectancy_ci_lo,
        "expectancy_ci_hi": result.expectancy_ci_hi,
        "max_daily_loss_mean": result.max_daily_loss_mean,
        "max_daily_loss_worst_p05": result.max_daily_loss_worst_p05,
        "dd_duration_mean": result.dd_duration_mean,
        "dd_duration_p95": result.dd_duration_p95,
        "max_dd_mean": result.max_dd_mean,
        "max_dd_p05": result.max_dd_p05,
        "max_dd_p95": result.max_dd_p95,
    }


__all__ = [
    "MCResult",
    "block_bootstrap_permute",
    "run_mc",
    "plot_mc_paths",
    "mc_summary_text",
    "mc_summary_dict",
]
