"""Parameter sensitivity sweep: vary each `param_grid` value and compare Combine pass rates.

Flags a cliff when a one-step neighbor drops more than `drop_threshold` (fraction
of pass-rate points) below the baseline — suggests an over-fit peak at defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    default_params: dict[str, Any]          # actual default param dict used in sweep


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
        default_params=dict(default_params),
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


def _describe_gradient(values_sorted: list[Any], rates: list[float], default_val: Any) -> str:
    """One-line text description of the rate gradient for a single param."""
    if len(rates) < 2:
        return "insufficient values"
    spread = max(rates) - min(rates)
    if spread < 3.0:
        return "flat (< 3pp spread)"
    default_idx = next((i for i, v in enumerate(values_sorted) if v == default_val), None)
    peak_idx = rates.index(max(rates))
    if default_idx is not None and peak_idx == default_idx:
        return f"peak at selected value ({default_val})"
    if peak_idx == 0:
        return "decreasing (best at min value)"
    if peak_idx == len(rates) - 1:
        return "increasing (best at max value)"
    return f"peak at {values_sorted[peak_idx]} ({spread:.1f}pp spread)"


def plot_sensitivity_heatmap(
    report: SensitivityReport,
    output_path: Path | str,
) -> Path:
    """Save one subplot per param showing pass-rate vs param value.

    Highlights default/selected value in orange. Returns the saved path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    params = list(report.param_results.keys())
    n = len(params)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No param results", ha="center", va="center")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return out

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax_idx, param_name in enumerate(params):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        raw = report.param_results[param_name]

        try:
            sorted_items = sorted(raw.items(), key=lambda kv: float(kv[0]))
        except (ValueError, TypeError):
            sorted_items = sorted(raw.items(), key=lambda kv: kv[0])

        xs_labels = [kv[0] for kv in sorted_items]
        ys = [float(kv[1]) for kv in sorted_items]

        default_key = str(report.default_params.get(param_name, ""))
        colors = ["darkorange" if label == default_key else "steelblue" for label in xs_labels]

        bars = ax.bar(range(len(xs_labels)), ys, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(report.default_pass_rate, color="gray", linewidth=1.0, linestyle="--", alpha=0.7, label=f"default={report.default_pass_rate:.1f}%")
        ax.set_xticks(range(len(xs_labels)))
        ax.set_xticklabels(xs_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Pass rate (%)")
        ax.set_ylim(0, max(100, max(ys) * 1.1) if ys else 100)
        gradient_desc = _describe_gradient(
            [kv[0] for kv in sorted_items],
            ys,
            default_key,
        )
        ax.set_title(f"{param_name}\n{gradient_desc}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide empty subplots
    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Parameter Sensitivity — {report.strategy}\n"
        f"default_pass_rate={report.default_pass_rate:.1f}%  "
        f"min_neighbor={report.min_neighbor_pass_rate:.1f}%  "
        f"cliff={'YES' if report.is_cliff else 'no'}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def sensitivity_heatmap_text(report: SensitivityReport) -> str:
    """One-line gradient description per param."""
    lines = [f"Parameter sensitivity — {report.strategy}"]
    for param_name, raw in report.param_results.items():
        try:
            sorted_items = sorted(raw.items(), key=lambda kv: float(kv[0]))
        except (ValueError, TypeError):
            sorted_items = sorted(raw.items(), key=lambda kv: kv[0])
        xs = [kv[0] for kv in sorted_items]
        ys = [float(kv[1]) for kv in sorted_items]
        default_key = str(report.default_params.get(param_name, ""))
        desc = _describe_gradient(xs, ys, default_key)
        lines.append(f"  {param_name}: {desc}  (range {min(ys):.1f}%–{max(ys):.1f}%)")
    return "\n".join(lines)


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
    "apply_sensitivity_to_verdict",
    "plot_sensitivity_heatmap",
    "run_sensitivity",
    "sensitivity_heatmap_text",
    "sensitivity_summary_dict",
]
