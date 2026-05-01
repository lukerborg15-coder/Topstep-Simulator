from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .combine_simulator import CombineSimResult
from .config import VERDICT_THRESHOLDS, VerdictThresholds


Verdict = Literal["REJECT", "PROMISING", "COMBINE-READY"]


@dataclass(frozen=True)
class VerdictResult:
    strategy: str
    verdict: Verdict
    reject_reasons: tuple[str, ...]
    warn_reasons: tuple[str, ...]
    pass_rate_pct: float
    worst_max_drawdown: float
    pct_daily_limit_hit: float
    mean_max_drawdown: float
    sensitivity_flag: bool | None
    wf_robust_ok: bool | None = None
    holdout_net_pnl: float | None = None
    holdout_max_drawdown: float | None = None
    holdout_mc_pnl_p05: float | None = None


def _format_threshold(value: float) -> str:
    return f"{value:g}"


def compute_verdict(
    strategy: str,
    sim: CombineSimResult,
    thresholds: VerdictThresholds = VERDICT_THRESHOLDS,
) -> VerdictResult:
    """Apply threshold ladder to Combine simulation metrics (`compute_verdict` helper)."""
    # sensitivity_flag is set downstream by apply_sensitivity_to_verdict()
    sensitivity_flag = None
    reject_reasons: list[str] = []
    warn_reasons: list[str] = []

    if sim.pass_rate_pct < thresholds.reject_pass_rate_pct:
        reject_reasons.append(f"pass_rate_pct < {_format_threshold(thresholds.reject_pass_rate_pct)}")
    if sim.worst_max_drawdown > thresholds.reject_max_dd:
        reject_reasons.append(f"worst_max_drawdown > {_format_threshold(thresholds.reject_max_dd)}")
    if sim.pct_daily_limit_hit > thresholds.reject_daily_hit_pct:
        reject_reasons.append(f"pct_daily_limit_hit > {_format_threshold(thresholds.reject_daily_hit_pct)}")
    if sim.mean_max_drawdown > thresholds.reject_mean_dd:
        reject_reasons.append(f"mean_max_drawdown > {_format_threshold(thresholds.reject_mean_dd)}")

    if thresholds.ready_daily_hit_pct <= sim.pct_daily_limit_hit <= thresholds.reject_daily_hit_pct:
        warn_reasons.append(
            "pct_daily_limit_hit between "
            f"{_format_threshold(thresholds.ready_daily_hit_pct)} and "
            f"{_format_threshold(thresholds.reject_daily_hit_pct)}"
        )
    if sensitivity_flag is True:
        warn_reasons.append("sensitivity_flag is True")

    ready = (
        sim.pass_rate_pct >= thresholds.ready_pass_rate_pct
        and sim.worst_max_drawdown <= thresholds.ready_max_dd
        and sim.pct_daily_limit_hit <= thresholds.ready_daily_hit_pct
        and sim.mean_max_drawdown <= thresholds.ready_mean_dd
    )
    if ready and not reject_reasons and not warn_reasons:
        verdict: Verdict = "COMBINE-READY"
    elif reject_reasons:
        verdict = "REJECT"
    else:
        verdict = "PROMISING"

    return VerdictResult(
        strategy=strategy,
        verdict=verdict,
        reject_reasons=tuple(reject_reasons),
        warn_reasons=tuple(warn_reasons),
        pass_rate_pct=sim.pass_rate_pct,
        worst_max_drawdown=sim.worst_max_drawdown,
        pct_daily_limit_hit=sim.pct_daily_limit_hit,
        mean_max_drawdown=sim.mean_max_drawdown,
        sensitivity_flag=sensitivity_flag,
        wf_robust_ok=None,
        holdout_net_pnl=None,
        holdout_max_drawdown=None,
        holdout_mc_pnl_p05=None,
    )


def compute_pipeline_verdict(
    strategy: str,
    *,
    wf_robust_ok: bool,
    wf_all_folds_seq_ok: bool,
    sensitivity_is_cliff: bool | None,
    holdout_net_pnl: float,
    holdout_max_drawdown: float,
    holdout_mc_pnl_p05: float,
) -> VerdictResult:
    """Staged pipeline: WF robustness, per-fold seq pass rate, sensitivity, holdout, MC."""
    reject_reasons: list[str] = []
    warn_reasons: list[str] = []

    wf_ok = wf_robust_ok and wf_all_folds_seq_ok
    if not wf_robust_ok:
        reject_reasons.append("walk_forward_robust_criteria_not_met")
    if not wf_all_folds_seq_ok:
        reject_reasons.append("walk_forward_per_fold_seq_pass_rate_below_minimum")
    if sensitivity_is_cliff is True:
        reject_reasons.append("parameter_sensitivity_cliff")
    if holdout_net_pnl < 0.0:
        reject_reasons.append("holdout_total_net_pnl_negative")
    if holdout_mc_pnl_p05 < 0.0:
        reject_reasons.append("holdout_mc_pnl_p05_negative")

    if holdout_max_drawdown > 10_000_000.0:
        warn_reasons.append("holdout_max_drawdown_extremely_high")

    if reject_reasons:
        verdict: Verdict = "REJECT"
    elif warn_reasons:
        verdict = "PROMISING"
    else:
        verdict = "COMBINE-READY"

    return VerdictResult(
        strategy=strategy,
        verdict=verdict,
        reject_reasons=tuple(reject_reasons),
        warn_reasons=tuple(warn_reasons),
        pass_rate_pct=0.0,
        worst_max_drawdown=0.0,
        pct_daily_limit_hit=0.0,
        mean_max_drawdown=0.0,
        sensitivity_flag=bool(sensitivity_is_cliff) if sensitivity_is_cliff is not None else None,
        wf_robust_ok=wf_ok,
        holdout_net_pnl=holdout_net_pnl,
        holdout_max_drawdown=holdout_max_drawdown,
        holdout_mc_pnl_p05=holdout_mc_pnl_p05,
    )


def verdict_summary_dict(result: VerdictResult) -> dict[str, Any]:
    """Flat dict suitable for verdict logs and audit stamps."""
    return asdict(result)


__all__ = [
    "VerdictResult",
    "compute_pipeline_verdict",
    "compute_verdict",
    "verdict_summary_dict",
]
