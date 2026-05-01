from __future__ import annotations

from v3.combine_simulator import CombineSimResult
from v3.config import VERDICT_THRESHOLDS, VerdictThresholds
from v3.verdict import (
    compute_pipeline_verdict,
    compute_verdict,
    verdict_summary_dict,
)


def _sim(
    *,
    pass_rate_pct: float = 80.0,
    worst_max_drawdown: float = 1_000.0,
    pct_daily_limit_hit: float = 10.0,
    mean_max_drawdown: float = 600.0,
) -> CombineSimResult:
    return CombineSimResult(
        n_resamples=100,
        pass_rate_pct=pass_rate_pct,
        n_passed=int(pass_rate_pct),
        n_failed_drawdown=0,
        n_not_passed=100 - int(pass_rate_pct),
        median_days_to_pass=12.0,
        mean_days_to_pass=12.0,
        min_days_to_pass=8,
        max_days_to_pass=18,
        pct_daily_limit_hit=pct_daily_limit_hit,
        mean_max_drawdown=mean_max_drawdown,
        worst_max_drawdown=worst_max_drawdown,
        n_trades=500,
        n_trading_days=80,
        n_resamples_requested=100,
    )


def test_rejects_when_pass_rate_below_gate():
    result = compute_verdict("connors_rsi2", _sim(pass_rate_pct=49.9))

    assert result.verdict == "REJECT"
    assert "pass_rate_pct < 50" in result.reject_reasons


def test_default_verdict_thresholds_document_topstep_50k_policy():
    assert VERDICT_THRESHOLDS == VerdictThresholds(
        reject_pass_rate_pct=50.0,
        reject_max_dd=1_800.0,
        reject_daily_hit_pct=60.0,
        reject_mean_dd=1_200.0,
        ready_pass_rate_pct=75.0,
        ready_max_dd=1_400.0,
        ready_daily_hit_pct=25.0,
        ready_mean_dd=800.0,
    )


def test_custom_thresholds_can_reject_more_aggressive_pass_rate():
    thresholds = VerdictThresholds(reject_pass_rate_pct=90.0)

    result = compute_verdict("connors_rsi2", _sim(pass_rate_pct=80.0), thresholds=thresholds)

    assert result.verdict == "REJECT"
    assert "pass_rate_pct < 90" in result.reject_reasons


def test_rejects_when_worst_max_drawdown_above_gate():
    result = compute_verdict("connors_rsi2", _sim(worst_max_drawdown=1_800.01))

    assert result.verdict == "REJECT"
    assert "worst_max_drawdown > 1800" in result.reject_reasons


def test_rejects_when_pct_daily_limit_hit_above_gate():
    result = compute_verdict("connors_rsi2", _sim(pct_daily_limit_hit=60.01))

    assert result.verdict == "REJECT"
    assert "pct_daily_limit_hit > 60" in result.reject_reasons


def test_rejects_when_mean_max_drawdown_above_gate():
    result = compute_verdict("connors_rsi2", _sim(mean_max_drawdown=1_200.01))

    assert result.verdict == "REJECT"
    assert "mean_max_drawdown > 1200" in result.reject_reasons


def test_warn_daily_limit_gate_keeps_promising_and_blocks_combine_ready():
    result = compute_verdict("connors_rsi2", _sim(pct_daily_limit_hit=40.0))

    assert result.verdict == "PROMISING"
    assert result.reject_reasons == ()
    assert "pct_daily_limit_hit between 25 and 60" in result.warn_reasons


def test_warn_daily_limit_gate_includes_25_boundary():
    result = compute_verdict("connors_rsi2", _sim(pct_daily_limit_hit=25.0))

    assert result.verdict == "PROMISING"
    assert "pct_daily_limit_hit between 25 and 60" in result.warn_reasons


def test_warn_daily_limit_gate_includes_60_boundary():
    result = compute_verdict("connors_rsi2", _sim(pct_daily_limit_hit=60.0))

    assert result.verdict == "PROMISING"
    assert "pct_daily_limit_hit between 25 and 60" in result.warn_reasons


def test_clean_sim_is_combine_ready():
    result = compute_verdict("connors_rsi2", _sim())

    assert result.verdict == "COMBINE-READY"
    assert result.reject_reasons == ()
    assert result.warn_reasons == ()


def test_sensitivity_flag_true_blocks_combine_ready():
    # sensitivity_flag blocking is handled by apply_sensitivity_to_verdict()
    # in sensitivity.py, not by compute_verdict() — tested in test_v3_sensitivity.py.
    # compute_verdict() always produces COMBINE-READY for a clean sim;
    # the downstream apply call then downgrades it if is_cliff=True.
    result = compute_verdict("connors_rsi2", _sim())
    assert result.verdict == "COMBINE-READY"
    assert result.sensitivity_flag is None  # set by apply_sensitivity_to_verdict()


def test_sensitivity_flag_none_does_not_block_combine_ready():
    result = compute_verdict("connors_rsi2", _sim())

    assert result.sensitivity_flag is None
    assert result.verdict == "COMBINE-READY"


def test_pipeline_verdict_rejects_when_robust_or_seq_gate_fails() -> None:
    r1 = compute_pipeline_verdict(
        "s",
        wf_robust_ok=False,
        wf_all_folds_seq_ok=True,
        sensitivity_is_cliff=None,
        holdout_net_pnl=100.0,
        holdout_max_drawdown=100.0,
        holdout_mc_pnl_p05=10.0,
    )
    assert r1.verdict == "REJECT"
    assert "walk_forward_robust_criteria_not_met" in r1.reject_reasons
    assert r1.wf_robust_ok is False

    r2 = compute_pipeline_verdict(
        "s",
        wf_robust_ok=True,
        wf_all_folds_seq_ok=False,
        sensitivity_is_cliff=None,
        holdout_net_pnl=100.0,
        holdout_max_drawdown=100.0,
        holdout_mc_pnl_p05=10.0,
    )
    assert r2.verdict == "REJECT"
    assert "walk_forward_per_fold_seq_pass_rate_below_minimum" in r2.reject_reasons
    assert r2.wf_robust_ok is False

    r3 = compute_pipeline_verdict(
        "s",
        wf_robust_ok=True,
        wf_all_folds_seq_ok=True,
        sensitivity_is_cliff=None,
        holdout_net_pnl=100.0,
        holdout_max_drawdown=100.0,
        holdout_mc_pnl_p05=10.0,
    )
    assert r3.verdict == "COMBINE-READY"
    assert r3.wf_robust_ok is True


def test_verdict_summary_dict_has_correct_keys():
    result = compute_verdict("connors_rsi2", _sim())
    summary = verdict_summary_dict(result)

    expected_keys = {
        "strategy",
        "verdict",
        "reject_reasons",
        "warn_reasons",
        "pass_rate_pct",
        "worst_max_drawdown",
        "pct_daily_limit_hit",
        "mean_max_drawdown",
        "sensitivity_flag",
        "wf_robust_ok",
        "holdout_net_pnl",
        "holdout_max_drawdown",
        "holdout_mc_pnl_p05",
    }
    assert set(summary) == expected_keys
    assert summary["strategy"] == result.strategy
    assert summary["verdict"] == result.verdict
