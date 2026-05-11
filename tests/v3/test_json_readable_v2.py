"""Test new pipeline_result_bundle_to_readable_text."""

from __future__ import annotations

from v3.json_readable import pipeline_result_bundle_to_readable_text


def test_readable_text_basic():
    """Test basic output structure."""
    data = {
        "strategy": "test_strategy",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
    }

    text = pipeline_result_bundle_to_readable_text(data)
    assert "TOPSTEP PIPELINE" in text
    assert "test_strategy" in text
    assert "5min" in text


def test_readable_text_includes_verdict():
    """Test verdict block."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "verdict": {
            "final_verdict": "READY",
            "reasons": ["High pass rate", "Low drawdown"],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)
    assert "VERDICT:" in text
    assert "READY" in text
    assert "High pass rate" in text


def test_readable_text_uses_current_verdict_contract():
    """verdict.verdict and reject/warn reasons must render directly."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "verdict": {
            "verdict": "REJECT",
            "reject_reasons": ["holdout_total_net_pnl_negative"],
            "warn_reasons": ["holdout_max_drawdown_extremely_high"],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "VERDICT: REJECT" in text
    assert "UNKNOWN" not in text
    assert "holdout_total_net_pnl_negative" in text
    assert "holdout_max_drawdown_extremely_high" in text


def test_readable_text_uses_walk_forward_best_params():
    """CLI writes walk_forward.best_params, not selected_params."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "walk_forward": {
            "best_params": {"width": 2.0, "lookback": 15},
            "oos_folds": [],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "width=2.0" in text
    assert "lookback=15" in text


def test_readable_text_includes_speed_optimization():
    """Test speed optimization section."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "speed_optimization_aggregate": {
            "optimal_risk_dollars": 100.0,
            "median_oos_median_days_to_pass": 3.5,
            "median_oos_pass_rate_pct": 75.0,
            "n_folds": 2,
            "min_oos_utility": 0.70,
            "candidates": [
                {"risk_dollars": 100.0, "median_oos_median_days_to_pass": 3.5, "median_oos_pass_rate_pct": 75.0},
                {"risk_dollars": 75.0, "median_oos_median_days_to_pass": 4.0, "median_oos_pass_rate_pct": 70.0},
            ],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)
    assert "SPEED OPTIMIZER" in text
    assert ">>> EVAL SIZING" in text
    assert "$100/trade" in text


def test_readable_text_includes_longevity_optimization():
    """Test longevity optimization section."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "longevity_optimization": {
            "optimal_risk_dollars": 75.0,
            "median_longevity_score": 0.85,
            "p05_longevity_score": 0.70,
            "median_components": {
                "survival_score": 0.9,
                "drawdown_score": 0.8,
                "efficiency_score": 0.85,
                "capital_score": 0.8,
            },
            "p05_components": {
                "survival_score": 0.7,
                "drawdown_score": 0.6,
                "efficiency_score": 0.65,
                "capital_score": 0.6,
            },
            "per_account_summary": [
                {"survival_days": 25, "terminal_balance": 52000, "breached": False},
                {"survival_days": 15, "terminal_balance": 49000, "breached": True},
            ],
            "candidates": [],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)
    assert "LONGEVITY OPTIMIZER" in text
    assert ">>> FUNDED SIZING" in text
    assert "$75/trade" in text
    assert "Per-account survival" in text
    assert "25" in text  # survival days


def test_readable_text_includes_holdout():
    """Test holdout section with long/short."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "holdout": {
            "metrics": {
                "total_trades": 50,
                "long_trades": 30,
                "short_trades": 20,
                "win_rate": 0.60,
                "long_win_rate": 0.65,
                "short_win_rate": 0.50,
                "total_net_pnl": 5000.0,
                "long_net_pnl": 3500.0,
                "short_net_pnl": 1500.0,
                "profit_factor": 2.5,
            },
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)
    assert "HOLDOUT" in text
    assert "Trades:" in text
    assert "30" in text  # long trades
    assert "20" in text  # short trades
    assert "Win rate:" in text


def test_readable_text_includes_walk_forward():
    """Test walk-forward section."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "walk_forward": {
            "selected_params": {"sma_period": 20, "atr_mult": 1.5},
            "min_fold_seq_pass_rate_pct": 40.0,
            "oos_folds": [
                {
                    "window": "WF1_test",
                    "topstep": {
                        "topstep_passed": True,
                        "topstep_final_balance": 53000,
                        "seq_eval_pass_rate": 0.75,
                        "topstep_days_to_pass": 5,
                    },
                    "metrics": {
                        "long_trades": 15,
                        "short_trades": 10,
                    },
                },
                {
                    "window": "WF2_test",
                    "topstep": {
                        "topstep_passed": False,
                        "topstep_final_balance": 49500,
                        "seq_eval_pass_rate": 0.50,
                        "topstep_days_to_pass": None,
                    },
                    "metrics": {
                        "long_trades": 12,
                        "short_trades": 8,
                    },
                },
            ],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)
    assert "WALK-FORWARD" in text
    assert "WF1" in text
    assert "WF2" in text
    assert "PASS" in text or "FAIL" in text


def test_readable_text_includes_sizing_comparison():
    """Test sizing comparison section."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "sizing_comparison": {
            "track_a_optimizer": {
                "eval_track": {"pass_rate_pct": 75.0},
                "holdout_track": {"longevity_score": 0.85},
            },
            "track_b_fixed_risk": {
                "eval_track": {"pass_rate_pct": 70.0},
                "holdout_track": {"longevity_score": 0.80},
            },
            "sanity_flags": ["Small sample", "High variance"],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)
    assert "SIZING COMPARISON" in text
    assert "Optimizer" in text
    assert "Fixed $/trade" in text
    assert "SANITY FLAGS" in text
    assert "Small sample" in text


def test_readable_text_renders_optimizer_diagnostics_without_zero_recommendation():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "speed_optimization_aggregate": {
            "optimal_risk_dollars": 0.0,
            "median_oos_utility": 0.0,
            "min_oos_utility": 0.0,
            "median_oos_pass_rate_pct": 0.0,
            "median_oos_median_days_to_pass": 0.0,
            "viable_folds": 0,
            "n_folds": 2,
            "candidates": [
                {
                    "risk_dollars": 100.0,
                    "viable_folds": 0,
                    "median_oos_utility": 0.2,
                    "median_oos_pass_rate_pct": 35.0,
                    "median_oos_median_days_to_pass": 9.0,
                }
            ],
        },
        "longevity_optimization": {
            "optimal_risk_dollars": 0.0,
            "median_longevity_score": 0.0,
            "p05_longevity_score": 0.0,
            "median_components": {},
            "p05_components": {},
            "candidates": [
                {
                    "risk_dollars": 150.0,
                    "rejected": True,
                    "reject_reason": "profit_factor 1.00 < 1.2",
                    "median_longevity_score": 0.0,
                }
            ],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "Winner: n/a (no viable risk)" in text
    assert "$0/trade" not in text
    assert "$100" in text


def test_readable_text_handles_json_null_optimizer_metrics():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "speed_optimization_aggregate": {
            "optimal_risk_dollars": 50.0,
            "median_oos_utility": None,
            "min_oos_utility": None,
            "median_oos_pass_rate_pct": None,
            "median_oos_median_days_to_pass": None,
            "viable_folds": 0,
            "n_folds": 2,
            "candidates": [{"risk_dollars": 50.0, "median_oos_median_days_to_pass": None, "median_oos_pass_rate_pct": None}],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "Winner: $50/trade" in text
    assert "n/a" in text
    assert "0/2 folds" in text


def test_readable_text_renders_optimizer_vs_fixed_comparison_values():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "sizing_comparison": {
            "track_a_optimizer": {
                "eval_track": {"risk_dollars": 100.0, "pass_rate_pct": 75.0, "median_days_to_pass": 8.0},
                "holdout_track": {"risk_dollars": 150.0, "longevity_score": 1.25},
            },
            "track_b_fixed_risk": {
                "risk_dollars": 200.0,
                "eval_track": {"pass_rate_pct": 65.0, "median_days_to_pass": 11.0},
                "holdout_track": {"longevity_score": 0.90},
            },
            "track_c_fixed_contracts": {
                "fixed_contracts": 3,
                "eval_track": {"pass_rate_pct": 55.0, "median_days_to_pass": 14.0},
                "holdout_track": {"longevity_score": 0.60},
            },
            "deltas": {"fixed_risk_vs_optimizer": {"eval_pass_rate_delta": -10.0}},
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "$100 eval / $150 funded" in text
    assert "$200" in text
    assert "3 contracts" in text
    assert "fixed_risk_vs_optimizer" in text
