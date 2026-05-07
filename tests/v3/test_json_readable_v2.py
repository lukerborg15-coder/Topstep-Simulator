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


def test_readable_text_shows_sensitivity_gradient_artifacts():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "skip_sensitivity": False,
        "sensitivity": {
            "sensitivity_is_cliff": False,
            "sensitivity_default_pass_rate": 80.0,
            "sensitivity_min_neighbor_pass_rate": 72.0,
            "sensitivity_heatmap_path": "C:/tmp/test_sensitivity_heatmap.png",
            "sensitivity_heatmap_text": "Parameter sensitivity - test\n  width: increasing (range 72.0%-80.0%)",
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "PARAMETER SENSITIVITY" in text
    assert "C:/tmp/test_sensitivity_heatmap.png" in text
    assert "width: increasing" in text


def test_readable_text_explains_skipped_sensitivity():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "skip_sensitivity": True,
        "sensitivity_skip_reason": "--skip-sensitivity flag",
        "sensitivity": None,
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "PARAMETER SENSITIVITY" in text
    assert "Skipped: --skip-sensitivity flag" in text
    assert "--full" in text


def test_readable_text_prefers_sequential_median_days_when_single_run_failed():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "walk_forward": {
            "oos_folds": [
                {
                    "metrics": {"total_trades": 10, "total_net_pnl": 1000.0},
                    "topstep": {
                        "topstep_passed": False,
                        "topstep_days_to_pass": None,
                        "seq_eval_passes": 2,
                        "seq_eval_attempts": 4,
                        "seq_eval_pass_rate": 0.5,
                        "seq_eval_median_days_to_pass": 8.0,
                        "seq_eval_first_pass_days_to_pass": 6,
                    },
                }
            ],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "Seq median days" in text
    assert "8d" in text
    assert "Single days" in text
    assert "n/a" in text


def test_readable_text_includes_full_optimizer_candidate_tables():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "speed_optimization_aggregate": {
            "optimal_risk_dollars": 100.0,
            "median_oos_median_days_to_pass": 5.0,
            "median_oos_pass_rate_pct": 80.0,
            "min_oos_utility": 0.7,
            "viable_folds": 2,
            "n_folds": 2,
            "all_candidates": [
                {"risk_dollars": 50.0, "median_oos_utility": 0.6, "median_oos_pass_rate_pct": 70.0, "median_oos_median_days_to_pass": 7.0, "viable_folds": 2},
                {"risk_dollars": 100.0, "median_oos_utility": 0.8, "median_oos_pass_rate_pct": 80.0, "median_oos_median_days_to_pass": 5.0, "viable_folds": 2},
                {"risk_dollars": 150.0, "median_oos_utility": 0.4, "median_oos_pass_rate_pct": 50.0, "median_oos_median_days_to_pass": 12.0, "viable_folds": 1},
            ],
        },
        "longevity_optimization": {
            "optimal_risk_dollars": 50.0,
            "median_longevity_score": 0.9,
            "p05_longevity_score": 0.7,
            "median_components": {},
            "p05_components": {},
            "all_candidates": [
                {"risk_dollars": 50.0, "rejected": False, "median_longevity_score": 0.9, "p05_longevity_score": 0.7, "median_avg_pnl_per_trade": 200.0},
                {"risk_dollars": 100.0, "rejected": True, "reject_reason": "p05 survival 0.25 < 0.5", "median_longevity_score": 0.3, "p05_longevity_score": 0.1},
            ],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "Full candidate table" in text
    assert "$150" in text
    assert "p05 survival 0.25 < 0.5" in text


def test_readable_text_comparison_includes_eval_funded_and_deltas():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "sizing_comparison": {
            "track_a_optimizer": {
                "eval_track": {"risk_dollars": 100.0, "pass_rate_pct": 75.0, "median_days_to_pass": 6.0},
                "holdout_track": {"risk_dollars": 50.0, "longevity_score": 0.85, "accounts_used": 1, "accounts_blown": 0},
            },
            "track_b_fixed_risk": {
                "risk_dollars": 75.0,
                "eval_track": {"pass_rate_pct": 70.0, "median_days_to_pass": 8.0},
                "holdout_track": {"longevity_score": 0.80, "accounts_used": 2, "accounts_blown": 1},
            },
            "track_c_fixed_contracts": {
                "fixed_contracts": 2,
                "eval_track": {"pass_rate_pct": 65.0, "median_days_to_pass": 9.0},
                "holdout_track": {"longevity_score": 0.70, "accounts_used": 3, "accounts_blown": 2},
            },
            "deltas": {
                "fixed_risk_vs_optimizer": {
                    "eval_pass_rate_delta": -5.0,
                    "eval_days_delta": 2.0,
                    "holdout_longevity_delta": -0.05,
                },
                "fixed_contracts_vs_optimizer": {
                    "eval_pass_rate_delta": -10.0,
                    "eval_days_delta": 3.0,
                    "holdout_longevity_delta": -0.15,
                },
            },
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "Eval days" in text
    assert "Accounts" in text
    assert "Delta pass" in text
    assert "Fixed contracts" in text


def test_readable_text_includes_rich_wf_and_funded_account_details():
    """Readable summary should expose audit-useful WF and funded-account metrics."""
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "timestamp": "2024-01-15",
        "wf_development_window": {"start": "2021-03-19", "end": "2024-08-31"},
        "walk_forward": {
            "best_params": {"sma_period": 20, "atr_mult": 1.5},
            "aggregate": {
                "wf_oos_total_pnl": 4200.0,
                "wf_avg_net_pnl": 2100.0,
                "wf_avg_sharpe": 1.25,
            },
            "oos_folds": [
                {
                    "window": "WF1_test",
                    "metrics": {
                        "total_trades": 25,
                        "total_net_pnl": 2500.0,
                        "max_drawdown": 600.0,
                        "profit_factor": 1.8,
                        "sharpe": 1.4,
                        "avg_r": 0.22,
                        "long_trades": 14,
                        "short_trades": 11,
                    },
                    "topstep": {
                        "topstep_passed": True,
                        "topstep_days_to_pass": 9,
                        "topstep_reason": "profit target reached",
                        "seq_eval_passes": 3,
                        "seq_eval_attempts": 4,
                        "seq_eval_pass_rate": 0.75,
                    },
                },
            ],
        },
        "speed_optimization_aggregate": {
            "optimal_risk_dollars": 100.0,
            "median_oos_median_days_to_pass": 6.0,
            "median_oos_pass_rate_pct": 80.0,
            "median_oos_utility": 0.8,
            "min_oos_utility": 0.7,
            "viable_folds": 2,
            "n_folds": 2,
            "per_fold_oos": [
                {"fold_index": 0, "median_days_to_pass": 6.0, "pass_rate_pct": 80.0, "utility": 0.8, "viable": True}
            ],
        },
        "holdout": {
            "metrics": {
                "total_trades": 40,
                "total_net_pnl": 3500.0,
                "max_drawdown": 750.0,
                "profit_factor": 2.2,
                "sharpe": 1.7,
                "avg_r": 0.18,
                "long_trades": 24,
                "short_trades": 16,
                "win_rate": 0.62,
                "long_win_rate": 0.66,
                "short_win_rate": 0.56,
                "long_net_pnl": 2300.0,
                "short_net_pnl": 1200.0,
            },
            "topstep": {
                "topstep_passed": True,
                "topstep_days_to_pass": 11,
                "topstep_reason": "profit target reached",
            },
        },
        "express_funded_reset_sim": {
            "funded_accounts_used": 2,
            "funded_accounts_failed": 1,
            "current_account_active": True,
            "accrued_pnl_bank": 1400.0,
            "stints_summary": [
                {
                    "stint_index": 0,
                    "breached": True,
                    "survival_days": 7,
                    "trades_applied_count": 12,
                    "terminal_balance": 48900.0,
                    "bank_increment": -1100.0,
                    "win_rate_pct": 41.7,
                    "profit_factor": 0.8,
                    "avg_r_multiple": -0.12,
                    "sharpe_annualized": -0.9,
                    "stint_worst_daily_dd": 900.0,
                    "stint_worst_peak_to_trough": 1600.0,
                },
                {
                    "stint_index": 1,
                    "breached": False,
                    "survival_days": 14,
                    "trades_applied_count": 18,
                    "terminal_balance": 52500.0,
                    "bank_increment": 2500.0,
                    "win_rate_pct": 66.7,
                    "profit_factor": 2.4,
                    "avg_r_multiple": 0.25,
                    "sharpe_annualized": 1.6,
                    "stint_worst_daily_dd": 300.0,
                    "stint_worst_peak_to_trough": 500.0,
                },
            ],
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "BEST WALK-FORWARD PARAMETERS" in text
    assert "sma_period=20" in text
    assert "WF AGGREGATE" in text
    assert "Sharpe" in text
    assert "Single days" in text
    assert "Speed median days" in text
    assert "FUNDED ACCOUNT DETAIL" in text
    assert "Accounts used: 2" in text
    assert "failed: 1" in text
    assert "Acct" in text
    assert "1.60" in text


def test_readable_text_includes_best_market_regime():
    data = {
        "strategy": "test",
        "timeframe": "5min",
        "regime_fit": {
            "regime_verdict": "prefers_volatile",
            "total_trades": 42,
            "window": "holdout",
            "calm": {
                "count": 18,
                "win_rate": 0.44,
                "expectancy_r": -0.05,
                "mean_net_pnl": -25.0,
            },
            "volatile": {
                "count": 24,
                "win_rate": 0.62,
                "expectancy_r": 0.21,
                "mean_net_pnl": 115.0,
            },
        },
    }

    text = pipeline_result_bundle_to_readable_text(data)

    assert "MARKET REGIME FIT" in text
    assert "Best regime: Volatile" in text
    assert "Calm" in text
    assert "Volatile" in text
    assert "0.2100R" in text
