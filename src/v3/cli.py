from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import evaluator as evaluator_module
from .audit_stamp import write_audit_stamp
from .combine_simulator import run_combine_simulator
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_FUNDED_EXPRESS_SIM,
    DEFAULT_MAX_CONTRACTS,
    DEFAULT_MC_BLOCK_SIZE,
    DEFAULT_MC_CI_PCT,
    DEFAULT_MC_N_PERMS,
    DEFAULT_MIN_FOLD_SEQ_PASS_RATE_PCT,
    DEFAULT_RISK_DOLLARS,
    DEFAULT_STRICT_WF_GATE,
    OUTPUT_DIR,
    ScoringWeights,
    VERDICT_THRESHOLDS,
    VerdictThresholds,
)
from .data import load_ohlcv
from .evaluator import (
    EvaluationResult,
    aggregate_wf_metrics,
    all_folds_meet_min_seq_pass_rate,
    evaluate_strategy,
    run_walk_forward,
    walk_forward_development_window,
    wf_oos_folds_for_selected_params,
    wf_train_test_trades_for_selected_params,
)
from .freeze import freeze_params
from .funded_express_sim import express_funded_reset_sim_summary_dict, simulate_express_funded_resets
from .holdout_monte_carlo import (
    holdout_monte_carlo_summary_dict,
    plot_holdout_mc_paths,
    run_holdout_trade_monte_carlo,
)
from .json_readable import write_readable_text_from_json_file
from .monte_carlo import MCResult, mc_summary_dict, mc_summary_text, plot_mc_paths, run_mc
from .pipeline_config import resolve_windows
from .position_sizing import (
    DEFAULT_RISK_GRID,
    LongevityOptimizationMCResult,
    LongevityOptimizationResult,
    SpeedOptimizationAggregateResult,
    SpeedOptimizationResult,
    optimize_for_longevity_holdout,
    optimize_for_speed_wf,
    optimize_longevity_holdout_mc,
    optimize_speed_wf_aggregate,
)
from .sizing_comparison import SizingComparisonResult, run_sizing_comparison
from .regime_classifier import classify_regime_fit, regime_summary_dict, regime_summary_text
from .sensitivity import (
    SensitivityReport,
    plot_sensitivity_heatmap,
    run_sensitivity,
    sensitivity_heatmap_text,
    sensitivity_summary_dict,
)
from .strategies import STRATEGIES, load_user_strategies
from .trades import TradeResult
from .validator import StrategyValidationError, validate_filter_references, validate_strategy
from .verdict import compute_pipeline_verdict, verdict_summary_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topstep evaluation pipeline (validate → WF → sensitivity → holdout → MC → regime → verdict)."
    )
    parser.add_argument("--strategy", default=None, help="Registered strategy key (required unless --list-strategies).")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "holdout-only"],
        default="quick",
        help=(
            "Pipeline mode: quick: stages 1→2 then 4→7 (skip stage 3 parameter sensitivity). "
            "full: all stages 1→7 including sensitivity. holdout-only: 1→2 then 4→7 "
            "(frozen/default params; skips walk-forward and sensitivity)."
        ),
    )
    parser.add_argument("--timeframe", default="5min", help="Bar timeframe suffix (matches data file naming).")
    parser.add_argument("--max-grid", type=int, default=None, help="Cap walk-forward candidate count (omit for full grid).")
    parser.add_argument(
        "--min-wf-passes",
        type=int,
        default=2,
        help="Require params to meet sequential OOS eval threshold on >= N folds (M of F). Default 2.",
    )
    parser.add_argument(
        "--min-eval-passes-per-fold",
        type=int,
        default=2,
        help="Each qualifying fold must have >= this many sequential Combine eval passes (K). Default 2.",
    )
    parser.add_argument(
        "--pipeline-config",
        default=None,
        help="JSON file with walk_forward and holdout windows (see config/ in project root).",
    )
    parser.add_argument("--holdout-mc-iterations", type=int, default=DEFAULT_MC_N_PERMS, help="Holdout trade-order block-bootstrap permutations (MC2). Default 1000.")
    parser.add_argument("--holdout-mc-seed", type=int, default=42, help="RNG seed for holdout Monte Carlo.")
    parser.add_argument("--mc-block-size", type=int, default=DEFAULT_MC_BLOCK_SIZE, help=f"Block size for block-bootstrap MC (default {DEFAULT_MC_BLOCK_SIZE}).")
    parser.add_argument("--mc-ci-pct", type=float, default=DEFAULT_MC_CI_PCT, help=f"Confidence interval percentage for MC (default {DEFAULT_MC_CI_PCT}).")
    parser.add_argument("--topstep-weight", type=float, default=1.0, help="Weight for Topstep score in walk-forward selection (default 1.0)")
    parser.add_argument("--avg-r-weight", type=float, default=25.0, help="Weight for avg_r in walk-forward selection (default 25.0)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing MNQ_* CSV bundles.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Run output root (JSON→json/, text summaries→txt summaries/, frozen→frozen_params/).")
    parser.add_argument("--frozen-params-dir", default=None, help="Overrides output-dir/frozen_params as the freeze and audit stamp directory.")
    parser.add_argument("--list-strategies", action="store_true", help="List registered strategies and exit.")
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward optimization; evaluate holdout using default_params.")
    parser.add_argument("--force", action="store_true", help="Continue pipeline even when walk-forward gates fail (verdict may still REJECT).")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=DEFAULT_STRICT_WF_GATE,
        help="Legacy no-op for WF exit behavior; walk-forward failure exits unless --force.",
    )
    parser.add_argument(
        "--eval-risk",
        type=float,
        default=DEFAULT_RISK_DOLLARS,
        dest="eval_risk_dollars",
        help=f"Target dollars at stop per trade before contract cap (default {DEFAULT_RISK_DOLLARS}).",
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=DEFAULT_MAX_CONTRACTS,
        help=f"Max contracts per signal after risk sizing (default {DEFAULT_MAX_CONTRACTS}).",
    )
    parser.add_argument(
        "--min-fold-seq-pass-rate-pct",
        type=float,
        default=DEFAULT_MIN_FOLD_SEQ_PASS_RATE_PCT,
        help=(
            "Each OOS fold (selected params) must have seq_eval_passes/seq_eval_attempts "
            f">= this %% (default {DEFAULT_MIN_FOLD_SEQ_PASS_RATE_PCT}); attempts==0 fails the fold."
        ),
    )
    sensitivity_group = parser.add_mutually_exclusive_group()
    sensitivity_group.add_argument("--full", dest="skip_sensitivity", action="store_false", help="Run full pipeline, including the optional sensitivity sweep.")
    sensitivity_group.add_argument("--skip-sensitivity", dest="skip_sensitivity", action="store_true", help="Skip parameter sensitivity (stage 3).")
    parser.set_defaults(skip_sensitivity=None)
    parser.add_argument("--sensitivity-resamples", type=int, default=200, help="Bootstrap iterations per sensitivity sweep point (default 200).")
    parser.add_argument("--sensitivity-mc-iterations", type=int, default=DEFAULT_MC_N_PERMS, help="Block-bootstrap permutations for sensitivity MC1 (default 1000).")
    parser.add_argument("--sensitivity-mc-seed", type=int, default=99, help="RNG seed for sensitivity MC1 (default 99).")
    parser.add_argument("--reject-pass-rate", type=float, default=VERDICT_THRESHOLDS.reject_pass_rate_pct, help="Reject if Combine pass rate is below this percentage.")
    parser.add_argument("--reject-max-dd", type=float, default=VERDICT_THRESHOLDS.reject_max_dd, help="Reject if worst-case drawdown exceeds this dollar amount.")
    parser.add_argument("--reject-daily-hit-pct", type=float, default=VERDICT_THRESHOLDS.reject_daily_hit_pct, help="Reject if daily loss limit hit rate exceeds this percentage.")
    parser.add_argument("--reject-mean-dd", type=float, default=VERDICT_THRESHOLDS.reject_mean_dd, help="Reject if average max drawdown exceeds this dollar amount.")
    parser.add_argument("--ready-pass-rate", type=float, default=VERDICT_THRESHOLDS.ready_pass_rate_pct, help="Require at least this pass rate percentage for COMBINE-READY.")
    parser.add_argument("--ready-max-dd", type=float, default=VERDICT_THRESHOLDS.ready_max_dd, help="Require worst-case drawdown at or below this dollar amount for COMBINE-READY.")
    parser.add_argument("--ready-daily-hit-pct", type=float, default=VERDICT_THRESHOLDS.ready_daily_hit_pct, help="Require daily loss limit hit rate at or below this percentage for COMBINE-READY.")
    parser.add_argument("--ready-mean-dd", type=float, default=VERDICT_THRESHOLDS.ready_mean_dd, help="Require average max drawdown at or below this dollar amount for COMBINE-READY.")
    parser.add_argument("--optimize-sizing-for-speed", action="store_true", help="Run walk-forward fold sizing optimization for fastest Combine pass.")
    parser.add_argument("--pass-floor-pct", type=float, default=40.0, help="Minimum pass rate percentage for walk-forward speed sizing candidates (default 40).")
    parser.add_argument("--pass-target-pct", type=float, default=75.0, help="Target pass rate percentage for walk-forward speed sizing reporting (default 75).")
    parser.add_argument("--optimize-sizing-for-longevity", action="store_true", help="Run holdout sizing optimization for funded-account longevity.")
    parser.add_argument("--min-profit-per-trade", type=float, default=150.0, help="Minimum average P&L per trade for holdout longevity sizing candidates (default 150).")
    parser.add_argument("--speed-attempt-budget", type=int, default=10, help="Max sequential eval attempts per fold/risk for speed optimization (default 10).")
    parser.add_argument("--speed-target-days", type=float, default=10.0, help="Target days to pass for speed optimization utility decay (default 10).")
    parser.add_argument("--longevity-mc-iterations", type=int, default=500, help="Monte Carlo iterations for longevity optimization (default 500).")
    parser.add_argument("--longevity-mc-block-size", type=int, default=5, help="Block size for block-bootstrap in longevity MC (default 5).")
    parser.add_argument("--longevity-bootstrap-iterations", type=int, default=1000, help="Bootstrap iterations for P05 pnl estimation (default 1000).")
    parser.add_argument("--longevity-confidence-level", type=float, default=0.05, help="Percentile for hard filters in longevity MC (default 0.05 = 5th percentile).")
    parser.add_argument("--longevity-weight-survival", type=float, default=0.4, help="Weight for survival score in longevity optimization (default 0.4).")
    parser.add_argument("--longevity-weight-drawdown", type=float, default=0.2, help="Weight for drawdown score in longevity optimization (default 0.2).")
    parser.add_argument("--longevity-weight-efficiency", type=float, default=0.2, help="Weight for efficiency score in longevity optimization (default 0.2).")
    parser.add_argument("--longevity-weight-capital", type=float, default=0.2, help="Weight for capital score in longevity optimization (default 0.2).")
    parser.add_argument("--longevity-min-profit-factor", type=float, default=1.2, help="Min profit factor for longevity candidates (default 1.2).")
    parser.add_argument("--risk-coverage-threshold", type=float, default=0.5, help="Min fraction of trades needing >=1 contract for risk levels (default 0.5).")
    parser.add_argument("--compare-fixed-risk", type=float, default=None, help="Fixed risk dollars for sizing comparison (optional).")
    parser.add_argument("--compare-fixed-contracts", type=int, default=None, help="Fixed contracts for sizing comparison (optional).")
    return parser


def _apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.skip_sensitivity is None:
        args.skip_sensitivity = args.mode != "full"
    if args.mode == "holdout-only":
        args.skip_wf = True


def _trade_json(trade: TradeResult) -> dict[str, Any]:
    return {
        "strategy": trade.strategy,
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "direction": trade.direction,
        "entry": trade.entry,
        "stop": trade.stop,
        "target": trade.target,
        "exit": trade.exit,
        "contracts": trade.contracts,
        "gross_pnl": trade.gross_pnl,
        "commission": trade.commission,
        "net_pnl": trade.net_pnl,
        "r_multiple": trade.r_multiple,
        "exit_reason": trade.exit_reason,
        "bars_held": trade.bars_held,
        "regime": trade.regime,
        "params": trade.params,
    }


def _eval_json(result: EvaluationResult) -> dict[str, Any]:
    return {
        "strategy": result.strategy,
        "timeframe": result.timeframe,
        "window": result.window,
        "params": result.params,
        "metrics": result.metrics,
        "topstep": result.topstep,
        "score": result.score,
        "trades": [_trade_json(t) for t in result.trades],
        "combine_sim": asdict(result.combine_sim) if result.combine_sim is not None else None,
    }


def _print_stage_header(stage: str) -> None:
    print()
    print(f"=== {stage} ===")


def _print_summary_table(rows: list[tuple[str, str]]) -> None:
    if not rows:
        return
    key_w = max(len(k) for k, _ in rows)
    print()
    print("Summary")
    print("-" * max(48, key_w + 3 + max(len(v) for _, v in rows)))
    for key, val in rows:
        print(f"{key.ljust(key_w)} | {val}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_optimization_json(path: Path, result: SpeedOptimizationResult | LongevityOptimizationResult) -> None:
    payload = _json_safe(asdict(result))
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _contracts_range(min_contracts: int, max_contracts: int) -> str:
    return f"{min_contracts}-{max_contracts}"


def _print_speed_optimization(result: SpeedOptimizationResult) -> None:
    print()
    print(f"=== {result.window} Speed Optimization ===")
    if result.optimal_risk_dollars <= 0:
        print("Optimal Risk: n/a")
        print(f"Pass Rate: {result.pass_rate_pct:.1f}%")
        print("Mean Days to Pass: n/a")
        print("Contracts: n/a")
    else:
        print(f"Optimal Risk: {_money(result.optimal_risk_dollars)}/trade")
        print(f"Pass Rate: {result.pass_rate_pct:.1f}%")
        print(f"Mean Days to Pass: {result.mean_days_to_pass:.1f} ± {result.std_days_to_pass:.1f}")
        print(f"Contracts: {_contracts_range(result.min_contracts_used, result.max_contracts_used)}")
    print()
    print("Top 5:")
    if not result.candidates:
        print("(no candidates)")
        return
    for idx, candidate in enumerate(result.candidates[:5], 1):
        mean_days = candidate.get("mean_days_to_pass")
        mean_s = (
            "n/a"
            if mean_days is None or not math.isfinite(float(mean_days))
            else f"{float(mean_days):.1f} days"
        )
        print(
            f"{idx}. {_money(float(candidate['risk_dollars']))}  "
            f"{float(candidate['pass_rate_pct']):.1f}%   {mean_s}   "
            f"{_contracts_range(int(candidate.get('min_contracts', 0)), int(candidate.get('max_contracts', 0)))} contracts"
        )


def _print_speed_optimization_aggregate(result: SpeedOptimizationAggregateResult) -> None:
    print()
    print(f"=== Walk-Forward Speed Optimization (Aggregate) ===")
    print(f"Optimal Risk: {_money(result.optimal_risk_dollars)}/trade")
    print(f"Median OOS Utility: {result.median_oos_utility:.4f}")
    print(f"Min OOS Utility: {result.min_oos_utility:.4f}")
    print(f"Median OOS Pass Rate: {result.median_oos_pass_rate_pct:.1f}%")
    print(f"Median OOS Median Days: {result.median_oos_median_days_to_pass:.1f} days")
    print(f"Viable in {result.viable_folds}/{result.n_folds} folds")
    if result.adaptive_floor_applied:
        print(
            f"Pass floor (effective/user): {result.effective_pass_floor_pct:.1f}% / "
            f"{result.pass_floor_pct:.1f}% (adaptive applied)"
        )
    else:
        print(f"Pass floor: {result.effective_pass_floor_pct:.1f}%")
    print()
    print("Top 5 alternatives:")
    if not result.candidates:
        print("(no candidates)")
        return
    for idx, candidate in enumerate(result.candidates[:5], 1):
        print(
            f"{idx}. {_money(float(candidate['risk_dollars']))}  "
            f"utility={float(candidate['median_oos_utility']):.4f}  "
            f"{float(candidate['median_oos_pass_rate_pct']):.1f}%  "
            f"{float(candidate['median_oos_median_days_to_pass']):.1f}d"
        )


def _print_longevity_optimization(result: LongevityOptimizationResult) -> None:
    print()
    print("=== Holdout Longevity Optimization ===")
    if result.optimal_risk_dollars <= 0:
        print("Optimal Risk: n/a")
    else:
        print(f"Optimal Risk: {_money(result.optimal_risk_dollars)}/trade")
    print(f"Avg P&L per Trade: {_money(result.avg_pnl_per_trade)}")
    print(f"Total P&L: {_money(result.total_pnl)}")
    print(f"Accounts Used: {result.funded_accounts_used}")
    print(f"Longevity Score: {result.longevity_score:.2f}")
    print()
    print("Top 5:")
    if not result.candidates:
        print("(no candidates)")
        return
    for idx, candidate in enumerate(result.candidates[:5], 1):
        accounts = int(candidate.get("funded_accounts_used", 0))
        account_word = "account" if accounts == 1 else "accounts"
        print(
            f"{idx}. {_money(float(candidate['risk_dollars']))}   "
            f"{_money(float(candidate['avg_pnl_per_trade']))}/trade   "
            f"{accounts} {account_word}   score={float(candidate['longevity_score']):.2f}"
        )


def _print_longevity_optimization_mc(result: LongevityOptimizationMCResult) -> None:
    print()
    print("=== Holdout Longevity Optimization (Monte Carlo) ===")
    print(f"Optimal Risk: {_money(result.optimal_risk_dollars)}/trade")
    print(f"Median Longevity Score: {result.median_longevity_score:.4f}  (p05: {result.p05_longevity_score:.4f})")
    print(f"Median Avg P&L/Trade: {_money(result.median_avg_pnl_per_trade)}  (p05: {_money(result.p05_avg_pnl_per_trade)})")
    print(f"Median Accounts Used: {result.median_accounts_used:.1f}  Blown: {result.median_accounts_blown:.1f}")
    print()
    print("Component scores (median / p05):")
    for comp in ["survival_score", "drawdown_score", "efficiency_score", "capital_score"]:
        med = result.median_components.get(comp, 0.0)
        p05 = result.p05_components.get(comp, 0.0)
        print(f"  {comp:.<30} {med:.4f} / {p05:.4f}")
    print()
    print("Top 5 alternatives:")
    if not result.candidates:
        print("(no candidates)")
        return
    for idx, candidate in enumerate(result.candidates[:5], 1):
        if candidate.get("rejected"):
            print(
                f"{idx}. {_money(float(candidate['risk_dollars']))}  "
                f"REJECTED — {candidate.get('reject_reason', 'unknown')}"
            )
        else:
            print(
                f"{idx}. {_money(float(candidate['risk_dollars']))}  "
                f"med_long={float(candidate['median_longevity_score']):.4f}  "
                f"p05_long={float(candidate['p05_longevity_score']):.4f}"
            )


def _resolve_frozen_dir(output_dir: Path, frozen_explicit: Path | None) -> Path:
    default = Path(output_dir) / "frozen_params"
    return default if frozen_explicit is None else Path(frozen_explicit)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_mode_defaults(args)
    verdict_thresholds = VerdictThresholds(
        reject_pass_rate_pct=args.reject_pass_rate,
        reject_max_dd=args.reject_max_dd,
        reject_daily_hit_pct=args.reject_daily_hit_pct,
        reject_mean_dd=args.reject_mean_dd,
        ready_pass_rate_pct=args.ready_pass_rate,
        ready_max_dd=args.ready_max_dd,
        ready_daily_hit_pct=args.ready_daily_hit_pct,
        ready_mean_dd=args.ready_mean_dd,
    )
    if args.min_wf_passes < 1:
        parser.error("--min-wf-passes must be >= 1")
    if args.min_eval_passes_per_fold < 1:
        parser.error("--min-eval-passes-per-fold must be >= 1")
    if args.holdout_mc_iterations < 1:
        parser.error("--holdout-mc-iterations must be >= 1")
    if args.eval_risk_dollars <= 0:
        parser.error("--eval-risk must be positive")
    if args.max_contracts < 1:
        parser.error("--max-contracts must be >= 1")
    if not (0.0 < args.min_fold_seq_pass_rate_pct <= 100.0):
        parser.error("--min-fold-seq-pass-rate-pct must be in (0, 100]")
    if not (0.0 <= args.pass_floor_pct <= 100.0):
        parser.error("--pass-floor-pct must be in [0, 100]")
    if not (0.0 <= args.pass_target_pct <= 100.0):
        parser.error("--pass-target-pct must be in [0, 100]")
    if args.min_profit_per_trade < 0:
        parser.error("--min-profit-per-trade must be >= 0")
    if args.topstep_weight != 1.0 or args.avg_r_weight != 25.0:
        evaluator_module.SCORING_WEIGHTS = ScoringWeights(
            topstep_weight=args.topstep_weight,
            avg_r_weight=args.avg_r_weight,
        )
    load_user_strategies()

    try:
        validate_filter_references(STRATEGIES)
    except StrategyValidationError as exc:
        print(f"REJECT validation: filter references — {exc}", file=sys.stderr)
        return 1

    if args.list_strategies:
        _print_strategies()
        return 0

    if not args.strategy:
        parser.error("the following arguments are required: --strategy (unless --list-strategies is set)")

    strategy_key = args.strategy.strip()
    if strategy_key not in STRATEGIES:
        parser.error(
            f"Unknown strategy {strategy_key!r}. Available: {', '.join(sorted(STRATEGIES))}",
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    frozen_explicit = Path(args.frozen_params_dir).resolve() if args.frozen_params_dir else None
    frozen_root = _resolve_frozen_dir(output_dir, frozen_explicit)

    spec = STRATEGIES[strategy_key]

    _print_stage_header("Stage 1 — Validate")
    try:
        validate_strategy(spec)
    except StrategyValidationError as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 1
    print("OK — strategy specification passed Level B validation.")

    data_dir_path = Path(args.data_dir)
    frame = load_ohlcv(instrument="mnq", timeframe=args.timeframe, data_dir=data_dir_path, session_only=True)
    pipeline_windows = resolve_windows(args.pipeline_config)
    eval_risk = float(args.eval_risk_dollars)
    max_contracts = int(args.max_contracts)

    # Derive WF development window for sensitivity and logging.
    wf_dev_window = walk_forward_development_window(pipeline_windows)

    _print_stage_header("Stage 2 — Walk-forward")
    wf_aggregate: dict[str, Any]
    wf_folds_serial: list[dict[str, Any]]
    selected_params: dict[str, Any]
    wf_oos_final: list[EvaluationResult]
    wf_robust_ok: bool
    wf_all_folds_seq_ok: bool
    fold_rates: list[float] = []
    speed_optimization_paths: list[str] = []
    longevity_optimization_path: str | None = None
    speed_optimization_aggregate_result: SpeedOptimizationAggregateResult | None = None
    longevity_optimization_mc_result: LongevityOptimizationMCResult | None = None
    sizing_comparison_result: SizingComparisonResult | None = None
    fold_trade_pairs: list[tuple[list[TradeResult], list[TradeResult]]] = []

    if args.skip_wf:
        selected_params = dict(spec.default_params)
        wf_oos_final = []
        wf_aggregate = aggregate_wf_metrics([])
        wf_folds_serial = []
        wf_robust_ok = True
        wf_all_folds_seq_ok = True
        print("Walk-forward skipped (--skip-wf). Using strategy default_params for holdout.")
    else:
        selected_params, _wf_oos_per_fold_selection, wf_robust_ok = run_walk_forward(
            frame,
            strategy_key,
            args.timeframe,
            args.max_grid,
            min_eval_passes_per_fold=args.min_eval_passes_per_fold,
            min_folds_meeting_passes=args.min_wf_passes,
            windows=pipeline_windows,
            risk_dollars=eval_risk,
            max_contracts=max_contracts,
        )
        wf_oos_final = wf_oos_folds_for_selected_params(
            frame,
            strategy_key,
            args.timeframe,
            selected_params,
            pipeline_windows,
            risk_dollars=eval_risk,
            max_contracts=max_contracts,
        )
        wf_all_folds_seq_ok, fold_rates = all_folds_meet_min_seq_pass_rate(
            wf_oos_final, args.min_fold_seq_pass_rate_pct
        )
        wf_ok_pre = wf_robust_ok and wf_all_folds_seq_ok
        print(
            f"Walk-forward folds: {len(wf_oos_final)}  wf_robust_ok={wf_robust_ok}  "
            f"wf_all_folds_seq_ok={wf_all_folds_seq_ok}  wf_ok={wf_ok_pre}"
        )
        # Extract train-test trade pairs for aggregate optimizers
        if args.optimize_sizing_for_speed or (args.compare_fixed_risk is not None or args.compare_fixed_contracts is not None):
            fold_trade_pairs = wf_train_test_trades_for_selected_params(
                frame,
                strategy_key,
                args.timeframe,
                selected_params,
                pipeline_windows,
                risk_dollars=eval_risk,
                max_contracts=max_contracts,
            )

        for fold_idx, fold in enumerate(wf_oos_final, 1):
            tn = fold.metrics["total_trades"]
            pn = fold.metrics["total_net_pnl"]
            tsp = fold.topstep.get("topstep_passed")
            seq = fold.topstep.get("seq_eval_passes", 0)
            att = fold.topstep.get("seq_eval_attempts", 0)
            rate = fold.topstep.get("seq_eval_pass_rate")
            rate_s = f"{float(rate):.3f}" if rate is not None else "n/a"
            print(
                f"  {fold.window}: OOS trades={tn} net_pnl={pn:.2f} topstep_pass={tsp} "
                f"seq_eval_passes={seq} seq_attempts={att} seq_rate={rate_s} score={fold.score:.4f}"
            )

        # Run aggregate speed optimization if enabled
        if args.optimize_sizing_for_speed and fold_trade_pairs:
            speed_optimization_aggregate_result = optimize_speed_wf_aggregate(
                fold_trade_pairs,
                strategy=strategy_key,
                risk_levels=list(DEFAULT_RISK_GRID),
                pass_floor_pct=args.pass_floor_pct,
                speed_target_days=args.speed_target_days,
                attempt_budget=args.speed_attempt_budget,
                coverage_threshold=args.risk_coverage_threshold,
            )
            speed_path = json_dir / f"{strategy_key}_wf_speed_optimization_aggregate.json"
            _write_optimization_json(speed_path, speed_optimization_aggregate_result)
            speed_optimization_paths.append(str(speed_path.resolve()))
            _print_speed_optimization_aggregate(speed_optimization_aggregate_result)

        wf_aggregate = aggregate_wf_metrics(wf_oos_final)
        wf_folds_serial = [_eval_json(f) for f in wf_oos_final]
        print("WF aggregate:", ", ".join(f"{k}={v}" for k, v in wf_aggregate.items()))
        print("best_params(selected):", json.dumps(selected_params, sort_keys=True, default=str))
        print(f"WF development window: {wf_dev_window.start} → {wf_dev_window.end}")

    wf_ok = (wf_robust_ok and wf_all_folds_seq_ok) if not args.skip_wf else True
    if not args.skip_wf and not wf_ok:
        p_th = args.min_fold_seq_pass_rate_pct / 100.0
        lines: list[str] = []
        if not wf_robust_ok:
            lines.append("robust_param_selection_failed (M-of-F / min_eval_passes_per_fold).")
        if not wf_all_folds_seq_ok:
            lines.append(
                f"per_fold_seq_pass_rate below {args.min_fold_seq_pass_rate_pct}% "
                "(selected params re-evaluated on each OOS window)."
            )
            for fold, rate in zip(wf_oos_final, fold_rates, strict=True):
                if rate < p_th:
                    ts = fold.topstep
                    lines.append(
                        f"  {fold.window}: seq_eval_passes={int(ts.get('seq_eval_passes', 0))} "
                        f"seq_eval_attempts={int(ts.get('seq_eval_attempts', 0))} "
                        f"rate={rate:.4f} (need >= {p_th:.4f})"
                    )
        detail = " ".join(lines)
        if not args.force:
            print("REJECT: walk-forward gates not met.", file=sys.stderr)
            print(f"  wf_robust_ok={wf_robust_ok} wf_all_folds_seq_ok={wf_all_folds_seq_ok}", file=sys.stderr)
            print(f"  {detail}", file=sys.stderr)
            return 1
        print(f"WARN: walk-forward gates not met (--force). {detail}")

    sensitivity_report: SensitivityReport | None = None
    sensitivity_serial: dict[str, Any] | None = None
    sensitivity_mc: MCResult | None = None
    sensitivity_heatmap_path: str | None = None

    if args.skip_sensitivity or not spec.param_grid:
        skip_reason = "--skip-sensitivity flag" if args.skip_sensitivity else "no param_grid defined"
        _print_stage_header("Stage 3 — Parameter sensitivity (skipped)")
        print(f"Skipped: {skip_reason}")
    else:
        _print_stage_header("Stage 3 — Parameter sensitivity")

        def _trades_fn(params: dict) -> list[TradeResult]:
            result = evaluate_strategy(
                frame,
                strategy_key,
                args.timeframe,
                params,
                wf_dev_window,
                risk_dollars=eval_risk,
                max_contracts=max_contracts,
            )
            return result.trades

        sensitivity_report = run_sensitivity(
            strategy_name=strategy_key,
            default_params=selected_params,
            param_grid=spec.param_grid,
            sim_fn=run_combine_simulator,
            trades_fn=_trades_fn,
            n_resamples=args.sensitivity_resamples,
            seed=42,
        )
        sensitivity_serial = sensitivity_summary_dict(sensitivity_report)
        cliff_label = "CLIFF DETECTED" if sensitivity_report.is_cliff else "flat"
        print(
            f"sensitivity={cliff_label}  default_pass_rate={sensitivity_report.default_pass_rate:.1f}%  "
            f"min_neighbor={sensitivity_report.min_neighbor_pass_rate:.1f}%",
        )
        if sensitivity_report.is_cliff:
            print(f"  cliff params: {', '.join(sensitivity_report.cliff_params)}")

        # MC1: block bootstrap on sensitivity (WF dev window) trades
        _sens_trades = _trades_fn(selected_params)
        if _sens_trades:
            sensitivity_mc = run_mc(
                _sens_trades,
                n_perms=args.sensitivity_mc_iterations,
                block_size=args.mc_block_size,
                seed=args.sensitivity_mc_seed,
                ci_pct=args.mc_ci_pct,
            )
            _sens_mc_path = graphs_dir / f"{strategy_key}_{args.timeframe}_sensitivity_mc_paths.png"
            try:
                plot_mc_paths(sensitivity_mc, _sens_mc_path, title=f"Sensitivity MC — {strategy_key} ({wf_dev_window.start}→{wf_dev_window.end})")
                sensitivity_heatmap_path = str(_sens_mc_path.resolve())
                print(f"sensitivity_mc_paths={sensitivity_heatmap_path}")
            except Exception as exc:
                print(f"WARN: sensitivity MC graph not written ({exc})", file=sys.stderr)
            print(mc_summary_text(sensitivity_mc, title="Sensitivity MC1"))

        # Sensitivity heatmap
        _heatmap_path = graphs_dir / f"{strategy_key}_{args.timeframe}_sensitivity_heatmap.png"
        try:
            plot_sensitivity_heatmap(sensitivity_report, _heatmap_path)
            sensitivity_serial["sensitivity_heatmap_path"] = str(_heatmap_path.resolve())
            sensitivity_serial["sensitivity_heatmap_text"] = sensitivity_heatmap_text(sensitivity_report)
            print(f"sensitivity_heatmap={_heatmap_path.resolve()}")
        except Exception as exc:
            print(f"WARN: sensitivity heatmap not written ({exc})", file=sys.stderr)

    _print_stage_header("Stage 4 — Holdout")
    holdout_eval = evaluate_strategy(
        frame,
        strategy_key,
        args.timeframe,
        selected_params,
        pipeline_windows.holdout,
        risk_dollars=eval_risk,
        max_contracts=max_contracts,
    )
    print(
        f"holdout net_pnl={float(holdout_eval.metrics['total_net_pnl']):.2f} "
        f"max_dd={float(holdout_eval.metrics['max_drawdown']):.2f}",
    )
    if args.optimize_sizing_for_longevity:
        # Use new MC-based longevity optimizer
        longevity_optimization_mc_result = optimize_longevity_holdout_mc(
            holdout_eval.trades,
            strategy=strategy_key,
            window="holdout",
            min_profit_per_trade=args.min_profit_per_trade,
            min_profit_factor=args.longevity_min_profit_factor,
            weights={
                "survival_score": args.longevity_weight_survival,
                "drawdown_score": args.longevity_weight_drawdown,
                "efficiency_score": args.longevity_weight_efficiency,
                "capital_score": args.longevity_weight_capital,
            },
            mc_iterations=args.longevity_mc_iterations,
            mc_block_size=args.longevity_mc_block_size,
            bootstrap_iterations=args.longevity_bootstrap_iterations,
            confidence_level=args.longevity_confidence_level,
            coverage_threshold=args.risk_coverage_threshold,
        )
        longevity_path = json_dir / f"{strategy_key}_holdout_longevity_optimization_mc.json"
        _write_optimization_json(longevity_path, longevity_optimization_mc_result)
        longevity_optimization_path = str(longevity_path.resolve())
        _print_longevity_optimization_mc(longevity_optimization_mc_result)

        # Run sizing comparison if both optimizers are available and comparison flags are set
        if (
            speed_optimization_aggregate_result is not None
            and longevity_optimization_mc_result is not None
            and (args.compare_fixed_risk is not None or args.compare_fixed_contracts is not None)
            and fold_trade_pairs
        ):
            sizing_comparison_result = run_sizing_comparison(
                fold_trade_pairs,
                holdout_eval.trades,
                speed_optimization_aggregate_result,
                longevity_optimization_mc_result,
                fixed_risk_dollars=args.compare_fixed_risk,
                fixed_contracts=args.compare_fixed_contracts,
            )
            comparison_path = json_dir / f"{strategy_key}_sizing_comparison.json"
            payload = _json_safe(asdict(sizing_comparison_result))
            comparison_path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
            print()
            print("=== Sizing Comparison ===")
            print(f"Optimizer eval pass rate: {sizing_comparison_result.track_a_optimizer['eval_track']['pass_rate_pct']:.1f}%")
            if sizing_comparison_result.track_b_fixed_risk:
                print(f"Fixed risk eval pass rate: {sizing_comparison_result.track_b_fixed_risk['eval_track']['pass_rate_pct']:.1f}%")
            if sizing_comparison_result.track_c_fixed_contracts:
                print(f"Fixed contracts eval pass rate: {sizing_comparison_result.track_c_fixed_contracts['eval_track']['pass_rate_pct']:.1f}%")
            if sizing_comparison_result.sanity_flags:
                print("Sanity flags:")
                for flag in sizing_comparison_result.sanity_flags:
                    print(f"  - {flag}")

    ho_express_sim = simulate_express_funded_resets(
        holdout_eval.trades,
        rules=DEFAULT_FUNDED_EXPRESS_SIM,
    )
    express_sim_summary = express_funded_reset_sim_summary_dict(ho_express_sim)
    express_sim_summary["rules"] = {
        k: getattr(DEFAULT_FUNDED_EXPRESS_SIM, k)
        for k in (
            "account_size",
            "max_drawdown",
            "daily_loss_limit",
            "lock_trigger_balance",
            "locked_floor_balance",
        )
    }
    print(
        f"express_funded_reset breaches={ho_express_sim.funded_accounts_failed} "
        f"accounts_used={ho_express_sim.funded_accounts_used} "
        f"stints_simulated={ho_express_sim.stints_opened} bank_accrued=${ho_express_sim.accrued_pnl_bank:.2f} "
        f"peak_nominal=${ho_express_sim.max_nominal_peak_balance:.2f}",
    )

    _print_stage_header("Stage 5 — Monte Carlo (trade-order, holdout)")
    holdout_mc = run_holdout_trade_monte_carlo(
        holdout_eval.trades,
        n=args.holdout_mc_iterations,
        seed=args.holdout_mc_seed,
        block_size=args.mc_block_size,
        ci_pct=args.mc_ci_pct,
    )
    ho_mc_d = holdout_monte_carlo_summary_dict(holdout_mc)
    print(mc_summary_text(holdout_mc, title="Holdout MC2"))

    holdout_mc_graph_path: str | None = None
    _ho_mc_path = graphs_dir / f"{strategy_key}_{args.timeframe}_holdout_mc_paths.png"
    try:
        plot_holdout_mc_paths(holdout_mc, _ho_mc_path)
        holdout_mc_graph_path = str(_ho_mc_path.resolve())
        print(f"holdout_mc_paths={holdout_mc_graph_path}")
    except Exception as exc:
        print(f"WARN: holdout MC graph not written ({exc})", file=sys.stderr)

    _print_stage_header("Stage 6 — Regime classifier")
    regime_result = classify_regime_fit(
        frame,
        holdout_eval.trades,
        pipeline_windows.holdout,
    )
    regime_d = regime_summary_dict(regime_result)
    print(regime_summary_text(regime_result))

    cliff_flag = sensitivity_report.is_cliff if sensitivity_report is not None else None
    _print_stage_header("Stage 7 — Verdict")
    verdict = compute_pipeline_verdict(
        strategy_key,
        wf_robust_ok=wf_robust_ok,
        wf_all_folds_seq_ok=wf_all_folds_seq_ok,
        sensitivity_is_cliff=cliff_flag,
        holdout_net_pnl=float(holdout_eval.metrics["total_net_pnl"]),
        holdout_max_drawdown=float(holdout_eval.metrics["max_drawdown"]),
        holdout_mc_pnl_p05=float(holdout_mc.pnl_p05),
    )
    reason_parts: list[str] = []
    if verdict.reject_reasons:
        reason_parts.extend([f"reject: {r}" for r in verdict.reject_reasons])
    if verdict.warn_reasons:
        reason_parts.extend([f"warn: {r}" for r in verdict.warn_reasons])
    if not reason_parts:
        reason_parts.append("(no disqualifiers)")
    print(f"VERDICT: {verdict.verdict}")
    print("Reasons:", "; ".join(reason_parts))

    params_hash: str | None = None
    audit_path: Path | None = None
    log_path = frozen_root / "audit_log.jsonl"
    if verdict.verdict == "REJECT":
        print("Freeze skipped: verdict is REJECT")
    else:
        _print_stage_header("Stage 8 — Freeze + audit stamp")
        params_hash = freeze_params(output_dir, strategy_key, args.timeframe, selected_params, frozen_params_dir=frozen_explicit)
        audit_path = write_audit_stamp(strategy_key, params_hash, verdict, frozen_root)
        print(f"audit_stamp={audit_path.resolve()}")
        print(f"audit_log.jsonl={log_path.resolve()}")

    sensitivity_mc_serial: dict[str, Any] | None = (
        mc_summary_dict(sensitivity_mc) if sensitivity_mc is not None else None
    )
    if sensitivity_mc_serial is not None and sensitivity_heatmap_path is not None:
        sensitivity_mc_serial["graph_path"] = sensitivity_heatmap_path

    if holdout_mc_graph_path is not None:
        ho_mc_d["graph_path"] = holdout_mc_graph_path

    result_bundle: dict[str, Any] = {
        "strategy": strategy_key,
        "timeframe": args.timeframe,
        "data_dir": str(data_dir_path),
        "output_dir": str(output_dir),
        "frozen_params_dir": str(frozen_root),
        "max_grid": args.max_grid,
        "pipeline_config": str(args.pipeline_config) if args.pipeline_config else None,
        "skip_walk_forward": bool(args.skip_wf),
        "skip_sensitivity": bool(args.skip_sensitivity),
        "sizing": {
            "eval_risk_dollars": eval_risk,
            "max_contracts": max_contracts,
            "speed_optimization_paths": speed_optimization_paths,
            "longevity_optimization_path": longevity_optimization_path,
        },
        "min_fold_seq_pass_rate_pct": float(args.min_fold_seq_pass_rate_pct),
        "wf_development_window": {"name": wf_dev_window.name, "start": wf_dev_window.start, "end": wf_dev_window.end},
        "stage_validate": {"status": "ok"},
        "walk_forward": {
            "aggregate": wf_aggregate,
            "best_params": selected_params,
            "oos_folds": wf_folds_serial,
            "wf_robust_ok": wf_robust_ok,
            "wf_all_folds_seq_ok": wf_all_folds_seq_ok,
        },
        "sensitivity": sensitivity_serial,
        "sensitivity_mc": sensitivity_mc_serial,
        "holdout": _eval_json(holdout_eval),
        "express_funded_reset_sim": express_sim_summary,
        "holdout_monte_carlo": ho_mc_d,
        "regime_fit": regime_d,
        "verdict": verdict_summary_dict(verdict),
        "verdict_thresholds": asdict(verdict_thresholds),
        "freeze": (
            {"sha256": params_hash, "audit_stamp": str(audit_path.resolve())}
            if params_hash is not None and audit_path is not None
            else None
        ),
    }

    out_json = json_dir / f"{strategy_key}_{args.timeframe}_result.json"
    out_json.write_text(json.dumps(result_bundle, indent=2, sort_keys=False, default=str) + "\n")
    txt_dir = output_dir / "txt summaries"
    try:
        summary_path = write_readable_text_from_json_file(
            out_json,
            txt_dir / f"{out_json.stem}_summary.txt",
            style="pipeline",
        )
    except Exception as exc:  # noqa: BLE001
        summary_path = None
        print(f"WARN: readable summary not written ({exc})", file=sys.stderr)

    print(f"result_json={out_json.resolve()}")
    if summary_path is not None:
        print(f"readable_summary_txt={summary_path.resolve()}")

    sensitivity_flag_label = (
        str(verdict.sensitivity_flag) if verdict.sensitivity_flag is not None else "n/a"
    )
    table_rows: list[tuple[str, str]] = [
        ("strategy", strategy_key),
        ("timeframe", args.timeframe),
        ("verdict", verdict.verdict),
        ("regime_fit", regime_result.verdict),
        ("wf_robust_ok", str(wf_robust_ok)),
        ("wf_all_folds_seq_ok", str(wf_all_folds_seq_ok)),
        ("holdout_net_pnl", f"{float(holdout_eval.metrics['total_net_pnl']):.2f}"),
        ("holdout_mc_pnl_p05", f"{holdout_mc.pnl_p05:.2f}"),
        ("sensitivity_flag", sensitivity_flag_label),
        ("wf_oos_total_pnl", f"{wf_aggregate.get('wf_oos_total_pnl', 0.0):.2f}"),
        ("wf_dev_window", f"{wf_dev_window.start} → {wf_dev_window.end}"),
        ("result_json", str(out_json.resolve())),
    ]
    if summary_path is not None:
        table_rows.append(("readable_summary_txt", str(summary_path.resolve())))
    if holdout_mc_graph_path is not None:
        table_rows.append(("holdout_mc_graph", holdout_mc_graph_path))
    if sensitivity_heatmap_path is not None:
        table_rows.append(("sensitivity_mc_graph", sensitivity_heatmap_path))
    for idx, path in enumerate(speed_optimization_paths, 1):
        table_rows.append((f"wf{idx}_speed_optimization", path))
    if longevity_optimization_path is not None:
        table_rows.append(("holdout_longevity_optimization", longevity_optimization_path))
    if params_hash is not None and audit_path is not None:
        table_rows.extend(
            [
                ("frozen_params_sha256", params_hash[:16] + "..."),
                ("audit_stamp", str(audit_path.resolve())),
                ("audit_log_jsonl", str(log_path.resolve())),
            ]
        )
    _print_summary_table(table_rows)
    return 0


def _print_strategies() -> None:
    for key in sorted(STRATEGIES):
        spec = STRATEGIES[key]
        requires = ", ".join(spec.requires) if spec.requires else "none"
        filt = spec.filter_of if spec.filter_of is not None else "none"
        print(f"{spec.name}: requires={requires}; filter_of={filt}")


if __name__ == "__main__":
    raise SystemExit(main())
