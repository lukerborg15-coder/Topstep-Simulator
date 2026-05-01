from __future__ import annotations

import argparse
import json
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
    run_in_sample_sanity,
    run_walk_forward,
    wf_oos_folds_for_selected_params,
)
from .freeze import freeze_params
from .funded_express_sim import express_funded_reset_sim_summary_dict, simulate_express_funded_resets
from .json_readable import write_readable_text_from_json_file
from .holdout_monte_carlo import holdout_monte_carlo_summary_dict, run_holdout_trade_monte_carlo
from .pipeline_config import resolve_windows
from .sensitivity import (
    SensitivityReport,
    run_sensitivity,
    sensitivity_summary_dict,
)
from .strategies import STRATEGIES, load_user_strategies
from .trades import TradeResult
from .validator import StrategyValidationError, validate_filter_references, validate_strategy
from .verdict import compute_pipeline_verdict, verdict_summary_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topstep evaluation pipeline (backtest → WF → sensitivity → holdout → MC → verdict).")
    parser.add_argument("--strategy", default=None, help="Registered strategy key (required unless --list-strategies).")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "holdout-only"],
        default="quick",
        help=(
            "Pipeline mode: quick: stages 1→3 then 5→8 (skip stage 4 parameter sensitivity). ~12 min. "
            "full: all stages 1→8 including sensitivity. holdout-only: 1→2 then 5→8 "
            "(frozen/default params; skips walk-forward and sensitivity). ~2 min."
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
        help="JSON file with in_sample_sanity, walk_forward, holdout windows (see config/ in project root).",
    )
    parser.add_argument("--holdout-mc-iterations", type=int, default=1000, help="Holdout trade-order shuffles (Monte Carlo v1).")
    parser.add_argument("--holdout-mc-seed", type=int, default=42, help="RNG seed for holdout Monte Carlo.")
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
            f"≥ this %% (default {DEFAULT_MIN_FOLD_SEQ_PASS_RATE_PCT}); attempts==0 fails the fold."
        ),
    )
    sensitivity_group = parser.add_mutually_exclusive_group()
    sensitivity_group.add_argument("--full", dest="skip_sensitivity", action="store_false", help="Run full pipeline, including the optional sensitivity sweep.")
    sensitivity_group.add_argument("--skip-sensitivity", dest="skip_sensitivity", action="store_true", help="Skip parameter sensitivity (stage 4).")
    parser.set_defaults(skip_sensitivity=None)
    parser.add_argument("--sensitivity-resamples", type=int, default=200, help="Bootstrap iterations per sensitivity sweep point (default 200).")
    parser.add_argument("--reject-pass-rate", type=float, default=VERDICT_THRESHOLDS.reject_pass_rate_pct, help="Reject if Combine pass rate is below this percentage.")
    parser.add_argument("--reject-max-dd", type=float, default=VERDICT_THRESHOLDS.reject_max_dd, help="Reject if worst-case drawdown exceeds this dollar amount.")
    parser.add_argument("--reject-daily-hit-pct", type=float, default=VERDICT_THRESHOLDS.reject_daily_hit_pct, help="Reject if daily loss limit hit rate exceeds this percentage.")
    parser.add_argument("--reject-mean-dd", type=float, default=VERDICT_THRESHOLDS.reject_mean_dd, help="Reject if average max drawdown exceeds this dollar amount.")
    parser.add_argument("--ready-pass-rate", type=float, default=VERDICT_THRESHOLDS.ready_pass_rate_pct, help="Require at least this pass rate percentage for COMBINE-READY.")
    parser.add_argument("--ready-max-dd", type=float, default=VERDICT_THRESHOLDS.ready_max_dd, help="Require worst-case drawdown at or below this dollar amount for COMBINE-READY.")
    parser.add_argument("--ready-daily-hit-pct", type=float, default=VERDICT_THRESHOLDS.ready_daily_hit_pct, help="Require daily loss limit hit rate at or below this percentage for COMBINE-READY.")
    parser.add_argument("--ready-mean-dd", type=float, default=VERDICT_THRESHOLDS.ready_mean_dd, help="Require average max drawdown at or below this dollar amount for COMBINE-READY.")
    return parser


def _apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.skip_sensitivity is None:
        args.skip_sensitivity = args.mode != "full"
    if args.mode == "holdout-only":
        args.skip_wf = True

    # --skip-sensitivity and --skip-wf can still be set individually for power users.


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

    _print_stage_header("Stage 2 — Backtest (in-sample)")
    sanity = run_in_sample_sanity(
        frame,
        strategy_key,
        args.timeframe,
        pipeline_windows,
        risk_dollars=eval_risk,
        max_contracts=max_contracts,
    )
    top_ok = sanity.topstep.get("topstep_passed")
    tp_label = "passed" if top_ok else "failed"
    trades_n = int(sanity.metrics["total_trades"])
    win_rate = float(sanity.metrics.get("win_rate", 0.0))
    profit_factor = float(sanity.metrics.get("profit_factor", 0.0))
    net_pnl = sanity.metrics["total_net_pnl"]
    avg_r = sanity.metrics["avg_r"]
    print(
        f"trade_count={trades_n} win_rate={win_rate:.3f} "
        f"profit_factor={profit_factor:.3f} net_pnl={net_pnl:.2f} "
        f"avg_r={avg_r:.4f} topstep_{tp_label}"
    )
    print("Gate checks: trades >= 20, win_rate >= 30%, profit_factor >= 1.0")
    if trades_n < 20:
        if not args.force:
            print("REJECT: insufficient sample size")
            return 1
        print("WARN: insufficient sample size (--force)")
    if win_rate < 0.30:
        if not args.force:
            print("REJECT: win rate < 30%")
            return 1
        print("WARN: win rate < 30% (--force)")
    if profit_factor < 1.0:
        if not args.force:
            print("REJECT: profit factor < 1.0")
            return 1
        print("WARN: profit factor < 1.0 (--force)")

    _print_stage_header("Stage 3 — Walk-forward")
    wf_aggregate: dict[str, Any]
    wf_folds_serial: list[dict[str, Any]]
    selected_params: dict[str, Any]
    wf_oos_final: list[EvaluationResult]
    wf_robust_ok: bool
    wf_all_folds_seq_ok: bool
    fold_rates: list[float] = []

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
        for fold in wf_oos_final:
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
        wf_aggregate = aggregate_wf_metrics(wf_oos_final)
        wf_folds_serial = [_eval_json(f) for f in wf_oos_final]
        print("WF aggregate:", ", ".join(f"{k}={v}" for k, v in wf_aggregate.items()))
        print("best_params(selected):", json.dumps(selected_params, sort_keys=True, default=str))

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

    if args.skip_sensitivity or not spec.param_grid:
        skip_reason = "--skip-sensitivity flag" if args.skip_sensitivity else "no param_grid defined"
        _print_stage_header("Stage 4 — Parameter sensitivity (skipped)")
        print(f"Skipped: {skip_reason}")
    else:
        _print_stage_header("Stage 4 — Parameter sensitivity")

        def _trades_fn(params: dict) -> list[TradeResult]:
            result = evaluate_strategy(
                frame,
                strategy_key,
                args.timeframe,
                params,
                pipeline_windows.in_sample_sanity,
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

    _print_stage_header("Stage 5 — Holdout")
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

    _print_stage_header("Stage 6 — Monte Carlo (trade-order)")
    holdout_mc = run_holdout_trade_monte_carlo(
        holdout_eval.trades,
        n=args.holdout_mc_iterations,
        seed=args.holdout_mc_seed,
    )
    ho_mc_d = holdout_monte_carlo_summary_dict(holdout_mc)
    print(
        f"holdout_mc pnl p05/p50/p95="
        f"{holdout_mc.pnl_p05:.2f}/{holdout_mc.pnl_p50:.2f}/{holdout_mc.pnl_p95:.2f}",
    )

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
        "sizing": {"eval_risk_dollars": eval_risk, "max_contracts": max_contracts},
        "min_fold_seq_pass_rate_pct": float(args.min_fold_seq_pass_rate_pct),
        "stage_validate": {"status": "ok"},
        "in_sample_sanity": _eval_json(sanity),
        "walk_forward": {
            "aggregate": wf_aggregate,
            "best_params": selected_params,
            "oos_folds": wf_folds_serial,
            "wf_robust_ok": wf_robust_ok,
            "wf_all_folds_seq_ok": wf_all_folds_seq_ok,
        },
        "sensitivity": sensitivity_serial,
        "holdout": _eval_json(holdout_eval),
        "express_funded_reset_sim": express_sim_summary,
        "holdout_monte_carlo": ho_mc_d,
        "verdict": verdict_summary_dict(verdict),
        "verdict_thresholds": asdict(verdict_thresholds),
        "freeze": (
            {"sha256": params_hash, "audit_stamp": str(audit_path.resolve())}
            if params_hash is not None and audit_path is not None
            else None
        ),
    }

    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
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
        ("wf_robust_ok", str(wf_robust_ok)),
        ("wf_all_folds_seq_ok", str(wf_all_folds_seq_ok)),
        ("holdout_net_pnl", f"{float(holdout_eval.metrics['total_net_pnl']):.2f}"),
        ("holdout_mc_pnl_p05", f"{holdout_mc.pnl_p05:.2f}"),
        ("sensitivity_flag", sensitivity_flag_label),
        ("wf_oos_total_pnl", f"{wf_aggregate.get('wf_oos_total_pnl', 0.0):.2f}"),
        ("in_sample_net_pnl", f"{sanity.metrics['total_net_pnl']:.2f}"),
        ("result_json", str(out_json.resolve())),
    ]
    if summary_path is not None:
        table_rows.append(("readable_summary_txt", str(summary_path.resolve())))
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
