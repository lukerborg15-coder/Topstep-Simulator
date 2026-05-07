"""Pretty-print pipeline JSON blobs and arbitrary JSON-compatible data as text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "format_titled_json_block",
    "json_object_to_readable_text",
    "pipeline_result_bundle_to_readable_text",
    "write_readable_text_from_json_file",
]


def json_object_to_readable_text(obj: Any, *, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, sort_keys=False, ensure_ascii=False, default=str) + "\n"


def format_titled_json_block(title: str, data: Any, *, indent: int = 2) -> str:
    sep = "=" * 72
    body = json.dumps(data, indent=indent, sort_keys=False, ensure_ascii=False, default=str)
    return f"{sep}\n{title}\n{sep}\n{body}\n\n"


def _strip_trades(obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return obj
    out = {k: v for k, v in obj.items() if k != "trades"}
    if "trades" in obj and isinstance(obj["trades"], list):
        out["trade_count"] = len(obj["trades"])
    return out


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{digits}f}"


def _fmt_pct(value: Any, *, input_is_fraction: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if input_is_fraction:
        v *= 100.0
    return f"{v:.1f}%"


def _fmt_days(value: Any) -> str:
    if value in (None, "", "n/a"):
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == float("inf"):
        return "n/a"
    return f"{v:.1f}d" if not v.is_integer() else f"{int(v)}d"


def _fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _section(parts: list[str], title: str) -> None:
    parts.append("-" * 80 + "\n")
    parts.append(f" {title}\n")
    parts.append("-" * 80 + "\n")


def _params_line(params: dict[str, Any]) -> str:
    if not params:
        return "n/a"
    return "  ".join(f"{k}={v}" for k, v in params.items())


def _regime_label(verdict: Any) -> str:
    labels = {
        "prefers_calm": "Calm",
        "prefers_volatile": "Volatile",
        "mixed": "Mixed / no clear edge",
        "insufficient_data": "Insufficient data",
    }
    return labels.get(str(verdict), str(verdict).replace("_", " ").title())


def _seq_or_single_days(topstep: dict[str, Any]) -> Any:
    seq_days = topstep.get("seq_eval_median_days_to_pass")
    if seq_days not in (None, "", "n/a"):
        return seq_days
    return topstep.get("topstep_days_to_pass")


def _append_sensitivity_section(parts: list[str], data: dict[str, Any]) -> None:
    if "sensitivity" not in data and "skip_sensitivity" not in data:
        return
    _section(parts, "PARAMETER SENSITIVITY")
    sensitivity = data.get("sensitivity")
    if isinstance(sensitivity, dict):
        cliff = "YES" if sensitivity.get("sensitivity_is_cliff") else "no"
        parts.append(f"  Cliff detected: {cliff}\n")
        parts.append(f"  Default pass rate: {_fmt_pct(sensitivity.get('sensitivity_default_pass_rate'))}\n")
        parts.append(f"  Min neighbor pass rate: {_fmt_pct(sensitivity.get('sensitivity_min_neighbor_pass_rate'))}\n")
        heatmap_path = sensitivity.get("sensitivity_heatmap_path")
        if heatmap_path:
            parts.append(f"  Sensitivity heatmap: {heatmap_path}\n")
        heatmap_text = sensitivity.get("sensitivity_heatmap_text")
        if heatmap_text:
            parts.append("\n")
            for line in str(heatmap_text).splitlines():
                parts.append(f"  {line}\n")
        parts.append("\n")
        return

    if data.get("skip_sensitivity") or data.get("sensitivity_skip_reason"):
        reason = data.get("sensitivity_skip_reason") or "--skip-sensitivity flag"
        parts.append(f"  Skipped: {reason}\n")
        parts.append("  Run with --full to produce the parameter sensitivity heatmap and gradient summary.\n\n")
    else:
        parts.append("  Not available for this run.\n\n")


def _append_candidate_tables(parts: list[str], speed_data: dict[str, Any] | None, long_data: dict[str, Any] | None) -> None:
    if isinstance(speed_data, dict):
        speed_candidates = speed_data.get("all_candidates") or speed_data.get("candidates") or []
        if speed_candidates:
            parts.append("  Full candidate table (speed):\n")
            parts.append("    Risk    Utility  Pass rate  Median days  Viable folds\n")
            for c in speed_candidates:
                if not isinstance(c, dict):
                    continue
                parts.append(
                    f"    {_fmt_money(c.get('risk_dollars')):>7} "
                    f"{_fmt_float(c.get('median_oos_utility'), 4):>8} "
                    f"{_fmt_pct(c.get('median_oos_pass_rate_pct')):>10} "
                    f"{_fmt_days(c.get('median_oos_median_days_to_pass')):>11} "
                    f"{_fmt_int(c.get('viable_folds', 0)):>6}\n"
                )
            parts.append("\n")

    if isinstance(long_data, dict):
        long_candidates = long_data.get("all_candidates") or long_data.get("candidates") or []
        if long_candidates:
            parts.append("  Full candidate table (longevity):\n")
            parts.append("    Risk    Status    Median score  P05 score  Avg PnL/trade  Notes\n")
            for c in long_candidates:
                if not isinstance(c, dict):
                    continue
                rejected = bool(c.get("rejected"))
                status = "REJECT" if rejected else "OK"
                notes = str(c.get("reject_reason", "")) if rejected else ""
                parts.append(
                    f"    {_fmt_money(c.get('risk_dollars')):>7} "
                    f"{status:<8} "
                    f"{_fmt_float(c.get('median_longevity_score'), 4):>12} "
                    f"{_fmt_float(c.get('p05_longevity_score'), 4):>9} "
                    f"{_fmt_money(c.get('median_avg_pnl_per_trade')):>14} "
                    f"{notes}\n"
                )
            parts.append("\n")


def _append_sizing_comparison(parts: list[str], comparison: Any) -> None:
    if not isinstance(comparison, dict):
        return
    _section(parts, "SIZING COMPARISON (Optimizer vs Fixed)")
    parts.append(
        "  Track             Eval risk   Eval pass   Eval days   Funded risk  "
        "Longevity   Accounts   Blown   Delta pass   Delta days   Delta long\n"
    )

    rows: list[tuple[str, dict[str, Any], str | None]] = []
    if isinstance(comparison.get("track_a_optimizer"), dict):
        rows.append(("Optimizer", comparison["track_a_optimizer"], None))
    if isinstance(comparison.get("track_b_fixed_risk"), dict):
        rows.append(("Fixed $/trade", comparison["track_b_fixed_risk"], "fixed_risk_vs_optimizer"))
    if isinstance(comparison.get("track_c_fixed_contracts"), dict):
        rows.append(("Fixed contracts", comparison["track_c_fixed_contracts"], "fixed_contracts_vs_optimizer"))

    deltas = comparison.get("deltas", {}) if isinstance(comparison.get("deltas"), dict) else {}
    for label, track, delta_key in rows:
        eval_track = track.get("eval_track", {}) if isinstance(track.get("eval_track"), dict) else {}
        holdout_track = track.get("holdout_track", {}) if isinstance(track.get("holdout_track"), dict) else {}
        delta = deltas.get(delta_key, {}) if delta_key and isinstance(deltas.get(delta_key), dict) else {}
        eval_risk = eval_track.get("risk_dollars", track.get("risk_dollars"))
        funded_risk = holdout_track.get("risk_dollars", track.get("risk_dollars", track.get("fixed_contracts")))
        accounts = holdout_track.get("accounts_used", 0)
        blown = holdout_track.get("accounts_blown", 0)
        parts.append(
            f"  {label:<16} "
            f"{_fmt_money(eval_risk):>9} "
            f"{_fmt_pct(eval_track.get('pass_rate_pct')):>10} "
            f"{_fmt_days(eval_track.get('median_days_to_pass')):>9} "
            f"{_fmt_money(funded_risk):>11} "
            f"{_fmt_float(holdout_track.get('longevity_score'), 4):>10} "
            f"{_fmt_int(accounts):>8} "
            f"{_fmt_int(blown):>7} "
            f"{_fmt_pct(delta.get('eval_pass_rate_delta')):>11} "
            f"{_fmt_days(delta.get('eval_days_delta')):>10} "
            f"{_fmt_float(delta.get('holdout_longevity_delta'), 4):>10}\n"
        )

    if comparison.get("sanity_flags"):
        parts.append("\n  [SANITY FLAGS]:\n")
        for flag in comparison["sanity_flags"]:
            parts.append(f"    - {flag}\n")
    parts.append("\n")


def pipeline_result_bundle_to_readable_text(data: dict[str, Any]) -> str:
    """Turn a v3.cli result-bundle dict into staged human-readable ASCII blocks."""

    parts: list[str] = []

    # Header
    strategy = data.get("strategy", "unknown")
    timeframe = data.get("timeframe", "unknown")
    run_date = data.get("timestamp", "unknown date")
    parts.append("\n" + "=" * 80 + "\n")
    parts.append(f" TOPSTEP PIPELINE — {strategy} ({timeframe})\n")
    parts.append(f" Run: {run_date}  |  Mode: optimizer\n")
    parts.append("=" * 80 + "\n\n")

    # Verdict summary (if available)
    verdict_data = data.get("verdict", {})
    if isinstance(verdict_data, dict):
        verdict = verdict_data.get("final_verdict", "UNKNOWN")
        parts.append(f"VERDICT: {verdict}\n")
        reasons = verdict_data.get("reasons", [])
        if reasons:
            parts.append("Reasons: " + ", ".join(str(r) for r in reasons) + "\n")
        parts.append("\n")

    # Frozen parameters
    _section(parts, "FROZEN PARAMETERS (recommended)")
    if "walk_forward" in data and isinstance(data["walk_forward"], dict):
        wf_data = data["walk_forward"]
        selected_params = wf_data.get("best_params", wf_data.get("selected_params", {}))
        if isinstance(selected_params, dict):
            parts.append("  BEST WALK-FORWARD PARAMETERS\n")
            parts.append("  Strategy params:    " + _params_line(selected_params) + "\n\n")

    # Speed optimization
    if "speed_optimization_aggregate" in data:
        speed_data = data["speed_optimization_aggregate"]
        if isinstance(speed_data, dict):
            parts.append("  >>> EVAL SIZING (speed-optimized):    ${:.0f}/trade".format(speed_data.get("optimal_risk_dollars", 0)))
            parts.append("\n")
            parts.append("        → tied to: pass_rate, median_days_to_pass\n\n")

    # Longevity optimization
    if "longevity_optimization" in data:
        long_data = data["longevity_optimization"]
        if isinstance(long_data, dict):
            parts.append("  >>> FUNDED SIZING (longevity-optim):  ${:.0f}/trade".format(long_data.get("optimal_risk_dollars", 0)))
            parts.append("\n")
            parts.append("        → tied to: longevity_score, accounts_used, per-account survival days\n\n")

    _append_sensitivity_section(parts, data)

    # Sizing comparison (if present)
    if "sizing_comparison" in data and data["sizing_comparison"]:
        _append_sizing_comparison(parts, data["sizing_comparison"])

    # Detailed walk-forward audit
    if "walk_forward" in data and isinstance(data["walk_forward"], dict):
        wf = data["walk_forward"]
        _section(parts, "WALK-FORWARD DETAIL")
        dev = data.get("wf_development_window", {})
        if isinstance(dev, dict) and (dev.get("start") or dev.get("end")):
            parts.append(f"  Development window: {dev.get('start', 'n/a')} -> {dev.get('end', 'n/a')}\n")

        aggregate = wf.get("aggregate", {})
        if isinstance(aggregate, dict):
            parts.append("  WF AGGREGATE\n")
            parts.append(f"    Total OOS PnL:  {_fmt_money(aggregate.get('wf_oos_total_pnl', 0.0))}\n")
            parts.append(f"    Avg fold PnL:   {_fmt_money(aggregate.get('wf_avg_net_pnl', 0.0))}\n")
            parts.append(f"    Avg Sharpe:     {_fmt_float(aggregate.get('wf_avg_sharpe', 0.0))}\n")
            parts.append(f"    Consistency SD: {_fmt_money(aggregate.get('wf_consistency', 0.0))}\n\n")

        speed_by_fold: dict[int, dict[str, Any]] = {}
        speed_data = data.get("speed_optimization_aggregate")
        if isinstance(speed_data, dict):
            for item in speed_data.get("per_fold_oos", []) or []:
                if isinstance(item, dict) and item.get("fold_index") is not None:
                    try:
                        speed_by_fold[int(item["fold_index"])] = item
                    except (TypeError, ValueError):
                        pass

        oos_folds = wf.get("oos_folds", [])
        if isinstance(oos_folds, list) and oos_folds:
            parts.append(
                "  Fold     Trades  OOS PnL   MaxDD    PF   Sharpe  AvgR  "
                "Seq       Single days  Seq median days  Speed median days  L/S    Result\n"
            )
            for i, fold in enumerate(oos_folds, 1):
                if not isinstance(fold, dict):
                    continue
                metrics = fold.get("metrics", {})
                topstep = fold.get("topstep", {})
                speed_fold = speed_by_fold.get(i - 1, {})
                passed = topstep.get("topstep_passed", False)
                status = "PASS" if passed else "FAIL"
                seq = f"{topstep.get('seq_eval_passes', 0)}/{topstep.get('seq_eval_attempts', 0)}"
                seq_rate = _fmt_pct(topstep.get("seq_eval_pass_rate", 0.0), input_is_fraction=True)
                parts.append(
                    f"  WF{i:<6} "
                    f"{int(metrics.get('total_trades', 0)):>5} "
                    f"{_fmt_money(metrics.get('total_net_pnl', 0)):>8} "
                    f"{_fmt_money(metrics.get('max_drawdown', 0)):>7} "
                    f"{_fmt_float(metrics.get('profit_factor', 0)):>5} "
                    f"{_fmt_float(metrics.get('sharpe', 0)):>7} "
                    f"{_fmt_float(metrics.get('avg_r', 0)):>5} "
                    f"{seq:>5} {seq_rate:>7} "
                    f"{_fmt_days(topstep.get('topstep_days_to_pass')):>11} "
                    f"{_fmt_days(topstep.get('seq_eval_median_days_to_pass')):>15} "
                    f"{_fmt_days(speed_fold.get('median_days_to_pass')):>18} "
                    f"{int(metrics.get('long_trades', 0)):>2}/{int(metrics.get('short_trades', 0)):<2} "
                    f"{status}\n"
                )
                reason = topstep.get("topstep_reason")
                if reason:
                    parts.append(f"           reason: {reason}\n")
            parts.append("\n")

    # Walk-forward summary
    if "walk_forward" in data and isinstance(data["walk_forward"], dict):
        wf = data["walk_forward"]
        parts.append("-" * 80 + "\n")
        parts.append(" WALK-FORWARD (N folds)\n")
        parts.append("-" * 80 + "\n")
        oos_folds = wf.get("oos_folds", [])
        if isinstance(oos_folds, list):
            parts.append("  Fold     OOS PnL     Pass Rate   Days→Pass   Long/Short    Topstep\n")
            for i, fold in enumerate(oos_folds, 1):
                if isinstance(fold, dict):
                    pnl = fold.get("topstep", {}).get("topstep_final_balance", 50000) - 50000
                    seq_pass_rate = fold.get("topstep", {}).get("seq_eval_pass_rate", 0) * 100
                    days = _fmt_days(_seq_or_single_days(fold.get("topstep", {})))
                    long_cnt = fold.get("metrics", {}).get("long_trades", 0)
                    short_cnt = fold.get("metrics", {}).get("short_trades", 0)
                    passed = fold.get("topstep", {}).get("topstep_passed", False)
                    status = "PASS" if passed else "FAIL"
                    parts.append(f"  WF{i}      ${pnl:>8,.0f}   {seq_pass_rate:>6.1f}%    {days:>6}     {long_cnt:>3}/{short_cnt:<3}     {status}\n")

        # Aggregate row
        if oos_folds and isinstance(oos_folds[0], dict):
            total_pnl = sum(f.get("topstep", {}).get("topstep_final_balance", 50000) - 50000 for f in oos_folds)
            avg_pass = sum(f.get("topstep", {}).get("seq_eval_pass_rate", 0) for f in oos_folds) / len(oos_folds) * 100
            total_long = sum(f.get("metrics", {}).get("long_trades", 0) for f in oos_folds)
            total_short = sum(f.get("metrics", {}).get("short_trades", 0) for f in oos_folds)
            passed_count = sum(1 for f in oos_folds if f.get("topstep", {}).get("topstep_passed", False))
            parts.append(f"  Aggreg   ${total_pnl:>8,.0f}   {avg_pass:>6.1f}%    —           {total_long:>3}/{total_short:<3}     {passed_count}/{len(oos_folds)} passed\n")

        # Robust check
        min_seq_pass = wf.get("min_fold_seq_pass_rate_pct", 0)
        robust = "YES" if all(f.get("topstep", {}).get("seq_eval_pass_rate", 0) * 100 >= min_seq_pass for f in oos_folds) else "NO"
        parts.append(f"\n  Robust params? {robust} — N/N folds met seq_eval floor ({min_seq_pass:.0f}%)\n\n")

    # Speed optimizer detail
    if "speed_optimization_aggregate" in data:
        speed_data = data["speed_optimization_aggregate"]
        if isinstance(speed_data, dict):
            parts.append("-" * 80 + "\n")
            parts.append(" SPEED OPTIMIZER (aggregate winner across N folds)\n")
            parts.append("-" * 80 + "\n")
            parts.append(f"  Winner: ${speed_data.get('optimal_risk_dollars', 0):.0f}/trade\n")
            parts.append(f"    Median days→pass:  {speed_data.get('median_oos_median_days_to_pass', 0):.1f}\n")
            parts.append(f"    Pass rate:         {speed_data.get('median_oos_pass_rate_pct', 0):.1f}%\n")
            parts.append(
                f"    Viable in:         {speed_data.get('viable_folds', 0)}/"
                f"{speed_data.get('n_folds', 0)} folds\n"
            )
            parts.append(f"    Worst fold utility: {speed_data.get('min_oos_utility', 0):.2f}\n")
            eff_floor = speed_data.get("effective_pass_floor_pct")
            user_floor = speed_data.get("pass_floor_pct")
            if eff_floor is not None and user_floor is not None:
                if speed_data.get("adaptive_floor_applied"):
                    parts.append(
                        f"    Pass floor:        {eff_floor:.1f}% (adaptive; user {user_floor:.1f}%)\n"
                    )
                else:
                    parts.append(f"    Pass floor:        {eff_floor:.1f}%\n")
            candidates = speed_data.get("candidates", [])
            if candidates:
                top_alts = candidates[:3]
                alt_str = " | ".join(f"${c.get('risk_dollars', 0):.0f} ({c.get('median_oos_median_days_to_pass', 0):.1f}d, {c.get('median_oos_pass_rate_pct', 0):.0f}%)" for c in top_alts)
                parts.append(f"  Top 3 alternates: {alt_str}\n\n")
            _append_candidate_tables(parts, speed_data, None)

    # Detailed holdout and funded-account audit
    if "holdout" in data and isinstance(data["holdout"], dict):
        _section(parts, "HOLDOUT DETAIL")
        holdout = data["holdout"]
        metrics = holdout.get("metrics", {})
        topstep = holdout.get("topstep", {})
        parts.append(f"  Trades:       {int(metrics.get('total_trades', 0))}\n")
        parts.append(f"  Net PnL:      {_fmt_money(metrics.get('total_net_pnl', 0.0))}\n")
        parts.append(f"  Max DD:       {_fmt_money(metrics.get('max_drawdown', 0.0))}\n")
        parts.append(f"  Profit factor:{_fmt_float(metrics.get('profit_factor', 0.0)):>8}\n")
        parts.append(f"  Sharpe:       {_fmt_float(metrics.get('sharpe', 0.0))}\n")
        parts.append(f"  Avg R:        {_fmt_float(metrics.get('avg_r', 0.0))}\n")
        parts.append(
            f"  Topstep:      {'PASS' if topstep.get('topstep_passed') else 'FAIL'}  "
            f"days_to_pass={_fmt_days(topstep.get('topstep_days_to_pass'))}  "
            f"reason={topstep.get('topstep_reason', 'n/a')}\n\n"
        )

    express = data.get("express_funded_reset_sim")
    if isinstance(express, dict):
        _section(parts, "FUNDED ACCOUNT DETAIL")
        parts.append(
            f"  Accounts used: {int(express.get('funded_accounts_used', 0))}  "
            f"failed: {int(express.get('funded_accounts_failed', 0))}  "
            f"current_active: {bool(express.get('current_account_active', False))}\n"
        )
        parts.append(f"  Bank accrued:  {_fmt_money(express.get('accrued_pnl_bank', 0.0))}\n")
        stints = express.get("stints_summary", [])
        if stints:
            parts.append(
                "  Acct  Status  Days  Trades  Terminal   PnL     Win%   PF   AvgR  "
                "Sharpe  DailyDD  PeakDD\n"
            )
            for i, acct in enumerate(stints, 1):
                if not isinstance(acct, dict):
                    continue
                status = "FAILED" if acct.get("breached") else "ACTIVE"
                parts.append(
                    f"  {i:>4}  {status:<6} "
                    f"{int(acct.get('survival_days', 0)):>4} "
                    f"{int(acct.get('trades_applied_count', 0)):>7} "
                    f"{_fmt_money(acct.get('terminal_balance', 0.0)):>9} "
                    f"{_fmt_money(acct.get('bank_increment', 0.0)):>7} "
                    f"{_fmt_pct(acct.get('win_rate_pct', 0.0)):>6} "
                    f"{_fmt_float(acct.get('profit_factor', 0.0)):>5} "
                    f"{_fmt_float(acct.get('avg_r_multiple', 0.0)):>5} "
                    f"{_fmt_float(acct.get('sharpe_annualized', 0.0)):>7} "
                    f"{_fmt_money(acct.get('stint_worst_daily_dd', 0.0)):>8} "
                    f"{_fmt_money(acct.get('stint_worst_peak_to_trough', 0.0)):>7}\n"
                )
        else:
            parts.append("  No funded-account stints were simulated.\n")
        parts.append("\n")

    regime = data.get("regime_fit")
    if isinstance(regime, dict):
        _section(parts, "MARKET REGIME FIT")
        verdict = regime.get("regime_verdict", "insufficient_data")
        parts.append(f"  Best regime: {_regime_label(verdict)}\n")
        parts.append(f"  Window: {regime.get('window', 'n/a')}  Trades classified: {int(regime.get('total_trades', 0))}\n")
        parts.append("  Regime      Trades  Win rate  Expectancy  Mean PnL\n")
        for label, key in (("Calm", "calm"), ("Volatile", "volatile")):
            stats = regime.get(key, {})
            if not isinstance(stats, dict):
                stats = {}
            parts.append(
                f"  {label:<10} "
                f"{int(stats.get('count', 0)):>6} "
                f"{_fmt_pct(stats.get('win_rate', 0.0), input_is_fraction=True):>9} "
                f"{_fmt_float(stats.get('expectancy_r', 0.0), 4)}R".rjust(12)
                + f" {_fmt_money(stats.get('mean_net_pnl', 0.0)):>9}\n"
            )
        parts.append("\n")

    # Holdout summary
    if "holdout" in data:
        parts.append("-" * 80 + "\n")
        parts.append(" HOLDOUT\n")
        parts.append("-" * 80 + "\n")
        holdout = data["holdout"]
        if isinstance(holdout, dict):
            metrics = holdout.get("metrics", {})
            total_trades = metrics.get("total_trades", 0)
            long_trades = metrics.get("long_trades", 0)
            short_trades = metrics.get("short_trades", 0)
            win_rate = metrics.get("win_rate", 0) * 100
            long_wr = metrics.get("long_win_rate", 0) * 100
            short_wr = metrics.get("short_win_rate", 0) * 100
            pnl = metrics.get("total_net_pnl", 0)
            long_pnl = metrics.get("long_net_pnl", 0)
            short_pnl = metrics.get("short_net_pnl", 0)
            pf = metrics.get("profit_factor", float("inf"))
            parts.append(f"  Trades:       {total_trades:>3} (Long: {long_trades:>3} | Short: {short_trades:>3})\n")
            parts.append(f"  Win rate:     {win_rate:>5.1f}% (Long: {long_wr:>5.1f}% | Short: {short_wr:>5.1f}%)\n")
            parts.append(f"  Net PnL:      ${pnl:>8,.0f} (Long: ${long_pnl:>7,.0f} | Short: ${short_pnl:>7,.0f})\n")
            parts.append(f"  Profit factor: {pf:>6.2f}\n\n")

    # Longevity optimizer detail
    if "longevity_optimization" in data:
        long_data = data["longevity_optimization"]
        if isinstance(long_data, dict):
            parts.append("-" * 80 + "\n")
            parts.append(" LONGEVITY OPTIMIZER (holdout)\n")
            parts.append("-" * 80 + "\n")
            parts.append(f"  Winner: ${long_data.get('optimal_risk_dollars', 0):.0f}/trade\n")
            components = long_data.get("median_components", {})
            for comp_name in ["survival_score", "drawdown_score", "efficiency_score", "capital_score"]:
                p05_val = long_data.get("p05_components", {}).get(comp_name, 0)
                med_val = components.get(comp_name, 0)
                parts.append(f"    {comp_name.replace('_', ' ').title():.<20} {med_val:.2f}   p05: {p05_val:.2f}\n")
            parts.append(f"    Composite score: {long_data.get('median_longevity_score', 0):.2f}   p05: {long_data.get('p05_longevity_score', 0):.2f}\n\n")

            # Per-account survival
            per_account = long_data.get("per_account_summary", [])
            if per_account:
                parts.append("  Per-account survival (baseline):\n")
                parts.append("    Account #   Survival days   Terminal balance   Result\n")
                for i, acct in enumerate(per_account, 1):
                    survival_days = acct.get("survival_days", 0)
                    balance = acct.get("terminal_balance", 50000)
                    breached = acct.get("breached", False)
                    result = "BLOWN" if breached else "ACTIVE"
                    parts.append(f"    {i:>8}       {survival_days:>3}            ${balance:>9,.0f}        {result}\n")
                parts.append("\n")

            candidates = long_data.get("candidates", [])
            if candidates:
                top_alts = candidates[:3]
                alt_str = " | ".join(f"${c.get('risk_dollars', 0):.0f} ({c.get('median_longevity_score', 0):.2f})" for c in top_alts)
                parts.append(f"  Top 3 alternates: {alt_str}\n\n")
            _append_candidate_tables(parts, None, long_data)

    # Supporting artifacts
    parts.append("-" * 80 + "\n")
    parts.append(" SUPPORTING ARTIFACTS\n")
    parts.append("-" * 80 + "\n")
    parts.append("  [Output files would be listed here]\n\n")

    # Notes
    parts.append("-" * 80 + "\n")
    parts.append("[End of summary. Full structured data in companion JSON.]\n")

    return "".join(parts)


def write_readable_text_from_json_file(
    json_path: Path | str,
    output_path: Path | str | None = None,
    *,
    style: str = "pipeline",
) -> Path:
    in_path = Path(json_path)
    payload: Any = json.loads(in_path.read_text(encoding="utf-8"))

    if style == "pretty":
        text = json_object_to_readable_text(payload)
        default_suffix = "_readable.txt"
    elif style == "pipeline":
        if not isinstance(payload, dict):
            raise TypeError(f'pipeline style requires JSON object at root, got {type(payload).__name__}')
        text = pipeline_result_bundle_to_readable_text(payload)
        default_suffix = "_summary.txt"
    else:
        raise ValueError(style)

    out = Path(output_path) if output_path is not None else in_path.with_name(in_path.stem + default_suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out
