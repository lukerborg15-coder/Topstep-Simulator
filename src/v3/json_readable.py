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
    parts.append("-" * 80 + "\n")
    parts.append(" FROZEN PARAMETERS (recommended)\n")
    parts.append("-" * 80 + "\n")
    if "walk_forward" in data and isinstance(data["walk_forward"], dict):
        wf_data = data["walk_forward"]
        selected_params = wf_data.get("selected_params", {})
        if isinstance(selected_params, dict):
            parts.append("  Strategy params:    " + "  ".join(f"{k}={v}" for k, v in list(selected_params.items())[:5]) + "\n\n")

    # Speed optimization
    if "speed_optimization_aggregate" in data:
        speed_data = data["speed_optimization_aggregate"]
        if isinstance(speed_data, dict):
            parts.append("  >>> EVAL SIZING (speed-optimized):    ${:.0f}/trade".format(speed_data.get("optimal_risk_dollars", 0)))
            parts.append("   (X-X contracts)\n")
            parts.append("        → tied to: pass_rate, median_days_to_pass\n\n")

    # Longevity optimization
    if "longevity_optimization" in data:
        long_data = data["longevity_optimization"]
        if isinstance(long_data, dict):
            parts.append("  >>> FUNDED SIZING (longevity-optim):  ${:.0f}/trade".format(long_data.get("optimal_risk_dollars", 0)))
            parts.append("   (X-X contracts)\n")
            parts.append("        → tied to: longevity_score, accounts_used, per-account survival days\n\n")

    # Sizing comparison (if present)
    if "sizing_comparison" in data and data["sizing_comparison"]:
        parts.append("\n" + "-" * 80 + "\n")
        parts.append(" SIZING COMPARISON (Optimizer vs Fixed)\n")
        parts.append("-" * 80 + "\n")
        comparison = data["sizing_comparison"]
        if isinstance(comparison, dict):
            parts.append("  Track             Eval pass rate   Funded longevity\n")
            if "track_a_optimizer" in comparison:
                opt = comparison["track_a_optimizer"]
                eval_pass = opt.get("eval_track", {}).get("pass_rate_pct", 0)
                long_score = opt.get("holdout_track", {}).get("longevity_score", 0)
                parts.append(f"  Optimizer         {eval_pass:6.1f}%            {long_score:.2f}\n")
            if "track_b_fixed_risk" in comparison and comparison["track_b_fixed_risk"]:
                fixed_b = comparison["track_b_fixed_risk"]
                eval_pass_b = fixed_b.get("eval_track", {}).get("pass_rate_pct", 0)
                long_score_b = fixed_b.get("holdout_track", {}).get("longevity_score", 0)
                parts.append(f"  Fixed $/trade     {eval_pass_b:6.1f}%            {long_score_b:.2f}\n")
            if "track_c_fixed_contracts" in comparison and comparison["track_c_fixed_contracts"]:
                fixed_c = comparison["track_c_fixed_contracts"]
                eval_pass_c = fixed_c.get("eval_track", {}).get("pass_rate_pct", 0)
                long_score_c = fixed_c.get("holdout_track", {}).get("longevity_score", 0)
                parts.append(f"  Fixed contracts   {eval_pass_c:6.1f}%            {long_score_c:.2f}\n")
            if "sanity_flags" in comparison and comparison["sanity_flags"]:
                parts.append("\n  [SANITY FLAGS]:\n")
                for flag in comparison["sanity_flags"]:
                    parts.append(f"    - {flag}\n")
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
                    days = fold.get("topstep", {}).get("topstep_days_to_pass") or "n/a"
                    long_cnt = fold.get("metrics", {}).get("long_trades", 0)
                    short_cnt = fold.get("metrics", {}).get("short_trades", 0)
                    passed = fold.get("topstep", {}).get("topstep_passed", False)
                    status = "PASS" if passed else "FAIL"
                    parts.append(f"  WF{i}      ${pnl:>8,.0f}   {seq_pass_rate:>6.1f}%    {str(days):>6}     {long_cnt:>3}/{short_cnt:<3}     {status}\n")

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
            parts.append(f"    Viable in:         X/{speed_data.get('n_folds', 0)} folds\n")
            parts.append(f"    Worst fold utility: {speed_data.get('min_oos_utility', 0):.2f}\n")
            candidates = speed_data.get("candidates", [])
            if candidates:
                top_alts = candidates[:3]
                alt_str = " | ".join(f"${c.get('risk_dollars', 0):.0f} ({c.get('median_oos_median_days_to_pass', 0):.1f}d, {c.get('median_oos_pass_rate_pct', 0):.0f}%)" for c in top_alts)
                parts.append(f"  Top 3 alternates: {alt_str}\n\n")

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
