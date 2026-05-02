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
    """Turn a v3.cli result-bundle dict into staged human-readable blocks."""

    parts: list[str] = []

    meta = {
        k: data[k]
        for k in (
            "strategy",
            "timeframe",
            "data_dir",
            "output_dir",
            "frozen_params_dir",
            "max_grid",
            "pipeline_config",
            "skip_walk_forward",
            "skip_sensitivity",
            "sizing",
            "min_fold_seq_pass_rate_pct",
            "wf_development_window",
        )
        if k in data
    }
    parts.append(format_titled_json_block("RUN INPUTS / METADATA", meta))
    parts.append(format_titled_json_block("STAGE 1 — validate", data.get("stage_validate")))

    if "walk_forward" in data and isinstance(data["walk_forward"], dict):
        wf = dict(data["walk_forward"])
        oos = wf.get("oos_folds")
        if isinstance(oos, list):
            wf["oos_folds"] = [_strip_trades(x) if isinstance(x, dict) else x for x in oos]
        parts.append(format_titled_json_block("STAGE 2 — walk-forward (oos trade lists omitted)", wf))

    if "sensitivity" in data and data["sensitivity"] is not None:
        sens = dict(data["sensitivity"])
        sens.pop("sensitivity_param_results", None)
        parts.append(format_titled_json_block("STAGE 3 — sensitivity (param sweep summary)", sens))
        if "sensitivity_heatmap_text" in data.get("sensitivity", {}):
            parts.append(
                format_titled_json_block(
                    "STAGE 3 — sensitivity gradient text",
                    data["sensitivity"]["sensitivity_heatmap_text"],
                )
            )
    else:
        parts.append(format_titled_json_block("STAGE 3 — sensitivity", data.get("sensitivity")))

    if "sensitivity_mc" in data and data["sensitivity_mc"] is not None:
        parts.append(format_titled_json_block("STAGE 3 — sensitivity MC1 (block bootstrap)", data.get("sensitivity_mc")))

    if "holdout" in data:
        parts.append(
            format_titled_json_block(
                "STAGE 4 — holdout (trade list omitted)",
                _strip_trades(data["holdout"]),
            )
        )

    if "express_funded_reset_sim" in data:
        parts.append(
            format_titled_json_block(
                "STAGE 4 — Express funded reset sim (milestones / bank across breaches)",
                data.get("express_funded_reset_sim"),
            )
        )

    if "holdout_monte_carlo" in data:
        ho_mc = dict(data["holdout_monte_carlo"]) if isinstance(data["holdout_monte_carlo"], dict) else data["holdout_monte_carlo"]
        parts.append(format_titled_json_block("STAGE 5 — holdout Monte Carlo MC2 (block bootstrap)", ho_mc))

    if "regime_fit" in data:
        parts.append(format_titled_json_block("STAGE 6 — regime fit", data.get("regime_fit")))

    parts.append(format_titled_json_block("STAGE 7 — verdict", data.get("verdict")))
    parts.append(format_titled_json_block("VERDICT THRESHOLDS (CLI)", data.get("verdict_thresholds")))
    parts.append(format_titled_json_block("STAGE 8 — freeze / audit", data.get("freeze")))

    # Graph paths summary
    graph_paths: list[str] = []
    if isinstance(data.get("sensitivity_mc"), dict) and data["sensitivity_mc"].get("graph_path"):
        graph_paths.append(f"Sensitivity MC paths: {data['sensitivity_mc']['graph_path']}")
    if isinstance(data.get("sensitivity"), dict) and data["sensitivity"].get("sensitivity_heatmap_path"):
        graph_paths.append(f"Sensitivity heatmap: {data['sensitivity']['sensitivity_heatmap_path']}")
    if isinstance(data.get("holdout_monte_carlo"), dict) and data["holdout_monte_carlo"].get("graph_path"):
        graph_paths.append(f"Holdout MC paths: {data['holdout_monte_carlo']['graph_path']}")
    if graph_paths:
        parts.append("\nGraphs generated:\n" + "\n".join(f"  {p}" for p in graph_paths) + "\n")

    parts.append(
        "\nNotes:\n"
        "- Full fills and outcomes are still in the .json inside `trades` arrays.\n"
        "- WF OOS summaries above include `trade_count` per fold instead of listing trades.\n"
        "- Stage 2 in-sample sanity removed; WF development window covers full pre-holdout calendar.\n"
    )

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
