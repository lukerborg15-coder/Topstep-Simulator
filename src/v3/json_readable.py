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
    """Format any JSON-serializable value as indented text (uses ``default=str`` for odd types)."""

    return json.dumps(obj, indent=indent, sort_keys=False, ensure_ascii=False, default=str) + "\n"


def format_titled_json_block(title: str, data: Any, *, indent: int = 2) -> str:
    """Wrap a title and pretty JSON body in separators."""

    sep = "=" * 72
    body = json.dumps(data, indent=indent, sort_keys=False, ensure_ascii=False, default=str)
    return f"{sep}\n{title}\n{sep}\n{body}\n\n"


def _strip_trades(obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return obj  # pragma: no cover
    out = {k: v for k, v in obj.items() if k != "trades"}
    if "trades" in obj and isinstance(obj["trades"], list):
        out["trade_count"] = len(obj["trades"])
    return out


def pipeline_result_bundle_to_readable_text(data: dict[str, Any]) -> str:
    """Turn a ``v3.cli`` result-bundle dict into staged human-readable blocks (trade arrays omitted).

    Mirrors the historical ``scripts/summarize_result_json.py`` layout.
    """

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
        )
        if k in data
    }
    parts.append(format_titled_json_block("RUN INPUTS / METADATA", meta))
    parts.append(format_titled_json_block("STAGE 1 — validate", data.get("stage_validate")))

    if "in_sample_sanity" in data:
        parts.append(
            format_titled_json_block(
                "STAGE 2 — in-sample sanity (trade list omitted)",
                _strip_trades(data["in_sample_sanity"]),
            )
        )

    if "walk_forward" in data and isinstance(data["walk_forward"], dict):
        wf = dict(data["walk_forward"])
        oos = wf.get("oos_folds")
        if isinstance(oos, list):
            wf["oos_folds"] = [_strip_trades(x) if isinstance(x, dict) else x for x in oos]
        parts.append(format_titled_json_block("STAGE 3 — walk-forward (oos trade lists omitted)", wf))

    parts.append(format_titled_json_block("STAGE 4 — sensitivity", data.get("sensitivity")))

    if "holdout" in data:
        parts.append(
            format_titled_json_block(
                "STAGE 5 — holdout (trade list omitted)",
                _strip_trades(data["holdout"]),
            )
        )

    if "express_funded_reset_sim" in data:
        parts.append(
            format_titled_json_block(
                "HOLDOUT — Express funded reset sim (milestones / bank across breaches)",
                data.get("express_funded_reset_sim"),
            )
        )

    parts.append(format_titled_json_block("STAGE 6 — holdout Monte Carlo (trade-order)", data.get("holdout_monte_carlo")))
    parts.append(format_titled_json_block("STAGE 7 — verdict", data.get("verdict")))
    parts.append(format_titled_json_block("VERDICT THRESHOLDS (CLI)", data.get("verdict_thresholds")))
    parts.append(format_titled_json_block("STAGE 8 — freeze / audit", data.get("freeze")))

    parts.append(
        "\nNotes:\n"
        "- Full fills and outcomes are still in the .json inside `trades` arrays.\n"
        "- WF OOS summaries above include `trade_count` per fold instead of listing trades.\n"
    )

    return "".join(parts)


def write_readable_text_from_json_file(
    json_path: Path | str,
    output_path: Path | str | None = None,
    *,
    style: str = "pipeline",
) -> Path:
    """Read a JSON file, format it, write UTF-8 text.

    Args:
        json_path: Path to `.json`.
        output_path: Target `.txt` path. If omitted, uses ``<stem>_summary.txt`` (pipeline) or
            ``<stem>_readable.txt`` (pretty).
        style: ``"pipeline"`` — ``pipeline_result_bundle_to_readable_text`` when root is an object,
            indent-staged summaries and strip trade spam. ``"pretty"`` —whole file as one indented JSON tree.
    """

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
