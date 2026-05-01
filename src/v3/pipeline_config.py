"""Load walk-forward date windows from JSON (optional pipeline profile)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DateWindow,
    PipelineWindows,
    WalkForwardWindow,
    WINDOWS,
)


def _parse_date_window(name: str, obj: dict[str, Any]) -> DateWindow:
    return DateWindow(name=str(obj.get("name", name)), start=str(obj["start"]), end=str(obj["end"]))


def load_pipeline_windows(path: str | Path) -> PipelineWindows:
    """Load `PipelineWindows` from JSON. Schema mirrors `WINDOWS` in config."""
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    in_sample = _parse_date_window("in_sample_sanity", data["in_sample_sanity"])
    holdout = _parse_date_window("holdout", data["holdout"])
    wf_items: list[WalkForwardWindow] = []
    for wf in data["walk_forward"]:
        wf_items.append(
            WalkForwardWindow(
                name=str(wf["name"]),
                train=_parse_date_window(str(wf["train"].get("name", "train")), wf["train"]),
                test=_parse_date_window(str(wf["test"].get("name", "test")), wf["test"]),
            )
        )
    return PipelineWindows(
        in_sample_sanity=in_sample,
        walk_forward=tuple(wf_items),
        holdout=holdout,
    )


def windows_to_jsonable(windows: PipelineWindows) -> dict[str, Any]:
    """Serialize defaults for `windows.example.json`."""
    return {
        "in_sample_sanity": {"name": windows.in_sample_sanity.name, "start": windows.in_sample_sanity.start, "end": windows.in_sample_sanity.end},
        "holdout": {"name": windows.holdout.name, "start": windows.holdout.start, "end": windows.holdout.end},
        "walk_forward": [
            {
                "name": wf.name,
                "train": {"name": wf.train.name, "start": wf.train.start, "end": wf.train.end},
                "test": {"name": wf.test.name, "start": wf.test.start, "end": wf.test.end},
            }
            for wf in windows.walk_forward
        ],
    }


def resolve_windows(pipeline_config: str | Path | None) -> PipelineWindows:
    if pipeline_config is None:
        return WINDOWS
    return load_pipeline_windows(pipeline_config)


__all__ = [
    "load_pipeline_windows",
    "resolve_windows",
    "windows_to_jsonable",
]
