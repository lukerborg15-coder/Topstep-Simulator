from __future__ import annotations

from pathlib import Path

from v3.json_readable import (
    json_object_to_readable_text,
    pipeline_result_bundle_to_readable_text,
    write_readable_text_from_json_file,
)


def test_json_object_to_readable_text():
    txt = json_object_to_readable_text({"a": 1, "b": ["x"]})
    assert '"a"' in txt
    assert "1," in txt or "1\n" in txt


def test_pipeline_result_bundle_is_string():
    d = pipeline_result_bundle_to_readable_text({"stage_validate": {"status": "ok"}})
    # New layout uses TOPSTEP PIPELINE header and ASCII section dividers
    assert "TOPSTEP PIPELINE" in d
    assert "End of summary" in d


def test_write_readable_text_from_json_file(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text('{"strategy": "s", "stage_validate": null}', encoding="utf-8")
    out = write_readable_text_from_json_file(p)
    assert out.read_text(encoding="utf-8")
    assert out.name == "x_summary.txt"
