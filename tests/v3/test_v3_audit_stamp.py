from __future__ import annotations

import json
from datetime import datetime

import pytest

from v3.audit_stamp import write_audit_stamp
from v3.verdict import VerdictResult


def _verdict(verdict: str = "COMBINE-READY") -> VerdictResult:
    return VerdictResult(
        strategy="connors_rsi2",
        verdict=verdict,
        reject_reasons=(),
        warn_reasons=(),
        pass_rate_pct=80.0,
        worst_max_drawdown=1_000.0,
        pct_daily_limit_hit=10.0,
        mean_max_drawdown=600.0,
        sensitivity_flag=None,
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_write_audit_stamp_creates_individual_json_file_with_correct_fields(tmp_path):
    path = write_audit_stamp("connors_rsi2", "abc123", _verdict(), tmp_path)

    assert path == tmp_path / "connors_rsi2_audit.json"
    data = json.loads(path.read_text())
    assert data["strategy"] == "connors_rsi2"
    assert data["params_hash"] == "abc123"
    assert data["verdict"] == "COMBINE-READY"
    assert data["pass_rate_pct"] == 80.0
    assert data["worst_max_drawdown"] == 1_000.0
    assert data["pct_daily_limit_hit"] == 10.0
    assert data["mean_max_drawdown"] == 600.0
    assert data["sensitivity_flag"] is None
    assert data["sprint"] == "3"
    assert data["timestamp"]


def test_write_audit_stamp_appends_to_audit_log_jsonl_on_repeated_calls(tmp_path):
    write_audit_stamp("connors_rsi2", "first", _verdict("PROMISING"), tmp_path)
    write_audit_stamp("connors_rsi2", "second", _verdict("COMBINE-READY"), tmp_path)

    rows = _read_jsonl(tmp_path / "audit_log.jsonl")
    assert len(rows) == 2
    assert rows[0]["params_hash"] == "first"
    assert rows[0]["verdict"] == "PROMISING"
    assert rows[1]["params_hash"] == "second"
    assert rows[1]["verdict"] == "COMBINE-READY"


def test_write_audit_stamp_overwrites_individual_json_file_on_repeated_calls(tmp_path):
    write_audit_stamp("connors_rsi2", "first", _verdict("PROMISING"), tmp_path)
    path = write_audit_stamp("connors_rsi2", "second", _verdict("COMBINE-READY"), tmp_path)

    data = json.loads(path.read_text())
    assert data["params_hash"] == "second"
    assert data["verdict"] == "COMBINE-READY"


def test_params_hash_round_trips_correctly(tmp_path):
    path = write_audit_stamp("connors_rsi2", "sha256:deadbeef", _verdict(), tmp_path)

    data = json.loads(path.read_text())
    rows = _read_jsonl(tmp_path / "audit_log.jsonl")
    assert data["params_hash"] == "sha256:deadbeef"
    assert rows[0]["params_hash"] == "sha256:deadbeef"


def test_timestamp_is_present_and_parseable_as_iso8601(tmp_path):
    path = write_audit_stamp("connors_rsi2", "abc123", _verdict(), tmp_path)

    data = json.loads(path.read_text())
    parsed = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_strategy_must_match_verdict_strategy(tmp_path):
    with pytest.raises(ValueError, match="strategy mismatch"):
        write_audit_stamp("other_strategy", "abc123", _verdict(), tmp_path)


def test_strategy_rejects_path_separator_filename_escape(tmp_path):
    verdict = VerdictResult(
        strategy="../escape",
        verdict="PROMISING",
        reject_reasons=(),
        warn_reasons=(),
        pass_rate_pct=60.0,
        worst_max_drawdown=1_000.0,
        pct_daily_limit_hit=10.0,
        mean_max_drawdown=600.0,
        sensitivity_flag=None,
    )

    with pytest.raises(ValueError, match="invalid strategy"):
        write_audit_stamp("../escape", "abc123", verdict, tmp_path)
