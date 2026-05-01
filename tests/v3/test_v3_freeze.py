from __future__ import annotations

import json

import pytest

from v3.freeze import FrozenParamsViolation, freeze_params, verify_frozen_params


def test_freeze_params_writes_expected_json_keys(tmp_path):
    frozen_hash = freeze_params(tmp_path, "connors_rsi2", "5min", {"rsi": 2})

    path = tmp_path / "frozen_params" / "connors_rsi2_5min.json"
    data = json.loads(path.read_text())

    assert set(data) == {"strategy", "timeframe", "params", "sha256", "frozen_at"}
    assert data["strategy"] == "connors_rsi2"
    assert data["timeframe"] == "5min"
    assert data["params"] == {"rsi": 2}
    assert data["sha256"] == frozen_hash
    assert data["frozen_at"]


def test_verify_frozen_params_with_matching_params_returns_hash(tmp_path):
    params = {"rsi": 2, "levels": (1, 2)}
    frozen_hash = freeze_params(tmp_path, "connors_rsi2", "5min", params)

    assert verify_frozen_params(tmp_path, "connors_rsi2", "5min", params) == frozen_hash


def test_verify_frozen_params_with_different_params_raises_hash_mismatch(tmp_path):
    freeze_params(tmp_path, "connors_rsi2", "5min", {"rsi": 2})

    with pytest.raises(FrozenParamsViolation, match="hash mismatch"):
        verify_frozen_params(tmp_path, "connors_rsi2", "5min", {"rsi": 3})


def test_verify_frozen_params_with_no_file_raises_no_frozen_file(tmp_path):
    with pytest.raises(FrozenParamsViolation, match="no frozen file"):
        verify_frozen_params(tmp_path, "connors_rsi2", "5min", {"rsi": 2})


def test_freeze_params_same_params_is_idempotent(tmp_path):
    params = {"rsi": 2}
    first_hash = freeze_params(tmp_path, "connors_rsi2", "5min", params)
    second_hash = freeze_params(tmp_path, "connors_rsi2", "5min", params)

    assert second_hash == first_hash


def test_freeze_params_different_params_for_same_strategy_timeframe_raises(tmp_path):
    freeze_params(tmp_path, "connors_rsi2", "5min", {"rsi": 2})

    with pytest.raises(FrozenParamsViolation, match="already frozen"):
        freeze_params(tmp_path, "connors_rsi2", "5min", {"rsi": 3})
