# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from benchmarks.webarena import prepare_native_v3 as webarena_prepare
from benchmarks.webvoyager import prepare_native_v3 as webvoyager_prepare


def test_webvoyager_native_prepare_delegates_optional_source(monkeypatch, tmp_path) -> None:
    output = tmp_path / "prepared.jsonl"
    calls = []
    monkeypatch.setattr(
        webvoyager_prepare,
        "prepare_native",
        lambda source_path, output_path: calls.append((source_path, output_path)) or output_path,
    )

    assert webvoyager_prepare.prepare(output=output) == output
    assert calls == [(None, output)]


def test_webvoyager_native_prepare_delegates_explicit_source(monkeypatch, tmp_path) -> None:
    source = tmp_path / "webvoyager.jsonl"
    output = tmp_path / "prepared.jsonl"
    calls = []
    monkeypatch.setattr(
        webvoyager_prepare,
        "prepare_native",
        lambda source_path, output_path: calls.append((source_path, output_path)) or output_path,
    )

    assert webvoyager_prepare.prepare(source=source, output=output) == output
    assert calls == [(source, output)]


def test_webarena_native_prepare_uses_environment_source(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared.jsonl"
    records = [{"id": index} for index in range(812)]
    calls = {}
    monkeypatch.setenv("WEBARENA_NATIVE_SOURCE_JSONL", str(source))
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(webarena_prepare, "NATIVE_V3_SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())

    def load_records(source_path):
        calls["source"] = source_path
        return records

    monkeypatch.setattr(webarena_prepare, "load_json_records", load_records)
    monkeypatch.setattr(webarena_prepare, "adapt_native_webarena_records", lambda value: value)
    monkeypatch.setattr(webarena_prepare, "write_jsonl", lambda rows, output_path: len(rows))

    assert webarena_prepare.prepare(output=output) == output
    assert calls == {"source": str(source)}


def test_webarena_native_prepare_rejects_unpinned_source(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("WEBARENA_NATIVE_SOURCE_JSONL", str(source))

    with pytest.raises(ValueError, match="native WebArena source hash mismatch"):
        webarena_prepare.prepare()


def test_webarena_native_prepare_fails_closed_without_source(monkeypatch) -> None:
    monkeypatch.delenv("WEBARENA_NATIVE_SOURCE_JSONL", raising=False)

    with pytest.raises(RuntimeError, match="WEBARENA_NATIVE_SOURCE_JSONL"):
        webarena_prepare.prepare()
