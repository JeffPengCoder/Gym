# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from benchmarks.visualwebarena import prepare_native_v3 as visualwebarena_prepare
from benchmarks.webarena import prepare_native_v3 as webarena_prepare
from benchmarks.webvoyager import prepare_native_v3 as webvoyager_prepare


def test_webvoyager_native_prepare_uses_environment_source(monkeypatch, tmp_path) -> None:
    source = tmp_path / "webvoyager.jsonl"
    output = tmp_path / "prepared.jsonl"
    calls = []
    monkeypatch.setenv("WEBVOYAGER_SOURCE_JSONL", str(source))
    monkeypatch.setattr(
        webvoyager_prepare,
        "prepare_native",
        lambda source_path, output_path: calls.append((source_path, output_path)) or output_path,
    )

    assert webvoyager_prepare.prepare(output=output) == output
    assert calls == [(source, output)]


@pytest.mark.parametrize(
    ("module", "environment_name", "expected_tasks", "adapter_name"),
    [
        (webarena_prepare, "WEBARENA_NATIVE_SOURCE_JSONL", 812, "adapt_native_webarena_records"),
        (
            visualwebarena_prepare,
            "VISUALWEBARENA_NATIVE_SOURCE_JSONL",
            908,
            "adapt_native_visualwebarena_records",
        ),
    ],
)
def test_webarena_family_native_prepare_uses_environment_source(
    monkeypatch,
    tmp_path,
    module,
    environment_name,
    expected_tasks,
    adapter_name,
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared.jsonl"
    records = [{"id": index} for index in range(expected_tasks)]
    calls = {}
    monkeypatch.setenv(environment_name, str(source))
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "NATIVE_V3_SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())

    def load_records(source_path):
        calls["source"] = source_path
        return records

    monkeypatch.setattr(module, "load_json_records", load_records)
    monkeypatch.setattr(module, adapter_name, lambda value: value)
    monkeypatch.setattr(module, "write_jsonl", lambda rows, output_path: len(rows))

    assert module.prepare(output=output) == output
    assert calls == {"source": str(source)}


@pytest.mark.parametrize(
    ("module", "environment_name", "benchmark_name"),
    [
        (webarena_prepare, "WEBARENA_NATIVE_SOURCE_JSONL", "WebArena"),
        (
            visualwebarena_prepare,
            "VISUALWEBARENA_NATIVE_SOURCE_JSONL",
            "VisualWebArena",
        ),
    ],
)
def test_webarena_family_native_prepare_rejects_unpinned_source(
    monkeypatch,
    tmp_path,
    module,
    environment_name,
    benchmark_name,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(environment_name, str(source))

    with pytest.raises(ValueError, match=rf"native {benchmark_name} source hash mismatch"):
        module.prepare()


@pytest.mark.parametrize(
    ("module", "environment_name"),
    [
        (webvoyager_prepare, "WEBVOYAGER_SOURCE_JSONL"),
        (webarena_prepare, "WEBARENA_NATIVE_SOURCE_JSONL"),
        (visualwebarena_prepare, "VISUALWEBARENA_NATIVE_SOURCE_JSONL"),
    ],
)
def test_native_prepare_fails_closed_without_source(monkeypatch, module, environment_name) -> None:
    monkeypatch.delenv(environment_name, raising=False)

    with pytest.raises(RuntimeError, match=environment_name):
        module.prepare()
