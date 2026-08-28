# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import tarfile

import pytest
from omegaconf import OmegaConf

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
    ],
)
def test_native_prepare_fails_closed_without_source(monkeypatch, module, environment_name) -> None:
    monkeypatch.delenv(environment_name, raising=False)

    with pytest.raises(RuntimeError, match=environment_name):
        module.prepare()


def test_visualwebarena_native_prepare_downloads_and_reuses_pinned_source(monkeypatch, tmp_path) -> None:
    source_root = tmp_path / "cache" / "pinned-source"
    output = tmp_path / "prepared.jsonl"
    image_reference = "visualwebarena/shopping/task_86/input_0.png"
    records = [
        {
            "id": f"visualwebarena-{index}",
            "ques": f"Task {index}",
            "image": [image_reference] if index == 0 else [],
            "eval": {},
        }
        for index in range(visualwebarena_prepare.EXPECTED_TASKS)
    ]
    source_bytes = b"".join(json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n" for record in records)
    monkeypatch.setattr(
        visualwebarena_prepare,
        "NATIVE_V3_SOURCE_SHA256",
        hashlib.sha256(source_bytes).hexdigest(),
    )
    monkeypatch.delenv("VISUALWEBARENA_NATIVE_SOURCE_JSONL", raising=False)
    monkeypatch.delenv("VISUALWEBARENA_NATIVE_SOURCE_ROOT", raising=False)

    archive_bytes = io.BytesIO()
    archive_root = f"webarena_benchmarks-{visualwebarena_prepare.NATIVE_V3_SOURCE_COMMIT}"
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for relative_path, contents in (
            (visualwebarena_prepare.SOURCE_JSONL_NAME, source_bytes),
            (image_reference, b"image"),
        ):
            member = tarfile.TarInfo(f"{archive_root}/{relative_path}")
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))

    downloads = []

    def urlopen(url, timeout):
        downloads.append((url, timeout))
        return io.BytesIO(archive_bytes.getvalue())

    monkeypatch.setattr(visualwebarena_prepare.urllib.request, "urlopen", urlopen)

    assert visualwebarena_prepare.prepare(output=output, source_root=source_root) == output
    assert (source_root / visualwebarena_prepare.SOURCE_JSONL_NAME).read_bytes() == source_bytes
    assert (source_root / image_reference).read_bytes() == b"image"
    assert len(output.read_text(encoding="utf-8").splitlines()) == visualwebarena_prepare.EXPECTED_TASKS

    assert visualwebarena_prepare.prepare(output=output, source_root=source_root) == output
    assert downloads == [(visualwebarena_prepare.SOURCE_ARCHIVE_URL, 120)]


def test_visualwebarena_native_config_shares_prepared_image_root(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VISUALWEBARENA_NATIVE_SOURCE_ROOT", raising=False)
    config = OmegaConf.load(visualwebarena_prepare.BENCHMARK_DIR / "configs" / "native_v3.yaml")
    config.cache_dir = str(tmp_path / "gym-cache")
    OmegaConf.resolve(config)

    source_root = config.prepare_script_args.source_root
    assert source_root == str(
        tmp_path / "gym-cache" / "webarena_benchmarks" / visualwebarena_prepare.NATIVE_V3_SOURCE_COMMIT
    )
    assert config.native_visualwebarena.resources_servers.native_web.task_image_root == source_root
    assert config.native_visualwebarena_agent.responses_api_agents.web_agent.task_image_root == source_root
