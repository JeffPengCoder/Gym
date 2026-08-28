# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the maintained 908-task VisualWebArena native-runner population."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from nemo_gym import CACHE_DIR
from nemo_gym.web.datasets import (
    adapt_native_visualwebarena_records,
    load_json_records,
    write_jsonl,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT = BENCHMARK_DIR / "data" / "visualwebarena_native_v3.jsonl"
EXPECTED_TASKS = 908
NATIVE_V3_SOURCE_COMMIT = "6a2977939b157b0ab9de7799bb089c721f1ac115"  # pragma: allowlist secret
NATIVE_V3_SOURCE_SHA256 = (
    "923a4ec5a2a306d497a0a2f0d267db2c47b40b57c6be1965de0b19dd5041e04a"  # pragma: allowlist secret
)
SOURCE_REPOSITORY = "jayl940712/webarena_benchmarks"
SOURCE_ARCHIVE_URL = f"https://github.com/{SOURCE_REPOSITORY}/archive/{NATIVE_V3_SOURCE_COMMIT}.tar.gz"
DEFAULT_SOURCE_ROOT = CACHE_DIR / "webarena_benchmarks" / NATIVE_V3_SOURCE_COMMIT
SOURCE_JSONL_NAME = "visualwebarena.jsonl"


def _source_records(source_path: Path, image_root: Path) -> list[dict]:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != NATIVE_V3_SOURCE_SHA256:
        raise ValueError(
            f"native VisualWebArena source hash mismatch: expected {NATIVE_V3_SOURCE_SHA256}, got {digest}"
        )
    records = load_json_records(str(source_path))
    if len(records) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} native VisualWebArena tasks, found {len(records)}")

    root = image_root.expanduser().resolve()
    missing: list[str] = []
    for record in records:
        image_value = record.get("image") or record.get("images") or []
        references = [image_value] if isinstance(image_value, str) else image_value
        for reference_value in references:
            reference = str(reference_value).strip()
            if not reference or urlparse(reference).scheme:
                continue
            candidate = (root / reference).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"native VisualWebArena image path escapes source root: {reference!r}") from exc
            if not candidate.is_file():
                missing.append(reference)
    if missing:
        examples = ", ".join(repr(path) for path in missing[:3])
        raise FileNotFoundError(
            f"native VisualWebArena source is missing {len(missing)} referenced image(s) below {root}; "
            f"examples: {examples}"
        )
    return records


def _download_pinned_source(source_root: Path) -> Path:
    """Download and atomically cache the public JSONL plus reference images."""

    source_root = source_root.expanduser().resolve()
    source_path = source_root / SOURCE_JSONL_NAME
    if source_root.exists():
        if not source_root.is_dir():
            raise RuntimeError(f"native VisualWebArena source root is not a directory: {source_root}")
        try:
            _source_records(source_path, source_root)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"native VisualWebArena cache is incomplete or invalid: {source_root}. "
                "Remove that benchmark cache directory or select a fresh "
                "VISUALWEBARENA_NATIVE_SOURCE_ROOT."
            ) from exc
        print(f"Using cached native VisualWebArena source at {source_root}", flush=True)
        return source_path

    source_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="visualwebarena-source-", dir=source_root.parent) as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            archive_path = temp_dir / "source.tar.gz"
            extracted_dir = temp_dir / "extracted"
            extracted_dir.mkdir()
            print(f"Downloading pinned VisualWebArena source from {SOURCE_ARCHIVE_URL}", flush=True)
            with urllib.request.urlopen(SOURCE_ARCHIVE_URL, timeout=120) as response:  # noqa: S310
                with archive_path.open("wb") as archive_handle:
                    shutil.copyfileobj(response, archive_handle, length=1024 * 1024)
            with tarfile.open(archive_path, mode="r:gz") as archive:
                archive.extractall(extracted_dir, filter="data")

            candidates = [path for path in extracted_dir.iterdir() if path.is_dir()]
            if len(candidates) != 1:
                raise RuntimeError(f"expected one source directory in the pinned archive, found {len(candidates)}")
            extracted_root = candidates[0]
            _source_records(extracted_root / SOURCE_JSONL_NAME, extracted_root)
            try:
                extracted_root.replace(source_root)
            except OSError:
                # Another prepare process may have populated the immutable cache
                # while this process was downloading it.
                if not source_root.is_dir():
                    raise
                _source_records(source_path, source_root)
    except (OSError, tarfile.TarError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"could not download pinned VisualWebArena data from {SOURCE_ARCHIVE_URL}; "
            "set VISUALWEBARENA_NATIVE_SOURCE_ROOT to a local checkout of "
            f"{SOURCE_REPOSITORY}@{NATIVE_V3_SOURCE_COMMIT}"
        ) from exc

    print(f"Cached native VisualWebArena source at {source_root}", flush=True)
    return source_path


def prepare(
    source: str | Path | None = None,
    output: str | Path = OUTPUT,
    source_root: str | Path | None = None,
) -> Path:
    """Prepare the maintained population through Gym's benchmark CLI contract."""

    source_value = str(source or os.environ.get("VISUALWEBARENA_NATIVE_SOURCE_JSONL", "")).strip()
    configured_root = str(os.environ.get("VISUALWEBARENA_NATIVE_SOURCE_ROOT", "")).strip()
    if source_value:
        source_path = Path(source_value).expanduser().resolve()
        image_root = Path(configured_root).expanduser().resolve() if configured_root else source_path.parent
    else:
        image_root = Path(configured_root or source_root or DEFAULT_SOURCE_ROOT).expanduser().resolve()
        source_path = _download_pinned_source(image_root)

    records = _source_records(source_path, image_root)
    rows = adapt_native_visualwebarena_records(records)
    count = write_jsonl(rows, output)
    print(
        f"Wrote {count} native VisualWebArena tasks to {output}\nTask image root: {image_root}",
        flush=True,
    )
    return Path(output)


if __name__ == "__main__":
    prepare()
