# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the official WebVoyager JSONL into normalized Gym rows."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from nemo_gym.web.datasets import (
    adapt_native_webvoyager_record,
    adapt_webvoyager_record,
    load_json_records,
    write_jsonl,
)


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "webvoyager_benchmark.jsonl"
DEFAULT_SOURCE = BENCHMARK_DIR.parents[2] / "WebVoyager" / "data" / "WebVoyager_data.jsonl"
NATIVE_V3_SOURCE_COMMIT = "6a2977939b157b0ab9de7799bb089c721f1ac115"  # pragma: allowlist secret
NATIVE_V3_SOURCE_SHA256 = (
    "f635a9b27fa1980a63b39bbf64ae8e9e766159cb70fa765451d3d3c0b948ff98"  # pragma: allowlist secret
)


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT_FPATH) -> Path:
    source_path = Path(source or os.environ.get("WEBVOYAGER_SOURCE_JSONL", DEFAULT_SOURCE))
    rows = [adapt_webvoyager_record(record) for record in load_json_records(source_path)]
    count = write_jsonl(rows, output)
    print(f"Wrote {count} WebVoyager tasks to {output}", flush=True)
    return Path(output)


def prepare_native(source: str | Path, output: str | Path) -> Path:
    """Prepare the pinned maintained population for Nano Omni native runs."""

    source_path = Path(source)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != NATIVE_V3_SOURCE_SHA256:
        raise ValueError(f"native WebVoyager source hash mismatch: expected {NATIVE_V3_SOURCE_SHA256}, got {digest}")
    records = load_json_records(source_path)
    rows = [adapt_native_webvoyager_record(record) for record in records]
    if len(rows) != 552:
        raise ValueError(f"native WebVoyager v3 requires exactly 552 tasks, got {len(rows)}")
    count = write_jsonl(rows, output)
    print(f"Wrote {count} native WebVoyager tasks to {output}", flush=True)
    return Path(output)


if __name__ == "__main__":
    profile = os.environ.get("WEBVOYAGER_PREPARE_PROFILE", "legacy")
    if profile == "native_v3":
        source = os.environ.get("WEBVOYAGER_SOURCE_JSONL")
        if not source:
            raise RuntimeError("WEBVOYAGER_SOURCE_JSONL is required for native_v3 preparation")
        prepare_native(source, os.environ.get("WEBVOYAGER_OUTPUT_JSONL", OUTPUT_FPATH))
    elif profile == "legacy":
        prepare()
    else:
        raise RuntimeError(f"unsupported WEBVOYAGER_PREPARE_PROFILE: {profile}")
