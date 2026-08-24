# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the maintained 812-task WebArena native-runner population."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from nemo_gym.web.datasets import adapt_native_webarena_records, load_json_records, write_jsonl


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT = BENCHMARK_DIR / "data" / "webarena_native_v3.jsonl"
EXPECTED_TASKS = 812
NATIVE_V3_SOURCE_COMMIT = "6a2977939b157b0ab9de7799bb089c721f1ac115"  # pragma: allowlist secret
NATIVE_V3_SOURCE_SHA256 = (
    "6a6dae6809c84a08881ac37cadbfd4d50eac66e3578ebdb6ba8a308eebe1e0a3"  # pragma: allowlist secret
)


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT) -> Path:
    """Prepare the maintained population through Gym's benchmark CLI contract."""

    source_path = str(source or os.environ.get("WEBARENA_NATIVE_SOURCE_JSONL", "")).strip()
    if not source_path:
        raise RuntimeError(
            "WEBARENA_NATIVE_SOURCE_JSONL must point to webarena.jsonl from "
            "jayl940712/webarena_benchmarks@6a2977939b157b0ab9de7799bb089c721f1ac115"
        )
    digest = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    if digest != NATIVE_V3_SOURCE_SHA256:
        raise ValueError(f"native WebArena source hash mismatch: expected {NATIVE_V3_SOURCE_SHA256}, got {digest}")
    records = load_json_records(source_path)
    if len(records) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} native WebArena tasks, found {len(records)}")
    rows = adapt_native_webarena_records(records)
    count = write_jsonl(rows, output)
    print(f"Wrote {count} native WebArena tasks to {output}", flush=True)
    return Path(output)


if __name__ == "__main__":
    prepare()
