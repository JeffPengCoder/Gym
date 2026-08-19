# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the hash-pinned maintained WebVoyager population."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.webvoyager.prepare import BENCHMARK_DIR, prepare_native


OUTPUT = BENCHMARK_DIR / "data" / "webvoyager_native_v3.jsonl"


if __name__ == "__main__":
    source = os.environ.get("WEBVOYAGER_SOURCE_JSONL", "").strip()
    if not source:
        raise RuntimeError(
            "WEBVOYAGER_SOURCE_JSONL must point to webvoyager.jsonl from "
            "jayl940712/webarena_benchmarks@6a2977939b157b0ab9de7799bb089c721f1ac115"
        )
    prepare_native(Path(source), OUTPUT)
