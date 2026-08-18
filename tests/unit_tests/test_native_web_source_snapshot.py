# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "resources_servers" / "native_web"


def test_native_source_lock_pins_independent_runtime_branches():
    source_lock = json.loads((NATIVE_ROOT / "SOURCE_LOCK.json").read_text())

    assert source_lock["snapshots"]["nemotron_v3"]["commit"] == (
        "3b775dc538931ead0cb6b4922349da9c6d493dab"
    )
    assert source_lock["snapshots"]["visualwebarena"]["commit"] == (
        "267b8d95243c9832990b4da1e6f1f328b0496a6b"
    )
    assert source_lock["benchmark_data"]["task_counts"] == {
        "webarena": 812,
        "visualwebarena": 908,
        "webvoyager": 552,
    }


def test_each_snapshot_contains_its_native_launcher_and_runtime_modules():
    nemotron = NATIVE_ROOT / "baselines" / "nemotron_v3"
    visual = NATIVE_ROOT / "baselines" / "visualwebarena"

    expected = [
        nemotron / "launch_nemotron_webarena_parallel.sh",
        nemotron / "launch_nemotron_webvoyager_parallel.sh",
        nemotron / "webarena" / "nvidia" / "nemotron_toolcall_agent.py",
        nemotron / "webarena" / "common" / "browser.py",
        nemotron / "webarena" / "common" / "classic_evaluation.py",
        nemotron / "webarena" / "common" / "webvoyager_evaluation.py",
        visual / "launch_nemotron_visualwebarena_parallel.sh",
        visual / "webarena" / "nvidia" / "nemotron_toolcall_agent.py",
        visual / "webarena" / "common" / "visualwebarena_evaluation.py",
    ]

    assert all(path.is_file() for path in expected)
