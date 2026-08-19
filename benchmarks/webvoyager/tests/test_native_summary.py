# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.webvoyager.summarize_native_v3 import summarize


def test_native_summary_keeps_fixed_denominator_and_exposes_masked_failures() -> None:
    report = summarize(
        [
            {"task_id": "a", "task_success": True, "mask_sample": False},
            {
                "task_id": "b",
                "task_success": False,
                "mask_sample": True,
                "failure_kind": "judge_unparseable",
            },
        ]
    )

    assert report["success"] == 1
    assert report["strict_sr"] == 1 / 552
    assert report["missing"] == 550
    assert report["failure_kinds"] == {"judge_unparseable": 1}
    assert report["comparable"] is False
