# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from benchmarks.webvoyager.summarize_native_v3 import (
    load_dataset,
    load_rows,
    summarize,
    write_missing_rows,
)


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


def test_native_summary_merges_worker_outputs_and_builds_exact_cleanup_input(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset_rows = [
        {"responses_create_params": {"metadata": {"task_id": task_id}}, "payload": task_id}
        for task_id in ("a", "b", "c")
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in dataset_rows), encoding="utf-8")
    worker_root = tmp_path / "workers"
    for worker, rows in {
        "worker-00": [{"task_id": "a", "task_success": True, "mask_sample": False}],
        "worker-01": [{"task_id": "b", "task_success": False, "mask_sample": False}],
    }.items():
        output = worker_root / worker / "rollouts.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded_dataset, expected = load_dataset(dataset)
    report = summarize(load_rows([worker_root]), expected_task_ids=expected)
    cleanup = tmp_path / "cleanup.jsonl"
    write_missing_rows(loaded_dataset, set(report["missing_task_ids"]), cleanup)

    assert report["expected"] == 3
    assert report["completed_unique"] == 2
    assert report["missing_task_ids"] == ["c"]
    assert report["success"] == 1
    assert [json.loads(line)["payload"] for line in cleanup.read_text().splitlines()] == ["c"]


def test_native_summary_marks_duplicate_worker_results_non_comparable() -> None:
    report = summarize(
        [
            {"task_id": "a", "task_success": True, "mask_sample": False},
            {"task_id": "a", "task_success": True, "mask_sample": False},
        ],
        expected_task_ids={"a"},
    )

    assert report["duplicate_task_ids"] == ["a"]
    assert report["comparable"] is False


def test_native_summary_retries_masked_rows_as_well_as_missing_rows(tmp_path) -> None:
    dataset_rows = [
        {"responses_create_params": {"metadata": {"task_id": task_id}}, "payload": task_id}
        for task_id in ("a", "b", "c")
    ]
    report = summarize(
        [
            {"task_id": "a", "task_success": False, "mask_sample": True},
            {"task_id": "b", "task_success": False, "mask_sample": False},
        ],
        expected_task_ids={"a", "b", "c"},
    )
    cleanup = tmp_path / "cleanup.jsonl"
    write_missing_rows(dataset_rows, set(report["retry_task_ids"]), cleanup)

    assert report["invalid_task_ids"] == ["a"]
    assert report["missing_task_ids"] == ["c"]
    assert report["retry_task_ids"] == ["a", "c"]
    assert [json.loads(line)["payload"] for line in cleanup.read_text().splitlines()] == ["a", "c"]
