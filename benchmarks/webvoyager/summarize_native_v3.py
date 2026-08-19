# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Account for every native WebVoyager task without hiding failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_TASKS = 552


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        duplicates[task_id] += 1
        by_task[task_id] = row

    success = sum(bool(row.get("task_success")) for row in by_task.values())
    invalid = sum(bool(row.get("mask_sample")) for row in by_task.values())
    failures = Counter(
        str(row.get("failure_kind") or "policy_failure")
        for row in by_task.values()
        if not row.get("task_success")
    )
    completed = len(by_task)
    return {
        "expected": EXPECTED_TASKS,
        "completed_unique": completed,
        "missing": max(0, EXPECTED_TASKS - completed),
        "success": success,
        "strict_sr": success / EXPECTED_TASKS,
        "invalid_or_infrastructure": invalid,
        "duplicate_task_ids": sorted(task_id for task_id, count in duplicates.items() if count > 1),
        "failure_kinds": dict(sorted(failures.items())),
        "comparable": completed == EXPECTED_TASKS and invalid == 0,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(load_rows(args.rollouts))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
