# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned native WebArena-family evaluator implementation."""

from resources_servers.native_web.reference_evaluation.classic_evaluation import (
    evaluate_classic_task_sync,
)
from resources_servers.native_web.reference_evaluation.eval_snapshots import (
    build_snapshot_context,
    collect_browser_snapshots_sync,
    collect_snapshots,
    merge_snapshots,
)
from resources_servers.native_web.reference_evaluation.visualwebarena_evaluation import (
    evaluate_visualwebarena_task_sync,
)
