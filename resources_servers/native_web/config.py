# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the native Playwright/PyAutoGUI web runtime."""

from __future__ import annotations

from pydantic import Field

from nemo_gym.web.models import WebBenchmark
from nemo_gym.web.native_browser import NativeBrowserResourcesServerConfig


class NativeWebResourcesServerConfig(NativeBrowserResourcesServerConfig):
    """Native-browser configuration scoped to WebArena and VisualWebArena."""

    artifact_dir: str = "cache/native-web/artifacts"
    allowed_benchmarks: list[WebBenchmark] = Field(
        default_factory=lambda: [WebBenchmark.WEBARENA, WebBenchmark.VISUALWEBARENA]
    )
