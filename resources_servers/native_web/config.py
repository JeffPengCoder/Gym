# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the native Playwright/PyAutoGUI web runtime."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, model_validator

from nemo_gym.web.models import WebBenchmark
from nemo_gym.web.resource_config import WebResourcesServerConfig


class NativeWebResourcesServerConfig(WebResourcesServerConfig):
    artifact_dir: str = "cache/native-web/artifacts"
    headless: bool = False
    viewport_width: int = Field(default=1920, ge=640)
    viewport_height: int = Field(default=1080, ge=480)
    action_delay_seconds: float = Field(default=2.0, ge=0, le=30)
    # Context-wide Playwright deadline, matching the reference runner's
    # PW_DEFAULT_TIMEOUT_MS. It bounds navigation and every other page operation.
    default_timeout_ms: int = Field(default=45_000, ge=1_000, le=600_000)
    terminate_on_action_error: bool = True
    max_computer_actions: int = Field(default=20, ge=1, le=100)
    record_video: bool = False
    browser_proxy_env: str = "WA_BROWSER_PROXY_SERVER"
    captcha_api_key_env: str = "CAPSOLVER_API_KEY"
    captcha_provider_env: str = "WA_CAPTCHA_PROVIDER"
    captcha_solver_env: str = "WA_CAPTCHA_SOLVER"
    require_captcha_solver: bool = False
    proxy_mode: Literal["webvoyager_domains", "always", "disabled"] = "webvoyager_domains"
    browser_channel: str | None = None
    # Mounted by native VisualWebArena profiles for task and evaluator images.
    task_image_root: str | None = None
    max_task_image_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    allowed_benchmarks: list[WebBenchmark] = Field(default_factory=lambda: [WebBenchmark.WEBVOYAGER])
    max_evidence_screenshots: int = Field(default=200, ge=1, le=500)

    @model_validator(mode="after")
    def validate_native(self) -> "NativeWebResourcesServerConfig":
        if self.headless:
            raise ValueError("native visual actions require headed Chromium under Xvfb")
        if self.max_sessions != 1:
            raise ValueError("PyAutoGUI native runtimes require max_sessions=1 per isolated DISPLAY")
        return self

    def browser_proxy(self) -> str:
        return os.environ.get(self.browser_proxy_env, "").strip()

    def captcha_api_key(self) -> str:
        return os.environ.get(self.captcha_api_key_env, "").strip()

    def captcha_solver(self) -> str:
        explicit = os.environ.get(self.captcha_solver_env, "").strip()
        if explicit:
            return explicit
        if self.captcha_api_key() and os.environ.get(self.captcha_provider_env, "capsolver").lower() == "capsolver":
            return "builtin:capsolver"
        return ""
