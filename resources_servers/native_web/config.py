# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the native Playwright/PyAutoGUI web runtime."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, model_validator

from nemo_gym.web.models import WebBenchmark
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig


class NativeWebResourcesServerConfig(BrowserGymWebResourcesServerConfig):
    artifact_dir: str = "cache/native-web/artifacts"
    headless: bool = False
    viewport_width: int = Field(default=1920, ge=640)
    viewport_height: int = Field(default=1080, ge=480)
    action_delay_seconds: float = Field(default=2.0, ge=0, le=30)
    terminate_on_action_error: bool = True
    max_computer_actions: int = Field(default=20, ge=1, le=100)
    browser_proxy_env: str = "WA_BROWSER_PROXY_SERVER"
    captcha_api_key_env: str = "CAPSOLVER_API_KEY"
    captcha_provider_env: str = "WA_CAPTCHA_PROVIDER"
    captcha_solver_env: str = "WA_CAPTCHA_SOLVER"
    require_captcha_solver: bool = False
    proxy_mode: Literal["webvoyager_domains", "always", "disabled"] = "webvoyager_domains"
    browser_channel: str | None = None
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
