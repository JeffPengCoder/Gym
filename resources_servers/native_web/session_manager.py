# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-runtime specialization of the backend-neutral web session manager."""

from __future__ import annotations

from nemo_gym.web.models import WebActionProfile, WebBenchmark, WebRuntimeProfile, WebTask
from resources_servers.browsergym_web.session_manager import BrowserGymSessionManager
from resources_servers.native_web.backend import native_backend_factory
from resources_servers.native_web.config import NativeWebResourcesServerConfig


class NativeWebSessionManager(BrowserGymSessionManager):
    def __init__(self, config: NativeWebResourcesServerConfig) -> None:
        super().__init__(config, backend_factory=native_backend_factory)

    def _validate_task(self, task: WebTask) -> None:
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("native_web currently supports WebVoyager only")
        if task.runtime_profile != WebRuntimeProfile.NATIVE_VISUAL:
            raise ValueError("native_web requires runtime_profile=native_visual")
        if task.action_profile != WebActionProfile.NATIVE_TOOLCALL:
            raise ValueError("native_web requires action_profile=native_toolcall")
