# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-visual specialization of the common web session manager."""

from __future__ import annotations

from nemo_gym.web.models import WebActionProfile, WebBenchmark, WebRuntimeProfile, WebTask
from nemo_gym.web.session_manager import WebSessionManager
from resources_servers.native_web.backend import native_backend_factory
from resources_servers.native_web.config import NativeWebResourcesServerConfig


NATIVE_VERIFIER_PROFILES = {
    WebBenchmark.WEBARENA: "native_webarena_classic",
    WebBenchmark.VISUALWEBARENA: "native_visualwebarena",
}


class NativeWebSessionManager(WebSessionManager):
    def __init__(self, config: NativeWebResourcesServerConfig) -> None:
        super().__init__(config, backend_factory=native_backend_factory)

    def _validate_task(self, task: WebTask) -> None:
        super()._validate_task(task)
        if task.runtime_profile != WebRuntimeProfile.NATIVE_VISUAL:
            raise ValueError("native_web requires runtime_profile=native_visual")
        if task.action_profile != WebActionProfile.NATIVE_TOOLCALL:
            raise ValueError("native_web requires action_profile=native_toolcall")
        if task.benchmark not in NATIVE_VERIFIER_PROFILES:
            raise ValueError("native_web only accepts WebArena and VisualWebArena tasks")
        expected_verifier = NATIVE_VERIFIER_PROFILES[task.benchmark]
        if task.verifier_profile != expected_verifier:
            raise ValueError(
                f"native_web requires verifier_profile={expected_verifier} for benchmark={task.benchmark.value}"
            )
