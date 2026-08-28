# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym resource server for native visual browser sessions."""

from __future__ import annotations

from nemo_gym.web.resources_server import WebResourcesServer
from resources_servers.native_web.config import NativeWebResourcesServerConfig
from resources_servers.native_web.session_manager import NativeWebSessionManager


class NativeWebResourcesServer(WebResourcesServer):
    config: NativeWebResourcesServerConfig

    def make_session_manager(self) -> NativeWebSessionManager:
        return NativeWebSessionManager(self.config)


if __name__ == "__main__":
    NativeWebResourcesServer.run_webserver()
