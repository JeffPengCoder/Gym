# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym resource server for native WebVoyager browser sessions."""

from __future__ import annotations

from pydantic import PrivateAttr

from resources_servers.browsergym_web.app import BrowserGymWebResourcesServer
from resources_servers.native_web.config import NativeWebResourcesServerConfig
from resources_servers.native_web.session_manager import NativeWebSessionManager


class NativeWebResourcesServer(BrowserGymWebResourcesServer):
    config: NativeWebResourcesServerConfig
    _manager: NativeWebSessionManager = PrivateAttr()

    def model_post_init(self, _context) -> None:
        self._manager = NativeWebSessionManager(self.config)


if __name__ == "__main__":
    NativeWebResourcesServer.run_webserver()
