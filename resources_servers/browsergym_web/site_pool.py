# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Site-stack lease boundary.

The first implementation deliberately exposes an unmanaged lease. It is safe
for one rollout at a time and makes the missing reset/isolation layer visible
instead of implying that a fresh browser context resets mutable websites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nemo_gym.web.models import WebTask


@dataclass(frozen=True, slots=True)
class SiteLease:
    lease_id: str
    isolated: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class SitePool(Protocol):
    async def acquire(self, session_id: str, task: WebTask) -> SiteLease: ...

    async def release(self, lease: SiteLease, *, healthy: bool) -> None: ...

    async def health(self) -> dict[str, Any]: ...


class UnmanagedSitePool:
    """Pass through the URLs configured by BrowserGym environment variables."""

    def __init__(self) -> None:
        self._active: set[str] = set()

    async def acquire(self, session_id: str, task: WebTask) -> SiteLease:
        self._active.add(session_id)
        return SiteLease(
            lease_id=f"unmanaged:{session_id}",
            isolated=False,
            metadata={"benchmark": task.benchmark.value, "sites": task.sites},
        )

    async def release(self, lease: SiteLease, *, healthy: bool) -> None:
        del healthy
        self._active.discard(lease.lease_id.removeprefix("unmanaged:"))

    async def health(self) -> dict[str, Any]:
        return {
            "mode": "unmanaged",
            "isolated": False,
            "active_leases": len(self._active),
        }
