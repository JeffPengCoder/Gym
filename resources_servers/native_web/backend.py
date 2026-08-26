# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebArena-family policy for the shared headed native-browser driver."""

from __future__ import annotations

import logging
import time
from typing import Any

from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.composed_backend import ComposedWebBackend
from nemo_gym.web.models import WebTask
from nemo_gym.web.native_browser import (
    NAVIGATION_WAIT_UNTIL,
    RESET_WAIT_UNTIL,
    NativeBrowserDriver,
    NativeBrowserEvaluationContext,
    _paste_unicode,
    _type_browser_text,
    _url_origin,
    is_retryable_navigation_transport_error,
)
from resources_servers.native_web.config import NativeWebResourcesServerConfig
from resources_servers.native_web.site_auth import configured_site_urls, login_sites, resolve_start_urls


LOG = logging.getLogger("nemo_gym.resources_servers.native_web")

# WebArena-family policy-driven navigation keeps the existing bounded transport
# recovery, while initial local-site setup additionally retries any Playwright
# failure because containers can still be settling after a reset.
NAVIGATION_RETRY_DELAYS_S = (4.0, 4.0, 4.0, 8.0)
LOCAL_SETUP_RETRY_DELAYS_S = (1.0, 2.0)


class NativeWebDriver(NativeBrowserDriver):
    """Shared native browser specialized for local WebArena-family sites."""

    config: NativeWebResourcesServerConfig

    def _navigation_retry_delays(self) -> tuple[float, ...]:
        return NAVIGATION_RETRY_DELAYS_S

    def _should_retry_navigation(self, exc: Exception) -> bool:
        return is_retryable_navigation_transport_error(exc)

    def _prepare_task(self, task: WebTask) -> list[str]:
        site_urls = configured_site_urls(task)
        login_sites(
            task,
            context=self._context,
            site_urls=site_urls,
            goto=lambda page, url: self._goto(page, url, wait_until=RESET_WAIT_UNTIL),
        )
        return resolve_start_urls(task, site_urls)

    def _after_start_navigation(self, page: Any, url: str) -> None:
        del url
        time.sleep(1)
        page.bring_to_front()

    def _reset_metadata(self) -> dict[str, Any]:
        return {"site_login_enabled": True}

    def _goto_task_start(self, page: Any, url: str) -> Any:
        attempts = len(LOCAL_SETUP_RETRY_DELAYS_S) + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._goto(page, url, wait_until=RESET_WAIT_UNTIL)
            except Exception as exc:
                if attempt >= attempts:
                    raise
                delay_seconds = LOCAL_SETUP_RETRY_DELAYS_S[attempt - 1]
                LOG.warning(
                    "event=native_local_site_setup_retry session=%s task=%s origin=%s "
                    "attempt=%d/%d error_type=%s sleep_seconds=%.1f",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    _url_origin(url),
                    attempt,
                    attempts,
                    type(exc).__name__,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
        raise RuntimeError("local site setup retry loop exited without a result")


def native_backend_factory(config, session_id: str, artifacts: WebArtifactStore):
    """Compose the local-site browser with its benchmark evaluator."""

    from resources_servers.native_web.evaluators import NativeTaskEvaluator

    return ComposedWebBackend(
        NativeWebDriver(config, session_id, artifacts),
        NativeTaskEvaluator(config=config),
    )


__all__ = [
    "LOCAL_SETUP_RETRY_DELAYS_S",
    "NAVIGATION_RETRY_DELAYS_S",
    "NAVIGATION_WAIT_UNTIL",
    "NativeBrowserEvaluationContext",
    "NativeWebDriver",
    "_paste_unicode",
    "_type_browser_text",
    "native_backend_factory",
]
