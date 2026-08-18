"""Helpers for agent-side captcha solving callbacks."""

from __future__ import annotations

import asyncio
import os
from typing import Any


def _captcha_solver_call_timeout_s(logger) -> float:
    configured = os.environ.get("WA_AGENT_CAPTCHA_TIMEOUT")
    if configured is not None:
        try:
            return max(0.1, float(configured))
        except ValueError:
            logger.warning(
                "Invalid WA_AGENT_CAPTCHA_TIMEOUT=%r; falling back to WA_CAPTCHA_TIMEOUT",
                configured,
            )

    try:
        challenge_timeout = float(os.environ.get("WA_CAPTCHA_TIMEOUT", "45"))
    except ValueError:
        challenge_timeout = 45.0
    return max(1.0, challenge_timeout + 5.0)


def maybe_solve_captcha(
    captcha_solver,
    page: Any | None,
    loop: Any | None,
    logger,
    phase: str,
) -> bool:
    """Run an async captcha callback from a synchronous agent loop."""
    if captcha_solver is None or page is None:
        return False
    if loop is None:
        logger.info("Captcha solver unavailable at %s: no event loop", phase)
        return False
    timeout_s = _captcha_solver_call_timeout_s(logger)
    try:
        solved = loop.run_until_complete(
            asyncio.wait_for(captcha_solver(page), timeout=timeout_s)
        )
        return bool(solved)
    except asyncio.TimeoutError:
        logger.warning("Captcha solver timed out at %s after %.1fs", phase, timeout_s)
        return False
    except Exception as exc:
        logger.info("Captcha solver failed at %s: %s", phase, exc)
        return False
