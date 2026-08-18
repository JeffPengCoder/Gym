"""Helpers for agent-side captcha solving callbacks."""

from __future__ import annotations

from typing import Any


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
    try:
        return bool(loop.run_until_complete(captcha_solver(page)))
    except Exception as exc:
        logger.info("Captcha solver failed at %s: %s", phase, exc)
        return False
