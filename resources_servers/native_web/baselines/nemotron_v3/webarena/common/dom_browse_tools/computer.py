"""Small browser action helpers used by DOM tools."""

from __future__ import annotations

from typing import Any

from common.pyautogui_utils import execute_action


SCROLL_UNITS_PER_AMOUNT = 5


async def page_scroll(
    page: Any,
    *,
    index: int | None = None,
    coordinate: Any = None,
    scroll_direction: str = "down",
    scroll_amount: int = 1,
    pages: float | None = None,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> str:
    """Scroll the page with the same pyautogui wheel path as existing agents."""

    try:
        if pages is not None:
            page_count = float(pages)
            scroll_amount = 0 if page_count <= 0 else max(1, round(page_count))
        direction = str(scroll_direction or "down").lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("scroll_direction must be up, down, left, or right")
        amount = max(0, int(scroll_amount))

        if coordinate is None:
            x = int(screen_width / 2)
            y = int(screen_height / 2)
        else:
            x = int(float(coordinate[0]) * max(1, screen_width - 1))
            y = int(float(coordinate[1]) * max(1, screen_height - 1))

        clicks = amount * SCROLL_UNITS_PER_AMOUNT
        if direction in {"down", "left"}:
            clicks = -clicks

        if direction in {"left", "right"}:
            import pyautogui

            pyautogui.moveTo(x, y)
            pyautogui.hscroll(clicks)
        else:
            execute_action(
                {"type": "scroll", "clicks": clicks, "x": x, "y": y},
                screen_width,
                screen_height,
            )
        target = "screen center" if index is None else f"index [{index}]"
        return f"Scrolled {direction} {amount} wheel click(s) at {target}."
    except Exception as exc:
        return f"[ERROR] page_scroll failed: {type(exc).__name__}: {exc}"


__all__ = ["page_scroll"]
