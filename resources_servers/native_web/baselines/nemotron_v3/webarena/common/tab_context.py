"""Playwright tab context helpers for WebArena trajectory logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_tab_context(page: Any | None, loop: Any | None) -> dict[str, Any] | None:
    """Return structured tab context for the active Playwright page."""
    if page is None or loop is None:
        return None

    try:
        pages = list(page.context.pages)
    except Exception:
        return None

    tabs: list[dict[str, Any]] = []
    for idx, tab_page in enumerate(pages):
        try:
            title = loop.run_until_complete(tab_page.title())
        except Exception:
            title = "unknown"
        try:
            url = tab_page.url
        except Exception:
            url = "unknown"
        tabs.append({"tab_id": idx, "title": title, "url": url})

    try:
        current_tab_id = pages.index(page)
    except ValueError:
        current_tab_id = -1

    return {
        "current_tab_id": current_tab_id,
        "tab_count": len(pages),
        "tabs": tabs,
    }


async def get_tab_context_async(page: Any | None) -> dict[str, Any] | None:
    """Return structured tab context from inside an active async Playwright loop."""
    if page is None:
        return None

    try:
        pages = list(page.context.pages)
    except Exception:
        return None

    tabs: list[dict[str, Any]] = []
    for idx, tab_page in enumerate(pages):
        try:
            title = await tab_page.title()
        except Exception:
            title = "unknown"
        try:
            url = tab_page.url
        except Exception:
            url = "unknown"
        tabs.append({"tab_id": idx, "title": title, "url": url})

    try:
        current_tab_id = pages.index(page)
    except ValueError:
        current_tab_id = -1

    return {
        "current_tab_id": current_tab_id,
        "tab_count": len(pages),
        "tabs": tabs,
    }


def format_tab_context(ctx: dict[str, Any] | None) -> str:
    """Format tab context for model prompts."""
    if ctx is None:
        return "Tab Context:\n- (unavailable)"

    lines = [
        "Tab Context:",
        f"- current_tab_id: {ctx['current_tab_id']}",
        f"- tab_count: {ctx['tab_count']}",
        "- available_tabs:",
    ]
    for tab in ctx["tabs"]:
        lines.append(
            f"  - tab_id: {tab['tab_id']}, title: {tab['title']}, url: {tab['url']}"
        )
    if not ctx["tabs"]:
        lines.append("  - (none)")
    return "\n".join(lines)


async def add_tab_context_to_info_async(
    info: dict[str, Any] | None,
    page: Any | None,
) -> dict[str, Any]:
    """Return info with tab_context added from inside an async Playwright loop."""
    merged = dict(info or {})
    ctx = await get_tab_context_async(page)
    if ctx is not None:
        merged["tab_context"] = ctx
    return merged


def add_tab_context_to_info(
    info: dict[str, Any] | None,
    page: Any | None,
    loop: Any | None,
) -> dict[str, Any]:
    """Return info with tab_context added when Playwright handles are available."""
    merged = dict(info or {})
    ctx = get_tab_context(page, loop)
    if ctx is not None:
        merged["tab_context"] = ctx
    return merged


def append_traj(
    task_dir: Path,
    entry: dict[str, Any],
    *,
    page: Any | None = None,
    loop: Any | None = None,
) -> None:
    """Append one JSON line to traj.jsonl, enriching info with tab context."""
    record = dict(entry)
    record["info"] = add_tab_context_to_info(record.get("info"), page, loop)
    with open(task_dir / "traj.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str))
        handle.write("\n")


async def append_traj_async(
    task_dir: Path,
    entry: dict[str, Any],
    *,
    page: Any | None = None,
) -> None:
    """Append one JSON line to traj.jsonl from inside an active async loop."""
    record = dict(entry)
    record["info"] = await add_tab_context_to_info_async(record.get("info"), page)
    with open(task_dir / "traj.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str))
        handle.write("\n")
