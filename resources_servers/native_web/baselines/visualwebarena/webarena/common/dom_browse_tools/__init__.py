"""Shared DOM browsing tools for WebArena agents."""

from .browser_use import (
    browser_use_click,
    browser_use_element_center,
    browser_use_fill_form,
    browser_use_read_page,
    find_browser_use_dom_lines,
    normalize_index,
    normalize_url,
)
from .computer import page_scroll
from .schemas import BROWSER_TOOL_DEFINITIONS, TEXT_TOOL_DEFINITIONS, TOOL_DEFINITIONS
from .text import (
    PageTextDocument,
    get_page_text_document,
    read_page_text,
    search_page_text,
    split_markdown_parts,
)

__all__ = [
    "BROWSER_TOOL_DEFINITIONS",
    "PageTextDocument",
    "TEXT_TOOL_DEFINITIONS",
    "TOOL_DEFINITIONS",
    "browser_use_click",
    "browser_use_element_center",
    "browser_use_fill_form",
    "browser_use_read_page",
    "find_browser_use_dom_lines",
    "get_page_text_document",
    "normalize_index",
    "normalize_url",
    "page_scroll",
    "read_page_text",
    "search_page_text",
    "split_markdown_parts",
]
