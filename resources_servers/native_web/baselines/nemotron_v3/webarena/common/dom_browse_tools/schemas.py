"""OpenAI-compatible schemas for DOM browser tools."""

from __future__ import annotations

from typing import Any


BROWSER_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate to a URL, or go forward/back in browser history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": 'URL to navigate to. Use "back" or "forward" for browser history.',
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Tab ID to navigate. Defaults to the current tab.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": (
                "Find currently visible/interactable DOM elements by searching the latest DOM "
                "representation. Returns matching lines with [index] refs for click, fill_form, "
                "or other DOM-indexed actions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text, role, tag, id, placeholder, label, or attribute to match.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 20,
                        "description": "Maximum matching DOM lines to return.",
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Tab ID to search in. Defaults to the current tab.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_form",
            "description": "Set the value of a form element by DOM ref index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "DOM ref index from the latest visible DOM, for example [12].",
                    },
                    "value": {
                        "anyOf": [{"type": "string"}, {"type": "boolean"}],
                        "description": "Value to set. Use booleans for checkboxes/switches.",
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Tab ID to operate in. Defaults to the current tab.",
                    },
                },
                "required": ["index", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an element by DOM ref index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "DOM ref index from the latest visible DOM, for example [12].",
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "middle", "right"],
                        "default": "left",
                        "description": "Mouse button to click.",
                    },
                    "clicks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "default": 1,
                        "description": "Number of clicks to perform.",
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Tab ID to operate in. Defaults to the current tab.",
                    },
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": (
                "Scroll with the original pyautogui-style mouse wheel behavior. "
                "Use a DOM ref index to scroll at an element, or null to scroll at screen center."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}],
                        "description": (
                            "Use null to scroll at screen center. Use a DOM ref index from the "
                            "latest visible DOM to move the mouse to that element before scrolling."
                        ),
                    },
                    "scroll_parameters": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "scroll_direction": {
                                        "type": "string",
                                        "enum": ["up", "down", "left", "right"],
                                        "default": "down",
                                    },
                                    "scroll_amount": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "default": 1,
                                        "description": "Number of 500px scroll units.",
                                    },
                                },
                                "required": ["scroll_direction", "scroll_amount"],
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Tab ID to operate in. Defaults to the current tab.",
                    },
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tabs_create",
            "description": "Create a new tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "default": "about:blank",
                        "description": "Start URL for the new tab.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tabs_focus",
            "description": "Focus an existing tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab_id": {"type": "integer", "description": "Tab ID to focus."},
                },
                "required": ["tab_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for the page to load, update, or settle before observing again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 30,
                        "default": 5,
                        "description": "Seconds to wait. Values are capped at 30 seconds.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "Terminate the current task and report its completion status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                        "description": "Task completion status.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "Final answer, or a brief failure explanation.",
                    },
                },
                "required": ["status"],
            },
        },
    },
]


TEXT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_page_text",
            "description": (
                "Extract the entire current page, including non-visible/off-viewport content, "
                "as readable Markdown/text and return one roughly equal part."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Zero-based part index to read.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 500,
                        "default": 6000,
                        "description": "Approximate maximum characters to return for one part.",
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Tab ID to read from. Defaults to the current tab.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_page_text",
            "description": (
                "Search the entire extracted page text, including non-visible/off-viewport "
                "content, and return matching text blocks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for."},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "description": "Maximum matching blocks to return.",
                    },
                    "target_chars": {
                        "type": "integer",
                        "minimum": 200,
                        "default": 1400,
                        "description": "Approximate characters per search block.",
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Tab ID to search in. Defaults to the current tab.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


TOOL_DEFINITIONS = BROWSER_TOOL_DEFINITIONS + TEXT_TOOL_DEFINITIONS

__all__ = ["BROWSER_TOOL_DEFINITIONS", "TEXT_TOOL_DEFINITIONS", "TOOL_DEFINITIONS"]
