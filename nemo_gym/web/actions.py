# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe parsing for BrowserGym and legacy WebVoyager model actions."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Literal
from urllib.parse import urlparse

from nemo_gym.web.models import WebAction, WebActionProfile


ALLOWED_ACTIONS = frozenset(
    {
        "clear",
        "click",
        "dblclick",
        "drag_and_drop",
        "fill",
        "focus",
        "go_back",
        "go_forward",
        "goto",
        "hover",
        "keyboard_press",
        "new_tab",
        "noop",
        "press",
        "report_infeasible",
        "scroll",
        "select_option",
        "send_msg_to_user",
        "tab_close",
        "tab_focus",
        "upload_file",
    }
)
TERMINAL_ACTIONS = frozenset({"send_msg_to_user", "report_infeasible"})
NATIVE_TOOL_NAMES = frozenset({"computer", "navigate", "tabs_create", "tabs_focus", "terminate"})
NATIVE_COMPUTER_ACTIONS = frozenset(
    {
        "double_click",
        "key_press",
        "left_click",
        "left_click_drag",
        "middle_click",
        "mouse_move",
        "right_click",
        "scroll",
        "triple_click",
        "type",
        "wait",
    }
)
NATIVE_CLICK_ACTIONS = frozenset(
    {
        "double_click",
        "left_click",
        "middle_click",
        "right_click",
        "triple_click",
    }
)
NativeActionRecovery = Literal["strict", "decode_string", "repair_single_closing_bracket"]


class ActionParseError(ValueError):
    """Raised when model output does not contain a safe supported action."""


def _strip_model_scaffolding(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip()

    action_match = re.search(r"(?:^|\n)\s*Action\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if action_match:
        return action_match.group(1).strip()
    return text.strip()


def _first_legacy_action_section(text: str) -> str:
    """Return the first labelled action, matching upstream WebVoyager.

    The original Selenium loop splits on ``Thought:``, ``Action:``, and
    ``Observation:`` and executes the section immediately following the first
    ``Action:``. Some reasoning models continue by hallucinating later
    Thought/Action turns in one completion. Treating the remainder as one
    Python expression rejects an otherwise valid first legacy action.
    """

    action_match = re.search(
        r"(?:^|\n)\s*Action\s*:\s*(.*?)(?=\n\s*(?:Thought|Action|Observation)\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if action_match:
        return action_match.group(1).strip()
    return _strip_model_scaffolding(text)


def _literal(value: ast.AST) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ActionParseError("action arguments must be Python literals") from exc


def parse_browsergym_action(text: str, *, max_calls: int = 2) -> WebAction:
    candidate = _strip_model_scaffolding(text)
    if not candidate:
        raise ActionParseError("model output did not contain an action")
    try:
        tree = ast.parse(candidate, mode="exec")
    except SyntaxError as exc:
        raise ActionParseError(f"invalid action syntax: {exc.msg}") from exc

    if not 1 <= len(tree.body) <= max_calls:
        raise ActionParseError(f"expected between 1 and {max_calls} action calls")

    calls: list[tuple[str, list[Any], dict[str, Any], ast.Call]] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ActionParseError("each action must be a direct function call")
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id not in ALLOWED_ACTIONS:
            raise ActionParseError("unsupported browser action")
        if any(keyword.arg is None for keyword in call.keywords):
            raise ActionParseError("expanded keyword arguments are not allowed")
        args = [_literal(arg) for arg in call.args]
        kwargs = {keyword.arg: _literal(keyword.value) for keyword in call.keywords if keyword.arg is not None}
        calls.append((call.func.id, args, kwargs, call))

    names = [name for name, *_ in calls]
    terminal_names = [name for name in names if name in TERMINAL_ACTIONS]
    if terminal_names and names[-1] not in TERMINAL_ACTIONS:
        raise ActionParseError("a terminal action must be the final call")

    answer = None
    if terminal_names:
        _, args, kwargs, _ = calls[-1]
        value = args[0] if args else kwargs.get("text", kwargs.get("reason"))
        answer = "" if value is None else str(value)

    script = "\n".join(ast.unparse(call) for *_, call in calls)
    arguments: dict[str, Any]
    if len(calls) == 1:
        _, args, kwargs, _ = calls[0]
        arguments = {"args": args, "kwargs": kwargs}
    else:
        arguments = {"calls": [{"name": name, "args": args, "kwargs": kwargs} for name, args, kwargs, _ in calls]}
    return WebAction(
        name=names[0] if len(names) == 1 else "multi_action",
        script=script,
        arguments=arguments,
        terminal=bool(terminal_names),
        answer=answer,
        raw_model_output=text,
    )


def _legacy_webvoyager_action(text: str) -> WebAction:
    candidate = _first_legacy_action_section(text).strip().rstrip(".")

    answer_match = re.fullmatch(r"ANSWER\s*[;:]?\s*\[?(.*?)\]?", candidate, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        return parse_browsergym_action(f"send_msg_to_user({answer!r})")

    click_match = re.fullmatch(r"Click\s*\[([^\]]+)\]", candidate, flags=re.IGNORECASE)
    if click_match:
        return parse_browsergym_action(f"click({click_match.group(1).strip()!r})")

    type_match = re.fullmatch(
        r"Type\s*\[([^\]]+)\]\s*;\s*\[?(.*?)\]?",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if type_match:
        bid, value = type_match.group(1).strip(), type_match.group(2).strip()
        return parse_browsergym_action(f"fill({bid!r}, {value!r})\nkeyboard_press('Enter')")

    scroll_match = re.fullmatch(r"Scroll(?:\s*\[[^\]]+\])?\s*;?\s*(up|down)", candidate, flags=re.IGNORECASE)
    if scroll_match:
        dy = -500 if scroll_match.group(1).lower() == "up" else 500
        return parse_browsergym_action(f"scroll(0, {dy})")

    if re.fullmatch(r"Wait", candidate, flags=re.IGNORECASE):
        return parse_browsergym_action("noop()")
    if re.fullmatch(r"GoBack", candidate, flags=re.IGNORECASE):
        return parse_browsergym_action("go_back()")
    if re.fullmatch(r"Google", candidate, flags=re.IGNORECASE):
        return parse_browsergym_action("goto('https://www.google.com/')")

    return parse_browsergym_action(text)


def _json_container_balance(value: str) -> tuple[int, int, bool]:
    """Return square/curly balance while respecting JSON string literals."""

    square = 0
    curly = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            square += 1
        elif character == "]":
            square -= 1
        elif character == "{":
            curly += 1
        elif character == "}":
            curly -= 1
        if square < 0 or curly < 0:
            return square, curly, in_string
    return square, curly, in_string


def _decode_native_actions(value: Any, recovery: NativeActionRecovery) -> tuple[Any, str]:
    if not isinstance(value, str):
        return value, "strict"
    if recovery == "strict":
        return value, "strict"
    try:
        return json.loads(value), "decoded_inner_string"
    except json.JSONDecodeError as first_error:
        if recovery != "repair_single_closing_bracket":
            raise ActionParseError("native computer actions string is invalid JSON") from first_error
        square, curly, in_string = _json_container_balance(value)
        if not value.lstrip().startswith("[") or square != 1 or curly != 0 or in_string:
            raise ActionParseError(
                "native computer actions string is not eligible for one-bracket recovery"
            ) from first_error
        try:
            return json.loads(value + "]"), "closed_one_missing_bracket"
        except json.JSONDecodeError as recovery_error:
            raise ActionParseError("native computer actions string remains invalid after recovery") from recovery_error


def _native_number(value: Any, *, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionParseError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ActionParseError(f"{field} must be in [{minimum:g}, {maximum:g}]")
    return number


def _native_coordinate(value: Any, *, field: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ActionParseError(f"{field} must contain normalized x and y")
    _native_number(value[0], field=f"{field}[0]", minimum=0, maximum=1)
    _native_number(value[1], field=f"{field}[1]", minimum=0, maximum=1)


def _validate_native_computer_action(action: Any, index: int) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ActionParseError(f"native computer action[{index}] must be an object")
    name = action.get("action")
    if name not in NATIVE_COMPUTER_ACTIONS:
        raise ActionParseError(f"unsupported native computer action[{index}]: {name!r}")
    prefix = f"native computer action[{index}] ({name})"
    if name in NATIVE_CLICK_ACTIONS or name == "mouse_move":
        _native_coordinate(action.get("coordinate"), field=f"{prefix}.coordinate")
    elif name == "left_click_drag":
        _native_coordinate(action.get("start_coordinate"), field=f"{prefix}.start_coordinate")
        _native_coordinate(action.get("coordinate"), field=f"{prefix}.coordinate")
    elif name == "type":
        if not isinstance(action.get("text"), str):
            raise ActionParseError(f"{prefix}.text must be a string")
    elif name == "key_press":
        keys = action.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key for key in keys):
            raise ActionParseError(f"{prefix}.keys must be a non-empty string list")
    elif name == "wait":
        _native_number(action.get("duration"), field=f"{prefix}.duration", minimum=0, maximum=30)
    elif name == "scroll":
        coordinate = action.get("coordinate")
        if coordinate is not None:
            _native_coordinate(coordinate, field=f"{prefix}.coordinate")
        parameters = action.get("scroll_parameters")
        if not isinstance(parameters, dict):
            raise ActionParseError(f"{prefix}.scroll_parameters must be an object")
        direction = parameters.get("scroll_direction")
        if direction not in {"up", "down", "left", "right"}:
            raise ActionParseError(f"{prefix}.scroll_direction is unsupported")
        amount = parameters.get("scroll_amount")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ActionParseError(f"{prefix}.scroll_amount must be a non-negative integer")
    return dict(action)


def _validate_native_tool_arguments(
    name: str,
    arguments: dict[str, Any],
    *,
    recovery: NativeActionRecovery,
    max_computer_actions: int,
) -> tuple[dict[str, Any], str, int]:
    normalized = dict(arguments)
    if name == "computer":
        actions, recovery_mode = _decode_native_actions(arguments.get("actions"), recovery)
        if not isinstance(actions, list) or not actions:
            raise ActionParseError("native computer tool requires a non-empty actions list")
        if len(actions) > max_computer_actions:
            raise ActionParseError(f"native computer tool exceeded the {max_computer_actions}-action batch limit")
        normalized["actions"] = [
            _validate_native_computer_action(action, index) for index, action in enumerate(actions)
        ]
        return normalized, recovery_mode, len(actions)
    if name == "navigate":
        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            raise ActionParseError("native navigate.url must be a non-empty string")
        if url not in {"back", "forward"} and urlparse(url).scheme not in {"http", "https"}:
            raise ActionParseError("native navigate.url must use http(s), back, or forward")
        tab_id = arguments.get("tab_id")
        if tab_id is not None and (isinstance(tab_id, bool) or not isinstance(tab_id, int) or tab_id < 0):
            raise ActionParseError("native navigate.tab_id must be a non-negative integer or null")
    elif name == "tabs_create":
        url = arguments.get("url", "about:blank")
        if not isinstance(url, str) or (url != "about:blank" and urlparse(url).scheme not in {"http", "https"}):
            raise ActionParseError("native tabs_create.url must be about:blank or use http(s)")
    elif name == "tabs_focus":
        tab_id = arguments.get("tab_id")
        if isinstance(tab_id, bool) or not isinstance(tab_id, int) or tab_id < 0:
            raise ActionParseError("native tabs_focus.tab_id must be a non-negative integer")
    elif name == "terminate":
        if arguments.get("status") not in {"success", "failure"}:
            raise ActionParseError("native terminate.status must be success or failure")
        answer = arguments.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ActionParseError("native terminate.answer must be a string or null")
    return normalized, "strict", 0


def parse_native_tool_calls(
    items: list[Any],
    *,
    max_calls: int = 8,
    max_computer_actions: int = 20,
    recovery: NativeActionRecovery = "strict",
) -> WebAction:
    """Validate native Nano Omni function calls without executing arbitrary code."""

    calls: list[dict[str, Any]] = []
    parse_records: list[dict[str, Any]] = []
    for item in items:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "function_call":
            continue
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        raw_arguments = item.get("arguments") if isinstance(item, dict) else getattr(item, "arguments", None)
        call_id = item.get("call_id") if isinstance(item, dict) else getattr(item, "call_id", None)
        if name not in NATIVE_TOOL_NAMES:
            raise ActionParseError(f"unsupported native browser tool: {name!r}")
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"invalid JSON arguments for native tool {name!r}") from exc
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ActionParseError(f"native tool {name!r} arguments must be an object")
        arguments, recovery_mode, computer_actions = _validate_native_tool_arguments(
            name,
            arguments,
            recovery=recovery,
            max_computer_actions=max_computer_actions,
        )
        calls.append({"id": call_id, "name": name, "arguments": arguments})
        parse_records.append(
            {
                "call_id": call_id,
                "tool": name,
                "recovery_mode": recovery_mode,
                "computer_actions": computer_actions,
            }
        )

    if not calls:
        raise ActionParseError("model response did not contain a native function call")
    if len(calls) > max_calls:
        raise ActionParseError(f"native response exceeded the {max_calls}-call limit")
    terminal_indices = [index for index, call in enumerate(calls) if call["name"] == "terminate"]
    if terminal_indices and terminal_indices != [len(calls) - 1]:
        raise ActionParseError("native terminate must be the final tool call")

    terminal = bool(terminal_indices)
    terminal_args = calls[-1]["arguments"] if terminal else {}
    answer = terminal_args.get("answer") if terminal else None
    return WebAction(
        name=calls[0]["name"] if len(calls) == 1 else "native_tool_calls",
        script="",
        arguments={"calls": calls},
        terminal=terminal,
        answer=None if answer is None else str(answer),
        raw_model_output=json.dumps(calls, ensure_ascii=False),
        metadata={
            "native_parse": {
                "calls": parse_records,
                "recovered": any(record["recovery_mode"] != "strict" for record in parse_records),
            }
        },
    )


def parse_model_action(text: str, profile: WebActionProfile | str) -> WebAction:
    profile = WebActionProfile(profile)
    if profile == WebActionProfile.NATIVE_TOOLCALL:
        raise ActionParseError("native tool-call actions must be parsed from structured response output")
    if profile == WebActionProfile.WEBVOYAGER_LEGACY:
        action = _legacy_webvoyager_action(text)
        return action.model_copy(update={"raw_model_output": text})
    return parse_browsergym_action(text)
