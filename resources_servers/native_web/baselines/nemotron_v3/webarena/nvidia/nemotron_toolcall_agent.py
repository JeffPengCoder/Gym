"""
Nemotron tool-call agent for WebArena evaluation.

This variant uses the OpenAI/vLLM chat-completions tool-call interface instead
of asking the model to emit pyautogui code blocks. The backend chat template is
responsible for rendering tool definitions into the model prompt.
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.agent_captcha import maybe_solve_captcha
from common.cloudflare_handler import go_back, go_forward, goto
from common.pyautogui_utils import execute_action, take_screenshot
from common.tab_context import append_traj, format_tab_context, get_tab_context
from common.visualwebarena_task_images import (
    TASK_INPUT_IMAGE_MARKER,
    TASK_INPUT_IMAGE_REDACTION_NOTICE,
    load_task_input_image_parts,
)

_default_logger = logging.getLogger("nemotron_toolcall_eval.agent")
IMAGE_TOKEN_BUDGET = 2040
RESPONSE_TOKEN_HEADROOM = 4096
FALLBACK_CHARS_PER_TOKEN = 4


class FinishReasonLengthError(RuntimeError):
    """Raised when the API response is truncated before a usable tool call."""


SYSTEM_PROMPT = """
You are a GUI agent controlling a web browser. You are given a task instruction, a screenshot of the browser, and your previous interactions. You need to perform a series of actions to complete the task. The browser is already open and logged into the required websites.

<tool_guidelines>
- Operate via x,y coordinates from the latest screenshot using the `computer` tool.
- Coordinates are relative to the viewport in [0, 1], with (0, 0) at the top-left.
- Use `tabs_create` and `tabs_focus` to manage tabs.
- Use `navigate` to go to URLs or use "back"/"forward" for browser history.
- When the task is complete, call `terminate` with status and answer.
</tool_guidelines>
""".strip()


TOOL_DEFINITIONS: list[dict[str, Any]] = [
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
                        "description": 'The URL to navigate to. Use "forward" to go forward in history or "back" to go back in history.',
                    },
                    "tab_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "description": "Tab ID to navigate.",
                        "default": None,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer",
            "description": "Interact with the web browser with a sequence of computer actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "List of actions to perform sequentially.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "description": "The action to perform: `left_click`, `middle_click`, `right_click`, `double_click`, `triple_click`, `mouse_move` (coordinate), `type` (text), `key_press` (list of keys to press), `scroll` (direction + amount, optional coordinate), `left_click_drag` (start_coordinate to coordinate), or `wait` (duration in seconds).",
                                    "enum": [
                                        "left_click",
                                        "middle_click",
                                        "right_click",
                                        "double_click",
                                        "triple_click",
                                        "mouse_move",
                                        "type",
                                        "key_press",
                                        "wait",
                                        "scroll",
                                        "left_click_drag",
                                    ],
                                },
                                "coordinate": {
                                    "anyOf": [
                                        {
                                            "type": "array",
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "prefixItems": [
                                                {"type": "number", "minimum": 0, "maximum": 1},
                                                {"type": "number", "minimum": 0, "maximum": 1},
                                            ],
                                        },
                                        {"type": "null"},
                                    ],
                                    "default": None,
                                    "description": "(x, y) relative coordinates in the [0, 1] range, where (0, 0) is the top-left of the viewport and (1, 1) is the bottom-right. Required for click actions and `mouse_move`. For `scroll`, defaults to the screen center when omitted. For `left_click_drag`, this is the end position.",
                                },
                                "duration": {
                                    "anyOf": [{"type": "integer", "minimum": 0, "maximum": 30}, {"type": "null"}],
                                    "default": None,
                                    "description": "The number of seconds to wait. Required for `wait`. Maximum 30 seconds.",
                                },
                                "keys": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": 'List of keys to press for the `key_press` action. Use platform modifier keys such as "cmd" on Mac or "ctrl" on Windows/Linux, e.g., ["ctrl", "a"] for select all.',
                                },
                                "scroll_parameters": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "properties": {
                                                "scroll_amount": {
                                                    "type": "integer",
                                                    "minimum": 0,
                                                    "default": 1,
                                                    "description": "Number of mouse wheel clicks to scroll in the requested direction. This value is uncapped.",
                                                },
                                                "scroll_direction": {
                                                    "type": "string",
                                                    "enum": ["up", "down", "left", "right"],
                                                    "default": "down",
                                                    "description": "The direction to scroll in.",
                                                },
                                            },
                                            "required": ["scroll_direction", "scroll_amount"],
                                        },
                                        {"type": "null"},
                                    ],
                                    "default": None,
                                    "description": "The parameters to scroll with. Required for `scroll`.",
                                },
                                "start_coordinate": {
                                    "anyOf": [
                                        {
                                            "type": "array",
                                            "minItems": 2,
                                            "maxItems": 2,
                                            "prefixItems": [
                                                {"type": "number", "minimum": 0, "maximum": 1},
                                                {"type": "number", "minimum": 0, "maximum": 1},
                                            ],
                                        },
                                        {"type": "null"},
                                    ],
                                    "default": None,
                                    "description": "(x, y) relative starting coordinates in the [0, 1] range for `left_click_drag`.",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The text to type. Only used for the `type` action.",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                },
                "required": ["actions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tabs_create",
            "description": "Creates a new empty tab in the current tab group",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Start URL for new tab. Default about:blank.",
                        "default": "about:blank",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tabs_focus",
            "description": "Focus an existing tab in the current tab group.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab_id": {
                        "type": "integer",
                        "description": "Tab ID to focus.",
                    },
                },
                "required": ["tab_id"],
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
                        "description": "The status of the task.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer of the task.",
                    },
                },
                "required": ["status"],
            },
        },
    },
]


def _decode_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return json.loads(raw)
    raise TypeError(f"Unsupported tool arguments type: {type(raw).__name__}")


def _normalize_tool_call(tool_call: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    call = copy.deepcopy(tool_call)
    call.setdefault("id", fallback_id)
    call.setdefault("type", "function")
    fn = call.setdefault("function", {})
    if not isinstance(fn.get("arguments"), str):
        fn["arguments"] = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
    return call


def _shorten(value: Any, limit: int = 160) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_coord(coord: Any) -> str:
    if not isinstance(coord, list | tuple) or len(coord) != 2:
        return "(unknown)"
    try:
        return f"({float(coord[0]):.3f}, {float(coord[1]):.3f})"
    except (TypeError, ValueError):
        return f"({_shorten(coord[0], 20)}, {_shorten(coord[1], 20)})"


def _format_computer_action(action: dict[str, Any]) -> str:
    kind = action.get("action")
    coord = action.get("coordinate")

    if kind in {"left_click", "middle_click", "right_click", "double_click", "triple_click"}:
        label = kind.replace("_", " ")
        return f"{label} at {_format_coord(coord)}"
    if kind == "mouse_move":
        return f"move mouse to {_format_coord(coord)}"
    if kind == "type":
        return f'type "{_shorten(action.get("text", ""), 80)}"'
    if kind == "key_press":
        keys = [str(key) for key in action.get("keys", [])]
        return f"press {'+'.join(keys) if keys else '(no keys)'}"
    if kind == "wait":
        duration = action.get("duration")
        return f"wait {duration if duration is not None else 'default'} seconds"
    if kind == "scroll":
        params = action.get("scroll_parameters") or {}
        direction = params.get("scroll_direction", "down")
        amount = params.get("scroll_amount", 1)
        at_coord = f" at {_format_coord(coord)}" if coord else ""
        return f"scroll {direction} by {amount} mouse wheel clicks{at_coord}"
    if kind == "left_click_drag":
        return f"drag from {_format_coord(action.get('start_coordinate'))} to {_format_coord(coord)}"
    return f"{kind or 'unknown'} action"


def _format_tool_call_action(name: str, args: dict[str, Any]) -> str:
    if name == "computer":
        actions = args.get("actions") or []
        if not isinstance(actions, list):
            return "computer action with invalid actions payload"
        if not actions:
            return "computer action with no actions"
        return "; ".join(_format_computer_action(action) for action in actions)
    if name == "navigate":
        url = str(args.get("url", ""))
        tab_id = args.get("tab_id")
        tab_suffix = f" in tab {tab_id}" if tab_id is not None else ""
        if url.lower() in {"back", "forward"}:
            return f"go {url.lower()}{tab_suffix}"
        return f"navigate to {_shorten(url, 120)}{tab_suffix}"
    if name == "tabs_create":
        return f"create tab at {_shorten(args.get('url') or 'about:blank', 120)}"
    if name == "tabs_focus":
        return f"focus tab {args.get('tab_id')}"
    if name == "terminate":
        status = args.get("status", "failure")
        answer = args.get("answer")
        if answer is None:
            return f"terminate with status {status}"
        return f"terminate with status {status}; answer: {_shorten(answer, 120)}"
    return f"{name}({_shorten(args)})"


def _format_step_action(action_payload: list[dict[str, Any]]) -> str:
    if not action_payload:
        return "No action"
    parts = [
        _format_tool_call_action(str(call.get("name", "")), call.get("arguments") or {})
        for call in action_payload
    ]
    return "\n".join(parts)


class NemotronToolCallAgent:
    """Nemotron agent using vLLM/OpenAI tool calls for browser control."""

    def __init__(
        self,
        model: str,
        max_steps: int = 50,
        max_image_history: int = 3,
        max_tokens: int = 16384,
        top_p: float = 0.95,
        temperature: float = 1.0,
        thinking: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080,
        wait_seconds: float = 2.0,
        api_timeout: float = 1200.0,
        expanded_browser_tools: bool = True,
        max_model_len: int = 131072,
        tokenizer_model: str | None = None,
        logger_name: str | None = None,
        captcha_solver: Any | None = None,
    ) -> None:
        self.logger = logging.getLogger(logger_name) if logger_name else _default_logger
        self.model = model
        self.max_steps = max_steps
        self.max_image_history = max_image_history
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.thinking = thinking
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.wait_seconds = wait_seconds
        self.api_timeout = api_timeout
        self.expanded_browser_tools = expanded_browser_tools
        self.max_model_len = max_model_len
        self.tokenizer_model = tokenizer_model or model
        self.captcha_solver = captcha_solver

        self._messages: list[dict[str, Any]] = []
        self._page: Any | None = None
        self._loop: Any | None = None
        self._captcha_failures = 0
        self._last_captcha_failure_step: int | None = None
        self._tokenizer: Any | None = None
        self._tokenizer_load_attempted = False
        self._tokenizer_warning_logged = False
        self._task_input_image_parts: list[dict[str, Any]] = []

    def run(
        self,
        instruction: str,
        task_dir=None,
        *,
        page: Any | None = None,
        loop: Any | None = None,
        task_input_images: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        """Run the tool-call agent loop and save trajectory artifacts."""
        errors: list[str] = []
        final_status = "fail"
        final_answer = None
        self._page = page
        self._loop = loop
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._captcha_failures = 0
        self._last_captcha_failure_step = None
        self._task_input_image_parts = load_task_input_image_parts(
            task_input_images,
            mark_images=True,
        )

        if task_dir:
            task_dir = Path(task_dir)
            task_dir.mkdir(parents=True, exist_ok=True)

        if self._page is None or self._loop is None:
            errors.append("NemotronToolCallAgent requires Playwright page and loop for browser tools")
            return {"status": "error", "answer": None, "steps": 0, "errors": errors}

        self._maybe_solve_captcha("initial")
        try:
            obs = take_screenshot()
        except Exception as e:
            errors.append(f"Initial screenshot failed: {e}")
            self.logger.error(f"Initial screenshot failed: {e}")
            return {"status": "error", "answer": None, "steps": 0, "errors": errors}

        initial_ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        initial_screenshot = f"step_0_{initial_ts}.png"
        if task_dir:
            (task_dir / initial_screenshot).write_bytes(obs)
            append_traj(task_dir, {
                "step_num": 0,
                "action": None,
                "natural_language_action": "Initial state",
                "action_timestamp": initial_ts,
                "response": None,
                "reward": 0,
                "done": False,
                "info": {},
                "screenshot_file": initial_screenshot,
            }, page=self._page, loop=self._loop)

        step_idx = 0
        while step_idx < self.max_steps:
            step_num = step_idx + 1
            self.logger.info(f"===== Step {step_num}/{self.max_steps} =====")

            self._append_user_turn(instruction, step_num, obs)

            response = None
            assistant_message = None
            tool_calls = None
            truncated = False
            for api_attempt in range(3):
                try:
                    response = self._call_api(self._build_request_messages())
                    assistant_message, tool_calls = self._parse_assistant_message(response, step_num)
                    if not tool_calls:
                        if os.environ.get("NEMOTRON_DEBUG_EMPTY_TOOL_CALLS") == "1":
                            self.logger.warning(
                                "EMPTY_TOOL_CALLS_RAW_RESPONSE step=%s attempt=%s response=%s",
                                step_num,
                                api_attempt + 1,
                                json.dumps(response, ensure_ascii=False, default=str),
                            )
                        raise ValueError("Model returned no tool calls")
                    break
                except FinishReasonLengthError as e:
                    error_message = str(e)
                    if "decoder prompt" in error_message and "maximum model length" in error_message:
                        errors.append(f"API prompt too long at step {step_num}: {e}")
                        self.logger.warning("API prompt too long at step %d; marking task as fail", step_num)
                    else:
                        errors.append(f"API truncated at step {step_num}: {e}")
                        self.logger.warning("API truncated at step %d; marking task as fail", step_num)
                    final_status = "fail"
                    truncated = True
                    break
                except Exception as e:
                    errors.append(f"API/parse error at step {step_num} (attempt {api_attempt + 1}): {e}")
                    self.logger.warning(f"API/parse error (attempt {api_attempt + 1}): {e}")
                    time.sleep(1)

            if truncated:
                break

            if response is None or assistant_message is None or not tool_calls:
                final_status = "error"
                break

            self._messages.append(assistant_message)

            action_results: list[str] = []
            action_payload: list[dict[str, Any]] = []
            done = False
            try:
                for call_idx, tool_call in enumerate(tool_calls):
                    call_id = tool_call["id"]
                    name = tool_call["function"]["name"]
                    args = _decode_arguments(tool_call["function"].get("arguments"))
                    action_payload.append({
                        "id": call_id,
                        "name": name,
                        "arguments": args,
                    })

                    result = self._execute_tool(name, args)
                    action_results.append(result["content"])

                    if result["status"] in ("done", "fail"):
                        final_status = result["status"]
                        final_answer = result.get("answer")
                        done = True
                        break
                    if not self._maybe_solve_captcha(f"after {name}", step_num=step_num):
                        error = self._captcha_failure_error(step_num)
                        if error:
                            errors.append(error)
                            final_status = "error"
                            done = True
                            break
            except Exception as e:
                errors.append(f"Action error at step {step_num}: {e}")
                self.logger.error(f"Action error at step {step_num}: {e}", exc_info=True)
                final_status = "error"
                break

            parsed_action = _format_step_action(action_payload)
            self.logger.info("Action:\n%s", parsed_action)

            action_ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            screenshot_file = None
            if not done:
                time.sleep(self.wait_seconds)
                if not self._maybe_solve_captcha("before post-action screenshot", step_num=step_num):
                    error = self._captcha_failure_error(step_num)
                    if error:
                        errors.append(error)
                        final_status = "error"
                        done = True
                        step_idx += 1
                        break
                try:
                    obs = take_screenshot()
                except Exception as e:
                    errors.append(f"Screenshot failed after action at step {step_num}: {e}")
                    final_status = "error"
                    break
                screenshot_file = f"step_{step_num}_{action_ts}.png"
                if task_dir:
                    (task_dir / screenshot_file).write_bytes(obs)

            if task_dir:
                append_traj(task_dir, {
                    "step_num": step_num,
                    "action": action_payload,
                    "natural_language_action": self._assistant_text(assistant_message),
                    "action_timestamp": action_ts,
                    "response": response,
                    "reward": 0,
                    "done": done,
                    "info": {
                        "parsed_action": parsed_action,
                        "tool_results": action_results,
                        "status": final_status if done else None,
                        "answer": final_answer if done else None,
                    },
                    "screenshot_file": screenshot_file,
                }, page=self._page, loop=self._loop)

            step_idx += 1
            if done:
                break

        return {
            "status": final_status,
            "answer": final_answer,
            "steps": step_idx,
            "errors": errors,
        }

    def _call_api(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        api_key = os.environ.get("VLLM_API_KEY", "")
        if not api_key:
            raise RuntimeError("VLLM_API_KEY environment variable not set")
        endpoint = os.environ.get("VLLM_API_ENDPOINT", "")
        if not endpoint:
            raise RuntimeError("VLLM_API_ENDPOINT environment variable not set")
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "chat_template_kwargs": {
                "truncate_history_thinking": False,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        for attempt in range(20):
            try:
                with httpx.Client(timeout=self.api_timeout, verify=False) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)
                if resp.status_code != 200:
                    self.logger.error(f"API returned {resp.status_code} (attempt {attempt + 1}): {resp.text}")
                    if resp.status_code == 400 and (
                        "decoder prompt" in resp.text
                        and "maximum model length" in resp.text
                    ):
                        raise FinishReasonLengthError(resp.text)
                    time.sleep(5)
                    continue

                data = resp.json()
                choice = data["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason in ("stop", "tool_calls"):
                    return choice["message"]
                if finish_reason == "length":
                    self.logger.warning(
                        "API finish_reason='length' (attempt %d); marking task as fail",
                        attempt + 1,
                    )
                    raise FinishReasonLengthError("API finish_reason='length'")
                self.logger.warning(f"API finish_reason={finish_reason!r} (attempt {attempt + 1}), retrying")
                time.sleep(5)
            except FinishReasonLengthError:
                raise
            except RuntimeError as e:
                self.logger.error(f"API call error (attempt {attempt + 1}): {e}")
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"API call error (attempt {attempt + 1}): {e}")
                time.sleep(5)

        raise RuntimeError("vLLM API max retries exceeded")

    def _append_user_turn(self, instruction: str, step_num: int, screenshot: bytes) -> None:
        text_parts = []
        if step_num == 1 or self._task_input_image_parts:
            text_parts.append(f"# Task Instruction:\n\n{instruction}")
        if step_num > 1 and self._task_input_image_parts:
            text_parts.append(TASK_INPUT_IMAGE_REDACTION_NOTICE)
        text_parts.append(f"You are currently on Step {step_num}.")
        text_parts.append(format_tab_context(get_tab_context(self._page, self._loop)))
        screenshot_b64 = base64.b64encode(screenshot).decode()
        if self._task_input_image_parts and step_num == 1:
            instruction_text = f"# Task Instruction:\n\n{instruction}"
            context_text = "\n\n".join(text_parts[1:])
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": instruction_text},
                *copy.deepcopy(self._task_input_image_parts),
                {"type": "text", "text": context_text},
            ]
        else:
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": "\n\n".join(text_parts)},
            ]
        self._messages.append({
            "role": "user",
            "content": content,
        })

    def _build_request_messages(self) -> list[dict[str, Any]]:
        messages = copy.deepcopy(self._messages)
        if self.max_model_len > 0:
            messages = self._compact_messages_to_text_budget(messages)
        else:
            self.logger.info("Request text token budgeting disabled (max_model_len=%s)", self.max_model_len)
        self._keep_recent_images(messages)
        if self.max_model_len > 0:
            self._log_request_token_budget(messages)
        return self._strip_internal_part_metadata(messages)

    def _compact_messages_to_text_budget(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages or self.max_model_len <= 0:
            return messages

        system_count = 1 if messages[0].get("role") == "system" else 0
        final_user_idx = next(
            (idx for idx in range(len(messages) - 1, system_count - 1, -1) if messages[idx].get("role") == "user"),
            None,
        )
        if final_user_idx is None:
            return messages

        prefix = messages[:system_count]
        final_turn = messages[final_user_idx:]
        groups = self._prior_turn_groups(messages[system_count:final_user_idx])
        dropped_turns = 0
        compacted = prefix + [msg for group in groups for msg in group] + final_turn
        text_budget = self._available_request_text_budget()

        while groups and self._count_text_tokens(self._strip_images(compacted)) > text_budget:
            groups.pop(0)
            dropped_turns += 1
            compacted = prefix + [msg for group in groups for msg in group] + final_turn

        if dropped_turns:
            candidate = copy.deepcopy(compacted)
            self._prepend_redaction_notice(candidate, dropped_turns)
            while groups and self._count_text_tokens(self._strip_images(candidate)) > text_budget:
                groups.pop(0)
                dropped_turns += 1
                compacted = prefix + [msg for group in groups for msg in group] + final_turn
                candidate = copy.deepcopy(compacted)
                self._prepend_redaction_notice(candidate, dropped_turns)
            compacted = candidate

        final_text_tokens = self._count_text_tokens(self._strip_images(compacted))
        if final_text_tokens > text_budget:
            self.logger.warning(
                "Minimum request text is over budget: %s tokens > %s text budget "
                "(max_model_len=%s, image_budget=%s, response_headroom=%s)",
                final_text_tokens,
                text_budget,
                self.max_model_len,
                self.max_image_history * IMAGE_TOKEN_BUDGET,
                RESPONSE_TOKEN_HEADROOM,
            )
        elif dropped_turns:
            self.logger.info(
                "Redacted %s prior turn(s) to fit context text budget (%s/%s tokens)",
                dropped_turns,
                final_text_tokens,
                text_budget,
            )

        return compacted

    @staticmethod
    def _prior_turn_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        idx = 0
        while idx < len(messages):
            group = [messages[idx]]
            if (
                messages[idx].get("role") == "user"
                and idx + 1 < len(messages)
                and messages[idx + 1].get("role") == "assistant"
            ):
                group.append(messages[idx + 1])
                idx += 2
            else:
                idx += 1
            groups.append(group)
        return groups

    def _available_request_text_budget(self) -> int:
        image_budget = max(0, self.max_image_history) * IMAGE_TOKEN_BUDGET
        return max(0, self.max_model_len - image_budget - RESPONSE_TOKEN_HEADROOM)

    def _keep_recent_images(self, messages: list[dict[str, Any]]) -> None:
        image_user_indices = [
            idx for idx, msg in enumerate(messages)
            if msg.get("role") == "user" and any(
                self._is_browser_image_part(part)
                for part in msg.get("content", [])
            )
        ]
        keep_count = max(0, self.max_image_history)
        keep = set(image_user_indices[-keep_count:]) if keep_count else set()
        for idx in image_user_indices:
            if idx in keep:
                continue
            messages[idx] = self._message_without_browser_images(messages[idx])

    def _log_request_token_budget(self, messages: list[dict[str, Any]]) -> None:
        text_tokens = self._count_text_tokens(self._strip_images(messages))
        image_count = self._image_count(messages)
        self.logger.info(
            "Request text tokens: %s/%s (images=%s, image_budget=%s, max_model_len=%s, response_headroom=%s)",
            text_tokens,
            self._available_request_text_budget(),
            image_count,
            image_count * IMAGE_TOKEN_BUDGET,
            self.max_model_len,
            RESPONSE_TOKEN_HEADROOM,
        )

    @staticmethod
    def _image_count(messages: list[dict[str, Any]]) -> int:
        return sum(
            1
            for message in messages
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
            if isinstance(part, dict) and part.get("type") == "image_url"
        )

    def _strip_images(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._message_without_images(message) for message in messages]

    @staticmethod
    def _message_without_images(message: dict[str, Any]) -> dict[str, Any]:
        msg = copy.deepcopy(message)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                part for part in content
                if not (isinstance(part, dict) and part.get("type") == "image_url")
            ]
        return msg

    @staticmethod
    def _is_task_input_image_part(part: Any) -> bool:
        return (
            isinstance(part, dict)
            and part.get("type") == "image_url"
            and bool(part.get(TASK_INPUT_IMAGE_MARKER))
        )

    @classmethod
    def _is_browser_image_part(cls, part: Any) -> bool:
        return (
            isinstance(part, dict)
            and part.get("type") == "image_url"
            and not cls._is_task_input_image_part(part)
        )

    @classmethod
    def _message_without_browser_images(cls, message: dict[str, Any]) -> dict[str, Any]:
        msg = copy.deepcopy(message)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                part for part in content
                if not cls._is_browser_image_part(part)
            ]
        return msg

    @staticmethod
    def _strip_internal_part_metadata(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = copy.deepcopy(messages)
        for message in cleaned:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    part.pop(TASK_INPUT_IMAGE_MARKER, None)
        return cleaned

    def _prepend_redaction_notice(self, messages: list[dict[str, Any]], dropped_turns: int) -> None:
        if dropped_turns == 1:
            notice = "Earlier interaction step 1 was redacted to fit the context window."
        else:
            notice = f"Earlier interaction steps 1-{dropped_turns} were redacted to fit the context window."
        for message in messages:
            if message.get("role") == "user":
                self._prepend_text_to_user_message(message, notice + "\n\n")
                return

    @staticmethod
    def _prepend_text_to_user_message(message: dict[str, Any], prefix: str) -> None:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = prefix + str(part.get("text", ""))
                    return
            content.append({"type": "text", "text": prefix.rstrip()})
        elif isinstance(content, str):
            message["content"] = prefix + content
        else:
            message["content"] = prefix.rstrip()

    def _count_text_tokens(self, messages: list[dict[str, Any]]) -> int:
        tokenizer = self._get_tokenizer()
        text_messages = self._messages_for_tokenizer(messages)
        if tokenizer is not None:
            try:
                tokens = tokenizer.apply_chat_template(
                    text_messages,
                    tools=TOOL_DEFINITIONS,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                return len(tokens["input_ids"])
            except TypeError:
                try:
                    tokens = tokenizer.apply_chat_template(
                        text_messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    return len(tokens["input_ids"])
                except Exception as exc:
                    self._log_tokenizer_fallback(exc)
            except Exception as exc:
                self._log_tokenizer_fallback(exc)

        return self._fallback_token_count(text_messages)

    def _get_tokenizer(self) -> Any | None:
        if self._tokenizer_load_attempted:
            return self._tokenizer
        self._tokenizer_load_attempted = True
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, trust_remote_code=True)
        except Exception as exc:
            self._log_tokenizer_fallback(exc)
            self._tokenizer = None
        return self._tokenizer

    def _log_tokenizer_fallback(self, exc: Exception) -> None:
        if self._tokenizer_warning_logged:
            return
        self._tokenizer_warning_logged = True
        self.logger.warning("Falling back to rough token counting: %s", exc)

    def _fallback_token_count(self, messages: list[dict[str, Any]]) -> int:
        rendered = json.dumps(
            {"messages": messages, "tools": TOOL_DEFINITIONS},
            ensure_ascii=False,
            default=str,
        )
        return max(1, (len(rendered) + FALLBACK_CHARS_PER_TOKEN - 1) // FALLBACK_CHARS_PER_TOKEN)

    def _messages_for_tokenizer(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text_messages: list[dict[str, Any]] = []
        for message in messages:
            msg = {
                key: copy.deepcopy(value)
                for key, value in message.items()
                if key not in {"content", "tool_calls", "reasoning_content"}
            }
            content = self._message_text_content(message)
            if message.get("reasoning_content"):
                content = "\n".join([content, str(message["reasoning_content"])]).strip()
            msg["content"] = content
            if message.get("tool_calls"):
                msg["tool_calls"] = copy.deepcopy(message["tool_calls"])
            text_messages.append(msg)
        return text_messages

    @staticmethod
    def _message_text_content(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    def _parse_assistant_message(
        self,
        response: dict[str, Any],
        step_num: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raw_calls = response.get("tool_calls") or []
        tool_calls = [
            _normalize_tool_call(call, f"call_{step_num}_{idx}")
            for idx, call in enumerate(raw_calls)
        ]
        message = {
            "role": "assistant",
            "content": response.get("content") or "",
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if response.get("reasoning_content"):
            message["reasoning_content"] = response["reasoning_content"]
        return message, tool_calls

    def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "computer":
            actions = args.get("actions") or []
            if not isinstance(actions, list):
                raise ValueError("computer.actions must be a list")
            for action in actions:
                self._execute_computer_action(action)
            return {"status": "action", "content": "Computer actions executed."}

        if name == "navigate":
            result = self._navigate(str(args.get("url", "")), args.get("tab_id"))
            return {"status": "action", "content": result}

        if name == "tabs_create":
            result = self._tabs_create(str(args.get("url") or "about:blank"))
            return {"status": "action", "content": result}

        if name == "tabs_focus":
            result = self._tabs_focus(int(args["tab_id"]))
            return {"status": "action", "content": result}

        if name == "terminate":
            status = args.get("status", "failure")
            answer = args.get("answer")
            return {
                "status": "done" if status == "success" else "fail",
                "answer": answer,
                "content": f"Terminated with status={status}, answer={answer}",
            }

        raise ValueError(f"Unsupported tool call: {name}")

    def _execute_computer_action(self, action: dict[str, Any]) -> None:
        kind = action.get("action")
        coord = action.get("coordinate")

        if kind in {"left_click", "middle_click", "right_click", "double_click", "triple_click"}:
            x, y = self._relative_xy(coord)
            button = {
                "left_click": "left",
                "middle_click": "middle",
                "right_click": "right",
                "double_click": "left",
                "triple_click": "left",
            }[kind]
            clicks = 2 if kind == "double_click" else 3 if kind == "triple_click" else 1
            execute_action({"type": "click", "x": x, "y": y, "button": button, "clicks": clicks}, self.screen_width, self.screen_height)
        elif kind == "mouse_move":
            x, y = self._relative_xy(coord)
            execute_action({"type": "move", "x": x, "y": y}, self.screen_width, self.screen_height)
        elif kind == "type":
            execute_action({"type": "typewrite", "text": str(action.get("text", ""))}, self.screen_width, self.screen_height)
        elif kind == "key_press":
            keys = [str(k) for k in action.get("keys", [])]
            if not keys:
                raise ValueError("key_press requires keys")
            if len(keys) == 1:
                execute_action({"type": "press", "key": self._normalize_key(keys[0])}, self.screen_width, self.screen_height)
            else:
                execute_action({"type": "hotkey", "keys": [self._normalize_key(k) for k in keys]}, self.screen_width, self.screen_height)
        elif kind == "wait":
            duration = float(action.get("duration") or self.wait_seconds)
            execute_action({"type": "wait", "seconds": duration}, self.screen_width, self.screen_height)
        elif kind == "scroll":
            self._scroll(action)
        elif kind == "left_click_drag":
            start_x, start_y = self._relative_xy(action.get("start_coordinate"))
            end_x, end_y = self._relative_xy(coord)
            execute_action({"type": "move", "x": start_x, "y": start_y}, self.screen_width, self.screen_height)
            execute_action({"type": "drag", "start_x": start_x, "start_y": start_y, "x": end_x, "y": end_y}, self.screen_width, self.screen_height)
        else:
            raise ValueError(f"Unsupported computer action: {kind}")

    def _scroll(self, action: dict[str, Any]) -> None:
        import pyautogui

        params = action.get("scroll_parameters") or {}
        amount = int(params.get("scroll_amount", 1))
        direction = params.get("scroll_direction", "down")
        coord = action.get("coordinate")
        x, y = self._relative_xy(coord) if coord else (self.screen_width // 2, self.screen_height // 2)
        pyautogui.moveTo(x, y)
        if direction == "up":
            pyautogui.scroll(amount)
        elif direction == "down":
            pyautogui.scroll(-amount)
        elif direction == "left":
            pyautogui.hscroll(-amount)
        elif direction == "right":
            pyautogui.hscroll(amount)
        else:
            raise ValueError(f"Unsupported scroll direction: {direction}")
        time.sleep(0.3)

    def _relative_xy(self, coord: Any) -> tuple[int, int]:
        if not isinstance(coord, list | tuple) or len(coord) != 2:
            raise ValueError("Action requires coordinate=[x, y]")
        x = max(0, min(self.screen_width - 1, int(round(float(coord[0]) * self.screen_width))))
        y = max(0, min(self.screen_height - 1, int(round(float(coord[1]) * self.screen_height))))
        return x, y

    @staticmethod
    def _normalize_key(key: str) -> str:
        aliases = {
            "cmd": "ctrl",
            "command": "ctrl",
            "control": "ctrl",
            "return": "enter",
            "escape": "esc",
            "option": "alt",
        }
        return aliases.get(key.lower(), key.lower())

    def _run_coro(self, coro):
        if self._loop is None:
            raise RuntimeError("No event loop available for Playwright operation")
        return self._loop.run_until_complete(coro)

    def _maybe_solve_captcha(self, phase: str, step_num: int | None = None) -> bool:
        solved = maybe_solve_captcha(
            self.captcha_solver,
            self._page,
            self._loop,
            self.logger,
            phase,
        )
        if solved:
            return True
        if step_num is None or self.captcha_solver is None:
            return False
        if self._last_captcha_failure_step == step_num:
            return False
        self._last_captcha_failure_step = step_num
        self._captcha_failures += 1
        max_captcha_failures = int(os.environ.get("WA_MAX_CAPTCHA_FAILURES", "3"))
        self.logger.warning(
            "Captcha solver failed after VLM step %s (%s/%s failures)",
            step_num,
            self._captcha_failures,
            max_captcha_failures,
        )
        return False

    def _captcha_failure_error(self, step_num: int) -> str | None:
        max_captcha_failures = int(os.environ.get("WA_MAX_CAPTCHA_FAILURES", "3"))
        if self._captcha_failures <= max_captcha_failures:
            return None
        return (
            f"Captcha solver failed more than {max_captcha_failures} times "
            f"after VLM inference; aborting task at step {step_num}"
        )

    def _pages(self) -> list[Any]:
        if self._page is None:
            return []
        return list(self._page.context.pages)

    def _select_page(self, tab_id: Any | None) -> Any:
        if tab_id is None:
            return self._page
        pages = self._pages()
        idx = int(tab_id)
        if idx < 0 or idx >= len(pages):
            raise ValueError(f"Invalid tab_id {tab_id}; {len(pages)} tabs available")
        self._page = pages[idx]
        self._run_coro(self._page.bring_to_front())
        return self._page

    def _navigate(self, url: str, tab_id: Any | None = None) -> str:
        page = self._select_page(tab_id)
        normalized = url.strip()
        if normalized.lower() == "back":
            self._run_coro(go_back(page, wait_until="load"))
        elif normalized.lower() == "forward":
            self._run_coro(go_forward(page, wait_until="load"))
        else:
            self._run_coro(goto(page, normalized, wait_until="load"))
        self._run_coro(page.bring_to_front())
        return f"Navigated tab {self._current_tab_id()} to {page.url}."

    def _tabs_create(self, url: str = "about:blank") -> str:
        if self._page is None:
            raise RuntimeError("No active page")
        page = self._run_coro(self._page.context.new_page())
        self._page = page
        if url and url != "about:blank":
            self._run_coro(goto(page, url, wait_until="load"))
        self._run_coro(page.bring_to_front())
        return f"Created tab {self._current_tab_id()} at {page.url}."

    def _tabs_focus(self, tab_id: int) -> str:
        page = self._select_page(tab_id)
        return f"Focused tab {tab_id}: {page.url}."

    def _current_tab_id(self) -> int:
        if self._page is None:
            return -1
        try:
            return self._pages().index(self._page)
        except ValueError:
            return -1

    @staticmethod
    def _assistant_text(message: dict[str, Any]) -> str:
        content = message.get("content") or ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
