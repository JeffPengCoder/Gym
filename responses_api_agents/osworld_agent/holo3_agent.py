# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-owned Holo3 scaffold for OSWorld.

Holo3 is trained against a structured multi-step harness.  The model emits one
``{note, thought, tool_call}`` object per observation, keeps at most three
screenshots, receives tool results as user messages, and uses coordinates in a
normalized ``[0, 1000]`` space.  This module implements that model contract;
OSWorld continues to own the desktop environment and evaluator.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError


LOG = logging.getLogger("nemo_gym.osworld_agent.holo3_agent")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Goal(_StrictModel):
    title: str = Field(description="Clear, actionable goal description beginning with a verb")
    status: Literal["todo", "running", "done", "failed"] = "todo"


class UpdatePlan(_StrictModel):
    """Create or update the complete task plan."""

    tool_name: Literal["update_plan"]
    goals: List[Goal] = Field(min_length=1, description="Complete goal list, including completed goals")


class Answer(_StrictModel):
    """Finish the task and provide any requested answer."""

    tool_name: Literal["answer"]
    content: str = Field(description="Final answer or a concise description of the completed state")


class Wait(_StrictModel):
    """Wait briefly before observing the desktop again."""

    tool_name: Literal["wait"]
    duration: float = Field(default=5.0, ge=1.0, le=30.0, description="Seconds to wait")


class WriteDesktop(_StrictModel):
    """Type text into the currently focused element."""

    tool_name: Literal["write"]
    content: str
    press_enter: bool = False
    overwrite: bool = False


class ClickDesktop(_StrictModel):
    """Click a described element at normalized coordinates."""

    tool_name: Literal["click"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    button: Literal["left", "right", "middle"] = "left"


class DoubleClickDesktop(_StrictModel):
    """Double-click a described element at normalized coordinates."""

    tool_name: Literal["double_click"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class DragToDesktop(_StrictModel):
    """Drag from one normalized coordinate to another."""

    tool_name: Literal["drag"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    x2: int = Field(ge=0, le=1000)
    y2: int = Field(ge=0, le=1000)


class ScrollDesktop(_StrictModel):
    """Move to an element and scroll in the requested direction."""

    tool_name: Literal["scroll"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    direction: Literal["up", "down", "left", "right"]
    scroll_size: int = Field(default=5, ge=1, le=50, description="Mouse-wheel clicks")


class MoveToDesktop(_StrictModel):
    """Move the mouse to normalized coordinates."""

    tool_name: Literal["move"]
    element: str
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class KeyDownDesktop(_StrictModel):
    """Hold a keyboard key."""

    tool_name: Literal["key_down"]
    key: str


class KeyUpDesktop(_StrictModel):
    """Release a keyboard key."""

    tool_name: Literal["key_up"]
    key: str


class HotkeyDesktop(_StrictModel):
    """Press one or more keys together."""

    tool_name: Literal["hotkey"]
    keys: List[str] = Field(min_length=1, max_length=5)
    repeat_count: int = Field(default=1, ge=1, le=10)


class HoldAndTapKeyDesktop(_StrictModel):
    """Hold keys while tapping another key sequence."""

    tool_name: Literal["hold_and_tap_key"]
    hold_keys: List[str] = Field(min_length=1, max_length=3)
    tap_keys: List[str] = Field(min_length=1, max_length=5)


ToolCall = Annotated[
    Union[
        UpdatePlan,
        Answer,
        Wait,
        WriteDesktop,
        ClickDesktop,
        DoubleClickDesktop,
        DragToDesktop,
        ScrollDesktop,
        MoveToDesktop,
        KeyDownDesktop,
        KeyUpDesktop,
        HotkeyDesktop,
        HoldAndTapKeyDesktop,
    ],
    Field(discriminator="tool_name"),
]


class Holo3Step(_StrictModel):
    note: Optional[str] = Field(
        default=None,
        description="New task-relevant information from the observation; null when there is nothing to retain",
    )
    thought: str = Field(description="Concise reasoning about the single best next action")
    tool_call: ToolCall


@dataclass
class _HistoryStep:
    screenshot: bytes
    parsed: Holo3Step
    response_content: str
    reasoning: str
    actions: List[str]
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None


_SYSTEM_PROMPT = """You are Holo3, a meticulous computer-use agent operating an Ubuntu desktop.

# Core principles

1. Prefer accuracy and visible evidence over speed or inference.
2. Work methodically, adapt after failures, and avoid repeating an ineffective action.
3. Record durable facts in `note`; older screenshots are evicted, and hidden reasoning is not replayed.
4. Take one precise tool action per step. Coordinates are integers in [0, 1000] with a top-left origin.
5. Do not call `answer` until the requested state is visibly complete and verified.

# Environment

- The desktop is Ubuntu at {screen_width}x{screen_height} pixels.
- The computer password is {password}.
- Start and control applications through the visible GUI.
- Some actions need time to render; use `wait` when appropriate.
- Current UTC time: {start_time}.
- Budget: at most {max_steps} model steps and {max_time_s} seconds.

# Interaction protocol

At each step, inspect the `<observation>` screenshot, update durable `note` information, reason briefly in
`thought`, and select exactly one `tool_call`. Tool execution results arrive in a following user message as
`<tool_output>` or `<error>`. Emit only one JSON object matching the supplied schema. Never emit Markdown,
OpenAI `tool_calls`, an `args` wrapper, or extra keys.
"""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _append_parser_event(event: Dict[str, Any]) -> None:
    path = os.environ.get("OSWORLD_MODEL_IO_LOG", "").strip()
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        LOG.exception("Failed to append Holo3 parser event to %s", path)


def _response_parts(response: Any) -> tuple[str, str]:
    if isinstance(response, str):
        return response, ""
    if isinstance(response, Mapping):
        return str(response.get("content") or ""), str(
            response.get("reasoning_content") or response.get("reasoning") or ""
        )
    return str(getattr(response, "content", "") or ""), str(
        getattr(response, "reasoning_content", "") or getattr(response, "reasoning", "") or ""
    )


def _parse_step(content: str) -> Holo3Step:
    candidates = [content.strip()]
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL))
    decoder = json.JSONDecoder()
    for start, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        candidates.append(json.dumps(value, ensure_ascii=False))
        break

    last_error: Optional[Exception] = None
    for candidate in dict.fromkeys(item for item in candidates if item):
        try:
            return Holo3Step.model_validate_json(candidate)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Holo3 response does not match the structured schema: {last_error}")


def _scale_coordinate(value: int, size: int) -> int:
    return max(0, min(size - 1, int((value / 1000) * size)))


def _normalize_key(key: str) -> str:
    aliases = {
        "cmd": "ctrl",
        "command": "ctrl",
        "control": "ctrl",
        "return": "enter",
        "escape": "esc",
        "option": "alt",
    }
    normalized = key.lower()
    return aliases.get(normalized, normalized)


class Holo3Agent:
    """Local-vLLM Holo3 scaffold following H Company's structured agent loop."""

    def __init__(
        self,
        *,
        platform: str = "ubuntu",
        model: str = "Hcompany/Holo3-35B-A3B",
        max_steps: int = 100,
        max_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: Optional[float] = 0.95,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        screen_size: tuple[int, int] = (1920, 1080),
        client_password: str = "password",
        max_image_history_length: int = 3,
        parse_retries: int = 3,
        max_time_s: int = 7200,
        reasoning_effort: str = "medium",
        log_context: Optional[Mapping[str, Any]] = None,
        **_kwargs: Any,
    ) -> None:
        if platform.lower() != "ubuntu":
            raise ValueError("Holo3 OSWorld adapter currently requires platform='ubuntu'")
        if action_space != "pyautogui" or observation_type != "screenshot":
            raise ValueError("Holo3 OSWorld adapter requires screenshot observations and pyautogui actions")
        if max_image_history_length < 1:
            raise ValueError("max_image_history_length must be positive")
        if parse_retries < 1:
            raise ValueError("parse_retries must be positive")

        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.screen_size = screen_size
        self.client_password = client_password
        self.max_image_history_length = max_image_history_length
        self.parse_retries = parse_retries
        self.max_time_s = max_time_s
        self.reasoning_effort = reasoning_effort
        self.log_context = dict(log_context or {})
        self.schema = Holo3Step.model_json_schema()
        self.history: List[_HistoryStep] = []
        self.logger = LOG

    def reset(self, logger: Optional[logging.Logger] = None, **_kwargs: Any) -> None:
        self.history.clear()
        if logger is not None:
            self.logger = logger

    def _system_prompt(self) -> str:
        prompt = _SYSTEM_PROMPT.format(
            screen_width=self.screen_size[0],
            screen_height=self.screen_size[1],
            password=self.client_password,
            start_time=datetime.datetime.now(datetime.timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC"),
            max_steps=self.max_steps,
            max_time_s=self.max_time_s,
        )
        return (
            prompt
            + "\n<output_format>\n```json\n"
            + json.dumps(self.schema, ensure_ascii=False)
            + "\n```\n</output_format>\n"
        )

    @staticmethod
    def _image_observation(screenshot: bytes) -> Dict[str, Any]:
        encoded = base64.b64encode(screenshot).decode("ascii")
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "<observation>\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                {"type": "text", "text": "\n</observation>"},
            ],
        }

    @staticmethod
    def _evicted_observation() -> Dict[str, Any]:
        return {"role": "user", "content": "<observation>\n[screenshot evicted]\n</observation>"}

    def _messages(self, instruction: str, screenshot: bytes) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Task: {instruction}"},
        ]
        first_retained = max(0, len(self.history) + 1 - self.max_image_history_length)
        for index, step in enumerate(self.history):
            messages.append(
                self._image_observation(step.screenshot) if index >= first_retained else self._evicted_observation()
            )
            messages.append({"role": "assistant", "content": step.parsed.model_dump_json()})
            tool_name = step.parsed.tool_call.tool_name
            if step.tool_error:
                messages.append(
                    {"role": "user", "content": f'<error tool="{tool_name}">\n{step.tool_error}\n</error>'}
                )
            else:
                output = step.tool_output or "Tool execution result was not reported."
                messages.append(
                    {"role": "user", "content": f'<tool_output tool="{tool_name}">\n{output}\n</tool_output>'}
                )
        messages.append(self._image_observation(screenshot))
        return messages

    def _payload(self, messages: List[Dict[str, Any]], *, step: int, parse_attempt: int) -> Dict[str, Any]:
        context = dict(self.log_context)
        context.update({"step": step, "parse_attempt": parse_attempt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "_nemo_gym_return_message": True,
            "_nemo_gym_require_stop": True,
            "chat_template_kwargs": {"enable_thinking": True},
            "extra_body": {
                "reasoning_effort": self.reasoning_effort,
                "structured_outputs": {"json": self.schema},
            },
            "_osworld_log_context": context,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def predict(self, instruction: str, obs: Dict[str, Any]) -> tuple[str, List[str], Dict[str, Any]]:
        screenshot = obs.get("screenshot") or b""
        if not isinstance(screenshot, bytes) or not screenshot:
            raise ValueError("Holo3 requires a non-empty screenshot observation")
        messages = self._messages(instruction, screenshot)
        step_number = len(self.history) + 1

        last_error: Optional[Exception] = None
        for parse_attempt in range(1, self.parse_retries + 1):
            payload = self._payload(messages, step=step_number, parse_attempt=parse_attempt)
            try:
                response = self.call_llm(payload, self.model)
                content, reasoning = _response_parts(response)
                parsed = _parse_step(content)
                actions = self._tool_actions(parsed.tool_call)
                history_step = _HistoryStep(
                    screenshot=screenshot,
                    parsed=parsed,
                    response_content=content,
                    reasoning=reasoning,
                    actions=actions,
                )
                self.history.append(history_step)
                parser_event = {
                    **self.log_context,
                    "schema_version": 2,
                    "event": "agent_parse",
                    "adapter": "gym",
                    "runner": "holo3_agent",
                    "step": step_number,
                    "parse_attempt": parse_attempt,
                    "parser_input": content,
                    "reasoning": reasoning,
                    "parser_output": parsed.model_dump(mode="json"),
                    "normalized_actions": actions,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                _append_parser_event(parser_event)
                return (
                    content,
                    actions,
                    {
                        "note": parsed.note,
                        "thought": parsed.thought,
                        "tool_call": parsed.tool_call.model_dump(mode="json"),
                        "reasoning": reasoning,
                        "parse_attempt": parse_attempt,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - model/parse failures are retryable.
                last_error = exc
                self.logger.warning("Holo3 response attempt %d/%d failed: %s", parse_attempt, self.parse_retries, exc)
        raise ValueError(
            f"Holo3 returned no valid structured action after {self.parse_retries} attempts"
        ) from last_error

    def record_action_result(
        self,
        *,
        actions: List[Any],
        done: bool,
        info: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
        **_kwargs: Any,
    ) -> None:
        """Attach the real VM execution result to the most recent Holo3 turn."""

        if not self.history:
            raise RuntimeError("record_action_result called before predict")
        step = self.history[-1]
        if step.tool_output is not None or step.tool_error is not None:
            raise RuntimeError("record_action_result called twice for the same Holo3 turn")
        if error:
            step.tool_error = error
            return

        tool = step.parsed.tool_call
        if isinstance(tool, UpdatePlan):
            step.tool_output = "\n".join(
                f"{index}. {goal.title} [{goal.status}]" for index, goal in enumerate(tool.goals, 1)
            )
        elif isinstance(tool, Wait):
            step.tool_output = f"Waited {tool.duration:g} seconds."
        elif isinstance(tool, Answer):
            step.tool_output = "Final answer submitted."
        else:
            step.tool_output = f"{tool.tool_name} executed successfully."
        if done:
            step.tool_output += " The environment reported a terminal state."
        if info and info.get("error"):
            step.tool_error = str(info["error"])
            step.tool_output = None

    def _xy(self, x: int, y: int) -> tuple[int, int]:
        return _scale_coordinate(x, self.screen_size[0]), _scale_coordinate(y, self.screen_size[1])

    def _tool_actions(self, tool: ToolCall) -> List[str]:
        if isinstance(tool, Answer):
            return ["DONE"]
        if isinstance(tool, UpdatePlan):
            return ["pyautogui.sleep(0)"]
        if isinstance(tool, Wait):
            return [f"pyautogui.sleep({tool.duration!r})"]
        if isinstance(tool, WriteDesktop):
            commands = []
            if tool.overwrite:
                commands.append("pyautogui.hotkey('ctrl', 'a')")
            commands.append(f"pyautogui.write({tool.content!r}, interval=0.05)")
            if tool.press_enter:
                commands.append("pyautogui.press('enter')")
            return ["\n".join(commands)]
        if isinstance(tool, ClickDesktop):
            x, y = self._xy(tool.x, tool.y)
            return [f"pyautogui.click(x={x}, y={y}, button={tool.button!r})"]
        if isinstance(tool, DoubleClickDesktop):
            x, y = self._xy(tool.x, tool.y)
            return [f"pyautogui.doubleClick(x={x}, y={y}, interval=0.05)"]
        if isinstance(tool, DragToDesktop):
            x, y = self._xy(tool.x, tool.y)
            x2, y2 = self._xy(tool.x2, tool.y2)
            return [
                f"pyautogui.moveTo({x}, {y})\n"
                "pyautogui.mouseDown(button='left')\n"
                f"pyautogui.moveTo({x2}, {y2}, duration=1)\n"
                "pyautogui.mouseUp(button='left')"
            ]
        if isinstance(tool, ScrollDesktop):
            x, y = self._xy(tool.x, tool.y)
            amount = tool.scroll_size if tool.direction in {"up", "right"} else -tool.scroll_size
            method = "hscroll" if tool.direction in {"left", "right"} else "scroll"
            return [f"pyautogui.moveTo({x}, {y})\npyautogui.{method}({amount})"]
        if isinstance(tool, MoveToDesktop):
            x, y = self._xy(tool.x, tool.y)
            return [f"pyautogui.moveTo({x}, {y})"]
        if isinstance(tool, KeyDownDesktop):
            return [f"pyautogui.keyDown({_normalize_key(tool.key)!r})"]
        if isinstance(tool, KeyUpDesktop):
            return [f"pyautogui.keyUp({_normalize_key(tool.key)!r})"]
        if isinstance(tool, HotkeyDesktop):
            rendered_keys = ", ".join(repr(_normalize_key(key)) for key in tool.keys)
            return ["\n".join(f"pyautogui.hotkey({rendered_keys})" for _ in range(tool.repeat_count))]
        if isinstance(tool, HoldAndTapKeyDesktop):
            commands = [f"pyautogui.keyDown({_normalize_key(key)!r})" for key in tool.hold_keys]
            commands.extend(f"pyautogui.press({_normalize_key(key)!r})" for key in tool.tap_keys)
            commands.extend(f"pyautogui.keyUp({_normalize_key(key)!r})" for key in reversed(tool.hold_keys))
            return ["\n".join(commands)]
        raise TypeError(f"Unsupported Holo3 tool: {type(tool).__name__}")


__all__ = ["Holo3Agent", "Holo3Step"]
