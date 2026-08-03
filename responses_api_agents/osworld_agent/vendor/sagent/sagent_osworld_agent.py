"""Self-contained sagent "Forest" OSWorld agent for in-VM execution.

Reproduces the pure-policy Forest agent from WandB run ``sg11q7ck`` (76.2%) as a
single file that runs inside an OSWorld VM, mirroring the NVIDIA/Holotron
single-file agent (``holotron_osworld/osworld_agent.py``). No sagent /
hai_adapters / hai_drivers imports: only stdlib + requests + pydantic +
pyautogui + jinja2 + PIL.

The agent is a plain observe -> policy LLM -> parse -> execute loop with no
validator, no infeasible-checker and no callbacks. It reproduces sagent's
``structured_note`` output contract (a single JSON object with ``note``,
``thought`` and ``tool_call``), the ``0-1000`` coordinate system, the 16-tool
catalog of the sg11 recipe, and the structured message serialization
(``<message>`` / ``<observation>`` / assistant JSON / ``<tool_output>``).
"""

from __future__ import annotations

import base64
import copy
import datetime
import json
import logging
import operator
import os
import time
import traceback
from functools import reduce
from pathlib import Path
from typing import Any, Literal

import requests
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field, create_model

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL = "nemotron-3-nano-omni-ga-bf16-24000"
VLLM_API_ENDPOINT = "https://api.training.hcompany.ai/v1/models"

DEFAULT_MAX_STEPS = 100
DEFAULT_MAX_TIME_S = 3300
DEFAULT_MAX_IMAGES = 2
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_COMPLETION_TOKENS = 4096
DEFAULT_LLM_TIMEOUT_S = 1200
DEFAULT_WAIT_AFTER_S = 3.0

IMAGE_PLACEHOLDER = "[Image omitted by context cleaning]"
USER_CALLER_ID = "user"
ANSWER_TOOL_NAME = "answer"
INFEASIBLE_MARKER = "[TASK=INFEASIBLE]"

# Tools that produce no desktop side effect and whose result is echoed back to
# the model verbatim (mirrors sagent: the plan / wait text is shown, action
# tools show an empty <tool_output>).
NO_EFFECT_TOOLS = ("update_plan", "wait_desktop")

# Tools that sleep ``wait_after_s`` after execution in the sg11 config.
DESKTOP_ACTION_TOOLS = (
    "write_desktop",
    "click_desktop",
    "double_click_desktop",
    "write_at_desktop",
    "scroll_desktop",
    "move_to_desktop",
    "drag_and_drop",
    "key_down_desktop",
    "key_up_desktop",
    "hotkey_desktop",
    "hold_and_tap_key_desktop",
    "mouse_down_desktop",
    "mouse_up_desktop",
)


# ============================================================================
# Tool argument schemas (0-1000 coordinate variants)
# ----------------------------------------------------------------------------
# These mirror the pydantic arg-schemas in sagent/lib/tools/*.py used by the
# sg11 recipe (RELATIVE_0_1000 variants). Their field names, descriptions and
# constraints are reproduced verbatim so that ``.model_json_schema()`` on the
# assembled output format is byte-identical to sagent's.
# ============================================================================


GoalStatus = Literal["todo", "running", "done", "failed"]
MouseButton = Literal["left", "right", "middle"]
ScrollDirection = Literal["up", "down", "left", "right"]


class Goal(BaseModel):
    title: str = Field(description="Clear, actionable goal description (start with verb)")
    status: GoalStatus = Field(default="todo", description="Current status (todo/running/done/failed)")


class UpdatePlanArgs(BaseModel):
    goals: list[Goal] = Field(description="Complete list of goals (include done/failed goals when replanning)")


class WriteDesktopArgs(BaseModel):
    content: str = Field(description="Content to write")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing")
    overwrite: bool = Field(default=False, description="Whether to clear existing text before typing")


class Click1000DesktopSchema(BaseModel):
    element: str = Field(description="Detailed description of the target UI element to click on")
    x: int = Field(description="X coordinate as integer in [0, 1000]")
    y: int = Field(description="Y coordinate as integer in [0, 1000]")
    button: MouseButton = Field(description="Mouse button to click (left, right, middle)", default="left")


class DoubleClick1000DesktopSchema(BaseModel):
    element: str = Field(description="Detailed description of the target UI element to double click on")
    x: int = Field(description="The x coordinate as integer in [0, 1000]")
    y: int = Field(description="The y coordinate as integer in [0, 1000]")


class WriteAt1000DesktopSchema(BaseModel):
    content: str = Field(description="Content to write")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing")
    overwrite: bool = Field(default=False, description="Whether to clear existing text before typing")
    element: str = Field(description="Detailed description of the target UI element to write at")
    x: int = Field(description="X coordinate as integer in [0, 1000]")
    y: int = Field(description="Y coordinate as integer in [0, 1000]")


class Scroll1000DesktopSchema(BaseModel):
    element: str = Field(description="Detailed description of the target UI element to scroll on")
    x: int = Field(description="X coordinate as integer in [0, 1000]")
    y: int = Field(description="Y coordinate as integer in [0, 1000]")
    direction: ScrollDirection = Field(description="Direction to scroll in")
    scroll_size: int = Field(
        default=10,
        description=(
            "Scroll size in mouse wheel clicks (at most 100). "
            "Choose it carefully considering the OS and the objective of the scroll"
        ),
    )


class MoveTo1000DesktopSchema(BaseModel):
    element: str = Field(description="Detailed description of the target UI element to move the mouse to")
    x: int = Field(description="X coordinate as integer in [0, 1000]")
    y: int = Field(description="Y coordinate as integer in [0, 1000]")


class DragAndDropHoldKey1000Schema(BaseModel):
    description: str = Field(
        description="Precise description of the drag and drop action and what it should do. "
        "eg: drag the volume slider to set it to 10%/drag the map to the right to see the more the left side of the map"
    )
    x1: int = Field(description="The x coordinate of the start of the drag as integer in [0, 1000]")
    y1: int = Field(description="The y coordinate of the start of the drag as integer in [0, 1000]")
    x2: int = Field(description="The x coordinate of the end of the drag as integer in [0, 1000]")
    y2: int = Field(description="The y coordinate of the end of the drag as integer in [0, 1000]")
    hold_keys: list[str] | None = Field(default=None, description="List of keys to hold while dragging")


class KeyDownDesktopArgs(BaseModel):
    key: str = Field(description="Key to press")


class KeyUpDesktopArgs(BaseModel):
    key: str = Field(description="Key to release")


class HotkeyDesktopArgs(BaseModel):
    keys: list[str] = Field(
        min_length=1,
        max_length=5,
        description=(
            "List of 1 to 5 key(s) to press simultaneously: "
            "e.g. ['ctrl', 'alt', 't'] for Ubuntu, ['cmd', 't'] for MacOS..."
        ),
    )
    repeat_count: int = Field(default=1, description="Number of times to repeat the hotkey press")


class HoldAndTapKeyDesktopArgs(BaseModel):
    hold_keys: list[str] = Field(min_length=1, max_length=3, description="List of 1 to 3 key(s) to hold down")
    tap_keys: list[str] = Field(min_length=1, max_length=5, description="List of 1 to 5 key(s) to tap in sequence")


class MouseDownDesktopArgs(BaseModel):
    button: MouseButton = Field(description="Mouse button to press (left, right, middle)", default="left")


class MouseUpDesktopArgs(BaseModel):
    button: MouseButton = Field(description="Mouse button to release (left, right, middle)", default="left")


class WaitDesktopArgs(BaseModel):
    seconds: float = Field(description="Number of seconds to wait")


class AnswerArgs(BaseModel):
    content: str = Field(description="The answer content")


class NoteStructuredOutput(BaseModel):
    """Single tool call with note-taking (mirrors sagent NoteStructuredOutput)."""

    note: str | None = Field(
        default=None,
        description="Task-relevant information extracted from the previous observation. Keep empty if no new info.",
    )
    thought: str = Field(description="Reasoning about next steps")


# Ordered tool catalog for the sg11 recipe: (name, description, args_schema).
# The order defines the schema union order and must match the sg11 config.
TOOL_CATALOG: tuple[tuple[str, str, type[BaseModel]], ...] = (
    (
        "update_plan",
        (
            "Create and manage your task plan with hierarchical goals. "
            "Always provide the complete list of goals every time you call this tool. "
            "When creating an initial plan, include all goals with the first one as 'running' and others as 'todo'. "
            "When marking progress, include all goals with updated statuses "
            "(mark completed as 'done', set next as 'running'). "
            "When replanning after blockers, include your done/failed goals plus new goals. "
            "Maintain only one goal with status 'running' at any given time."
        ),
        UpdatePlanArgs,
    ),
    ("write_desktop", "Type text into the currently focused element without clicking first", WriteDesktopArgs),
    ("click_desktop", "Click at (x, y) coordinates", Click1000DesktopSchema),
    ("double_click_desktop", "Double click at (x, y) coordinates", DoubleClick1000DesktopSchema),
    (
        "write_at_desktop",
        "Click at (x, y) coordinates to focus an element, then type content",
        WriteAt1000DesktopSchema,
    ),
    ("scroll_desktop", "Scroll in a given direction, placing the cursor at (x, y) first", Scroll1000DesktopSchema),
    ("move_to_desktop", "Move mouse to (x, y) coordinates", MoveTo1000DesktopSchema),
    ("drag_and_drop", "Drag and drop from coordinates 1 to coordinates 2", DragAndDropHoldKey1000Schema),
    ("key_down_desktop", "Press and hold a key on the keyboard", KeyDownDesktopArgs),
    ("key_up_desktop", "Release a key on the keyboard", KeyUpDesktopArgs),
    (
        "hotkey_desktop",
        "Press multiple keys simultaneously (max 5). Adapt the keys depending on the operating system",
        HotkeyDesktopArgs,
    ),
    (
        "hold_and_tap_key_desktop",
        "Hold a list of key(s) and tap a list of key(s) in sequence. Adapt the keys depending on the operating system.",
        HoldAndTapKeyDesktopArgs,
    ),
    ("mouse_down_desktop", "Press mouse button down", MouseDownDesktopArgs),
    ("mouse_up_desktop", "Release mouse button", MouseUpDesktopArgs),
    (
        "wait_desktop",
        "Wait for a few seconds before getting a new observation. Useful to wait during loading.",
        WaitDesktopArgs,
    ),
    ("answer", "Provide a final answer", AnswerArgs),
)


# ============================================================================
# Schema building (reproduces sagent constrain_tool / _build_tool_call_variant)
# ============================================================================


class FlatToolCall(BaseModel):
    """Flattened tool call for LLM-friendly JSON generation (mirrors sagent)."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str


def build_output_schema() -> dict[str, Any]:
    """Build the ``structured_note`` JSON schema for the sg11 tool catalog.

    Reproduces sagent's ``NoteStructuredOutput.constrain(specs).model_json_schema()``
    by flattening each tool's args into a per-tool ``FlatToolCall`` subclass, taking
    their union as the ``tool_call`` field, and reading the resulting JSON schema.

    Returns:
        The JSON schema dict, byte-identical to sagent's structured_note schema.
    """
    variants = [_build_tool_call_variant(name, description, args) for name, description, args in TOOL_CATALOG]
    # Dynamically created models carry annotations (e.g. list[Goal]) that pydantic resolves lazily;
    # rebuild with this module's namespace so those references are available before schema generation.
    for variant in variants:
        variant.model_rebuild(_types_namespace=globals())
    union = variants[0] if len(variants) == 1 else reduce(operator.or_, variants[1:], variants[0])
    constrained = create_model(
        NoteStructuredOutput.__name__,
        __base__=NoteStructuredOutput,
        __config__=ConfigDict(extra="forbid"),
        tool_call=(union, Field(...)),
    )
    constrained.model_rebuild(_types_namespace=globals())
    return constrained.model_json_schema()


def _schema_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in name)
    if not sanitized:
        sanitized = "tool"
    if sanitized[0].isdigit():
        sanitized = f"tool_{sanitized}"
    return sanitized


def _build_tool_call_variant(name: str, description: str, args_schema: type[BaseModel]) -> type[FlatToolCall]:
    field_definitions: dict[str, tuple[Any, Any]] = {"tool_name": (Literal[name], Field(...))}
    for field_name, field in args_schema.model_fields.items():
        if field_name == "tool_name":
            continue
        field_definitions[field_name] = (field.annotation or Any, copy.deepcopy(field))

    model_class: type[FlatToolCall] = create_model(
        _schema_name(name),
        __base__=FlatToolCall,
        **field_definitions,
    )
    model_class.model_config = ConfigDict(extra="forbid", json_schema_extra={"description": description})
    return model_class


# ============================================================================
# Coordinate / key helpers
# ============================================================================


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("utf-8")


def _coord_to_pixel(value: int, size: int) -> int:
    """Convert a 0-1000 integer coordinate to a pixel coordinate."""
    pixel = int(int(value) / 1000 * size)
    return max(0, min(size - 1, pixel))


def _normalize_key(key: Any) -> str:
    """Normalize modifier/named keys to pyautogui names."""
    text = str(key).strip().lower()
    aliases = {
        "cmd": "ctrl",
        "command": "ctrl",
        "control": "ctrl",
        "return": "enter",
        "escape": "esc",
        "option": "alt",
    }
    return aliases.get(text, text)


# ============================================================================
# System prompt rendering
# ============================================================================


def _system_timestamp(dt: datetime.datetime) -> str:
    """Format a datetime for prompts: 'Monday, October 2, 2025 at 3:45 PM UTC'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    elif dt.tzinfo != datetime.timezone.utc:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt.strftime("%A, %B %d, %Y at %I:%M %p UTC")


def render_system_prompt(
    prompt_path: Path,
    output_schema: dict[str, Any],
    max_steps: int,
    max_time_s: float,
    start_time: datetime.datetime,
    services: dict[str, str],
    password: str,
) -> str:
    """Render the external prompt.j2 and append the structured output format block.

    Mirrors ``StructuredChatMapper.build_system_prompt``: render the jinja template
    with the ``StandardPolicyInput.dump_for_system_prompt`` context, then append
    ``<output_format>``. There are no skills or instructions in the sg11 recipe.

    Args:
        prompt_path: Path to the ``prompt.j2`` uploaded next to this agent.
        output_schema: The structured_note JSON schema (from ``build_output_schema``).
        max_steps: Maximum number of agent steps (jinja var).
        max_time_s: Maximum agent runtime in seconds (jinja var).
        start_time: Agent start time in UTC (jinja var, used by ``system_timestamp``).
        services: Service name -> type mapping (jinja var, e.g. ``{"desktop": ...}``).
        password: Computer password (jinja var, exposed as ``password``).

    Returns:
        The fully rendered system prompt string.

    Raises:
        FileNotFoundError: If ``prompt_path`` does not exist.
    """
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found at {prompt_path}. The sagent_osworld adapter must upload a prompt.j2 "
            "next to the agent and set 'prompt_path' in config.json."
        )

    env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.globals["system_timestamp"] = _system_timestamp
    template = env.from_string(prompt_path.read_text(encoding="utf-8"))

    context = {
        "agent_name": "OSWorldSurfer",
        "caller_id": USER_CALLER_ID,
        "current_step": 0,
        "max_steps": max_steps,
        "start_time": start_time,
        "elapsed_time_s": 0.0,
        "max_time_s": max_time_s,
        "context": {},
        "instructions": None,
        "callable_tool_names": None,
        "force_answer": False,
        "runtime_metadata": {},
        "services": services,
        "password": password,
        "tools": {name: True for name, _, _ in TOOL_CATALOG},
        "skills": {},
        "skill_hints": {},
    }
    prompt = template.render(**context)
    schema = json.dumps(output_schema)
    prompt += f"\n\n<output_format>\n```json\n{schema}\n```\n</output_format>"
    return prompt


# ============================================================================
# Agent
# ============================================================================


class SagentOSWorldAgent:
    """Pure-policy Forest OSWorld agent reproducing WandB run sg11q7ck.

    Observes a screenshot, calls the policy LLM with the structured_note contract,
    parses the single tool_call and maps it to pyautogui actions. No validator,
    infeasible-checker or callbacks.
    """

    def __init__(
        self,
        model: str,
        max_steps: int,
        prompt_path: str,
        screen_size: tuple[int, int] = (1920, 1080),
        max_images: int = DEFAULT_MAX_IMAGES,
        max_time_s: float = DEFAULT_MAX_TIME_S,
        temperature: float = DEFAULT_TEMPERATURE,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
        password: str = "osworld-public-evaluation",
    ) -> None:
        """Initialize the agent.

        Args:
            model: Model id for the OpenAI-compatible endpoint.
            max_steps: Maximum number of agent steps.
            prompt_path: Path to the external ``prompt.j2`` template.
            screen_size: (width, height) of the VM screen in pixels.
            max_images: Number of most recent observation images to keep in history.
            max_time_s: Maximum agent runtime in seconds (prompt var only).
            temperature: Sampling temperature.
            max_completion_tokens: Max completion tokens per LLM call.
            llm_timeout_s: Per-request LLM timeout in seconds.
            password: Computer password exposed to the prompt.
        """
        self.model = model
        self.max_steps = max_steps
        self.prompt_path = Path(prompt_path)
        self.screen_size = screen_size
        self.max_images = max_images
        self.max_time_s = max_time_s
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.llm_timeout_s = llm_timeout_s
        self.password = password
        self.endpoint = f"{VLLM_API_ENDPOINT}/{model}/chat/completions"
        self.logger = logging.getLogger("desktopenv.agent")

        self.output_schema = build_output_schema()
        self.start_time = datetime.datetime.now(datetime.timezone.utc)
        self.system_prompt = render_system_prompt(
            prompt_path=self.prompt_path,
            output_schema=self.output_schema,
            max_steps=self.max_steps,
            max_time_s=self.max_time_s,
            start_time=self.start_time,
            services={"desktop": "RemoteDesktopDriver"},
            password=self.password,
        )
        # History records: {"screenshot": bytes|None, "step": dict|None, "tool_name": str|None,
        #                   "tool_result": str}
        self.history: list[dict[str, Any]] = []

    def predict(
        self, instruction: str, screenshot: bytes, step_idx: int
    ) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        """Run one policy step: call the LLM and map its tool call to pyautogui code.

        Args:
            instruction: The task instruction (injected as the first user message).
            screenshot: PNG bytes of the current observation.
            step_idx: Zero-based step index (for logging).

        Returns:
            A ``(response, actions, info)`` tuple. ``response`` is the raw LLM message
            dict; ``actions`` is a list of pyautogui code strings or a sentinel
            (``["DONE"]``, ``["FAIL"]``, ``["WAIT"]``); ``info`` carries thought / note /
            action / tool_name / tool_call for the trajectory.
        """
        self.logger.info("========= %s Step %s =======", self.model, step_idx)

        messages = self._build_messages(instruction, screenshot)
        payload = self._build_payload(messages)

        parsed: dict[str, Any] | None = None
        response: dict[str, Any] | None = None
        actions: list[str] | None = None
        tool_result = ""
        last_error = ""

        for _ in range(5):
            try:
                response = self._call_llm(payload)
                parsed = self._parse_response(response)
                tool_call = parsed["tool_call"]
                actions, tool_result = self._tool_to_pyautogui(tool_call)
                break
            except Exception as exc:  # noqa: BLE001 - retry on any parse/action failure
                last_error = f"{exc}\n{traceback.format_exc()}"
                self.logger.error("Parse/action error: %s", last_error)
                time.sleep(1)

        if parsed is None or response is None or actions is None:
            return {"error": last_error}, ["FAIL"], {"action": last_error, "tool_name": "parse_error"}

        tool_call = parsed["tool_call"]
        tool_name = str(tool_call.get("tool_name") or "").strip().lower()
        info = {
            "thought": parsed.get("thought", ""),
            "note": parsed.get("note"),
            "action": self._describe_tool(tool_call),
            "tool_name": tool_name,
            "tool_call": tool_call,
        }

        self.history.append(
            {
                "screenshot": screenshot,
                "step": parsed,
                "tool_name": tool_name,
                "tool_result": self._history_tool_result(tool_name, tool_result),
            }
        )

        current_step = len(self.history)
        if current_step >= self.max_steps and actions and actions[0] not in {"DONE", "FAIL"}:
            self.logger.warning("Reached maximum steps %s. Forcing termination.", self.max_steps)
            actions = ["FAIL"]
            info["action"] = "Fail the task because reaching the maximum step limit."

        self.logger.info("Action:\n%s", info["action"])
        return response, actions, info

    def _build_messages(self, instruction: str, screenshot: bytes) -> list[dict[str, Any]]:
        """Assemble the chat messages in sagent structured-mode order."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

        # Task instruction reaches the model as a MessageEvent(caller_id="user").
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": f'<message from="{USER_CALLER_ID}">\n{instruction}\n</message>'}],
            }
        )

        records = [
            *self.history,
            {"screenshot": screenshot, "step": None, "tool_name": None, "tool_result": ""},
        ]
        keep_images = set(range(max(0, len(records) - self.max_images), len(records)))
        for idx, entry in enumerate(records):
            if entry.get("step") is not None:
                messages.append({"role": "assistant", "content": json.dumps(entry["step"], ensure_ascii=False)})
                tool_name = entry.get("tool_name") or "unknown"
                messages.append(
                    {
                        "role": "user",
                        "content": f'<tool_output tool="{tool_name}">\n{entry.get("tool_result") or ""}\n</tool_output>',
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": self._observation_content(
                        entry["screenshot"] if idx in keep_images else None,
                    ),
                }
            )

        return _merge_consecutive_user_messages(messages)

    def _observation_content(self, screenshot: bytes | None) -> list[dict[str, Any]]:
        """Build the observation user-message content (cursor_size=0 => no cursor)."""
        if screenshot is None:
            return [{"type": "text", "text": f"<observation>\n{IMAGE_PLACEHOLDER}\n</observation>"}]
        return [
            {"type": "text", "text": "<observation>\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(screenshot)}"}},
            {"type": "text", "text": "\n</observation>"},
        ]

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "NoteStructuredOutput", "schema": self.output_schema},
            },
        }

    def _call_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('HAI_API_KEY')}",
        }
        last_error = ""
        for _ in range(20):
            try:
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=self.llm_timeout_s, verify=False
                )
                if resp.status_code != 200:
                    last_error = resp.text
                    self.logger.error("Failed to call LLM: %s", resp.text)
                    time.sleep(5)
                    continue
                message = resp.json()["choices"][0]["message"]
                if (message.get("content") or "").strip():
                    return message
                last_error = "empty content"
                self.logger.error("LLM returned empty content, retrying.")
            except Exception as exc:  # noqa: BLE001 - retry on any transport/JSON error
                last_error = repr(exc)
                self.logger.error("LLM call error: %s", exc)
            time.sleep(5)
        raise RuntimeError(f"API max retries exceeded: {last_error}")

    def _parse_response(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content") or ""
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("tool_call"), dict):
            raise ValueError(f"Response missing tool_call object: {content[:500]}")
        tool_call = parsed["tool_call"]
        if not tool_call.get("tool_name"):
            raise ValueError(f"tool_call missing tool_name: {content[:500]}")
        return {"note": parsed.get("note"), "thought": parsed.get("thought", ""), "tool_call": tool_call}

    def _history_tool_result(self, tool_name: str, tool_result: str) -> str:
        """Result echoed into history: plan/wait text is kept, action tools are empty."""
        if tool_name in NO_EFFECT_TOOLS:
            return tool_result
        return ""

    def _tool_to_pyautogui(self, tool_call: dict[str, Any]) -> tuple[list[str], str]:
        """Map a parsed tool call to pyautogui code plus a tool-result string."""
        tool_name = str(tool_call.get("tool_name") or "").strip().lower()
        width, height = self.screen_size

        if tool_name == ANSWER_TOOL_NAME:
            content = str(tool_call.get("content") or "")
            return ["DONE"], content
        if tool_name == "update_plan":
            return ["WAIT"], self._update_plan_result(tool_call)
        if tool_name == "wait_desktop":
            seconds = float(tool_call.get("seconds") or 1.0)
            return [f"time.sleep({seconds})"], "Waited."

        if tool_name == "write_desktop":
            return [self._write_code(tool_call)], "Typed text."

        if tool_name == "click_desktop":
            x = _coord_to_pixel(tool_call["x"], width)
            y = _coord_to_pixel(tool_call["y"], height)
            button = str(tool_call.get("button") or "left")
            return [f"pyautogui.click({x}, {y}, button={button!r})"], "Clicked."

        if tool_name == "double_click_desktop":
            x = _coord_to_pixel(tool_call["x"], width)
            y = _coord_to_pixel(tool_call["y"], height)
            return [f"pyautogui.doubleClick({x}, {y}, button='left')"], "Double-clicked."

        if tool_name == "write_at_desktop":
            x = _coord_to_pixel(tool_call["x"], width)
            y = _coord_to_pixel(tool_call["y"], height)
            code = f"pyautogui.click({x}, {y}, button='left')\n" + self._write_code(tool_call)
            return [code], "Wrote at element."

        if tool_name == "move_to_desktop":
            x = _coord_to_pixel(tool_call["x"], width)
            y = _coord_to_pixel(tool_call["y"], height)
            return [f"pyautogui.moveTo({x}, {y})"], "Moved cursor."

        if tool_name == "scroll_desktop":
            return [self._scroll_code(tool_call, width, height)], "Scrolled."

        if tool_name == "drag_and_drop":
            return [self._drag_code(tool_call, width, height)], "Dragged."

        if tool_name == "key_down_desktop":
            return [f"pyautogui.keyDown({_normalize_key(tool_call.get('key'))!r})"], "Pressed key down."

        if tool_name == "key_up_desktop":
            return [f"pyautogui.keyUp({_normalize_key(tool_call.get('key'))!r})"], "Released key."

        if tool_name == "hotkey_desktop":
            return [self._hotkey_code(tool_call)], "Pressed hotkey."

        if tool_name == "hold_and_tap_key_desktop":
            return [self._hold_and_tap_code(tool_call)], "Held and tapped keys."

        if tool_name == "mouse_down_desktop":
            button = str(tool_call.get("button") or "left")
            return [f"pyautogui.mouseDown(button={button!r})"], "Pressed mouse button down."

        if tool_name == "mouse_up_desktop":
            button = str(tool_call.get("button") or "left")
            return [f"pyautogui.mouseUp(button={button!r})"], "Released mouse button."

        raise ValueError(f"unsupported tool: {tool_name!r}")

    def _write_code(self, tool_call: dict[str, Any]) -> str:
        content = str(tool_call.get("content") or "")
        lines = [f"text = {content!r}"]
        if bool(tool_call.get("overwrite", False)):
            lines.extend(["pyautogui.hotkey('ctrl', 'a')", "pyautogui.press('backspace')"])
        lines.append("pyautogui.write(text, interval=0.01)")
        if bool(tool_call.get("press_enter", False)):
            lines.append("pyautogui.press('enter')")
        return "\n".join(lines)

    def _scroll_code(self, tool_call: dict[str, Any], width: int, height: int) -> str:
        x = _coord_to_pixel(tool_call["x"], width)
        y = _coord_to_pixel(tool_call["y"], height)
        direction = str(tool_call.get("direction") or "down").lower()
        amount = max(1, min(100, int(tool_call.get("scroll_size") or 10)))
        move = f"pyautogui.moveTo({x}, {y})\n"
        if direction == "up":
            return f"{move}pyautogui.scroll({amount})"
        if direction == "down":
            return f"{move}pyautogui.scroll(-{amount})"
        if direction == "left":
            return f"{move}pyautogui.hscroll(-{amount})"
        if direction == "right":
            return f"{move}pyautogui.hscroll({amount})"
        raise ValueError(f"unsupported scroll direction: {direction!r}")

    def _drag_code(self, tool_call: dict[str, Any], width: int, height: int) -> str:
        x1 = _coord_to_pixel(tool_call["x1"], width)
        y1 = _coord_to_pixel(tool_call["y1"], height)
        x2 = _coord_to_pixel(tool_call["x2"], width)
        y2 = _coord_to_pixel(tool_call["y2"], height)
        hold_keys = [_normalize_key(k) for k in (tool_call.get("hold_keys") or [])]
        lines = [f"pyautogui.keyDown({k!r})" for k in hold_keys]
        lines.append(f"pyautogui.moveTo({x1}, {y1})")
        lines.append("pyautogui.mouseDown(button='left')")
        lines.append(f"pyautogui.moveTo({x2}, {y2})")
        lines.append("pyautogui.mouseUp(button='left')")
        lines.extend(f"pyautogui.keyUp({k!r})" for k in reversed(hold_keys))
        return "\n".join(lines)

    def _hotkey_code(self, tool_call: dict[str, Any]) -> str:
        keys = [_normalize_key(k) for k in (tool_call.get("keys") or [])]
        if not keys:
            raise ValueError("hotkey_desktop requires at least one key")
        repeat = max(1, min(20, int(tool_call.get("repeat_count") or 1)))
        keys_arg = ", ".join(repr(k) for k in keys)
        return "\n".join(f"pyautogui.hotkey({keys_arg}, interval=0.1)" for _ in range(repeat))

    def _hold_and_tap_code(self, tool_call: dict[str, Any]) -> str:
        hold_keys = [_normalize_key(k) for k in (tool_call.get("hold_keys") or [])]
        tap_keys = [_normalize_key(k) for k in (tool_call.get("tap_keys") or [])]
        if not hold_keys or not tap_keys:
            raise ValueError("hold_and_tap_key_desktop requires hold_keys and tap_keys")
        lines = [f"pyautogui.keyDown({k!r})" for k in hold_keys]
        lines.extend(f"pyautogui.press({k!r})" for k in tap_keys)
        lines.extend(f"pyautogui.keyUp({k!r})" for k in reversed(hold_keys))
        return "\n".join(lines)

    def _update_plan_result(self, tool_call: dict[str, Any]) -> str:
        goals = tool_call.get("goals")
        if not isinstance(goals, list) or not goals:
            return "Plan updated."
        lines = []
        for idx, goal in enumerate(goals, 1):
            if isinstance(goal, dict):
                lines.append(f"{idx}. {goal.get('title', '')} [{goal.get('status', '')}]".rstrip())
            else:
                lines.append(f"{idx}. {goal}")
        return "\n".join(lines)

    def _describe_tool(self, tool_call: dict[str, Any]) -> str:
        tool_name = str(tool_call.get("tool_name") or "").strip().lower()
        if tool_name in {"click_desktop", "double_click_desktop", "move_to_desktop", "write_at_desktop"}:
            return f"{tool_name}: {tool_call.get('element', 'target')} at ({tool_call.get('x')}, {tool_call.get('y')})"
        if tool_name == "scroll_desktop":
            return (
                f"scroll_desktop: {tool_call.get('direction')} on {tool_call.get('element', 'target')} "
                f"at ({tool_call.get('x')}, {tool_call.get('y')})"
            )
        if tool_name == "drag_and_drop":
            return (
                f"drag_and_drop: ({tool_call.get('x1')}, {tool_call.get('y1')}) -> "
                f"({tool_call.get('x2')}, {tool_call.get('y2')})"
            )
        if tool_name in {"write_desktop"}:
            return f"write_desktop: {tool_call.get('content', '')}"
        if tool_name == "hotkey_desktop":
            return f"hotkey_desktop: {tool_call.get('keys')}"
        if tool_name == ANSWER_TOOL_NAME:
            return f"answer: {tool_call.get('content', '')}"
        return f"{tool_name}: {json.dumps(tool_call, ensure_ascii=False)}"


def _merge_consecutive_user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent user messages into one (mirrors sagent merge_consecutive_user_messages).

    Text-only user messages are normalized to a single text chunk list before merging so
    that a merged message always carries a list of content chunks.

    Args:
        messages: The chat messages in OpenAI format.

    Returns:
        A new list with consecutive user messages coalesced.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "user" and merged and merged[-1]["role"] == "user":
            merged[-1]["content"] = _as_chunks(merged[-1]["content"]) + _as_chunks(message["content"])
            continue
        merged.append({"role": message["role"], "content": message["content"]})
    return merged


def _as_chunks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


# ============================================================================
# In-VM runner
# ----------------------------------------------------------------------------
# This file is copied into the OSWorld VM by the sagent_osworld adapter and
# invoked with ``--in-vm``: it runs the agent loop directly on the VM desktop
# with pyautogui, capturing every observation screenshot and cursor position
# into a trajectory.json that the adapter downloads afterwards. Module-level
# imports stay lightweight so the lean VM venv can import this file; pyautogui
# is imported lazily inside the runner.
# ============================================================================

VM_RUN_DIR = "/home/user/osworld_run"
VM_VENV_PYTHON = "/home/user/.venv/bin/python"
VM_CLIENT_PASSWORD = "osworld-public-evaluation"
RUN_RESULT_MARKER = "OSWORLD_RUN_RESULT "


def stringify_model_output(response: Any) -> str:
    """Render a raw model message into readable text for the report.

    Args:
        response: The raw message dict returned by the agent's LLM call, or an
            error payload.

    Returns:
        A truncated, human-readable string combining reasoning, content and any error.
    """
    if not isinstance(response, dict):
        return str(response)[:8000]
    parts: list[str] = []
    reasoning = response.get("reasoning_content") or response.get("reasoning")
    if reasoning:
        parts.append(f"[reasoning]\n{reasoning}")
    content = response.get("content")
    if content:
        parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    if response.get("error"):
        parts.append(f"[error]\n{response['error']}")
    text = "\n\n".join(parts) if parts else json.dumps(response, ensure_ascii=False)
    return text[:8000]


def run_agent_in_vm() -> None:
    """Run the agent loop inside the OSWorld VM (invoked with --in-vm).

    Reads ``config.json`` from the run directory, drives the desktop with pyautogui
    while capturing each observation screenshot and cursor position, and writes
    ``trajectory.json`` plus the per-step PNGs.
    """
    import io

    import pyautogui

    logging.basicConfig(level=logging.INFO)  # noqa: HAI001
    pyautogui.FAILSAFE = False

    run_dir = Path(VM_RUN_DIR)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    screenshots_dir = run_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    screen_size = (int(config["screen_size"][0]), int(config["screen_size"][1]))
    instruction = config["instruction"]
    max_steps = int(config.get("max_steps", DEFAULT_MAX_STEPS))
    wait_after_s = float(config.get("wait_after_s", DEFAULT_WAIT_AFTER_S))
    action_pause_s = float(config.get("action_pause_s", 0.2))

    prompt_path = config.get("prompt_path")
    if not prompt_path:
        raise ValueError("config.json must set 'prompt_path' pointing to the uploaded prompt.j2")

    agent = SagentOSWorldAgent(
        model=config.get("model", DEFAULT_MODEL),
        max_steps=max_steps,
        prompt_path=prompt_path,
        screen_size=screen_size,
        max_images=int(config.get("max_images", DEFAULT_MAX_IMAGES)),
        max_time_s=float(config.get("max_time_s", DEFAULT_MAX_TIME_S)),
        temperature=float(config.get("temperature", DEFAULT_TEMPERATURE)),
        max_completion_tokens=int(config.get("max_completion_tokens", DEFAULT_MAX_COMPLETION_TOKENS)),
        llm_timeout_s=float(config.get("llm_timeout_s", DEFAULT_LLM_TIMEOUT_S)),
        password=config.get("password", VM_CLIENT_PASSWORD),
    )

    def capture() -> tuple[bytes, tuple[int, int]]:
        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        position = pyautogui.position()
        return buffer.getvalue(), (int(position.x), int(position.y))

    exec_globals: dict[str, Any] = {"pyautogui": pyautogui, "time": time}
    steps: list[dict[str, Any]] = []
    feasible = False

    for step_idx in range(max_steps):
        screenshot_bytes, cursor = capture()
        filename = f"step_{step_idx:03d}.png"
        (screenshots_dir / filename).write_bytes(screenshot_bytes)

        response, actions, info = agent.predict(instruction, screenshot_bytes, step_idx)

        step_record: dict[str, Any] = {
            "step": step_idx,
            "screenshot": filename,
            "cursor": {"x": cursor[0], "y": cursor[1]},
            "thought": info.get("thought", ""),
            "note": info.get("note"),
            "action": info.get("action", ""),
            "tool_name": info.get("tool_name", ""),
            "tool_call": info.get("tool_call", {}),
            "code": actions,
            "model_output": stringify_model_output(response),
        }
        steps.append(step_record)

        if actions == ["DONE"]:
            feasible = INFEASIBLE_MARKER not in str(info.get("tool_call", {}).get("content", ""))
            break
        if actions == ["FAIL"]:
            feasible = False
            break

        exec_errors: list[str] = []
        for code in actions:
            try:
                exec(code, exec_globals)  # noqa: S102 - trusted agent-generated pyautogui code
            except Exception as exc:  # noqa: BLE001 - surface any action failure in the trajectory
                exec_errors.append(f"{code!r}: {exc}")
            time.sleep(action_pause_s)
        if exec_errors:
            step_record["exec_errors"] = exec_errors

        tool_name = info.get("tool_name", "")
        if tool_name in DESKTOP_ACTION_TOOLS:
            time.sleep(wait_after_s)

    final_bytes, final_cursor = capture()
    (screenshots_dir / "final.png").write_bytes(final_bytes)

    trajectory = {
        "instruction": instruction,
        "model": agent.model,
        "screen_size": list(screen_size),
        "feasible": feasible,
        "num_steps": len(steps),
        "steps": steps,
        "final_screenshot": "final.png",
        "final_cursor": {"x": final_cursor[0], "y": final_cursor[1]},
    }
    (run_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    print(RUN_RESULT_MARKER + json.dumps({"feasible": feasible, "num_steps": len(steps)}))


def build_report(trajectory: dict[str, Any], screenshots_dir: Path, evaluation: Any, output_path: Path) -> None:
    """Write a self-contained HTML report of the agent run.

    Embeds every screenshot (base64) with a marker at the cursor position and shows
    the agent's observations and outputs for each step.

    Args:
        trajectory: The parsed trajectory.json produced inside the VM.
        screenshots_dir: Local directory holding the downloaded PNG screenshots.
        evaluation: The TaskEvaluation produced from the environment score.
        output_path: Destination path for the HTML file.
    """
    import html

    screen_w, screen_h = trajectory["screen_size"]

    def image_block(filename: str, cursor: dict[str, int] | None) -> str:
        image_path = screenshots_dir / filename
        if not image_path.exists():
            return f"<div class='missing'>missing screenshot: {html.escape(filename)}</div>"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        marker = ""
        if cursor is not None:
            left = cursor["x"] / screen_w * 100
            top = cursor["y"] / screen_h * 100
            marker = (
                f"<div class='cursor' style='left:{left:.3f}%;top:{top:.3f}%' "
                f"title='cursor ({cursor['x']}, {cursor['y']})'></div>"
            )
        return f"<div class='shot'><img src='data:image/png;base64,{encoded}'/>{marker}</div>"

    rows: list[str] = []
    for step in trajectory["steps"]:
        cursor = step.get("cursor")
        note = step.get("note")
        exec_errors = step.get("exec_errors") or []
        errors_html = ""
        if exec_errors:
            joined = "<br>".join(html.escape(err) for err in exec_errors)
            errors_html = f"<div class='errors'><b>exec errors</b><br>{joined}</div>"
        rows.append(
            "<div class='step'>"
            "<div class='left'>"
            f"<div class='badge'>Step {step['step']} &middot; {html.escape(step.get('tool_name', ''))}</div>"
            f"{image_block(step['screenshot'], cursor)}"
            f"<div class='cursorpos'>cursor: ({cursor['x']}, {cursor['y']})</div>"
            "</div>"
            "<div class='right'>"
            f"<div class='action'><b>Action</b><br>{html.escape(step.get('action', ''))}</div>"
            + (f"<div class='note'><b>Note</b><br>{html.escape(str(note))}</div>" if note else "")
            + f"<details open><summary>Thought</summary><pre>{html.escape(step.get('thought', ''))}</pre></details>"
            f"<details><summary>Tool call</summary><pre>{html.escape(json.dumps(step.get('tool_call', {}), indent=2, ensure_ascii=False))}</pre></details>"
            f"<details><summary>pyautogui code</summary><pre>{html.escape(chr(10).join(step.get('code', [])))}</pre></details>"
            f"<details><summary>Raw model output</summary><pre>{html.escape(step.get('model_output', ''))}</pre></details>"
            f"{errors_html}"
            "</div>"
            "</div>"
        )

    final_block = image_block(trajectory.get("final_screenshot", "final.png"), trajectory.get("final_cursor"))

    success = getattr(evaluation, "success", None)
    score = getattr(evaluation, "score", None)
    verdict_class = "ok" if success else "bad"
    summary = (
        f"<tr><td>Instruction</td><td>{html.escape(trajectory['instruction'])}</td></tr>"
        f"<tr><td>Model</td><td>{html.escape(str(trajectory['model']))}</td></tr>"
        f"<tr><td>Steps taken</td><td>{trajectory['num_steps']}</td></tr>"
        f"<tr><td>Agent self-assessment (feasible)</td><td>{trajectory['feasible']}</td></tr>"
        f"<tr><td>Score</td><td>{score}</td></tr>"
        f"<tr><td class='{verdict_class}'>Success</td><td class='{verdict_class}'>{success}</td></tr>"
    )

    style = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
    .wrap{max-width:1400px;margin:0 auto;padding:24px}
    h1{font-size:20px} h2{font-size:16px;margin-top:32px;border-bottom:1px solid #333;padding-bottom:6px}
    table.summary{border-collapse:collapse;margin:12px 0;width:100%}
    table.summary td{border:1px solid #333;padding:6px 10px;vertical-align:top}
    table.summary td:first-child{width:240px;color:#9aa4b2}
    .ok{color:#3fb950;font-weight:600}.bad{color:#f85149;font-weight:600}
    .step{display:flex;gap:18px;padding:18px 0;border-bottom:1px solid #21262d}
    .left{flex:0 0 620px}.right{flex:1;min-width:0}
    .badge{display:inline-block;background:#1f6feb;color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;margin-bottom:8px}
    .shot{position:relative;display:inline-block;border:1px solid #333}
    .shot img{display:block;width:600px;height:auto}
    .cursor{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border:2px solid #f85149;border-radius:50%;box-shadow:0 0 0 2px rgba(248,81,73,.35)}
    .cursorpos{font-size:12px;color:#9aa4b2;margin-top:4px}
    .action{background:#161b22;border-left:3px solid #1f6feb;padding:8px 10px;margin-bottom:8px}
    .note{background:#161b22;border-left:3px solid #d29922;padding:8px 10px;margin-bottom:8px}
    .errors{background:#2d1416;border-left:3px solid #f85149;padding:8px 10px;margin-top:8px}
    details{margin:6px 0}summary{cursor:pointer;color:#9aa4b2}
    pre{white-space:pre-wrap;word-break:break-word;background:#161b22;padding:8px 10px;border-radius:4px;font-size:12px;overflow-x:auto}
    """

    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>OSWorld agent report</title>"
        f"<style>{style}</style></head><body><div class='wrap'>"
        "<h1>OSWorld sagent Forest agent report</h1>"
        f"<table class='summary'>{summary}</table>"
        "<h2>Steps</h2>"
        f"{''.join(rows)}"
        "<h2>Final state</h2>"
        f"<div class='step'><div class='left'>{final_block}"
        f"<div class='cursorpos'>cursor: ({trajectory['final_cursor']['x']}, {trajectory['final_cursor']['y']})</div>"
        "</div></div>"
        "</div></body></html>"
    )
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    run_agent_in_vm()


if __name__ == "__main__":
    main()
