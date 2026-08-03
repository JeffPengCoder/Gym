# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the adapter-owned Holo3 OSWorld scaffold."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent.holo3_agent import Holo3Agent, Holo3Step


def _step(tool_call: Dict[str, Any], *, note: str | None = None, thought: str = "Act carefully.") -> str:
    return json.dumps({"note": note, "thought": thought, "tool_call": tool_call})


def _image_count(messages: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


def test_holo3_schema_uses_documented_unsuffixed_tool_names() -> None:
    schema = Holo3Step.model_json_schema()
    constants: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("const"), str):
                constants.add(value["const"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(schema)
    assert {"click", "write", "hotkey", "answer", "wait"} <= constants
    assert not any(name.endswith("_desktop") for name in constants)


def test_holo3_request_uses_structured_output_thinking_and_normalized_coordinates() -> None:
    agent = Holo3Agent(
        model="Hcompany/Holo3-35B-A3B",
        screen_size=(1920, 1080),
        temperature=0.6,
        top_p=0.95,
    )
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, str]:
        payloads.append(payload)
        return {
            "content": _step(
                {
                    "tool_name": "click",
                    "element": "Settings button",
                    "x": 500,
                    "y": 250,
                    "button": "left",
                },
                note="Settings is visible.",
            ),
            "reasoning_content": "Hidden reasoning is logged but not replayed.",
        }

    agent.call_llm = call_llm
    content, actions, info = agent.predict("Open Settings.", {"screenshot": b"png-one"})

    assert json.loads(content)["tool_call"]["tool_name"] == "click"
    assert actions == ["pyautogui.click(x=960, y=270, button='left')"]
    assert info["note"] == "Settings is visible."
    assert info["reasoning"] == "Hidden reasoning is logged but not replayed."
    payload = payloads[0]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["extra_body"] == {
        "reasoning_effort": "medium",
        "structured_outputs": {"json": agent.schema},
    }
    assert payload["_nemo_gym_return_message"] is True
    assert payload["_nemo_gym_require_stop"] is True
    assert _image_count(payload["messages"]) == 1
    assert "<output_format>" in payload["messages"][0]["content"]
    assert "Settings button" not in payload["messages"][0]["content"]


def test_holo3_history_replays_parsed_json_and_keeps_only_three_images() -> None:
    agent = Holo3Agent(max_image_history_length=3)
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, str]:
        payloads.append(payload)
        index = len(payloads)
        return {
            "content": _step(
                {"tool_name": "wait", "duration": 1},
                note=f"durable-{index}",
                thought=f"wait-{index}",
            ),
            "reasoning_content": f"private-{index}",
        }

    agent.call_llm = call_llm
    for index in range(4):
        agent.predict("Observe the desktop.", {"screenshot": f"png-{index}".encode()})
        agent.record_action_result(actions=["pyautogui.sleep(1)"], reward=0.0, done=False, info={})

    assert [_image_count(payload["messages"]) for payload in payloads] == [1, 2, 3, 3]
    fourth_messages = payloads[3]["messages"]
    assert _image_count(fourth_messages) == 3
    assert any(
        message.get("content") == "<observation>\n[screenshot evicted]\n</observation>" for message in fourth_messages
    )
    assistant_contents = [message["content"] for message in fourth_messages if message["role"] == "assistant"]
    assert len(assistant_contents) == 3
    assert all(json.loads(content)["note"].startswith("durable-") for content in assistant_contents)
    assert all("private-" not in content for content in assistant_contents)
    tool_outputs = [
        message["content"]
        for message in fourth_messages
        if message["role"] == "user"
        and isinstance(message.get("content"), str)
        and "<tool_output" in message["content"]
    ]
    assert tool_outputs == ['<tool_output tool="wait">\nWaited 1 seconds.\n</tool_output>'] * 3


def test_holo3_retries_invalid_json_without_replaying_raw_output() -> None:
    agent = Holo3Agent(parse_retries=2)
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, str]:
        payloads.append(payload)
        if len(payloads) == 1:
            return {"content": '{"thought":"missing tool"}', "reasoning_content": "bad"}
        return {
            "content": _step({"tool_name": "answer", "content": "Completed."}),
            "reasoning_content": "good",
        }

    agent.call_llm = call_llm
    _content, actions, info = agent.predict("Finish.", {"screenshot": b"png"})

    assert actions == ["DONE"]
    assert info["parse_attempt"] == 2
    assert len(payloads) == 2
    assert payloads[0]["messages"] == payloads[1]["messages"]
    assert "missing tool" not in json.dumps(payloads[1]["messages"])


@pytest.mark.parametrize(
    ("tool_call", "expected"),
    [
        (
            {"tool_name": "write", "content": "hello", "press_enter": True, "overwrite": True},
            "pyautogui.hotkey('ctrl', 'a')\npyautogui.write('hello', interval=0.05)\npyautogui.press('enter')",
        ),
        (
            {
                "tool_name": "scroll",
                "element": "page",
                "x": 1000,
                "y": 1000,
                "direction": "down",
                "scroll_size": 7,
            },
            "pyautogui.moveTo(1919, 1079)\npyautogui.scroll(-7)",
        ),
        (
            {"tool_name": "hotkey", "keys": ["ctrl", "s"], "repeat_count": 2},
            "pyautogui.hotkey('ctrl', 's')\npyautogui.hotkey('ctrl', 's')",
        ),
        (
            {
                "tool_name": "hold_and_tap_key",
                "hold_keys": ["command"],
                "tap_keys": ["return", "escape"],
            },
            "pyautogui.keyDown('ctrl')\npyautogui.press('enter')\npyautogui.press('esc')\npyautogui.keyUp('ctrl')",
        ),
        ({"tool_name": "answer", "content": "done"}, "DONE"),
    ],
)
def test_holo3_tool_projection(tool_call: Dict[str, Any], expected: str) -> None:
    agent = Holo3Agent(screen_size=(1920, 1080))
    parsed = Holo3Step.model_validate_json(_step(tool_call))

    assert agent._tool_actions(parsed.tool_call) == [expected]


def test_holo3_action_result_error_is_returned_with_error_wrapper() -> None:
    agent = Holo3Agent()
    responses = [
        _step({"tool_name": "click", "element": "button", "x": 10, "y": 20}),
        _step({"tool_name": "answer", "content": "done"}),
    ]
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, str]:
        payloads.append(payload)
        return {"content": responses[len(payloads) - 1], "reasoning_content": ""}

    agent.call_llm = call_llm
    agent.predict("Click then finish.", {"screenshot": b"png-1"})
    agent.record_action_result(actions=[], reward=0.0, done=False, info={}, error="click failed")
    agent.predict("Click then finish.", {"screenshot": b"png-2"})

    assert any(
        message.get("content") == '<error tool="click">\nclick failed\n</error>'
        for message in payloads[1]["messages"]
    )
