# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Yi's Sagent/Holotron-3 Gym compatibility path."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent.sagent_holo3_agent import (
    PROMPT_SHA256,
    SCHEMA_SERIALIZED_SHA256,
    SOURCE_FIDELITY_GUARDRAIL,
    SOURCE_FIDELITY_GUARDRAIL_VERSION,
    SOURCE_PASSWORD,
    SagentHolo3Agent,
    load_frozen_schema,
    verify_sagent_assets,
)
from responses_api_agents.osworld_agent.vendor.sagent import sagent_osworld_agent as vendor


def _response(tool_call: Dict[str, Any], *, thought: str = "Act carefully.", note: str | None = None) -> Dict[str, str]:
    return {"content": json.dumps({"note": note, "thought": thought, "tool_call": tool_call})}


def _image_count(messages: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


def test_sagent_assets_and_pydantic_211_schema_are_frozen() -> None:
    hashes = verify_sagent_assets()
    schema = load_frozen_schema()

    assert hashes["prompt_sha256"] == PROMPT_SHA256
    assert hashlib.sha256(json.dumps(schema).encode("utf-8")).hexdigest() == SCHEMA_SERIALIZED_SHA256
    assert len(schema["$defs"]) == 17
    assert len(schema["properties"]["tool_call"]["anyOf"]) == 16


def test_sagent_prompt_adapts_password_without_mutating_source() -> None:
    agent = SagentHolo3Agent(
        max_steps=3,
        client_password="password",  # pragma: allowlist secret
        transport_retries=1,
        transport_retry_sleep_s=0,
    )

    assert agent.prompt_source_sha256 == PROMPT_SHA256
    assert agent.prompt_effective_sha256 == "099e37c8e0cbe65e351630c0d234df9b3e2828399a3b41a1273e01138750f997"
    assert agent.prompt_password_replacement_count == 2
    assert SOURCE_PASSWORD not in agent.system_prompt
    assert "The computer's password is: 'password'" in agent.system_prompt
    assert json.dumps(agent.schema) in agent.system_prompt


def test_sagent_request_and_two_image_history_match_yi_recipe() -> None:
    agent = SagentHolo3Agent(
        model="Hcompany/Holotron-3-Nano",
        max_steps=3,
        max_tokens=4096,
        temperature=0.8,
        top_p=0.95,
        max_image_history_length=2,
        transport_retries=1,
        transport_retry_sleep_s=0,
    )
    payloads: List[Dict[str, Any]] = []
    responses = [
        _response(
            {
                "tool_name": "update_plan",
                "goals": [{"title": "Inspect the desktop", "status": "running"}],
            }
        ),
        _response(
            {
                "tool_name": "click_desktop",
                "element": "center button",
                "x": 500,
                "y": 250,
                "button": "left",
            }
        ),
        _response({"tool_name": "answer", "content": "Completed."}),
    ]

    def call_llm(payload: Dict[str, Any], _model: str | None = None) -> Dict[str, str]:
        payloads.append(payload)
        return responses[len(payloads) - 1]

    agent.call_llm = call_llm
    first = agent.predict("Complete the task.", {"screenshot": b"png-1"})
    second = agent.predict("Complete the task.", {"screenshot": b"png-2"})
    third = agent.predict("Complete the task.", {"screenshot": b"png-3"})

    assert first[1] == []
    assert first[2]["tool_name"] == "update_plan"
    assert second[1] == ["pyautogui.click(960, 270, button='left')"]
    assert second[2]["_osworld_sleep_after_execution"] == pytest.approx(3.2)
    assert third[1] == ["DONE"]
    assert third[2]["_osworld_sleep_after_execution"] == 0.0
    assert [_image_count(payload["messages"]) for payload in payloads] == [1, 2, 2]
    assert any(
        part.get("text") == f"<observation>\n{vendor.IMAGE_PLACEHOLDER}\n</observation>"
        for message in payloads[2]["messages"]
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(part, dict)
    )

    request = payloads[0]
    assert request["model"] == "Hcompany/Holotron-3-Nano"
    assert request["temperature"] == 0.8
    assert request["top_p"] == 0.95
    assert request["max_completion_tokens"] == 4096
    assert "max_tokens" not in request
    assert "chat_template_kwargs" not in request
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "NoteStructuredOutput", "schema": agent.schema},
    }
    assert request["_nemo_gym_return_message"] is True
    assert request["_nemo_gym_require_stop"] is True
    assert SOURCE_FIDELITY_GUARDRAIL not in json.dumps(request["messages"])
    assert agent.history[0]["tool_result"] == "1. Inspect the desktop [running]"
    assert agent.history[1]["tool_result"] == ""


def test_source_fidelity_guardrail_is_explicit_and_does_not_mutate_vendor_prompt() -> None:
    baseline = SagentHolo3Agent(
        max_steps=1,
        transport_retries=1,
        transport_retry_sleep_s=0,
    )
    agent = SagentHolo3Agent(
        max_steps=1,
        preserve_source_fidelity=True,
        transport_retries=1,
        transport_retry_sleep_s=0,
    )
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str | None = None) -> Dict[str, str]:
        payloads.append(payload)
        return _response({"tool_name": "answer", "content": "Done."})

    agent.call_llm = call_llm
    _response_value, actions, info = agent.predict(
        "Complete the supplied source file.",
        {"screenshot": b"png"},
    )

    request_text = "\n".join(
        str(part.get("text") or "")
        for message in payloads[0]["messages"]
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(part, dict) and part.get("type") == "text"
    )
    assert actions == ["DONE"]
    assert request_text.count("Complete the supplied source file.") == 1
    assert request_text.count(SOURCE_FIDELITY_GUARDRAIL) == 1
    assert '<source_editing_guardrail version="2">' in request_text
    assert "caret is already after leading whitespace" in request_text
    assert "prefer Undo or one localized correction" in request_text
    assert "reconstruct the complete original faithfully" in request_text
    assert agent.system_prompt == baseline.system_prompt
    assert agent.prompt_effective_sha256 == baseline.prompt_effective_sha256
    assert info["_osworld_instruction_guardrail_version"] == SOURCE_FIDELITY_GUARDRAIL_VERSION
    assert info["_osworld_instruction_guardrail_sha256"] == agent.instruction_guardrail_sha256


def test_sagent_infeasible_answer_becomes_osworld_fail() -> None:
    agent = SagentHolo3Agent(max_steps=2, transport_retries=1, transport_retry_sleep_s=0)
    agent.call_llm = lambda *_args: _response(
        {"tool_name": "answer", "content": f"Cannot be done. {vendor.INFEASIBLE_MARKER}"}
    )

    _response_value, actions, info = agent.predict("Impossible task.", {"screenshot": b"png"})

    assert actions == ["FAIL"]
    assert info["_osworld_outcome"] == "infeasible"
    assert info["_osworld_sleep_after_execution"] == 0.0


def test_sagent_wait_desktop_keeps_yi_pyautogui_pause() -> None:
    agent = SagentHolo3Agent(
        max_steps=2,
        action_pause_s=0.2,
        transport_retries=1,
        transport_retry_sleep_s=0,
    )
    agent.call_llm = lambda *_args: _response({"tool_name": "wait_desktop", "seconds": 2})

    _response_value, actions, info = agent.predict("Wait.", {"screenshot": b"png"})

    assert actions == ["time.sleep(2.0)"]
    assert info["_osworld_sleep_after_execution"] == pytest.approx(0.2)


def test_sagent_max_step_exhaustion_is_evaluated_without_fail() -> None:
    agent = SagentHolo3Agent(max_steps=1, transport_retries=1, transport_retry_sleep_s=0)
    agent.call_llm = lambda *_args: _response(
        {"tool_name": "click_desktop", "element": "button", "x": 10, "y": 20, "button": "left"}
    )

    _response_value, actions, info = agent.predict("Click.", {"screenshot": b"png"})

    assert actions == []
    assert info["_osworld_outcome"] == "max_steps_exhausted"
    assert "maximum step limit" in info["action"].lower()


def test_sagent_parse_failure_is_score_zero_not_infeasible(monkeypatch) -> None:
    monkeypatch.setattr(vendor.time, "sleep", lambda _seconds: None)
    agent = SagentHolo3Agent(max_steps=2, transport_retries=1, transport_retry_sleep_s=0)
    call_count = 0

    def invalid_response(*_args: Any) -> Dict[str, str]:
        nonlocal call_count
        call_count += 1
        return {"content": "{}"}

    agent.call_llm = invalid_response
    _response_value, actions, info = agent.predict("Do the task.", {"screenshot": b"png"})

    assert call_count == 5
    assert actions == []
    assert info["tool_name"] == "parse_error"
    assert info["_osworld_force_score_zero"] is True
    assert info["_osworld_failure_reason"] == "parse_error"


def test_sagent_optional_thinking_and_unconstrained_modes_are_explicit() -> None:
    agent = SagentHolo3Agent(
        max_steps=1,
        enable_thinking=False,
        drop_response_format=True,
        transport_retries=1,
        transport_retry_sleep_s=0,
    )
    payloads: List[Dict[str, Any]] = []

    def call_llm(payload: Dict[str, Any], _model: str | None = None) -> Dict[str, str]:
        payloads.append(payload)
        return _response({"tool_name": "answer", "content": "Done."})

    agent.call_llm = call_llm
    agent.predict("Finish.", {"screenshot": b"png"})

    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payloads[0]
