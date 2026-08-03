# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from nemo_gym.server_utils import ServerClient
from nemo_gym.web.models import WebBenchmark, WebTask
from responses_api_agents.web_agent.app import (
    WebAgent,
    WebAgentConfig,
    WebAgentRunRequest,
    _redact_old_images,
)


def _model_response(text: str) -> dict:
    return {
        "id": "response-1",
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "output": [
            {
                "id": "message-1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _observation(url="https://example.test") -> dict:
    return {
        "goal": [{"type": "text", "text": "Complete the task"}],
        "axtree_text": "[a1] button 'Done'",
        "screenshot": {"data_url": "data:image/png;base64,abc"},
        "url": url,
    }


class _FakeHttpResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    @property
    def content(self):
        payload = self._payload

        class _Body:
            async def read(self):
                return json.dumps(payload).encode()

        return _Body()

    def raise_for_status(self):
        return None


def _agent(*, parse_retries=1, judge=False):
    config = WebAgentConfig(
        name="web_agent",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="browser"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        webvoyager_judge_server=(ResourcesServerRef(type="resources_servers", name="judge") if judge else None),
        max_parse_retries=parse_retries,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return WebAgent(config=config, server_client=client)


def _wire(agent, payloads):
    calls = []

    async def post(server_name, url_path, json=None, cookies=None, **kwargs):
        calls.append((server_name, url_path, json))
        return _FakeHttpResponse(payloads[url_path].pop(0))

    agent.server_client.post = AsyncMock(side_effect=post)
    return calls


def _seed(task_id="0"):
    return {
        "session_id": "session-a",
        "task_id": task_id,
        "status": "ready",
        "observation": _observation(),
        "info": {},
    }


def test_redact_old_visual_observation_removes_both_image_and_page_text():
    old = NeMoGymEasyInputMessage(
        role="user",
        content=[
            {"type": "input_text", "text": "large page tree"},
            {"type": "input_image", "image_url": "data:image/png;base64,old", "detail": "high"},
        ],
    )
    current = NeMoGymEasyInputMessage(
        role="user",
        content=[
            {"type": "input_text", "text": "current page tree"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,current",
                "detail": "high",
            },
        ],
    )

    redacted = _redact_old_images(
        [old, current],
        1,
        redact_observation_text=True,
    )

    assert redacted[0].content == [
        {
            "type": "input_text",
            "text": "[Earlier screenshot and page text omitted from context.]",
        }
    ]
    assert len(redacted[1].content) == 2
    assert old.content[0]["text"] == "large page tree"


@pytest.mark.asyncio
async def test_webarena_rollout_uses_colocated_evaluator_and_closes_session():
    agent = _agent()
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed()],
            "/v1/responses": [_model_response("Thought: done\nAction: send_msg_to_user('answer')")],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation(),
                    "execution_ok": True,
                    "benchmark_reward": 1.0,
                    "terminated": True,
                    "truncated": False,
                    "info": {},
                }
            ],
            "/evaluate": [
                {
                    "result": {
                        "reward": 1.0,
                        "raw_score": 1.0,
                        "task_success": True,
                        "valid_sample": True,
                    }
                }
            ],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve the task"},
        web_task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0"),
    )

    result = await agent.run(request, body)

    assert result.reward == 1.0
    assert result.task_success is True
    assert result.environment_steps == 1
    assert result.mask_sample is False
    step_body = next(body for _server, path, body in calls if path == "/step")
    assert step_body["action"]["script"] == "send_msg_to_user('answer')"
    assert calls[-1][1] == "/close"


@pytest.mark.asyncio
async def test_action_parse_failure_is_retried_without_stepping_browser():
    agent = _agent(parse_retries=1)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed()],
            "/v1/responses": [
                _model_response("I forgot the action"),
                _model_response("Thought: retry\nAction: send_msg_to_user('answer')"),
            ],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation(),
                    "execution_ok": True,
                    "benchmark_reward": 0.0,
                    "terminated": True,
                    "truncated": False,
                }
            ],
            "/evaluate": [{"result": {"valid_sample": True}}],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0"),
    )

    result = await agent.run(request, body)

    assert result.model_turns == 2
    assert result.environment_steps == 1
    assert [path for _server, path, _body in calls].count("/step") == 1


@pytest.mark.asyncio
async def test_webvoyager_routes_final_evidence_to_external_judge():
    agent = _agent(judge=True)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed("Allrecipes--0")],
            "/v1/responses": [_model_response("Thought: done\nAction: ANSWER; [42]")],
            "/step": [
                {
                    "operation_id": "step-0",
                    "observation": _observation("https://example.test/result"),
                    "execution_ok": True,
                }
            ],
            "/evaluate": [
                {
                    "result": {
                        "valid_sample": False,
                        "failure_kind": "external_judge_required",
                    }
                }
            ],
            "/verify_webvoyager": [
                {
                    "result": {
                        "reward": 1.0,
                        "raw_score": 1.0,
                        "task_success": True,
                        "valid_sample": True,
                    }
                }
            ],
            "/close": [{"closed": True}],
        },
    )
    request = MagicMock()
    request.cookies = {}
    body = WebAgentRunRequest(
        responses_create_params={"input": "Solve"},
        web_task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Allrecipes--0",
            intent="Find the answer",
            start_urls=["https://example.test"],
            action_profile="webvoyager_legacy",
        ),
    )

    result = await agent.run(request, body)

    assert result.reward == 1.0
    judge_call = next(call for call in calls if call[1] == "/verify_webvoyager")
    assert judge_call[0] == "judge"
    assert judge_call[2]["final_answer"] == "42"
    assert len(judge_call[2]["screenshots"]) == 2
