# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from nemo_gym.web.models import WebBenchmark, WebTask
from resources_servers.webvoyager_judge.app import (
    WebVoyagerJudgeResourcesServer,
    parse_native_verdict,
    parse_verdict,
)
from resources_servers.webvoyager_judge.config import WebVoyagerJudgeConfig
from resources_servers.webvoyager_judge.models import WebVoyagerJudgeRequest


def _model_response(text: str) -> dict:
    return {
        "id": "judge-response",
        "created_at": 0.0,
        "model": "judge",
        "object": "response",
        "output": [
            {
                "id": "judge-message",
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


class _FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    @property
    def content(self):
        payload = self.payload

        class _Body:
            async def read(self):
                return json.dumps(payload).encode()

        return _Body()

    async def read(self):
        return json.dumps(self.payload).encode()


def _server():
    config = WebVoyagerJudgeConfig(
        name="webvoyager_judge",
        host="localhost",
        port=8002,
        entrypoint="app.py",
        domain="agent",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge-model"),
        judge_responses_create_params={"input": "placeholder", "temperature": 0},
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return WebVoyagerJudgeResourcesServer(config=config, server_client=client)


def _request(answer="answer"):
    return WebVoyagerJudgeRequest(
        task=WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="site--0",
            intent="Find the requested fact",
            start_urls=["https://example.test"],
        ),
        final_answer=answer,
        screenshots=["data:image/png;base64,abc"],
    )


def test_not_success_is_checked_before_success():
    assert parse_verdict("The task is NOT SUCCESS") is False
    assert parse_verdict("The task is a SUCCESS") is True
    assert parse_verdict("unclear") is None


def test_native_verdict_requires_json_success_or_failure():
    assert parse_native_verdict('{"thought":"ok","verdict":"SUCCESS"}') == (
        True,
        {"thought": "ok", "verdict": "SUCCESS"},
    )
    assert parse_native_verdict("SUCCESS") is None
    assert parse_native_verdict('{"verdict":"NOT SUCCESS"}') is None


@pytest.mark.asyncio
async def test_empty_answer_is_valid_policy_failure_without_judge_call():
    server = _server()
    server.server_client.post = AsyncMock()

    response = await server.verify_webvoyager(_request(answer=""))

    assert response.result.valid_sample is True
    assert response.result.reward == 0.0
    assert response.result.failure_kind == "agent_no_final_answer"
    server.server_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_judge_success_returns_binary_reward():
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(_model_response("The screenshot proves completion. SUCCESS"))
    )

    response = await server.verify_webvoyager(_request())

    assert response.result.valid_sample is True
    assert response.result.reward == 1.0
    assert response.result.task_success is True


@pytest.mark.asyncio
async def test_judge_logs_lifecycle_without_screenshot_payload(caplog):
    server = _server()
    server.server_client.post = AsyncMock(
        return_value=_FakeHttpResponse(_model_response("The screenshot proves completion. SUCCESS"))
    )

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.webvoyager_judge"):
        response = await server.verify_webvoyager(_request())

    assert response.result.task_success is True
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=webvoyager_judge_start" in messages
    assert "event=webvoyager_judge_model_start" in messages
    assert "event=webvoyager_judge_model_complete" in messages
    assert "event=webvoyager_judge_complete" in messages
    assert "origins=none" in messages
    assert "data:image/png;base64,abc" not in messages
    assert "The screenshot proves completion" not in messages
