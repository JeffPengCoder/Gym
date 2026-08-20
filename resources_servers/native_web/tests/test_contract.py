# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.web.datasets import adapt_native_webvoyager_record
from nemo_gym.web.models import WebTask
from nemo_gym.web.native_webvoyager import NATIVE_WEBVOYAGER_SYSTEM_PROMPT, NATIVE_WEBVOYAGER_TOOLS
from resources_servers.webvoyager_judge.prompts import NATIVE_WEBVOYAGER_JUDGE_PROMPT
from resources_servers.native_web.backend import NativeWebDriver
from resources_servers.native_web.config import NativeWebResourcesServerConfig
from resources_servers.native_web.session_manager import NativeWebSessionManager


NATIVE_WEB_ROOT = Path(__file__).resolve().parents[1]


def _config(**updates) -> NativeWebResourcesServerConfig:
    return NativeWebResourcesServerConfig.model_validate(
        {
            "name": "native",
            "host": "localhost",
            "port": 8010,
            "entrypoint": "app.py",
            "domain": "agent",
            "num_workers": 1,
            "headless": False,
            **updates,
        }
    )


def test_native_dataset_row_binds_prompt_tools_and_runtime() -> None:
    row = adapt_native_webvoyager_record(
        {
            "id": "Allrecipes--0",
            "ques": "Find a recipe",
            "web": "https://example.com",
            "web_name": "Allrecipes",
        }
    )

    task = WebTask.model_validate(row["web_task"])
    params = row["responses_create_params"]
    NeMoGymResponseCreateParamsNonStreaming.model_validate(params)
    assert task.runtime_profile.value == "native_visual"
    assert task.observation_profile.value == "screenshot"
    assert task.action_profile.value == "native_toolcall"
    assert params["input"] == []
    assert params["parallel_tool_calls"] is True
    assert [tool["name"] for tool in params["tools"]] == [tool["name"] for tool in NATIVE_WEBVOYAGER_TOOLS]


def test_native_recipe_prompt_and_tool_hashes_are_pinned() -> None:
    tools = json.dumps(NATIVE_WEBVOYAGER_TOOLS, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(NATIVE_WEBVOYAGER_SYSTEM_PROMPT.encode()).hexdigest() == (
        "8332b42f09c577837b1e50bb5c04c857f8942eda6ea692b32eba38deb8cb0d36"
    )
    assert hashlib.sha256(tools.encode()).hexdigest() == (
        "12d525341f568cf3638e1b9dc99058fadf59e3bfa3719d9e88cb021e0e192f09"
    )
    assert hashlib.sha256(NATIVE_WEBVOYAGER_JUDGE_PROMPT.encode()).hexdigest() == (
        "d5548ef2bb6f0641bc9ff116fe721bf540d096502e2040890b2bf1c8560d3325"
    )


def test_native_resource_rejects_browsergym_task() -> None:
    manager = NativeWebSessionManager(_config())
    with pytest.raises(ValueError, match="runtime_profile=native_visual"):
        manager._validate_task(WebTask(benchmark="webvoyager", task_id="0"))


def test_native_config_rejects_headless_execution() -> None:
    with pytest.raises(ValueError, match="headed Chromium"):
        _config(headless=True)


def test_native_config_rejects_multiple_sessions_on_one_display() -> None:
    with pytest.raises(ValueError, match="max_sessions=1"):
        _config(max_sessions=2)


def test_native_component_declares_parent_gym_runtime() -> None:
    """The isolated component venv must not borrow FastAPI from its launcher."""

    project = tomllib.loads((NATIVE_WEB_ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    assert "nemo-gym" in dependencies
    assert "nemo-gym[dev]" not in dependencies
    assert "playwright==1.55.0" in dependencies
    assert project["tool"]["uv"]["sources"]["nemo-gym"] == {
        "path": "../..",
        "editable": True,
    }


def test_native_driver_defers_transient_captcha_solver_error(caplog) -> None:
    class _FailingSolver:
        def maybe_solve(self, _page, *, phase: str) -> bool:
            assert phase == "before post-action screenshot"
            raise TimeoutError("provider detail must not escape")

    driver = NativeWebDriver(_config(), "session-test", object())
    driver._captcha_solver = _FailingSolver()
    driver._page = type("Page", (), {"url": "https://example.test/private?query=secret"})()
    driver._task = WebTask(benchmark="webvoyager", task_id="GitHub--14")

    with caplog.at_level(logging.WARNING, logger="nemo_gym.resources_servers.native_web"):
        assert driver._maybe_solve_captcha("before post-action screenshot") is False

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_solver_deferred" in messages
    assert "error_type=TimeoutError" in messages
    assert "provider detail must not escape" not in messages
    assert "private?query=secret" not in messages
