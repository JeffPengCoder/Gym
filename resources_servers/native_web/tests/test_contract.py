# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from nemo_gym.web.models import WebObservation, WebTask
from nemo_gym.web.native_browser import NativeBrowserEvaluationContext
from nemo_gym.web.session import EvaluatorConfigurationError
from resources_servers.browsergym_web.app import BrowserGymWebResourcesServer
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.session_manager import BrowserGymSessionManager
from resources_servers.native_web.app import NativeWebResourcesServer
from resources_servers.native_web.backend import LOCAL_SETUP_RETRY_DELAYS_S, NativeWebDriver
from resources_servers.native_web.config import NativeWebResourcesServerConfig
from resources_servers.native_web.evaluators import NativeTaskEvaluator
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


def _webarena_task(**updates) -> WebTask:
    task = WebTask(
        benchmark="webarena",
        task_id="0",
        intent="Return the expected value",
        runtime_profile="native_visual",
        action_profile="native_toolcall",
        verifier_profile="native_webarena_classic",
        original_metadata={
            "id": "webarena-0",
            "eval": {
                "eval_types": ["string_match"],
                "reference_answers": {"exact_match": "expected"},
            },
        },
    )
    return task.model_copy(update=updates)


def test_native_resource_rejects_non_webarena_benchmark() -> None:
    manager = NativeWebSessionManager(_config())
    with pytest.raises(ValueError, match="benchmark 'webvoyager' is disabled"):
        manager._validate_task(WebTask(benchmark="webvoyager", task_id="0"))


def test_native_resource_rejects_mixed_verifier_profile() -> None:
    manager = NativeWebSessionManager(_config())
    with pytest.raises(ValueError, match="verifier_profile=native_webarena_classic"):
        manager._validate_task(_webarena_task(verifier_profile="browsergym_webarena"))


def test_native_and_browsergym_are_sibling_implementations() -> None:
    assert not issubclass(NativeWebResourcesServer, BrowserGymWebResourcesServer)
    assert not issubclass(NativeWebSessionManager, BrowserGymSessionManager)
    assert not issubclass(NativeWebResourcesServerConfig, BrowserGymWebResourcesServerConfig)


def test_native_evaluator_fails_closed_without_installed_plugin() -> None:
    context = NativeBrowserEvaluationContext(page=object(), browser_context=object(), evidence=())

    with pytest.raises(EvaluatorConfigurationError, match="not installed"):
        NativeTaskEvaluator().prepare(
            task=_webarena_task(),
            observation=WebObservation(),
            browser_context=context,
        )


def test_native_webarena_evaluator_scores_rule_only_task(monkeypatch) -> None:
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")  # pragma: allowlist secret
    context = NativeBrowserEvaluationContext(page=object(), browser_context=object(), evidence=())
    evaluator = NativeTaskEvaluator(config=_config())
    task = _webarena_task()

    evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
    result = evaluator.evaluate(
        task=task,
        observation=WebObservation(),
        final_answer="expected",
        browser_context=context,
    )

    assert result.reward == 1.0
    assert result.task_success
    assert result.valid_sample
    assert result.verifier_version == "native-webarena-3b775dc"


def test_native_webarena_evaluator_merges_api_and_browser_snapshots(monkeypatch) -> None:
    from resources_servers.native_web import reference_evaluation

    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")  # pragma: allowlist secret
    api_snapshots = iter(
        [
            {"shopping_orders": [{"increment_id": "1"}]},
            {"shopping_orders": [{"increment_id": "1"}, {"increment_id": "2"}]},
        ]
    )
    browser_snapshots = iter(
        [
            {"program_html": [{"key": "shared", "value": "before"}]},
            {"program_html": [{"key": "shared", "value": "after"}]},
        ]
    )
    captured = {}
    monkeypatch.setattr(reference_evaluation, "collect_snapshots", lambda _plan: next(api_snapshots))
    monkeypatch.setattr(
        reference_evaluation,
        "collect_browser_snapshots_sync",
        lambda _page, _plan: next(browser_snapshots),
    )

    def build_context(plan, before, after):
        captured.update(plan=plan, before=before, after=after)
        return {"snapshots": {"before": before, "after": after}}

    monkeypatch.setattr(reference_evaluation, "build_snapshot_context", build_context)
    monkeypatch.setattr(reference_evaluation, "evaluate_classic_task_sync", lambda *_args, **_kwargs: (1.0, "ok"))
    collision_plan = {
        "snapshot_adapters": {
            "shopping_orders": {},
            "program_html": {"targets": []},
        },
        "target_overrides": {},
    }
    task = _webarena_task(task_kwargs={"collision_plan": collision_plan})
    evaluator = NativeTaskEvaluator(config=_config())
    context = NativeBrowserEvaluationContext(page=object(), browser_context=object(), evidence=())

    evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
    evaluator.evaluate(
        task=task,
        observation=WebObservation(),
        final_answer="expected",
        browser_context=context,
    )

    assert captured["before"] == {
        "shopping_orders": [{"increment_id": "1"}],
        "program_html": [{"key": "shared", "value": "before"}],
    }
    assert captured["after"] == {
        "shopping_orders": [{"increment_id": "1"}, {"increment_id": "2"}],
        "program_html": [{"key": "shared", "value": "after"}],
    }


def test_native_config_enforces_one_headed_session_per_display() -> None:
    with pytest.raises(ValueError, match="headed Chromium"):
        _config(headless=True)
    with pytest.raises(ValueError, match="max_sessions=1"):
        _config(max_sessions=2)


class _RecordingPage:
    def __init__(self) -> None:
        self.url = "https://example.test/"
        self.goto_calls = []
        self._errors = [RuntimeError("Timeout 45000ms exceeded"), RuntimeError("container still settling")]

    def goto(self, url: str, wait_until: str = "load"):
        self.goto_calls.append((url, wait_until))
        if self._errors:
            raise self._errors.pop(0)


def test_webarena_setup_retries_initial_local_navigation(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("resources_servers.native_web.backend.time.sleep", sleeps.append)
    driver = NativeWebDriver(_config(), "session-test", object())
    driver._task = _webarena_task(task_id="17")
    page = _RecordingPage()

    driver._goto_task_start(page, "http://webarena.test/start")

    assert len(page.goto_calls) == 3
    assert sleeps == list(LOCAL_SETUP_RETRY_DELAYS_S)


def test_native_component_declares_isolated_runtime_dependencies() -> None:
    project = tomllib.loads((NATIVE_WEB_ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert "nemo-gym" in dependencies
    assert "nemo-gym[dev]" not in dependencies
    assert "playwright==1.55.0" in dependencies
    assert project["tool"]["uv"]["sources"]["nemo-gym"] == {
        "path": "../..",
        "editable": True,
    }
