# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.web.actions import MAX_NATIVE_SCROLL_AMOUNT
from nemo_gym.web.datasets import adapt_native_webvoyager_record
from nemo_gym.web.models import (
    BROWSER_TARGET_CLOSED_STATUS,
    CAPTCHA_BUDGET_EXHAUSTED_STATUS,
    WebAction,
    WebArtifactRef,
    WebObservation,
    WebTask,
)
from nemo_gym.web.native_webvoyager import NATIVE_WEBVOYAGER_SYSTEM_PROMPT, NATIVE_WEBVOYAGER_TOOLS
from nemo_gym.web.session import EvaluatorConfigurationError
from resources_servers.browsergym_web.app import BrowserGymWebResourcesServer
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.session_manager import BrowserGymSessionManager
from resources_servers.native_web.app import NativeWebResourcesServer
from resources_servers.native_web.backend import (
    LOCAL_SETUP_RETRY_DELAYS_S,
    NAVIGATION_RETRY_DELAYS_S,
    NAVIGATION_WAIT_UNTIL,
    NativeBrowserEvaluationContext,
    NativeWebDriver,
    _type_browser_text,
)
from resources_servers.native_web.config import NativeWebResourcesServerConfig
from resources_servers.native_web.evaluators import NativeTaskEvaluator
from resources_servers.native_web.session_manager import NativeWebSessionManager
from resources_servers.webvoyager_browser.backend import (
    WebVoyagerBrowserDriver,
    _is_playwright_target_closed_error,
)
from resources_servers.webvoyager_browser.config import WebVoyagerBrowserResourcesServerConfig
from resources_servers.webvoyager_browser.evaluators import WebVoyagerEvidenceEvaluator
from resources_servers.webvoyager_judge.prompts import NATIVE_WEBVOYAGER_JUDGE_PROMPT


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


def _webvoyager_config(**updates) -> WebVoyagerBrowserResourcesServerConfig:
    return WebVoyagerBrowserResourcesServerConfig.model_validate(
        {
            "name": "webvoyager-browser",
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
        "8332b42f09c577837b1e50bb5c04c857f8942eda6ea692b32eba38deb8cb0d36"  # pragma: allowlist secret
    )
    assert hashlib.sha256(tools.encode()).hexdigest() == (
        "48b135165e158eb0837f0ff6606e978a8a08ee70a1c41e35b2bc33c891af3d54"  # pragma: allowlist secret
    )
    assert hashlib.sha256(NATIVE_WEBVOYAGER_JUDGE_PROMPT.encode()).hexdigest() == (
        "d5548ef2bb6f0641bc9ff116fe721bf540d096502e2040890b2bf1c8560d3325"  # pragma: allowlist secret
    )


def test_native_resource_rejects_browsergym_task() -> None:
    manager = NativeWebSessionManager(_config())
    with pytest.raises(ValueError, match="benchmark 'webvoyager' is disabled"):
        manager._validate_task(WebTask(benchmark="webvoyager", task_id="0"))


def test_native_resource_rejects_mixed_verifier_profile() -> None:
    manager = NativeWebSessionManager(_config(allowed_benchmarks=["webarena"]))
    task = WebTask.model_validate(
        {
            "benchmark": "webarena",
            "task_id": "0",
            "runtime_profile": "native_visual",
            "action_profile": "native_toolcall",
            "verifier_profile": "browsergym_webarena",
        }
    )

    with pytest.raises(ValueError, match="verifier_profile=native_webarena_classic"):
        manager._validate_task(task)


def test_native_and_browsergym_are_sibling_implementations() -> None:
    assert not issubclass(NativeWebResourcesServer, BrowserGymWebResourcesServer)
    assert not issubclass(NativeWebSessionManager, BrowserGymSessionManager)
    assert not issubclass(NativeWebResourcesServerConfig, BrowserGymWebResourcesServerConfig)


def test_native_webvoyager_evaluator_preserves_external_judge_evidence() -> None:
    evidence = WebArtifactRef(
        uri="artifact://session/step-0.png",
        mime_type="image/png",
        size_bytes=123,
        sha256="0" * 64,
    )
    task = WebTask(
        benchmark="webvoyager",
        task_id="Allrecipes--0",
        runtime_profile="native_visual",
        action_profile="native_toolcall",
    )
    context = NativeBrowserEvaluationContext(
        page=object(),
        browser_context=object(),
        evidence=(evidence,),
    )
    evaluator = WebVoyagerEvidenceEvaluator()
    observation = WebObservation(url="https://example.test")

    evaluator.prepare(task=task, observation=observation, browser_context=context)
    result = evaluator.evaluate(
        task=task,
        observation=observation,
        final_answer="done",
        browser_context=context,
    )

    assert not result.valid_sample
    assert result.failure_kind == "external_judge_required"
    assert result.evidence == [evidence]
    assert result.metadata == {"final_answer": "done", "screenshots": 1}


def test_native_evaluator_fails_closed_when_benchmark_plugin_is_missing() -> None:
    task = WebTask(
        benchmark="webarena",
        task_id="0",
        runtime_profile="native_visual",
        action_profile="native_toolcall",
    )
    context = NativeBrowserEvaluationContext(
        page=object(),
        browser_context=object(),
        evidence=(),
    )

    with pytest.raises(EvaluatorConfigurationError, match="not installed"):
        NativeTaskEvaluator().prepare(
            task=task,
            observation=WebObservation(),
            browser_context=context,
        )


@pytest.mark.parametrize(
    ("benchmark", "reference_answers", "answer", "verifier_version"),
    [
        (
            "webarena",
            {"exact_match": "expected"},
            "expected",
            "native-webarena-3b775dc",
        ),
        (
            "visualwebarena",
            {"required_values": [">= 3"]},
            "4",
            "native-visualwebarena-3b775dc",
        ),
    ],
)
def test_native_webarena_family_evaluator_scores_rule_only_tasks(
    monkeypatch,
    benchmark,
    reference_answers,
    answer,
    verifier_version,
) -> None:
    # Classic exact_match has a judge fallback only when the deterministic
    # comparison misses. A non-secret sentinel proves the preflight while this
    # successful rule path makes no network request.
    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
    config = _config(allowed_benchmarks=[benchmark])
    task = WebTask(
        benchmark=benchmark,
        task_id="0",
        intent="Return the expected value",
        runtime_profile="native_visual",
        action_profile="native_toolcall",
        original_metadata={
            "id": f"{benchmark}-0",
            "eval": {
                "eval_types": ["string_match"],
                "reference_answers": reference_answers,
            },
        },
    )
    context = NativeBrowserEvaluationContext(
        page=object(),
        browser_context=object(),
        evidence=(),
    )
    observation = WebObservation()
    evaluator = NativeTaskEvaluator(config=config)

    evaluator.prepare(task=task, observation=observation, browser_context=context)
    result = evaluator.evaluate(
        task=task,
        observation=observation,
        final_answer=answer,
        browser_context=context,
    )

    assert result.reward == 1.0
    assert result.task_success
    assert result.valid_sample
    assert result.verifier_version == verifier_version


def test_native_webarena_evaluator_merges_api_and_browser_snapshots(monkeypatch) -> None:
    from resources_servers.native_web import reference_evaluation

    monkeypatch.setenv("WEBARENA_JUDGE_API_KEY", "test-only")
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
    task = WebTask(
        benchmark="webarena",
        task_id="0",
        intent="Create an order",
        runtime_profile="native_visual",
        action_profile="native_toolcall",
        task_kwargs={"collision_plan": collision_plan},
        original_metadata={
            "id": "webarena-0",
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "done"}},
        },
    )
    evaluator = NativeTaskEvaluator(config=_config(allowed_benchmarks=["webarena"]))
    context = NativeBrowserEvaluationContext(page=object(), browser_context=object(), evidence=())

    evaluator.prepare(task=task, observation=WebObservation(), browser_context=context)
    evaluator.evaluate(
        task=task,
        observation=WebObservation(),
        final_answer="done",
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


def test_native_config_rejects_headless_execution() -> None:
    with pytest.raises(ValueError, match="headed Chromium"):
        _config(headless=True)


def test_native_config_rejects_multiple_sessions_on_one_display() -> None:
    with pytest.raises(ValueError, match="max_sessions=1"):
        _config(max_sessions=2)


def test_webvoyager_browser_reads_proxy_in_always_mode(monkeypatch) -> None:
    monkeypatch.setenv("WA_BROWSER_PROXY_SERVER", "proxy.example.test:19407")
    driver = WebVoyagerBrowserDriver(_webvoyager_config(proxy_mode="always"), "session-test", object())

    assert driver._proxy_for_task(WebTask(benchmark="webvoyager", task_id="GitHub--0")) == ("proxy.example.test:19407")


def test_native_text_input_splits_special_keys() -> None:
    events = []

    class _PyAutoGUI:
        @staticmethod
        def write(text, *, interval):
            events.append(("write", text, interval))

        @staticmethod
        def press(key):
            events.append(("press", key))

        @staticmethod
        def hotkey(*keys):
            events.append(("hotkey", *keys))

    _type_browser_text(_PyAutoGUI(), "a\nb\t<c")

    assert events == [
        ("write", "a", 0.01),
        ("press", "enter"),
        ("write", "b", 0.01),
        ("press", "tab"),
        ("hotkey", "shift", ","),
        ("write", "c", 0.01),
    ]


def test_native_scroll_schema_and_runtime_clamp_excessive_amount(monkeypatch, caplog) -> None:
    computer_tool = next(tool for tool in NATIVE_WEBVOYAGER_TOOLS if tool["name"] == "computer")
    amount_schema = computer_tool["parameters"]["properties"]["actions"]["items"]["properties"][
        "scroll_parameters"
    ]["anyOf"][0]["properties"]["scroll_amount"]
    assert amount_schema["maximum"] == MAX_NATIVE_SCROLL_AMOUNT

    calls: list[int] = []
    pyautogui = SimpleNamespace(
        scroll=calls.append,
        hscroll=calls.append,
    )
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    monkeypatch.setattr("nemo_gym.web.native_browser.time.sleep", lambda _seconds: None)
    driver = NativeWebDriver(_config(), "session-test", object())
    driver._task = WebTask(benchmark="webarena", task_id="14")

    with caplog.at_level(logging.WARNING, logger="nemo_gym.web.native_browser"):
        driver._execute_computer(
            {
                "action": "scroll",
                "scroll_parameters": {"scroll_direction": "down", "scroll_amount": 100_000},
            }
        )

    assert calls == [-MAX_NATIVE_SCROLL_AMOUNT]
    assert "event=native_browser_scroll_clamped" in caplog.text
    assert "requested=100000 applied=50" in caplog.text


def test_native_text_input_uses_clipboard_for_unicode(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "nemo_gym.web.native_browser._paste_unicode",
        lambda _pyautogui, text: calls.append(text),
    )

    _type_browser_text(object(), "北京")

    assert calls == ["北京"]


def test_native_action_error_can_be_returned_for_policy_recovery(monkeypatch) -> None:
    driver = NativeWebDriver(_config(terminate_on_action_error=False, action_delay_seconds=0), "session", object())
    driver._page = object()
    driver._task = WebTask(benchmark="webarena", task_id="14")
    driver._observation = WebObservation.model_validate(
        {
            "goal": [],
            "screenshot": {"data_url": "data:image/png;base64,abc"},
            "url": "https://example.test",
        }
    )
    monkeypatch.setattr(driver, "_execute_call", lambda *_args: (_ for _ in ()).throw(ValueError("bad action")))
    monkeypatch.setattr(
        driver,
        "_capture",
        lambda: driver._observation.model_copy(
            update={
                "last_action": driver._last_action,
                "last_action_error": driver._last_error,
            }
        ),
    )

    result = driver.step(
        WebAction.model_validate(
            {
                "name": "computer",
                "script": "",
                "arguments": {
                    "calls": [
                        {
                            "name": "computer",
                            "arguments": {"actions": [{"action": "wait", "duration": 1}]},
                        }
                    ]
                },
            }
        )
    )

    assert result.execution_ok is False
    assert result.terminated is False
    assert result.observation.last_action_error == "ValueError: bad action"


def test_native_driver_validates_entire_batch_before_side_effect(monkeypatch) -> None:
    driver = NativeWebDriver(_config(terminate_on_action_error=False, action_delay_seconds=0), "session", object())
    driver._page = object()
    driver._task = WebTask(benchmark="webarena", task_id="14")
    driver._observation = WebObservation.model_validate(
        {
            "goal": [],
            "screenshot": {"data_url": "data:image/png;base64,abc"},
            "url": "https://example.test",
        }
    )
    executed = []
    monkeypatch.setattr(driver, "_execute_computer", lambda action: executed.append(action))
    monkeypatch.setattr(
        driver,
        "_capture",
        lambda: driver._observation.model_copy(update={"last_action_error": driver._last_error}),
    )

    result = driver.step(
        WebAction.model_validate(
            {
                "name": "computer",
                "script": "",
                "arguments": {
                    "calls": [
                        {
                            "name": "computer",
                            "arguments": {
                                "actions": [
                                    {"action": "left_click", "coordinate": [0.2, 0.3]},
                                    {"action": "left_click", "coordinate": [2, 3]},
                                ]
                            },
                        }
                    ]
                },
            }
        )
    )

    assert result.execution_ok is False
    assert result.terminated is False
    assert executed == []
    assert "action[1]" in result.observation.last_action_error


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

    driver = WebVoyagerBrowserDriver(_webvoyager_config(), "session-test", object())
    driver._captcha_solver = _FailingSolver()
    driver._page = type("Page", (), {"url": "https://example.test/private?query=secret"})()
    driver._task = WebTask(benchmark="webvoyager", task_id="GitHub--14")

    with caplog.at_level(logging.WARNING, logger="nemo_gym.resources_servers.webvoyager_browser"):
        assert driver._maybe_solve_captcha("before post-action screenshot") is False

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_solver_deferred" in messages
    assert "error_type=TimeoutError" in messages
    assert "provider detail must not escape" not in messages
    assert "private?query=secret" not in messages


def test_native_driver_caps_captcha_failures_by_vlm_step(monkeypatch, caplog) -> None:
    class _FailingSolver:
        def maybe_solve(self, _page, *, phase: str) -> bool:
            assert phase
            raise TimeoutError("provider detail must not escape")

    monkeypatch.setenv("WA_MAX_CAPTCHA_FAILURES", "1")
    driver = WebVoyagerBrowserDriver(_webvoyager_config(), "session-test", object())
    driver._captcha_solver = _FailingSolver()
    driver._page = type("Page", (), {"url": "https://example.test/private?query=secret"})()
    driver._task = WebTask(benchmark="webvoyager", task_id="GitHub--14")

    with caplog.at_level(logging.WARNING, logger="nemo_gym.resources_servers.webvoyager_browser"):
        assert driver._maybe_solve_captcha("after computer", failure_step=0) is False
        assert driver._maybe_solve_captcha("before post-action screenshot", failure_step=0) is False
        with pytest.raises(RuntimeError, match="failed more than 1 times"):
            driver._maybe_solve_captcha("after computer", failure_step=1)

    assert driver._captcha_failures == 2
    assert driver._captcha_budget_exhausted is True
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert messages.count("event=captcha_failure_counted") == 2
    assert "event=captcha_failure_budget_exhausted" in messages
    assert "provider detail must not escape" not in messages
    assert "private?query=secret" not in messages


def test_native_driver_does_not_charge_target_closed_to_captcha_budget(caplog) -> None:
    target_closed_error = type(
        "TargetClosedError",
        (Exception,),
        {"__module__": "playwright._impl._errors"},
    )

    class _ClosedTargetSolver:
        def maybe_solve(self, _page, *, phase: str) -> bool:
            assert phase == "after computer"
            raise target_closed_error("target detail must not escape")

    driver = WebVoyagerBrowserDriver(_webvoyager_config(), "session-test", object())
    driver._captcha_solver = _ClosedTargetSolver()
    driver._page = type("Page", (), {"url": "https://example.test/private?query=secret"})()
    driver._task = WebTask(benchmark="webvoyager", task_id="Google Map--35")

    with caplog.at_level(logging.WARNING, logger="nemo_gym.resources_servers.webvoyager_browser"):
        with pytest.raises(RuntimeError, match="browser target closed during CAPTCHA handling"):
            driver._maybe_solve_captcha("after computer", failure_step=0)

    assert _is_playwright_target_closed_error(target_closed_error("closed")) is True
    assert driver._browser_target_closed is True
    assert driver._captcha_failures == 0
    assert driver._captcha_budget_exhausted is False
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_browser_target_closed" in messages
    assert "failure_budget_counted=false" in messages
    assert "event=captcha_failure_counted" not in messages
    assert "target detail must not escape" not in messages
    assert "private?query=secret" not in messages


def test_native_driver_returns_retryable_status_when_captcha_target_closes(monkeypatch) -> None:
    target_closed_error = type(
        "TargetClosedError",
        (Exception,),
        {"__module__": "playwright._impl._errors"},
    )

    class _ClosedTargetSolver:
        calls = 0

        def maybe_solve(self, _page, *, phase: str) -> bool:
            assert phase == "after computer"
            self.calls += 1
            raise target_closed_error("target detail must not escape")

    driver = WebVoyagerBrowserDriver(
        _webvoyager_config(action_delay_seconds=0, terminate_on_action_error=False),
        "session-test",
        object(),
    )
    solver = _ClosedTargetSolver()
    driver._captcha_solver = solver
    driver._page = type("Page", (), {"url": "https://example.test"})()
    driver._task = WebTask(benchmark="webvoyager", task_id="Google Map--35")
    driver._observation = WebObservation.model_validate(
        {
            "goal": [],
            "screenshot": {"data_url": "data:image/png;base64,abc"},
            "url": "https://example.test",
        }
    )
    monkeypatch.setattr(driver, "_execute_call", lambda *_args: None)
    monkeypatch.setattr(driver, "_capture", lambda: pytest.fail("closed target must not be captured"))

    result = driver.step(
        WebAction.model_validate(
            {
                "name": "computer",
                "script": "",
                "arguments": {
                    "calls": [
                        {
                            "name": "computer",
                            "arguments": {"actions": [{"action": "wait", "duration": 1}]},
                        }
                    ]
                },
            }
        )
    )

    assert result.execution_ok is False
    assert result.terminated is True
    assert solver.calls == 1
    assert driver._captcha_failures == 0
    assert driver._captcha_budget_exhausted is False
    assert result.info["native_status"] == BROWSER_TARGET_CLOSED_STATUS
    assert "target detail must not escape" not in result.info["action_error"]


def test_native_driver_terminates_without_a_second_solve_after_budget_exhaustion(monkeypatch) -> None:
    class _FailingSolver:
        calls = 0

        def maybe_solve(self, _page, *, phase: str) -> bool:
            assert phase == "after computer"
            self.calls += 1
            raise TimeoutError("provider detail must not escape")

    monkeypatch.setenv("WA_MAX_CAPTCHA_FAILURES", "0")
    driver = WebVoyagerBrowserDriver(
        _webvoyager_config(action_delay_seconds=0, terminate_on_action_error=False),
        "session-test",
        object(),
    )
    solver = _FailingSolver()
    driver._captcha_solver = solver
    driver._page = type("Page", (), {"url": "https://example.test/private?query=secret"})()
    driver._task = WebTask(benchmark="webvoyager", task_id="GitHub--14")
    driver._observation = WebObservation.model_validate(
        {
            "goal": [],
            "screenshot": {"data_url": "data:image/png;base64,abc"},
            "url": "https://example.test",
        }
    )
    monkeypatch.setattr(driver, "_execute_call", lambda *_args: None)
    monkeypatch.setattr(
        driver,
        "_capture",
        lambda: driver._observation.model_copy(update={"last_action_error": driver._last_error}),
    )

    result = driver.step(
        WebAction.model_validate(
            {
                "name": "computer",
                "script": "",
                "arguments": {
                    "calls": [
                        {
                            "name": "computer",
                            "arguments": {"actions": [{"action": "wait", "duration": 1}]},
                        }
                    ]
                },
            }
        )
    )

    assert result.execution_ok is False
    assert result.terminated is True
    assert solver.calls == 1
    assert result.observation.last_action_error == (
        "RuntimeError: Captcha solver failed more than 0 times after VLM inference; aborting task at step 0"
    )
    # The agent masks on this status instead of judging a forced stop.
    assert result.info["native_status"] == CAPTCHA_BUDGET_EXHAUSTED_STATUS


def test_native_driver_returns_status_when_captcha_budget_exhausts_before_screenshot(
    monkeypatch,
) -> None:
    class _FailingOnSecondCheckSolver:
        calls = 0

        def maybe_solve(self, _page, *, phase: str) -> bool:
            self.calls += 1
            if self.calls == 1:
                assert phase == "after computer"
                return False
            assert phase == "before post-action screenshot"
            raise TimeoutError("provider detail must not escape")

    monkeypatch.setenv("WA_MAX_CAPTCHA_FAILURES", "0")
    driver = WebVoyagerBrowserDriver(
        _webvoyager_config(action_delay_seconds=0, terminate_on_action_error=False),
        "session-test",
        object(),
    )
    solver = _FailingOnSecondCheckSolver()
    driver._captcha_solver = solver
    driver._page = type("Page", (), {"url": "https://example.test/private?query=secret"})()
    driver._task = WebTask(benchmark="webvoyager", task_id="ESPN--13")
    driver._observation = WebObservation.model_validate(
        {
            "goal": [],
            "screenshot": {"data_url": "data:image/png;base64,abc"},
            "url": "https://example.test",
        }
    )
    monkeypatch.setattr(driver, "_execute_call", lambda *_args: None)
    monkeypatch.setattr(
        driver,
        "_capture",
        lambda: driver._observation.model_copy(update={"last_action_error": driver._last_error}),
    )

    result = driver.step(
        WebAction.model_validate(
            {
                "name": "computer",
                "script": "",
                "arguments": {
                    "calls": [
                        {
                            "name": "computer",
                            "arguments": {"actions": [{"action": "wait", "duration": 1}]},
                        }
                    ]
                },
            }
        )
    )

    assert solver.calls == 2
    assert result.execution_ok is False
    assert result.terminated is True
    assert result.info["native_status"] == CAPTCHA_BUDGET_EXHAUSTED_STATUS
    assert result.observation.last_action_error == (
        "RuntimeError: Captcha solver failed more than 0 times after VLM inference; aborting task at step 0"
    )


class _RecordingPage:
    """Minimal Playwright page double that records navigation calls."""

    def __init__(self, goto_errors: list[Exception] | None = None) -> None:
        self.url = "https://example.test/"
        self.goto_calls: list[tuple[str, str]] = []
        self.history_calls: list[tuple[str, str]] = []
        self._goto_errors = list(goto_errors or [])

    def goto(self, url: str, wait_until: str = "load"):
        self.goto_calls.append((url, wait_until))
        if self._goto_errors:
            raise self._goto_errors.pop(0)
        return None

    def go_back(self, wait_until: str = "load"):
        self.history_calls.append(("back", wait_until))

    def go_forward(self, wait_until: str = "load"):
        self.history_calls.append(("forward", wait_until))

    def bring_to_front(self) -> None:
        return None


def _navigation_driver() -> tuple[NativeWebDriver, _RecordingPage]:
    driver = NativeWebDriver(_config(), "session-test", object())
    driver._task = WebTask(benchmark="webarena", task_id="14")
    page = _RecordingPage()
    driver._page = page
    return driver, page


def test_policy_navigation_settles_on_load() -> None:
    """The reference runner waits for `load` on every policy-driven navigation."""

    driver, page = _navigation_driver()

    driver._execute_call("navigate", {"url": "https://example.test/page"})
    driver._execute_call("navigate", {"url": "back"})
    driver._execute_call("navigate", {"url": "forward"})

    assert NAVIGATION_WAIT_UNTIL == "load"
    assert page.goto_calls == [("https://example.test/page", "load")]
    assert page.history_calls == [("back", "load"), ("forward", "load")]


def test_navigation_retries_transport_faults_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("nemo_gym.web.native_browser.time.sleep", sleeps.append)
    driver = NativeWebDriver(_config(), "session-test", object())
    driver._task = WebTask(benchmark="webarena", task_id="14")
    page = _RecordingPage(
        goto_errors=[
            RuntimeError("Page.goto: net::ERR_CONNECTION_RESET at https://example.test/page"),
            RuntimeError("Page.goto: net::ERR_EMPTY_RESPONSE at https://example.test/page"),
        ]
    )
    driver._page = page

    driver._execute_call("navigate", {"url": "https://example.test/page"})

    assert len(page.goto_calls) == 3
    assert sleeps == [NAVIGATION_RETRY_DELAYS_S[0], NAVIGATION_RETRY_DELAYS_S[1]]


def test_navigation_does_not_retry_a_slow_page(monkeypatch) -> None:
    """A Playwright timeout is a slow page, so retrying only multiplies the wait."""

    sleeps: list[float] = []
    monkeypatch.setattr("nemo_gym.web.native_browser.time.sleep", sleeps.append)
    driver = NativeWebDriver(_config(), "session-test", object())
    driver._task = WebTask(benchmark="webarena", task_id="14")
    page = _RecordingPage(goto_errors=[RuntimeError("Page.goto: Timeout 45000ms exceeded")])
    driver._page = page

    with pytest.raises(RuntimeError, match="Timeout 45000ms exceeded"):
        driver._execute_call("navigate", {"url": "https://example.test/page"})

    assert len(page.goto_calls) == 1
    assert sleeps == []


def test_navigation_reraises_after_exhausting_retries(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("nemo_gym.web.native_browser.time.sleep", sleeps.append)
    driver = NativeWebDriver(_config(), "session-test", object())
    driver._task = WebTask(benchmark="webarena", task_id="14")
    attempts = len(NAVIGATION_RETRY_DELAYS_S) + 1
    page = _RecordingPage(
        goto_errors=[RuntimeError("Page.goto: net::ERR_TIMED_OUT") for _ in range(attempts)],
    )
    driver._page = page

    with pytest.raises(RuntimeError, match="net::ERR_TIMED_OUT"):
        driver._execute_call("navigate", {"url": "https://example.test/page"})

    assert len(page.goto_calls) == attempts
    assert sleeps == list(NAVIGATION_RETRY_DELAYS_S)


def test_webarena_setup_retries_any_initial_navigation_failure(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("resources_servers.native_web.backend.time.sleep", sleeps.append)
    driver = NativeWebDriver(_config(allowed_benchmarks=["webarena"]), "session-test", object())
    driver._task = WebTask(
        benchmark="webarena",
        task_id="17",
        runtime_profile="native_visual",
        action_profile="native_toolcall",
    )
    page = _RecordingPage(
        goto_errors=[
            RuntimeError("Page.goto: Timeout 45000ms exceeded"),
            RuntimeError("Page.goto: selector setup failed"),
        ]
    )

    driver._goto_task_start(page, "http://webarena.test/start")

    assert len(page.goto_calls) == 3
    assert sleeps == list(LOCAL_SETUP_RETRY_DELAYS_S)


def test_context_deadline_is_configurable_and_defaults_to_the_reference() -> None:
    assert _config().default_timeout_ms == 45_000
    assert _config(default_timeout_ms=90_000).default_timeout_ms == 90_000
