# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.models import WebAction, WebBenchmark, WebTask
from nemo_gym.web.session import EvaluatorConfigurationError, EvaluatorInfrastructureError
from resources_servers.browsergym_web import backend as backend_module
from resources_servers.browsergym_web.backend import BrowserGymBackend
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig


class FakeEnv:
    def __init__(self):
        self.actions = []
        self.closed = False

    @staticmethod
    def _observation(last_action="", last_action_error=""):
        return {
            "goal_object": ({"type": "text", "text": "Do the task"},),
            "open_pages_urls": ["https://example.test"],
            "open_pages_titles": ["Example"],
            "active_page_index": [0],
            "url": "https://example.test",
            "screenshot": b"not-a-real-png",
            "axtree_object": {},
            "extra_element_properties": {},
            "last_action": last_action,
            "last_action_error": last_action_error,
            "elapsed_time": [1.25],
        }

    def reset(self, seed):
        assert seed == 7
        return self._observation(), {"task_info": {}}

    def step(self, action):
        self.actions.append(action)
        return self._observation(last_action=action), 0.0, False, False, {}

    def close(self):
        self.closed = True


def _config(tmp_path):
    return BrowserGymWebResourcesServerConfig(
        name="browsergym_web",
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="agent",
        artifact_dir=str(tmp_path),
        allowed_benchmarks=[WebBenchmark.WEBARENA, WebBenchmark.VISUALWEBARENA, WebBenchmark.WEBVOYAGER],
    )


def _task() -> WebTask:
    return WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id="Allrecipes--0",
        intent="Find a recipe",
        start_urls=["https://www.allrecipes.com/"],
        action_profile="webvoyager_legacy",
        seed=7,
    )


def _webarena_task(**updates) -> WebTask:
    task = WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0", seed=7)
    return task.model_copy(update=updates)


def _visualwebarena_task(**updates) -> WebTask:
    task = WebTask(benchmark=WebBenchmark.VISUALWEBARENA, task_id="0", seed=7)
    return task.model_copy(update=updates)


def test_backend_collects_webvoyager_evidence_for_external_judge(tmp_path, monkeypatch):
    env = FakeEnv()
    backend = BrowserGymBackend(_config(tmp_path), "session-a", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: env)

    observation, info = backend.reset(_task())
    step = backend.step(WebAction(name="noop", script="noop()"))
    evaluation = backend.evaluate("final answer")

    assert observation.screenshot is not None
    assert observation.active_tab_index == 0
    assert observation.elapsed_time == 1.25
    assert info["env_id"] == "browsergym/openended"
    assert info["verifier_version"] == "webvoyager-llm-judge-v1"
    assert step.execution_ok is True
    assert step.benchmark_reward == 0.0
    assert evaluation.valid_sample is False
    assert evaluation.failure_kind == "external_judge_required"
    assert evaluation.verifier_version == "webvoyager-llm-judge-v1"
    assert evaluation.evidence

    backend.close()
    assert env.closed is True


def test_invalid_high_level_action_is_returned_to_agent(tmp_path, monkeypatch):
    class RejectingEnv(FakeEnv):
        def step(self, action):
            error = f"ValueError: invalid high-level action: {action}"
            return self._observation(last_action=action, last_action_error=error), 0.0, False, False, {}

    backend = BrowserGymBackend(_config(tmp_path), "session-invalid-action", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: RejectingEnv())
    backend.reset(_task())

    result = backend.step(WebAction(name="click", script="click('missing')"))

    assert result.execution_ok is False
    assert result.terminated is False
    assert result.truncated is False
    assert result.observation.last_action == "click('missing')"
    assert "invalid high-level action" in result.observation.last_action_error
    assert result.info["action_error"] == result.observation.last_action_error
    backend.close()


def test_exception_escaping_browsergym_step_is_an_infrastructure_failure(tmp_path, monkeypatch):
    class BrokenEnv(FakeEnv):
        def step(self, action):
            del action
            raise ValueError("browser runtime failed")

    backend = BrowserGymBackend(_config(tmp_path), "session-runtime-error", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: BrokenEnv())
    backend.reset(_task())

    with pytest.raises(EvaluatorInfrastructureError, match="browser runtime failed"):
        backend.step(WebAction(name="noop", script="noop()"))

    backend.close()


def test_reset_failure_closes_new_environment(tmp_path, monkeypatch):
    class BrokenResetEnv(FakeEnv):
        def reset(self, seed):
            del seed
            raise RuntimeError("reset failed")

    env = BrokenResetEnv()
    backend = BrowserGymBackend(_config(tmp_path), "session-reset-error", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: env)

    with pytest.raises(RuntimeError, match="reset failed"):
        backend.reset(_task())

    assert env.closed is True
    assert backend.env is None


def test_webarena_keeps_execution_and_terminal_benchmark_scores_separate(tmp_path, monkeypatch):
    class ScoringEnv(FakeEnv):
        def step(self, action):
            self.actions.append(action)
            terminal = action.startswith("send_msg_to_user")
            reward = 1.0 if terminal else 0.5
            return self._observation(last_action=action), reward, terminal, False, {}

    env = ScoringEnv()
    backend = BrowserGymBackend(_config(tmp_path), "session-wa", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: env)

    _observation, info = backend.reset(_webarena_task())
    step = backend.step(WebAction(name="noop", script="noop()"))
    evaluation = backend.evaluate("done")

    assert info["env_id"] == "browsergym/webarena.0"
    assert step.benchmark_reward == 0.5
    assert evaluation.reward == 1.0
    assert evaluation.metadata["best_observed_reward"] == 1.0
    assert evaluation.metadata["score_semantics"] == "terminal_native_evaluator_reward"
    backend.close()


def test_webarena_model_evaluator_requires_explicit_configuration(tmp_path):
    backend = BrowserGymBackend(_config(tmp_path), "session-missing-evaluator", WebArtifactStore(tmp_path))
    task = _webarena_task(
        task_id="8",
        original_metadata={"eval": {"reference_answers": {"fuzzy_match": ["answer"]}}},
    )

    with pytest.raises(EvaluatorConfigurationError, match="webarena_evaluator_model"):
        backend.reset(task)


def test_visualwebarena_model_evaluator_requires_explicit_configuration(tmp_path):
    backend = BrowserGymBackend(_config(tmp_path), "session-missing-vwa-evaluator", WebArtifactStore(tmp_path))
    task = _visualwebarena_task(
        original_metadata={"eval": {"reference_answers": {"fuzzy_match": ["answer"]}}},
    )

    with pytest.raises(EvaluatorConfigurationError, match="visualwebarena_evaluator_model"):
        backend.reset(task)


def test_rule_only_visualwebarena_uses_temporary_import_credential(tmp_path, monkeypatch):
    seen_keys = []

    class ImportSensitiveEnv(FakeEnv):
        def reset(self, seed):
            seen_keys.append(os.environ["OPENAI_API_KEY"])
            return super().reset(seed)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = BrowserGymBackend(_config(tmp_path), "session-vwa-rule-only", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: ImportSensitiveEnv())
    task = _visualwebarena_task(
        original_metadata={"eval": {"reference_answers": {"exact_match": "answer"}}},
    )

    backend.reset(task)

    assert seen_keys == ["unused-for-rule-only-evaluator"]
    assert "OPENAI_API_KEY" not in os.environ
    backend.close()


def test_webarena_evaluator_hook_runs_after_upstream_reset(tmp_path, monkeypatch):
    events = []

    class OrderedEnv(FakeEnv):
        def reset(self, seed):
            events.append("reset")
            return super().reset(seed)

    config = _config(tmp_path).model_copy(update={"webarena_evaluator_model": "local-judge"})
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")  # pragma: allowlist secret
    backend = BrowserGymBackend(config, "session-wa-hook", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: OrderedEnv())
    monkeypatch.setattr(
        backend_module,
        "configure_evaluator_environment",
        lambda **kwargs: events.append(f"environment:{kwargs['api_key']}"),
    )
    monkeypatch.setattr(
        backend_module,
        "configure_webarena_evaluator_model",
        lambda model: events.append(f"configure:{model}"),
    )

    _observation, info = backend.reset(_webarena_task())

    assert events == ["environment:test-only", "reset", "configure:local-judge"]
    assert info["evaluator_model"] == "local-judge"
    assert info["verifier_version"].endswith(":judge=local-judge")
    backend.close()


def test_visualwebarena_evaluator_hook_runs_after_upstream_reset(tmp_path, monkeypatch):
    events = []

    class OrderedEnv(FakeEnv):
        def reset(self, seed):
            events.append("reset")
            return super().reset(seed)

    config = _config(tmp_path).model_copy(update={"visualwebarena_evaluator_model": "local-visual-judge"})
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")  # pragma: allowlist secret
    backend = BrowserGymBackend(config, "session-vwa-hook", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: OrderedEnv())
    monkeypatch.setattr(
        backend_module,
        "configure_visualwebarena_evaluator_model",
        lambda model: events.append(f"configure:{model}"),
    )

    _observation, info = backend.reset(_visualwebarena_task())

    assert events == ["reset", "configure:local-visual-judge"]
    assert info["evaluator_model"] == "local-visual-judge"
    assert info["verifier_version"].endswith(":judge=local-visual-judge")
    backend.close()


def test_webarena_evaluator_hook_failure_closes_environment(tmp_path, monkeypatch):
    env = FakeEnv()
    config = _config(tmp_path).model_copy(update={"webarena_evaluator_model": "broken-evaluator"})
    backend = BrowserGymBackend(config, "session-hook-failure", WebArtifactStore(tmp_path))
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: env)
    monkeypatch.setattr(backend, "_prepare_evaluator", lambda _task: None)
    monkeypatch.setattr(
        backend_module,
        "configure_webarena_evaluator_model",
        lambda _model: (_ for _ in ()).throw(RuntimeError("evaluator hook failed")),
    )

    with pytest.raises(RuntimeError, match="evaluator hook failed"):
        backend.reset(_webarena_task())

    assert env.closed is True
    assert backend.env is None
