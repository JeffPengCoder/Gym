# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.web.models import WebAction, WebBenchmark, WebTask
from resources_servers.browsergym_web.artifacts import WebArtifactStore
from resources_servers.browsergym_web.backend import BrowserGymBackend
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig


class FakeEnv:
    def __init__(self):
        self.actions = []
        self.closed = False

    @staticmethod
    def _observation(last_action="", last_action_error=""):
        return {
            "goal_object": [{"type": "text", "text": "Do the task"}],
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
        terminal = action.startswith("send_msg_to_user")
        return self._observation(last_action=action), float(terminal), terminal, False, {}

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
    )


def test_backend_keeps_execution_and_benchmark_scores_separate(tmp_path, monkeypatch):
    env = FakeEnv()
    backend = BrowserGymBackend(
        _config(tmp_path),
        "session-a",
        WebArtifactStore(tmp_path),
    )
    monkeypatch.setattr(backend, "_make_environment", lambda _spec: env)
    task = WebTask(benchmark=WebBenchmark.WEBARENA, task_id="0", seed=7)

    observation, info = backend.reset(task)
    assert observation.screenshot is not None
    assert observation.active_tab_index == 0
    assert observation.elapsed_time == 1.25
    assert info["env_id"] == "browsergym/webarena.0"

    step = backend.step(WebAction(name="noop", script="noop()"))
    assert step.execution_ok is True
    assert step.benchmark_reward == 0.0

    evaluation = backend.evaluate("final answer")
    assert evaluation.valid_sample is True
    assert evaluation.raw_score == 1.0
    assert env.actions[-1] == "send_msg_to_user('final answer')"

    backend.close()
    assert env.closed is True
