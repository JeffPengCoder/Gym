# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.web.models import WebBenchmark, WebObservationProfile, WebRuntimeProfile, WebTask
from resources_servers.browsergym_web.profiles import resolve_browsergym_profile


def test_webarena_uses_pinned_environment_and_native_evaluator():
    profile = resolve_browsergym_profile(WebTask(benchmark=WebBenchmark.WEBARENA, task_id=310))

    assert profile.module == "browsergym.webarena"
    assert profile.env_id == "browsergym/webarena.310"
    assert profile.action_subsets == ("webarena",)
    assert profile.observation_profile == WebObservationProfile.A11Y
    assert profile.external_verifier is False


def test_visualwebarena_uses_pinned_environment_and_native_evaluator():
    profile = resolve_browsergym_profile(WebTask(benchmark=WebBenchmark.VISUALWEBARENA, task_id=721))

    assert profile.module == "browsergym.visualwebarena"
    assert profile.env_id == "browsergym/visualwebarena.721"
    assert profile.action_subsets == ("visualwebarena",)
    assert profile.observation_profile == WebObservationProfile.SOM
    assert profile.external_verifier is False


def test_webvoyager_uses_openended_task_and_external_verifier():
    profile = resolve_browsergym_profile(
        WebTask(
            benchmark=WebBenchmark.WEBVOYAGER,
            task_id="Allrecipes--0",
            intent="Find a recipe",
            start_urls=["https://www.allrecipes.com/"],
        )
    )

    assert profile.env_id == "browsergym/openended"
    assert profile.task_kwargs == {
        "start_url": "https://www.allrecipes.com/",
        "goal": "Find a recipe",
    }
    assert profile.external_verifier is True


def test_webvoyager_requires_a_start_url():
    with pytest.raises(ValueError, match="start URL"):
        resolve_browsergym_profile(WebTask(benchmark=WebBenchmark.WEBVOYAGER, task_id="missing-url"))


def test_browsergym_does_not_silently_accept_native_visual_tasks():
    task = WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id="0",
        runtime_profile=WebRuntimeProfile.NATIVE_VISUAL,
    )

    with pytest.raises(ValueError, match="unsupported runtime profile"):
        resolve_browsergym_profile(task)
