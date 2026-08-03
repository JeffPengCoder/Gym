# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.web.models import (
    WebBenchmark,
    WebImage,
    WebObservation,
    WebObservationProfile,
    WebTask,
)
from responses_api_agents.web_agent.render import compact_som_text, render_observation


def _block_types(message):
    return [
        block.get("type") if isinstance(block, dict) else getattr(block, "type", None) for block in message.content
    ]


def _block_text(block):
    return block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")


def test_a11y_profile_omits_page_screenshot():
    task = WebTask(
        benchmark=WebBenchmark.WEBARENA,
        task_id="0",
        observation_profile=WebObservationProfile.A11Y,
    )
    observation = WebObservation(
        goal=[{"type": "text", "text": "Find the answer"}],
        axtree_text="[a1] button 'Search'",
        screenshot=WebImage(data_url="data:image/png;base64,abc"),
    )

    message = render_observation(observation, task, step_index=0)

    assert _block_types(message) == ["input_text"]
    assert "[a1] button" in _block_text(message.content[0])


def test_som_profile_includes_page_and_goal_images():
    task = WebTask(
        benchmark=WebBenchmark.VISUALWEBARENA,
        task_id="0",
        observation_profile=WebObservationProfile.SOM,
    )
    observation = WebObservation(
        goal=[
            {"type": "text", "text": "Find the matching item"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,goal"}},
        ],
        screenshot=WebImage(data_url="data:image/png;base64,page"),
    )

    message = render_observation(observation, task, step_index=0)

    assert _block_types(message) == ["input_text", "input_image", "input_image"]


def test_som_only_text_keeps_only_labelled_interactive_elements():
    task = WebTask(
        benchmark=WebBenchmark.WEBVOYAGER,
        task_id="ArXiv--13",
        observation_profile=WebObservationProfile.SOM,
    )
    observation = WebObservation(
        axtree_text=(
            "RootWebArea 'Example'\n"
            "\t[10] link 'Search', som, expanded=False\n"
            "\t[11] heading 'News'\n"
            "\t\tStaticText 'body text'\n"
            "\t[12] textbox 'Query', som\n"
        ),
        screenshot=WebImage(data_url="data:image/png;base64,page"),
    )

    message = render_observation(
        observation,
        task,
        step_index=0,
        visual_observation_text="som_only",
    )
    text = _block_text(message.content[0])

    assert "[10] link 'Search', expanded=False" in text
    assert "[12] textbox 'Query'" in text
    assert "heading 'News'" not in text
    assert "body text" not in text
    assert "Accessibility tree" not in text


def test_compact_som_text_has_a_hard_character_budget():
    axtree = "\n".join(f"[{index}] link '{'x' * 200}', som" for index in range(100))

    compact = compact_som_text(axtree, max_chars=500)

    assert len(compact) < 550
    assert compact.endswith("[Additional labelled elements omitted.]")
