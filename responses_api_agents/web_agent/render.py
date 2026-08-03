# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render common web observations as Responses API multimodal messages."""

from __future__ import annotations

import re
from typing import Any

from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from nemo_gym.web.models import WebActionProfile, WebObservation, WebObservationProfile, WebTask


BROWSERGYM_ACTION_GUIDANCE = """Return one Action using BrowserGym high-level calls only.
Common calls:
- click('bid'), fill('bid', 'text'), select_option('bid', 'value'), hover('bid')
- scroll(0, 500), keyboard_press('Enter'), go_back(), go_forward(), goto('https://...')
- new_tab(), tab_focus(0), tab_close()
- send_msg_to_user('final answer') when the task is complete
- report_infeasible('reason') only when the task cannot be completed
You may return at most two calls on separate lines. Arguments must be literals; do not emit arbitrary Python.
Use this exact shape:
Thought: concise reasoning
Action: click('bid')"""


WEBVOYAGER_ACTION_GUIDANCE = """Return exactly one WebVoyager-style Action:
- Click [bid]
- Type [bid]; [text] (typing also submits with Enter)
- Scroll [WINDOW]; up or Scroll [WINDOW]; down
- Wait, GoBack, or Google
- ANSWER; [final answer] when complete
Use this exact shape:
Thought: concise reasoning
Action: Click [bid]"""


VISUAL_OBSERVATION_TEXT_MODES = frozenset({"full_axtree", "som_only", "none"})
_BID_LINE = re.compile(r"^\s*\[[^\]]+\]\s+")
_SOM_MARKER = re.compile(r",\s*som(?=,|$)")


def _goal_text(goal: list[dict[str, Any]], fallback: str) -> str:
    texts = [str(item.get("text", "")) for item in goal if item.get("type") == "text"]
    return "\n".join(text for text in texts if text).strip() or fallback


def _goal_images(goal: list[dict[str, Any]]) -> list[str]:
    images: list[str] = []
    for item in goal:
        if item.get("type") != "image_url":
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str) and image_url:
            images.append(image_url)
    return images


def action_guidance(task: WebTask) -> str:
    if task.action_profile == WebActionProfile.WEBVOYAGER_LEGACY:
        return WEBVOYAGER_ACTION_GUIDANCE
    return BROWSERGYM_ACTION_GUIDANCE


def compact_som_text(axtree_text: str, *, max_chars: int = 12_000) -> str:
    """Keep the BrowserGym-labelled interactive elements from a visual AXTree.

    BrowserGym marks the nodes actually represented in its SoM overlay with a
    ``som`` property.  WebVoyager's upstream prompt contains a similarly short
    list of labelled interactive elements, rather than the complete AXTree.
    """

    retained: list[str] = []
    retained_chars = 0
    for raw_line in axtree_text.splitlines():
        line = raw_line.strip()
        if not _BID_LINE.match(line) or _SOM_MARKER.search(line) is None:
            continue
        line = _SOM_MARKER.sub("", line)
        if len(line) > 220:
            line = f"{line[:217]}..."
        additional = len(line) + (1 if retained else 0)
        if retained_chars + additional > max_chars:
            retained.append("[Additional labelled elements omitted.]")
            break
        retained.append(line)
        retained_chars += additional
    return "\n".join(retained)


def render_observation(
    observation: WebObservation,
    task: WebTask,
    *,
    step_index: int,
    visual_observation_text: str = "full_axtree",
) -> NeMoGymEasyInputMessage:
    """Build one model turn without leaking raw BrowserGym Python objects."""

    if visual_observation_text not in VISUAL_OBSERVATION_TEXT_MODES:
        raise ValueError(f"unsupported visual observation text mode: {visual_observation_text}")

    profile = task.observation_profile
    if profile is None:
        profile = WebObservationProfile.A11Y if task.benchmark.value == "webarena" else WebObservationProfile.SOM
    text_parts = [
        f"Task: {_goal_text(observation.goal, task.intent)}",
        f"Step: {step_index}",
        f"Current URL: {observation.url}",
    ]
    if observation.tabs:
        text_parts.append(
            "Tabs:\n"
            + "\n".join(
                f"- [{tab.index}] {'ACTIVE ' if tab.active else ''}{tab.title} — {tab.url}" for tab in observation.tabs
            )
        )
    if observation.last_action:
        text_parts.append(f"Previous action: {observation.last_action}")
    if observation.last_action_error:
        text_parts.append(f"Previous action failed: {observation.last_action_error}")
    if observation.axtree_text and profile == WebObservationProfile.A11Y:
        text_parts.append(f"Accessibility tree (element ids are in brackets):\n{observation.axtree_text}")
    elif observation.axtree_text and visual_observation_text == "full_axtree":
        text_parts.append(f"Accessibility tree (element ids are in brackets):\n{observation.axtree_text}")
    elif observation.axtree_text and visual_observation_text == "som_only":
        compact_text = compact_som_text(observation.axtree_text)
        if compact_text:
            text_parts.append(f"Labelled interactive elements (ids match the screenshot):\n{compact_text}")
    text_parts.append(action_guidance(task))

    content: list[dict[str, Any]] = [{"type": "input_text", "text": "\n\n".join(text_parts)}]
    if step_index == 0:
        for image_url in [*_goal_images(observation.goal), *task.input_images]:
            content.append({"type": "input_image", "image_url": image_url, "detail": "high"})
    if profile in {WebObservationProfile.SCREENSHOT, WebObservationProfile.SOM}:
        screenshot = observation.screenshot
        if screenshot is not None and screenshot.data_url:
            content.append(
                {
                    "type": "input_image",
                    "image_url": screenshot.data_url,
                    "detail": "high",
                }
            )
    return NeMoGymEasyInputMessage(role="user", content=content)


def parse_error_message(error: Exception) -> NeMoGymEasyInputMessage:
    return NeMoGymEasyInputMessage(
        role="user",
        content=(
            f"Action parse error: {error}. Return a corrected Thought and Action only, "
            "using the required action grammar."
        ),
    )
