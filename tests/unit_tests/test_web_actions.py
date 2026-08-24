# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.web.actions import ActionParseError, parse_model_action, parse_native_tool_calls
from nemo_gym.web.models import WebActionProfile


def test_parses_fenced_browsergym_action() -> None:
    action = parse_model_action(
        "Thought: open the result\nAction:\n```python\nclick('a42')\n```",
        WebActionProfile.BROWSERGYM_HIGHLEVEL,
    )

    assert action.name == "click"
    assert action.script == "click('a42')"
    assert action.arguments["args"] == ["a42"]
    assert not action.terminal


def test_rejects_non_literal_or_arbitrary_python() -> None:
    with pytest.raises(ActionParseError, match="literal"):
        parse_model_action("click(get_target())", WebActionProfile.BROWSERGYM_HIGHLEVEL)
    with pytest.raises(ActionParseError, match="direct function call"):
        parse_model_action("import os", WebActionProfile.BROWSERGYM_HIGHLEVEL)


def test_translates_webvoyager_type_and_submit() -> None:
    action = parse_model_action("Action: Type [17]; [vegetarian lasagna]", WebActionProfile.WEBVOYAGER_LEGACY)

    assert action.name == "multi_action"
    assert action.script == "fill('17', 'vegetarian lasagna')\nkeyboard_press('Enter')"
    assert action.arguments["calls"][1]["name"] == "keyboard_press"


def test_translates_webvoyager_answer_to_terminal_action() -> None:
    action = parse_model_action("Action: ANSWER; [The result is 42]", WebActionProfile.WEBVOYAGER_LEGACY)

    assert action.name == "send_msg_to_user"
    assert action.terminal
    assert action.answer == "The result is 42"


def test_webvoyager_executes_only_the_first_labelled_action_section() -> None:
    action = parse_model_action(
        """Thought: Open the form.
Action: Click [61]
Thought: Pretend the click already happened.
Action: Type [99]; [SimCSE]""",
        WebActionProfile.WEBVOYAGER_LEGACY,
    )

    assert action.name == "click"
    assert action.script == "click('61')"


def test_webvoyager_does_not_cherry_pick_a_later_action_after_invalid_first_action() -> None:
    with pytest.raises(ActionParseError):
        parse_model_action(
            """Thought: Open the form.
Action: Click on the form button.
Thought: Pretend the click already happened.
Action: Click [61]""",
            WebActionProfile.WEBVOYAGER_LEGACY,
        )


def test_parses_native_computer_and_terminal_tool_calls() -> None:
    action = parse_native_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "computer",
                "arguments": '{"actions":[{"action":"left_click","coordinate":[0.25,0.75]}]}',
            },
            {
                "type": "function_call",
                "call_id": "call-2",
                "name": "terminate",
                "arguments": '{"status":"success","answer":"done"}',
            },
        ]
    )

    assert action.name == "native_tool_calls"
    assert action.arguments["calls"][0]["arguments"]["actions"][0]["action"] == "left_click"
    assert action.terminal is True
    assert action.answer == "done"
    assert action.metadata["native_parse"]["recovered"] is False


def test_native_robust_mode_decodes_actions_string_and_records_recovery() -> None:
    action = parse_native_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "computer",
                "arguments": '{"actions":"[{\\"action\\":\\"left_click\\",\\"coordinate\\":[0.25,0.75]}]"}',
            }
        ],
        recovery="decode_string",
    )

    assert action.arguments["calls"][0]["arguments"]["actions"][0]["action"] == "left_click"
    assert action.metadata["native_parse"]["recovered"] is True
    assert action.metadata["native_parse"]["calls"][0]["recovery_mode"] == "decoded_inner_string"


def test_native_robust_mode_repairs_only_one_missing_closing_bracket() -> None:
    action = parse_native_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "computer",
                "arguments": (
                    '{"actions":"[{\\"action\\":\\"left_click\\",'
                    '\\"coordinate\\":[0.325,0.5155]},'
                    '{\\"action\\":\\"type\\",\\"text\\":\\"Stockholm\\"}"}'
                ),
            }
        ],
        recovery="repair_single_closing_bracket",
    )

    actions = action.arguments["calls"][0]["arguments"]["actions"]
    assert [item["action"] for item in actions] == ["left_click", "type"]
    assert action.metadata["native_parse"]["calls"][0]["recovery_mode"] == "closed_one_missing_bracket"


def test_native_alignment_mode_still_rejects_actions_string() -> None:
    with pytest.raises(ActionParseError, match="non-empty actions list"):
        parse_native_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": '{"actions":"[{\\"action\\":\\"wait\\",\\"duration\\":1}]"}',
                }
            ]
        )


def test_native_alias_recovery_is_opt_in_and_records_nested_click_conversion() -> None:
    item = {
        "type": "function_call",
        "call_id": "call-click",
        "name": "computer",
        "arguments": '{"actions":[{"action":"click","coordinate":[0.25,0.75]}]}',
    }
    with pytest.raises(ActionParseError, match="unsupported native computer action"):
        parse_native_tool_calls([item])

    action = parse_native_tool_calls([item], alias_recovery="webvoyager_v3")

    call = action.arguments["calls"][0]
    assert call["name"] == "computer"
    assert call["arguments"]["actions"] == [{"action": "left_click", "coordinate": [0.25, 0.75]}]
    record = action.metadata["native_parse"]["calls"][0]
    assert record["original_tool"] == "computer"
    assert record["alias_recovery_modes"] == ["computer.click_to_left_click"]
    assert action.metadata["native_parse"]["recovered"] is True


@pytest.mark.parametrize("duration,expected", [(-4, 0.0), (45, 30.0)])
def test_native_alias_recovery_clamps_nested_wait_and_records_values(duration, expected) -> None:
    item = {
        "type": "function_call",
        "call_id": "call-wait",
        "name": "computer",
        "arguments": json.dumps({"actions": [{"action": "wait", "duration": duration}]}),
    }
    with pytest.raises(ActionParseError, match=r"must be in \[0, 30\]"):
        parse_native_tool_calls([item])

    action = parse_native_tool_calls([item], alias_recovery="webvoyager_v3")

    assert action.arguments["calls"][0]["arguments"]["actions"] == [{"action": "wait", "duration": expected}]
    record = action.metadata["native_parse"]["calls"][0]
    assert record["alias_recovery_modes"] == ["computer.wait_duration_clamped"]
    assert record["alias_recovery_details"] == [
        {
            "field": "computer.actions[0].duration",
            "original": duration,
            "normalized": expected,
            "minimum": 0,
            "maximum": 30,
        }
    ]


@pytest.mark.parametrize("duration", [True, "45", float("nan"), float("inf")])
def test_native_alias_recovery_rejects_unsafe_nested_wait_values(duration) -> None:
    item = {
        "type": "function_call",
        "name": "computer",
        "arguments": json.dumps({"actions": [{"action": "wait", "duration": duration}]}),
    }

    with pytest.raises(ActionParseError):
        parse_native_tool_calls([item], alias_recovery="webvoyager_v3")


@pytest.mark.parametrize(
    "name,arguments,expected_action,expected_payload,expected_mode",
    [
        (
            "click",
            '{"x":"0.25","y":"0.75"}',
            "left_click",
            {"coordinate": [0.25, 0.75]},
            "tool.click_xy_to_computer_left_click",
        ),
        (
            "left_click",
            '{"coordinate":"[0.4, 0.6]"}',
            "left_click",
            {"coordinate": [0.4, 0.6]},
            "tool.left_click_coordinate_to_computer_left_click",
        ),
        (
            "type",
            '{"text":"hello"}',
            "type",
            {"text": "hello"},
            "tool.type_to_computer_type",
        ),
        (
            "wait",
            '{"duration":"2"}',
            "wait",
            {"duration": 2.0},
            "tool.wait_to_computer_wait",
        ),
    ],
)
def test_native_alias_recovery_wraps_unambiguous_top_level_actions(
    name, arguments, expected_action, expected_payload, expected_mode
) -> None:
    with pytest.raises(ActionParseError, match="unsupported native browser tool"):
        parse_native_tool_calls(
            [{"type": "function_call", "call_id": "call-alias", "name": name, "arguments": arguments}]
        )

    action = parse_native_tool_calls(
        [{"type": "function_call", "call_id": "call-alias", "name": name, "arguments": arguments}],
        alias_recovery="webvoyager_v3",
    )

    call = action.arguments["calls"][0]
    assert call["name"] == "computer"
    computer_action = call["arguments"]["actions"][0]
    assert computer_action.pop("action") == expected_action
    assert computer_action == expected_payload
    record = action.metadata["native_parse"]["calls"][0]
    assert record["original_tool"] == name
    assert record["alias_recovery_modes"] == [expected_mode]


@pytest.mark.parametrize("duration,expected", [("-2", 0.0), (90, 30.0)])
def test_native_alias_recovery_clamps_top_level_wait_and_records_values(duration, expected) -> None:
    action = parse_native_tool_calls(
        [
            {
                "type": "function_call",
                "call_id": "call-wait",
                "name": "wait",
                "arguments": json.dumps({"duration": duration}),
            }
        ],
        alias_recovery="webvoyager_v3",
    )

    assert action.arguments["calls"][0]["arguments"]["actions"] == [{"action": "wait", "duration": expected}]
    record = action.metadata["native_parse"]["calls"][0]
    assert record["alias_recovery_modes"] == [
        "tool.wait_to_computer_wait",
        "tool.wait_duration_clamped",
    ]
    assert record["alias_recovery_details"] == [
        {
            "field": "tool.wait.duration",
            "original": duration,
            "normalized": expected,
            "minimum": 0,
            "maximum": 30,
        }
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"target":"Buy button"}',
        '{"x":"500","y":"300"}',
        '{"action":"double_click","coordinate":[0.4,0.6]}',
        '{"action":"type","coordinate":"[0.4,0.6]","text":"query"}',
    ],
)
def test_native_alias_recovery_rejects_ambiguous_or_non_normalized_clicks(arguments) -> None:
    with pytest.raises(ActionParseError):
        parse_native_tool_calls(
            [{"type": "function_call", "name": "click", "arguments": arguments}],
            alias_recovery="webvoyager_v3",
        )


def test_native_recovery_rejects_non_local_json_damage() -> None:
    with pytest.raises(ActionParseError, match="not eligible"):
        parse_native_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": '{"actions":"[{\\"action\\":\\"left_click\\",\\"coordinate\\":[0.2,0.3}"}',
                }
            ],
            recovery="repair_single_closing_bracket",
        )


def test_native_parser_validates_complete_batch_and_batch_limit() -> None:
    with pytest.raises(ActionParseError, match=r"action\[1\].*coordinate"):
        parse_native_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": (
                        '{"actions":['
                        '{"action":"left_click","coordinate":[0.2,0.3]},'
                        '{"action":"left_click","coordinate":[2,3]}]}'
                    ),
                }
            ]
        )
    with pytest.raises(ActionParseError, match="2-action batch limit"):
        parse_native_tool_calls(
            [
                {
                    "type": "function_call",
                    "name": "computer",
                    "arguments": (
                        '{"actions":['
                        '{"action":"wait","duration":1},'
                        '{"action":"wait","duration":1},'
                        '{"action":"wait","duration":1}]}'
                    ),
                }
            ],
            max_computer_actions=2,
        )


@pytest.mark.parametrize(
    "name,arguments,match",
    [
        ("navigate", '{"url":"example.com"}', "must use http"),
        ("tabs_focus", '{"tab_id":-1}', "non-negative integer"),
        ("terminate", '{"status":"done"}', "success or failure"),
    ],
)
def test_native_parser_validates_tool_arguments(name, arguments, match) -> None:
    with pytest.raises(ActionParseError, match=match):
        parse_native_tool_calls([{"type": "function_call", "name": name, "arguments": arguments}])


@pytest.mark.parametrize(
    "item,match",
    [
        ({"type": "function_call", "name": "shell", "arguments": "{}"}, "unsupported native browser tool"),
        (
            {"type": "function_call", "name": "computer", "arguments": '{"actions":[{"action":"exec"}]}'},
            "unsupported native computer action",
        ),
    ],
)
def test_rejects_unsafe_native_tool_calls(item, match) -> None:
    with pytest.raises(ActionParseError, match=match):
        parse_native_tool_calls([item])
