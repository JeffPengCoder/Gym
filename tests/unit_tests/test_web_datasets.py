# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.web.datasets import (
    adapt_native_visualwebarena_records,
    adapt_native_webarena_record,
    adapt_native_webarena_records,
    adapt_native_webvoyager_record,
    adapt_visualwebarena_records,
    adapt_webarena_record,
    adapt_webvoyager_record,
    load_json_records,
    write_jsonl,
)
from nemo_gym.web.models import WebTask


def test_webarena_record_preserves_source_and_splits_multi_page_start():
    record = {
        "task_id": 7,
        "intent": "Compare the pages",
        "sites": ["reddit", "gitlab"],
        "start_url": "__REDDIT__ |AND| __GITLAB__",
        "eval": {"reference_answers": {"exact_match": "secret"}},
    }

    row = adapt_webarena_record(record)
    task = WebTask.model_validate(row["web_task"])

    assert task.start_urls == ["__REDDIT__", "__GITLAB__"]
    assert task.original_metadata == record
    assert row["responses_create_params"]["metadata"]["task_id"] == "7"


def test_null_storage_state_does_not_become_string_auth_profile():
    row = adapt_webarena_record(
        {
            "task_id": 8,
            "intent": "Inspect the map",
            "require_login": True,
            "storage_state": None,
        }
    )

    task = WebTask.model_validate(row["web_task"])
    assert task.auth_profile is None


def test_webarena_auth_and_non_string_start_urls_are_normalized():
    row = adapt_webarena_record(
        {
            "task_id": 9,
            "require_login": True,
            "storage_state": 7,
            "start_url": ["https://one.example", "", 2],
        }
    )

    task = WebTask.model_validate(row["web_task"])
    assert task.auth_profile == "7"
    assert task.start_urls == ["https://one.example", "2"]

    scalar = WebTask.model_validate(adapt_webarena_record({"task_id": 10, "start_url": 123})["web_task"])
    assert scalar.start_urls == ["123"]


def test_visualwebarena_partitions_are_globally_reindexed():
    rows = adapt_visualwebarena_records(
        [
            ("classifieds", [{"task_id": 0, "intent": "c", "sites": ["classifieds"]}]),
            (
                "reddit",
                [
                    {"task_id": 0, "intent": "r0", "sites": ["reddit"]},
                    {"task_id": 1, "intent": "r1", "sites": ["wikipedia"]},
                ],
            ),
            ("shopping", [{"task_id": 0, "intent": "s", "sites": ["shopping"]}]),
        ]
    )

    tasks = [WebTask.model_validate(row["web_task"]) for row in rows]
    assert [task.task_id for task in tasks] == ["0", "1", "2", "3"]
    assert tasks[-1].original_metadata["_source_task_id"] == 0
    assert tasks[-1].original_metadata["_source_partition"] == "shopping"


def test_webvoyager_uses_legacy_action_surface_over_browsergym():
    row = adapt_webvoyager_record(
        {
            "web_name": "Allrecipes",
            "id": "Allrecipes--0",
            "ques": "Find a recipe",
            "web": "https://www.allrecipes.com/",
        }
    )
    task = WebTask.model_validate(row["web_task"])

    assert task.runtime_profile.value == "browsergym"
    assert task.action_profile.value == "webvoyager_legacy"
    assert task.start_urls == ["https://www.allrecipes.com/"]


def test_native_webvoyager_adapter_is_exposed_through_dataset_api():
    row = adapt_native_webvoyager_record({"id": "Allrecipes--0", "ques": "Find a recipe"})
    assert row["web_task"]["runtime_profile"] == "native_visual"


def test_native_webarena_uses_reference_numeric_task_id_and_shared_tools():
    row = adapt_native_webarena_record(
        {
            "id": "webarena-17",
            "ques": "Inspect an order",
            "web_name": ["shopping_admin"],
            "web": ["__SHOPPING_ADMIN__"],
            "eval": {"eval_types": ["string_match"]},
        }
    )

    assert row["web_task"]["task_id"] == "17"
    assert row["web_task"]["runtime_profile"] == "native_visual"
    assert row["web_task"]["verifier_profile"] == "native_webarena_classic"
    assert row["responses_create_params"]["tools"]


def test_native_webarena_batch_preserves_cross_task_collision_plan():
    rows = adapt_native_webarena_records(
        [
            {
                "id": f"webarena-{index}",
                "ques": "Update the shared profile",
                "web_name": ["gitlab"],
                "web": ["__GITLAB__/byteblaze"],
                "eval": {
                    "eval_types": ["program_html"],
                    "program_html": [
                        {
                            "url": "__GITLAB__/byteblaze",
                            "locator": "document.body.innerText",
                            "required_contents": {"must_include": ["updated"]},
                        }
                    ],
                },
            }
            for index in range(2)
        ]
    )

    for row in rows:
        targets = row["web_task"]["task_kwargs"]["collision_plan"]["snapshot_adapters"]["program_html"]["targets"]
        assert len(targets) == 1


def test_native_visualwebarena_matches_reference_contiguous_ids_and_images():
    rows = adapt_native_visualwebarena_records(
        [
            {
                "id": "visualwebarena-577",
                "ques": "First",
                "web_name": ["reddit"],
                "web": ["__REDDIT__"],
                "eval": {},
            },
            {
                "id": "visualwebarena-580",
                "ques": "Second",
                "web_name": ["shopping"],
                "web": ["__SHOPPING__"],
                "image": ["input-0.png", "input-1.png"],
                "eval": {},
            },
        ]
    )

    assert [row["web_task"]["task_id"] for row in rows] == ["0", "1"]
    assert rows[1]["web_task"]["input_images"] == ["input-0.png", "input-1.png"]
    assert rows[1]["web_task"]["original_metadata"]["id"] == "visualwebarena-580"


def test_write_jsonl_is_utf8_and_newline_delimited(tmp_path):
    output = tmp_path / "rows.jsonl"
    assert write_jsonl([{"text": "中文"}, {"text": "English"}], output) == 2
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"text": "中文"},
        {"text": "English"},
    ]


def test_load_json_records_accepts_json_and_jsonl_and_rejects_non_objects(tmp_path):
    json_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    invalid_path = tmp_path / "invalid.json"
    json_path.write_text('[{"id": 1}]', encoding="utf-8")
    jsonl_path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    invalid_path.write_text('[{"id": 1}, 2]', encoding="utf-8")

    assert load_json_records(json_path) == [{"id": 1}]
    assert load_json_records(jsonl_path) == [{"id": 1}, {"id": 2}]
    with pytest.raises(ValueError, match="JSON array or JSONL stream"):
        load_json_records(invalid_path)
