# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-independent OSWorld trajectory contracts.

Every OSWorld run has an environment trajectory, regardless of whether its
model endpoint exposes tokenizer-level evidence.  This module records that
semantic trajectory first and describes exact model-call evidence as an
optional capability.  Training is a consumer decision; Gym never changes the
agent's prompting policy merely because a caller may later train on the run.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


_CALLER_IDENTITY_FIELDS = (
    "context_compaction_rollout_id",
    "context_compaction_group_id",
    "context_compaction_task_id",
    "context_compaction_rollout_index",
    "context_compaction_attempt_index",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible evidence."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    """Return a deterministic, bounded identifier."""

    return f"{prefix}-{canonical_digest(parts)[:24]}"


def _nonnegative_int(value: Any, *, fallback: int, field: str) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _trajectory_identity(
    *,
    request_extra: Mapping[str, Any],
    verifier_metadata: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    caller_values = {
        field: request_extra.get(field) for field in _CALLER_IDENTITY_FIELDS
    }
    caller_stamped = (
        request_extra.get("context_compaction_contract_version") is not None
        or any(value is not None for value in caller_values.values())
    )
    if caller_stamped:
        if request_extra.get("context_compaction_contract_version") != 2:
            raise ValueError(
                "Caller-stamped OSWorld trajectory identity requires "
                "context_compaction_contract_version=2"
            )
        missing = [field for field, value in caller_values.items() if value is None]
        if missing:
            raise ValueError(
                "Caller-stamped OSWorld trajectory identity is incomplete: "
                + ", ".join(missing)
            )
        rollout_id = caller_values["context_compaction_rollout_id"]
        group_id = caller_values["context_compaction_group_id"]
        task_id = caller_values["context_compaction_task_id"]
        rollout_index = _nonnegative_int(
            caller_values["context_compaction_rollout_index"],
            fallback=0,
            field="context_compaction_rollout_index",
        )
        attempt_index = _nonnegative_int(
            caller_values["context_compaction_attempt_index"],
            fallback=0,
            field="context_compaction_attempt_index",
        )
        for field, value in (
            ("context_compaction_rollout_id", rollout_id),
            ("context_compaction_group_id", group_id),
            ("context_compaction_task_id", task_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        identity_source = "caller"
    else:
        task = verifier_metadata.get("osworld_task")
        task_id = verifier_metadata.get("task_id")
        if not task_id and isinstance(task, Mapping):
            task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            task_id = "unknown-task"
        rollout_index = _nonnegative_int(
            request_extra.get("_ng_rollout_index"),
            fallback=0,
            field="_ng_rollout_index",
        )
        attempt_index = _nonnegative_int(
            request_extra.get("_ng_attempt_index"),
            fallback=0,
            field="_ng_attempt_index",
        )
        group_id = stable_id("trajectory-group", task_id)
        rollout_id = stable_id(
            "rollout",
            task_id,
            group_id,
            rollout_index,
            attempt_index,
            model_name,
        )
        identity_source = "derived"

    return {
        "rollout_id": rollout_id,
        "group_id": group_id,
        "task_id": task_id,
        "rollout_index": rollout_index,
        "attempt_index": attempt_index,
        "identity_source": identity_source,
    }


def _exact_generation_arrays(
    response: Mapping[str, Any],
) -> tuple[dict[str, list[Any]] | None, list[str]]:
    reasons: list[str] = []
    prompt_ids = response.get("prompt_token_ids")
    generation_ids = response.get("generation_token_ids")
    generation_logprobs = response.get("generation_log_probs")
    if not isinstance(prompt_ids, (list, tuple)) or not prompt_ids:
        reasons.append("exact_prompt_token_ids_unavailable")
    elif any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in prompt_ids
    ):
        reasons.append("exact_prompt_token_ids_invalid")
    if not isinstance(generation_ids, (list, tuple)) or not generation_ids:
        reasons.append("exact_sampled_token_ids_unavailable")
    elif any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in generation_ids
    ):
        reasons.append("exact_sampled_token_ids_invalid")
    if not isinstance(generation_logprobs, (list, tuple)):
        reasons.append("exact_sampled_logprobs_unavailable")
    elif any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in generation_logprobs
    ):
        reasons.append("exact_sampled_logprobs_invalid")
    if not reasons and len(generation_ids) != len(generation_logprobs):
        reasons.append("sampled_token_logprob_length_mismatch")
    if reasons:
        return None, reasons
    return {
        "prompt_token_ids": [int(value) for value in prompt_ids],
        "sampled_token_ids": [int(value) for value in generation_ids],
        "sampled_logprobs": [float(value) for value in generation_logprobs],
    }, []


def collect_model_calls(
    steps: Sequence[Mapping[str, Any]],
    *,
    trajectory_id: str,
    sample_eligible: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect model-call records without assuming one call per env step."""

    model_calls: list[dict[str, Any]] = []
    incomplete_reasons: set[str] = set()
    for step_position, step in enumerate(steps):
        info = step.get("info")
        agent_info = info.get("agent") if isinstance(info, Mapping) else None
        raw_calls = agent_info.get("model_calls") if isinstance(agent_info, Mapping) else None
        if not isinstance(raw_calls, list) or not raw_calls:
            incomplete_reasons.add("model_call_evidence_unavailable")
            continue
        for local_call_index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, Mapping):
                incomplete_reasons.add("model_call_evidence_invalid")
                continue
            parse_attempt = raw_call.get("parse_attempt")
            if (
                isinstance(parse_attempt, bool)
                or not isinstance(parse_attempt, int)
                or parse_attempt <= 0
            ):
                parse_attempt = local_call_index + 1
                incomplete_reasons.add("model_call_parse_attempt_invalid")
            model_call_id = stable_id(
                "model-call",
                trajectory_id,
                step_position,
                parse_attempt,
            )
            prompt_messages = raw_call.get("prompt_messages")
            response = raw_call.get("response")
            call_reasons: list[str] = []
            if not isinstance(prompt_messages, list) or not prompt_messages:
                call_reasons.append("exact_prompt_messages_unavailable")
            if not isinstance(response, Mapping):
                call_reasons.append("structured_generation_response_unavailable")
                arrays = None
            else:
                arrays, array_reasons = _exact_generation_arrays(response)
                call_reasons.extend(array_reasons)
            incomplete_reasons.update(call_reasons)
            reward = step.get("reward", 0.0)
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise TypeError(f"OSWorld step {step_position} reward must be numeric")
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError(f"OSWorld step {step_position} reward must be finite")
            parsed_actions = raw_call.get("parsed_actions")
            if not isinstance(parsed_actions, list):
                parsed_actions = []
            model_calls.append(
                {
                    "model_call_id": model_call_id,
                    "turn_id": len(model_calls) + 1,
                    "environment_step": step.get("step", step_position),
                    "step_position": step_position,
                    "parse_attempt": parse_attempt,
                    "prompt_messages": prompt_messages,
                    "response": dict(response) if isinstance(response, Mapping) else None,
                    "exact_generation_arrays": arrays,
                    "exact_evidence": not call_reasons,
                    "accepted": raw_call.get("accepted") is True,
                    "parse_error": raw_call.get("parse_error"),
                    "parsed_actions": list(parsed_actions),
                    "reward": reward,
                    "done": bool(step.get("done", False)),
                    "eligible": sample_eligible,
                }
            )
    return model_calls, sorted(incomplete_reasons)


def build_trajectory_envelope(
    *,
    steps: Sequence[Mapping[str, Any]],
    request_extra: Mapping[str, Any],
    verifier_metadata: Mapping[str, Any],
    model_name: str,
    sample_eligible: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the universal semantic trajectory and evidence capability report."""

    identity = _trajectory_identity(
        request_extra=request_extra,
        verifier_metadata=verifier_metadata,
        model_name=model_name,
    )
    trajectory_id = stable_id("trajectory", identity, model_name)
    model_calls, evidence_reasons = collect_model_calls(
        steps,
        trajectory_id=trajectory_id,
        sample_eligible=sample_eligible,
    )
    calls_by_step: dict[int, list[dict[str, Any]]] = {}
    for model_call in model_calls:
        calls_by_step.setdefault(model_call["step_position"], []).append(model_call)

    transitions: list[dict[str, Any]] = []
    for step_position, step in enumerate(steps):
        reward = step.get("reward", 0.0)
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError(f"OSWorld step {step_position} reward must be numeric")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(f"OSWorld step {step_position} reward must be finite")
        actions = step.get("actions") or []
        if not isinstance(actions, list):
            raise TypeError(f"OSWorld step {step_position} actions must be a list")
        step_calls = calls_by_step.get(step_position, [])
        transition_id = stable_id("transition", trajectory_id, step_position)
        transitions.append(
            {
                "transition_id": transition_id,
                "turn_id": step_position + 1,
                "state": {
                    "observation": dict(step.get("state") or {}),
                    "model_call_ids": [call["model_call_id"] for call in step_calls],
                },
                "action": {
                    "raw_completion": str(step.get("model_text") or ""),
                    "parsed_actions": list(actions),
                    "accepted_model_call_id": next(
                        (
                            call["model_call_id"]
                            for call in reversed(step_calls)
                            if call["accepted"]
                        ),
                        None,
                    ),
                },
                "reward": reward,
                "next_state": {
                    "observation": dict(step.get("next_state") or {}),
                },
                "done": bool(step.get("done", False)),
                "eligible": sample_eligible,
            }
        )

    exact_model_call_evidence = bool(model_calls) and not evidence_reasons
    eligibility_reasons = list(evidence_reasons)
    if not sample_eligible:
        eligibility_reasons.append("rollout_sample_masked")
    if identity["identity_source"] != "caller":
        eligibility_reasons.append("caller_owned_rollout_identity_unavailable")
    status = (
        "requires_runtime_admission"
        if exact_model_call_evidence
        and sample_eligible
        and identity["identity_source"] == "caller"
        else "ineligible"
    )
    contract_without_id = {
        "schema_version": 2,
        "mode": "osworld_semantic_trajectory",
        **identity,
        "trajectory_id": trajectory_id,
        "model_name": model_name,
        "transition_count": len(transitions),
        "model_call_count": len(model_calls),
        "capabilities": {
            "semantic_trajectory": True,
            "exact_model_call_evidence": exact_model_call_evidence,
            "arbitrary_prompt_rewrites": exact_model_call_evidence,
            "trainable_token_reconstruction": exact_model_call_evidence,
        },
        "training_eligibility": {
            "status": status,
            "incomplete_reasons": sorted(set(eligibility_reasons)),
        },
    }
    trajectory_contract = {
        **contract_without_id,
        "trajectory_contract_id": stable_id(
            "trajectory-contract",
            contract_without_id,
        ),
    }
    model_call_summaries = [
        {
            "model_call_id": call["model_call_id"],
            "turn_id": call["turn_id"],
            "environment_step": call["environment_step"],
            "parse_attempt": call["parse_attempt"],
            "accepted": call["accepted"],
            "parse_error": call["parse_error"],
            "exact_evidence": call["exact_evidence"],
        }
        for call in model_calls
    ]
    return (
        {
            "trajectory_contract": trajectory_contract,
            "trajectory_transitions": transitions,
            "model_call_summaries": model_call_summaries,
        },
        model_calls,
    )
