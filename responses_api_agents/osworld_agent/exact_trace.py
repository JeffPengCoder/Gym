# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Arash NeMo-RL exact-trace evidence for OSWorld rollouts.

The OSWorld agent owns the semantic trajectory and the generation server owns
the exact tokenization.  This module binds both views without pretending that
successive prompts are append-only.  NeMo-RL may then split one logical
rollout into prefix-contiguous physical traces while retaining one logical
reward and advantage.

The wire contract intentionally matches schema v2 from
``aroshanghias/context-compaction-v2-clean``.  It is implemented locally so
the OSWorld Gym adapter can also continue serving the legacy Rohit branch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


_POLICY_NAME = "osworld_exact_prompt_trace"
_POLICY_VERSION = "1"
_SOURCE_UNIT_ID = "osworld-materialized-model-prompt"
_ALLOWED_IDENTITY_GAPS = (
    "exact_tokenizer_identity_not_reported_by_generation_server",
    "exact_chat_template_identity_not_reported_by_generation_server",
    "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
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
    """Return the digest convention used by the trace-aware NeMo-RL branch."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    """Return a deterministic, bounded identifier."""

    return f"{prefix}-{canonical_digest(parts)[:24]}"


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"OSWorld exact_trace requires non-empty {field}")
    return value


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"OSWorld exact_trace requires non-negative integer {field}")
    return value


def _rollout_identity(extra: Mapping[str, Any]) -> dict[str, Any]:
    contract_version = extra.get("context_compaction_contract_version")
    if contract_version != 2:
        raise ValueError(
            f"OSWorld exact_trace requires context_compaction_contract_version=2; received {contract_version!r}"
        )
    return {
        "rollout_id": _require_nonempty_string(
            extra.get("context_compaction_rollout_id"),
            field="context_compaction_rollout_id",
        ),
        "group_id": _require_nonempty_string(
            extra.get("context_compaction_group_id"),
            field="context_compaction_group_id",
        ),
        "task_id": _require_nonempty_string(
            extra.get("context_compaction_task_id"),
            field="context_compaction_task_id",
        ),
        "rollout_index": _require_nonnegative_int(
            extra.get("context_compaction_rollout_index"),
            field="context_compaction_rollout_index",
        ),
        "attempt_index": _require_nonnegative_int(
            extra.get("context_compaction_attempt_index"),
            field="context_compaction_attempt_index",
        ),
    }


def _training_record(step: Mapping[str, Any], *, turn_id: int) -> Mapping[str, Any]:
    info = step.get("info")
    agent_info = info.get("agent") if isinstance(info, Mapping) else None
    training = agent_info.get("training") if isinstance(agent_info, Mapping) else None
    if not isinstance(training, Mapping):
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has no training evidence")
    return training


def _token_list(value: Any, *, field: str, turn_id: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has invalid {field}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has non-integer {field}")
    return [int(item) for item in value]


def _logprob_list(value: Any, *, turn_id: int) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has invalid generation_log_probs")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"OSWorld exact_trace turn {turn_id} has non-finite generation_log_probs")
    return result


def _image_source(part: Mapping[str, Any]) -> dict[str, Any] | None:
    part_type = part.get("type")
    if part_type not in {"image_url", "input_image", "image"}:
        return None
    raw_source = part.get("image_url") or part.get("image") or part.get("url")
    detail = part.get("detail") or "high"
    if isinstance(raw_source, Mapping):
        detail = raw_source.get("detail") or detail
        raw_source = raw_source.get("url")
    if not isinstance(raw_source, str) or not raw_source:
        raise ValueError("OSWorld exact_trace encountered an image without a source URL")
    return {
        "type": "input_image",
        "image_url": raw_source,
        "detail": str(detail),
    }


def _register_media(
    source_part: Mapping[str, Any],
    *,
    media_assets: dict[str, dict[str, Any]],
) -> str:
    canonical = _canonical_json(source_part)
    content_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    media_id = f"media-{content_digest[:24]}"
    asset = {
        "media_id": media_id,
        "content_digest": content_digest,
        "source_part": dict(source_part),
        "original_dimensions": None,
        "color_mode": None,
        "source_format": None,
    }
    previous = media_assets.setdefault(media_id, asset)
    if previous.get("content_digest") != content_digest:
        raise RuntimeError(f"OSWorld exact_trace media ID collision for {media_id}")
    return media_id


def _prompt_media_ids(
    steps: Sequence[Mapping[str, Any]],
    *,
    step_index: int,
    media_assets: dict[str, dict[str, Any]],
) -> list[str]:
    turn_id = step_index + 1
    training = _training_record(steps[step_index], turn_id=turn_id)
    prompt_user_message_count = training.get("prompt_user_message_count", 1)
    if (
        isinstance(prompt_user_message_count, bool)
        or not isinstance(prompt_user_message_count, int)
        or prompt_user_message_count < 1
        or prompt_user_message_count > turn_id
    ):
        raise ValueError(
            f"OSWorld exact_trace turn {turn_id} has invalid prompt_user_message_count={prompt_user_message_count!r}"
        )

    media_ids: list[str] = []
    first_step = step_index - prompt_user_message_count + 1
    for prompt_step_index in range(first_step, step_index + 1):
        prompt_training = _training_record(steps[prompt_step_index], turn_id=prompt_step_index + 1)
        user_message = prompt_training.get("new_user_message")
        if not isinstance(user_message, Mapping):
            raise ValueError(
                f"OSWorld exact_trace turn {turn_id} cannot reconstruct prompt media from step {prompt_step_index + 1}"
            )
        content = user_message.get("content") or []
        if not isinstance(content, (list, tuple)):
            raise ValueError(f"OSWorld exact_trace step {prompt_step_index + 1} has invalid user content")
        for part in content:
            if not isinstance(part, Mapping):
                continue
            source = _image_source(part)
            if source is not None:
                media_ids.append(_register_media(source, media_assets=media_assets))
    return media_ids


def _generation_contract(
    *,
    model_name: str,
    sampling_config: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> dict[str, Any]:
    component_ids = {
        "model_contract_id": stable_id(
            "model-contract",
            {"model_name": model_name, "adapter": "osworld_agent"},
        ),
        "tokenizer_contract_id": stable_id(
            "tokenizer-contract",
            "server-authoritative-unavailable",
        ),
        "template_contract_id": stable_id(
            "template-contract",
            "server-authoritative-unavailable",
        ),
        "sampling_contract_id": stable_id("sampling-contract", dict(sampling_config)),
        "processor_contract_id": stable_id(
            "processor-contract",
            "server-authoritative-unavailable",
        ),
        "compaction_policy_id": stable_id("compaction-policy", dict(policy_config)),
    }
    return {
        "schema_version": 1,
        **component_ids,
        "generation_contract_id": stable_id(
            "generation-contract",
            canonical_digest(component_ids),
        ),
        "loss_normalization": "global_action_token_mean",
        "training_eligible": False,
        "incomplete_reasons": list(_ALLOWED_IDENTITY_GAPS),
    }


def _lineage_state_digest(record: Mapping[str, Any]) -> str:
    normalized = {
        "source_unit_id": record.get("source_unit_id"),
        "source_digest": record.get("source_digest"),
        "disposition": record.get("disposition"),
        "output_unit_ids": list(record.get("output_unit_ids") or []),
        "output_digests": list(record.get("output_digests") or []),
    }
    return canonical_digest([normalized])


def _policy_lineage(
    *,
    turn_id: int,
    view_digest: str,
    policy_config_digest: str,
    generation_contract_id: str,
    rollout_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    transformation_id = stable_id(
        "transformation",
        rollout_id,
        turn_id,
        view_digest,
        policy_config_digest,
    )
    unit_record = {
        "source_unit_id": _SOURCE_UNIT_ID,
        "source_digest": view_digest,
        "disposition": "retained",
        "output_unit_ids": [_SOURCE_UNIT_ID],
        "output_digests": [view_digest],
    }
    lineage = {
        "transformation_id": transformation_id,
        "transformation_type": "osworld_prompt_materialization",
        "transformation_version": _POLICY_VERSION,
        "configuration_digest": policy_config_digest,
        "deterministic": True,
        "lossy": True,
        "generator_contract_id": generation_contract_id,
        "unit_records": [unit_record],
        "validator_result": "passed",
    }
    decision = {
        "policy_name": _POLICY_NAME,
        "policy_version": _POLICY_VERSION,
        "config_digest": policy_config_digest,
        "protected_part_ids": [],
        "changed_part_ranges": [],
        "retained_part_count": 1,
        "omitted_part_count": 0,
        "selection_digest": view_digest,
        "inserted_artifact_ids": [],
        "decision_turn": turn_id,
        "lineage": lineage,
    }
    evidence = {
        "policy_name": _POLICY_NAME,
        "policy_version": _POLICY_VERSION,
        "config_digest": policy_config_digest,
        "decision_turn": turn_id,
        "selection_digest": view_digest,
        "transformation_id": transformation_id,
    }
    return decision, evidence, unit_record


def build_exact_trace_envelope(
    *,
    steps: Sequence[Mapping[str, Any]],
    request_extra: Mapping[str, Any],
    model_name: str,
    sampling_config: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact-trace response fields consumed by Arash NeMo-RL."""

    if not steps:
        raise ValueError("OSWorld exact_trace requires at least one model call")
    identity = _rollout_identity(request_extra)
    rollout_id = identity["rollout_id"]
    generation_contract = _generation_contract(
        model_name=model_name,
        sampling_config=sampling_config,
        policy_config=policy_config,
    )
    generation_contract_id = generation_contract["generation_contract_id"]
    policy_config_digest = canonical_digest(dict(policy_config))

    media_assets: dict[str, dict[str, Any]] = {}
    completion_evidence: list[dict[str, Any]] = []
    boundary_events: list[dict[str, Any]] = []
    lineage_deltas: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    previous_context: list[int] = []
    previous_media_ids: list[str] = []
    previous_view_digest: str | None = None
    previous_transformation_id: str | None = None
    segment_index = -1
    segment_id = ""
    final_policy_decision: dict[str, Any] | None = None

    for step_index, step in enumerate(steps):
        turn_id = step_index + 1
        training = _training_record(step, turn_id=turn_id)
        model_response = training.get("response")
        if not isinstance(model_response, Mapping):
            raise ValueError(f"OSWorld exact_trace turn {turn_id} has no model response evidence")

        prompt_token_ids = _token_list(
            model_response.get("prompt_token_ids"),
            field="prompt_token_ids",
            turn_id=turn_id,
        )
        sampled_token_ids = _token_list(
            model_response.get("generation_token_ids"),
            field="generation_token_ids",
            turn_id=turn_id,
        )
        sampled_logprobs = _logprob_list(
            model_response.get("generation_log_probs"),
            turn_id=turn_id,
        )
        if len(sampled_token_ids) != len(sampled_logprobs):
            raise ValueError(
                f"OSWorld exact_trace turn {turn_id} token/logprob mismatch: "
                f"tokens={len(sampled_token_ids)} logprobs={len(sampled_logprobs)}"
            )
        eligible = training.get("eligible", True)
        if not isinstance(eligible, bool):
            raise ValueError(f"OSWorld exact_trace turn {turn_id} has non-boolean eligible")

        media_ids = _prompt_media_ids(
            steps,
            step_index=step_index,
            media_assets=media_assets,
        )
        token_append_compatible = previous_context == prompt_token_ids[: len(previous_context)]
        media_append_compatible = previous_media_ids == media_ids[: len(previous_media_ids)]
        append_compatible = turn_id > 1 and token_append_compatible and media_append_compatible
        view_digest = canonical_digest(
            {
                "prompt_token_ids": prompt_token_ids,
                "media_ids": media_ids,
            }
        )

        final_policy_decision, policy_decision, unit_record = _policy_lineage(
            turn_id=turn_id,
            view_digest=view_digest,
            policy_config_digest=policy_config_digest,
            generation_contract_id=generation_contract_id,
            rollout_id=rollout_id,
        )
        transformation_id = policy_decision["transformation_id"]
        lineage_deltas.append(
            {
                "transformation_id": transformation_id,
                "parent_transformation_id": previous_transformation_id,
                "transformation_type": "osworld_prompt_materialization",
                "transformation_version": _POLICY_VERSION,
                "configuration_digest": policy_config_digest,
                "deterministic": True,
                "lossy": True,
                "generator_contract_id": generation_contract_id,
                "unit_upserts": [unit_record],
                "source_unit_count": 1,
                "state_digest": _lineage_state_digest(unit_record),
                "validator_result": "passed",
            }
        )

        boundary_event_id = None
        if not append_compatible:
            segment_index += 1
            segment_id = stable_id("segment", rollout_id, segment_index, view_digest)
            if turn_id > 1:
                boundary_event_id = stable_id(
                    "rewrite-boundary",
                    rollout_id,
                    turn_id,
                    previous_view_digest,
                    view_digest,
                )
                boundary_events.append(
                    {
                        "event_id": boundary_event_id,
                        "trigger_after_step": turn_id - 1,
                        "applies_to_step": turn_id,
                        "reason": "prompt_or_media_not_append_compatible",
                        "policy_name": _POLICY_NAME,
                        "policy_version": _POLICY_VERSION,
                        "config_digest": policy_config_digest,
                        "previous_view_digest": previous_view_digest,
                        "current_view_digest": view_digest,
                        "changed_part_ranges": [],
                        "retained_part_count": 1,
                        "omitted_part_count": 0,
                        "retained_media_count": len(media_ids),
                        "removed_media_count": sum(media_id not in media_ids for media_id in previous_media_ids),
                        "inserted_artifact_ids": [],
                        "schedule_name": "per_action",
                        "schedule_version": _POLICY_VERSION,
                        "schedule_config_digest": policy_config_digest,
                        "chunk_id": segment_id,
                        "block_index": segment_index,
                    }
                )

        completion_id = stable_id(
            "completion",
            rollout_id,
            turn_id,
            prompt_token_ids,
            sampled_token_ids,
        )
        action_id = f"action-{turn_id:06d}"
        prepared_request_id = stable_id(
            "prepared-request",
            rollout_id,
            turn_id,
            view_digest,
            generation_contract_id,
        )
        request_id = stable_id(
            "request",
            prepared_request_id,
            prompt_token_ids,
            media_ids,
        )
        model_call_id = stable_id("model-call", request_id, completion_id)
        occurrence_counts: Counter[str] = Counter()
        media_occurrences = []
        for media_id in media_ids:
            occurrence_ordinal = occurrence_counts[media_id]
            occurrence_counts[media_id] += 1
            media_occurrences.append(
                {
                    "media_id": media_id,
                    "occurrence_ordinal": occurrence_ordinal,
                    "model_call_id": model_call_id,
                    "placeholder_span_or_position": None,
                    "processed_dimensions": None,
                    "model_specific_sidecars": {},
                }
            )
        span = {
            "policy_output_span_id": stable_id(
                "policy-output-span",
                model_call_id,
                action_id,
                len(sampled_token_ids),
            ),
            "model_call_id": model_call_id,
            "action_ids": [action_id],
            "start": 0,
            "end": len(sampled_token_ids),
            "eligible": eligible,
            "old_logprobs_alignment": "sampled_tokens",
        }
        completion_evidence.append(
            {
                "rollout_id": rollout_id,
                "completion_id": completion_id,
                "action_id": action_id,
                "turn_id": turn_id,
                "prepared_request_id": prepared_request_id,
                "request_id": request_id,
                "context_epoch": segment_index,
                "segment_index": segment_index,
                "segment_id": segment_id,
                "expected_append_compatible": append_compatible,
                "compaction_event_id": boundary_event_id,
                "prompt_token_ids": prompt_token_ids,
                "sampled_token_ids": sampled_token_ids,
                "sampled_logprobs": sampled_logprobs,
                "finish_reason": model_response.get("finish_reason"),
                "media_ids": media_ids,
                "policy_decision": policy_decision,
                "generation_contract_id": generation_contract_id,
                "policy_output_spans": [span],
                "media_occurrences": media_occurrences,
                "processor_fingerprint": model_response.get("processor_fingerprint"),
                "eligible": eligible,
                "evidence_source": "generation_response",
            }
        )
        transitions.append(
            {
                "turn_id": turn_id,
                "completion_id": completion_id,
                "action_id": action_id,
                "state": {
                    "prompt_view_digest": view_digest,
                    "media_ids": media_ids,
                },
                "action": {
                    "sampled_token_ids": sampled_token_ids,
                    "sampled_logprobs": sampled_logprobs,
                    "raw_completion": str(model_response.get("raw_content") or ""),
                    "parsed_actions": list(step.get("actions") or []),
                },
                "reward": float(step.get("reward") or 0.0),
                "done": bool(step.get("done", False)),
                "eligible": eligible,
            }
        )

        previous_context = [*prompt_token_ids, *sampled_token_ids]
        previous_media_ids = media_ids
        previous_view_digest = view_digest
        previous_transformation_id = transformation_id

    assert final_policy_decision is not None
    return {
        "media_assets": media_assets,
        "completion_evidence": completion_evidence,
        "final_policy_decision": final_policy_decision,
        "lineage_deltas": lineage_deltas,
        "chunk_records": [],
        "boundary_events": boundary_events,
        "guard_records": [],
        "trajectory_transitions": transitions,
        "context_compaction_contract": {
            "schema_version": 2,
            "mode": "exact_trace_authority",
            **identity,
            "generation_contract": generation_contract,
        },
    }
