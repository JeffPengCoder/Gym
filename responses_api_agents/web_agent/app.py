# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rollout loop shared by WebArena, VisualWebArena, and WebVoyager."""

from __future__ import annotations

from collections import deque
from typing import Any, Literal, Optional

from fastapi import Body, Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.web.actions import ActionParseError, parse_model_action
from nemo_gym.web.models import WebBenchmark, WebObservation, WebTask, WebVerifierResult
from resources_servers.browsergym_web.models import (
    WebEvaluateResponse,
    WebSeedSessionResponse,
    WebStepResponse,
)
from responses_api_agents.web_agent.render import parse_error_message, render_observation


class WebAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    webvoyager_judge_server: Optional[ResourcesServerRef] = None
    max_steps: int = Field(default=15, ge=1, le=200)
    max_parse_retries: int = Field(default=2, ge=0, le=10)
    max_image_history: int = Field(default=3, ge=1, le=20)
    judge_max_screenshots: int = Field(default=3, ge=1, le=20)
    visual_observation_text: Literal["full_axtree", "som_only", "none"] = "full_axtree"
    redact_old_visual_observations: bool = False


class WebAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    web_task: WebTask | None = None
    verifier_metadata: dict[str, Any] | None = None


class WebAgentRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    benchmark: str
    task_id: str
    raw_score: float = 0.0
    task_success: bool = False
    mask_sample: bool = False
    failure_kind: str | None = None
    terminated: bool = False
    truncated: bool = False
    environment_steps: int = 0
    model_turns: int = 0
    execution_failures: int = 0
    verifier_result: WebVerifierResult | None = None


def _extract_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if getattr(block, "type", None) == "output_text":
                parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part).strip()


def _resolve_task(body: WebAgentRunRequest) -> WebTask:
    if body.web_task is not None:
        return body.web_task
    metadata = body.verifier_metadata or {}
    candidate = metadata.get("web_task") or metadata.get("task")
    if candidate is None and "benchmark" in metadata and "task_id" in metadata:
        candidate = metadata
    if candidate is None:
        extra = body.model_extra or {}
        candidate = extra.get("web_task") or extra.get("task")
    if candidate is None:
        raise ValueError("a normalized web_task is required in the row or verifier_metadata")
    return WebTask.model_validate(candidate)


def _merge_usage(total, response: NeMoGymResponse):
    if response.usage is None:
        return total
    if total is None:
        return response.usage.model_copy(deep=True)
    total.input_tokens += response.usage.input_tokens
    total.output_tokens += response.usage.output_tokens
    total.total_tokens += response.usage.total_tokens
    total.input_tokens_details.cached_tokens = 0
    total.output_tokens_details.reasoning_tokens = 0
    return total


def _redact_old_images(
    items: list[Any],
    max_image_history: int,
    *,
    redact_observation_text: bool = False,
) -> list[Any]:
    """Keep only the newest N image-bearing messages in the next model call."""

    copied = [item.model_copy(deep=True) if hasattr(item, "model_copy") else item for item in items]
    image_message_indices: list[int] = []
    for index, item in enumerate(copied):
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        if any(
            (isinstance(block, dict) and block.get("type") == "input_image")
            or getattr(block, "type", None) == "input_image"
            for block in content
        ):
            image_message_indices.append(index)
    for index in image_message_indices[:-max_image_history]:
        item = copied[index]
        content = getattr(item, "content", None)
        if redact_observation_text:
            item.content = [
                {
                    "type": "input_text",
                    "text": "[Earlier screenshot and page text omitted from context.]",
                }
            ]
            continue
        retained = [
            block
            for block in content
            if not (
                (isinstance(block, dict) and block.get("type") == "input_image")
                or getattr(block, "type", None) == "input_image"
            )
        ]
        retained.append({"type": "input_text", "text": "[Earlier screenshot omitted from context.]"})
        item.content = retained
    return copied


class WebAgent(SimpleResponsesAPIAgent):
    config: WebAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        result = NeMoGymResponse.model_validate(await get_response_json(model_response))
        for key, value in model_response.cookies.items():
            response.set_cookie(key, value)
        return result

    async def run(self, request: Request, body: WebAgentRunRequest) -> WebAgentRunResponse:
        task = _resolve_task(body)
        env_cookies = request.cookies
        model_cookies = None
        seeded = False
        last_model_response: NeMoGymResponse | None = None
        usage = None
        trajectory: list[Any] = []
        screenshot_history: deque[str] = deque(maxlen=self.config.judge_max_screenshots)
        url_history: deque[str] = deque(maxlen=self.config.judge_max_screenshots)
        final_answer: str | None = None
        terminated = False
        truncated = False
        environment_steps = 0
        model_turns = 0
        execution_failures = 0
        verifier_result: WebVerifierResult | None = None
        infrastructure_error: Exception | None = None

        base_body = body.responses_create_params.model_copy(deep=True)
        if isinstance(base_body.input, str):
            base_body.input = [NeMoGymEasyInputMessage(role="user", content=base_body.input)]

        try:
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json={"task": task.model_dump(mode="json")},
                cookies=env_cookies,
            )
            await raise_for_status(seed_response)
            seed_data = WebSeedSessionResponse.model_validate(await get_response_json(seed_response))
            env_cookies = seed_response.cookies
            seeded = True
            observation = seed_data.observation
            self._remember_evidence(observation, screenshot_history, url_history)
            base_body.input = list(base_body.input) + [
                render_observation(
                    observation,
                    task,
                    step_index=0,
                    visual_observation_text=self.config.visual_observation_text,
                )
            ]

            rollout_finished = False
            for step_index in range(self.config.max_steps):
                action = None
                for parse_attempt in range(self.config.max_parse_retries + 1):
                    model_input = _redact_old_images(
                        list(base_body.input) + trajectory,
                        self.config.max_image_history,
                        redact_observation_text=self.config.redact_old_visual_observations,
                    )
                    model_body = base_body.model_copy(update={"input": model_input})
                    raw_model_response = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path=self.url_path_for_run("/v1/responses", body),
                        json=model_body,
                        cookies=model_cookies,
                    )
                    await raise_for_status(raw_model_response)
                    model_response = NeMoGymResponse.model_validate(await get_response_json(raw_model_response))
                    model_cookies = raw_model_response.cookies
                    last_model_response = model_response
                    model_turns += 1
                    usage = _merge_usage(usage, model_response)
                    trajectory.extend(model_response.output)
                    model_text = _extract_output_text(model_response)
                    try:
                        action = parse_model_action(model_text, task.action_profile)
                        break
                    except ActionParseError as exc:
                        if parse_attempt >= self.config.max_parse_retries:
                            trajectory.append(
                                NeMoGymEasyInputMessage(
                                    role="user",
                                    content=f"Action parsing failed permanently: {exc}",
                                )
                            )
                            break
                        trajectory.append(parse_error_message(exc))

                if action is None:
                    break

                step_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json={
                        "operation_id": f"step-{step_index}",
                        "action": action.model_dump(mode="json"),
                    },
                    cookies=env_cookies,
                )
                await raise_for_status(step_response)
                step_data = WebStepResponse.model_validate(await get_response_json(step_response))
                env_cookies = step_response.cookies
                environment_steps += 1
                if not step_data.execution_ok:
                    execution_failures += 1
                observation = step_data.observation
                self._remember_evidence(observation, screenshot_history, url_history)
                terminated = step_data.terminated
                truncated = step_data.truncated
                if action.terminal:
                    final_answer = action.answer
                if action.terminal or terminated or truncated:
                    rollout_finished = True
                    break
                trajectory.append(
                    render_observation(
                        observation,
                        task,
                        step_index=step_index + 1,
                        visual_observation_text=self.config.visual_observation_text,
                    )
                )

            if not rollout_finished:
                truncated = True

            evaluate_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/evaluate",
                json={"final_answer": final_answer},
                cookies=env_cookies,
            )
            await raise_for_status(evaluate_response)
            evaluation = WebEvaluateResponse.model_validate(await get_response_json(evaluate_response))
            env_cookies = evaluate_response.cookies
            verifier_result = evaluation.result

            if task.benchmark == WebBenchmark.WEBVOYAGER:
                verifier_result = await self._judge_webvoyager(
                    task=task,
                    final_answer=final_answer or "",
                    screenshots=list(screenshot_history),
                    urls=list(url_history),
                    body=body,
                )
        except Exception as exc:  # noqa: BLE001 - return a masked rollout when a trajectory exists.
            infrastructure_error = exc
            if last_model_response is None:
                raise
            verifier_result = WebVerifierResult(
                valid_sample=False,
                failure_kind=f"infrastructure_error:{type(exc).__name__}",
                metadata={"error": str(exc)},
            )
        finally:
            if seeded:
                try:
                    await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/close",
                        json={},
                        cookies=env_cookies,
                    )
                except Exception:  # noqa: BLE001 - cleanup must not replace a completed result.
                    pass

        if last_model_response is None:
            if infrastructure_error is not None:
                raise infrastructure_error
            raise RuntimeError("web rollout ended before the policy returned a response")
        if verifier_result is None:
            verifier_result = WebVerifierResult(
                valid_sample=False,
                failure_kind="missing_verifier_result",
            )

        last_model_response.output = trajectory
        last_model_response.usage = usage
        return WebAgentRunResponse(
            responses_create_params=base_body,
            response=last_model_response,
            reward=verifier_result.reward if verifier_result.valid_sample else 0.0,
            benchmark=task.benchmark.value,
            task_id=task.task_id,
            raw_score=verifier_result.raw_score,
            task_success=verifier_result.task_success,
            mask_sample=not verifier_result.valid_sample,
            failure_kind=verifier_result.failure_kind,
            terminated=terminated,
            truncated=truncated,
            environment_steps=environment_steps,
            model_turns=model_turns,
            execution_failures=execution_failures,
            verifier_result=verifier_result,
        )

    async def _judge_webvoyager(
        self,
        *,
        task: WebTask,
        final_answer: str,
        screenshots: list[str],
        urls: list[str],
        body: WebAgentRunRequest,
    ) -> WebVerifierResult:
        if self.config.webvoyager_judge_server is None:
            return WebVerifierResult(
                valid_sample=False,
                failure_kind="webvoyager_judge_not_configured",
                verifier_version="webvoyager-llm-judge-v1",
            )
        judge_response = await self.server_client.post(
            server_name=self.config.webvoyager_judge_server.name,
            url_path="/verify_webvoyager",
            json={
                "task": task.model_dump(mode="json"),
                "final_answer": final_answer,
                "screenshots": screenshots,
                "page_urls": urls,
            },
        )
        await raise_for_status(judge_response)
        payload = await get_response_json(judge_response)
        candidate = payload.get("result") if isinstance(payload, dict) else None
        if candidate is None:
            raise RuntimeError("WebVoyager judge response did not contain result")
        return WebVerifierResult.model_validate(candidate)

    @staticmethod
    def _remember_evidence(
        observation: WebObservation,
        screenshots: deque[str],
        urls: deque[str],
    ) -> None:
        if observation.screenshot is not None and observation.screenshot.data_url:
            screenshots.append(observation.screenshot.data_url)
        if observation.url:
            urls.append(observation.url)


if __name__ == "__main__":
    WebAgent.run_webserver()
