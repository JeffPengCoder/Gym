"""WebVoyager LLM-as-judge evaluation helpers."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx


WEBVOYAGER_JUDGE_SYSTEM_PROMPT = """You will evaluate whether a web assistant was able to successfully complete a given web task. As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'FAILURE'.

Return JSON only, using this schema:
{"thought": "Reasoning behind the verdict", "verdict": "SUCCESS" | "FAILURE"}"""


WEBVOYAGER_EXTEND_JUDGE_SYSTEM_PROMPT = """You will evaluate whether a web assistant was able to successfully complete a given web task. As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Evidence Timeline: This contains selected evidence from the assistant's browsing trajectory. Evidence may include screenshots and text-tool outputs. Screenshots are visual representations of the screen showing the result or intermediate state of performing a web task. Text-tool outputs from `find`, `get_page_text`, and `find_page_text` are textual evidence collected from the browser/page, including content that may not be visible in a screenshot.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the evidence timeline or result response.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the evidence timeline and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that screenshots are authentic visual evidence, and `find`, `get_page_text`, and `find_page_text` outputs are direct textual evidence from the browser/page. Some pure lookup steps may omit screenshots because these tools do not change the visible page state; do not penalize the assistant for a missing screenshot on those steps when text-tool evidence is provided.
-- Note the difference: 1) Result response may contradict screenshots or text-tool evidence, then the content of the screenshots or text-tool evidence prevails, 2) If the Result Response includes details that are not directly shown in the evidence timeline, you may accept those details only when they are plausible and do not conflict with the evidence or task requirements.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'FAILURE'.

Return JSON only, using this schema:
{"thought": "Reasoning behind the verdict", "verdict": "SUCCESS" | "FAILURE"}"""

LOOKUP_TOOLS_FOR_EXTENDED_JUDGE = {"find", "get_page_text", "find_page_text"}


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _screenshot_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"step_(\d+)_", path.name)
    if match:
        return int(match.group(1)), path.name
    match = re.search(r"screenshot_(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def collect_webvoyager_screenshots(result_dir: Path) -> list[Path]:
    traj_path = result_dir / "traj.jsonl"
    screenshots: list[Path] = []

    if traj_path.exists():
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            screenshot_file = entry.get("screenshot_file")
            if screenshot_file:
                screenshot_path = result_dir / screenshot_file
                if screenshot_path.exists():
                    screenshots.append(screenshot_path)

    if not screenshots:
        screenshots = sorted(result_dir.glob("step_*.png"), key=_screenshot_sort_key)

    # Preserve trajectory order but avoid duplicate paths if both sources overlap.
    seen: set[Path] = set()
    unique_screenshots: list[Path] = []
    for screenshot in screenshots:
        resolved = screenshot.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_screenshots.append(screenshot)

    return unique_screenshots


def _load_traj_entries(result_dir: Path) -> list[dict[str, Any]]:
    traj_path = result_dir / "traj.jsonl"
    if not traj_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _action_items(entry: dict[str, Any]) -> list[dict[str, Any]]:
    action = entry.get("action")
    if isinstance(action, list):
        return [item for item in action if isinstance(item, dict)]
    if isinstance(action, dict):
        return [action]
    return []


def _action_name(item: dict[str, Any]) -> str:
    if item.get("name"):
        return str(item.get("name"))
    function = item.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function.get("name"))
    return ""


def _action_arguments(item: dict[str, Any]) -> Any:
    if "arguments" in item:
        return item.get("arguments")
    function = item.get("function")
    if isinstance(function, dict):
        raw = function.get("arguments")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw
    return None


def _is_lookup_only_step(actions: list[dict[str, Any]]) -> bool:
    names = [_action_name(item) for item in actions]
    return bool(names) and all(name in LOOKUP_TOOLS_FOR_EXTENDED_JUDGE for name in names)


def _tool_feedback_outputs(entry: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    info = entry.get("info") or {}
    turns = info.get("tool_feedback_turns") or []
    if not isinstance(turns, list):
        return outputs
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        name = str(turn.get("name") or "")
        if name not in LOOKUP_TOOLS_FOR_EXTENDED_JUDGE:
            continue
        content = turn.get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            outputs[name] = str(content)
            continue
        if isinstance(parsed, dict):
            if "content" in parsed:
                outputs[name] = str(parsed["content"])
            elif "error" in parsed:
                outputs[name] = str(parsed["error"])
            else:
                outputs[name] = json.dumps(parsed, ensure_ascii=False, default=str)
        elif parsed is not None:
            outputs[name] = str(parsed)
    return outputs


def _aligned_tool_result_outputs(entry: dict[str, Any], actions: list[dict[str, Any]]) -> dict[int, str]:
    info = entry.get("info") or {}
    tool_results = info.get("tool_results") or []
    if not isinstance(tool_results, list):
        return {}
    outputs: dict[int, str] = {}
    for idx, action in enumerate(actions):
        if _action_name(action) in LOOKUP_TOOLS_FOR_EXTENDED_JUDGE and idx < len(tool_results):
            outputs[idx] = str(tool_results[idx])
    return outputs


def _format_lookup_tool_evidence(
    entry: dict[str, Any],
    actions: list[dict[str, Any]],
) -> str:
    feedback_by_name = _tool_feedback_outputs(entry)
    aligned_outputs = _aligned_tool_result_outputs(entry, actions)
    blocks: list[str] = []
    for idx, action in enumerate(actions):
        name = _action_name(action)
        if name not in LOOKUP_TOOLS_FOR_EXTENDED_JUDGE:
            continue
        args = _action_arguments(action)
        args_text = json.dumps(args, ensure_ascii=False, default=str) if not isinstance(args, str) else args
        output = feedback_by_name.get(name)
        if output is None:
            output = aligned_outputs.get(idx, "")
        blocks.append(
            f"Tool: {name}\n"
            f"Arguments: {args_text}\n"
            f"Output:\n{output if output else '[No tool output recorded]'}"
        )
    return "\n\n".join(blocks)


def collect_webvoyager_extended_evidence(result_dir: Path) -> list[dict[str, Any]]:
    """Collect ordered evidence items for the extended WebVoyager judge."""

    entries = _load_traj_entries(result_dir)
    if not entries:
        return [
            {"kind": "screenshot", "step": idx, "path": path}
            for idx, path in enumerate(collect_webvoyager_screenshots(result_dir), start=1)
        ]

    evidence: list[dict[str, Any]] = []
    seen_screenshots: set[Path] = set()
    for entry in entries:
        step = entry.get("step_num")
        actions = _action_items(entry)
        lookup_text = _format_lookup_tool_evidence(entry, actions)
        if lookup_text:
            evidence.append({"kind": "text", "step": step, "text": lookup_text})

        if _is_lookup_only_step(actions):
            continue

        screenshot_file = entry.get("screenshot_file")
        if not screenshot_file:
            continue
        path = result_dir / screenshot_file
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen_screenshots:
            continue
        seen_screenshots.add(resolved)
        evidence.append({"kind": "screenshot", "step": step, "path": path})

    return evidence


def _judge_chat(messages: list[dict[str, Any]]) -> str:
    api_key = os.environ.get("WEBARENA_JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError("WEBARENA_JUDGE_API_KEY is required for WebVoyager evaluation")

    model = os.environ.get("WEBARENA_JUDGE_MODEL", "gpt-4-1106-preview")
    base_url = os.environ.get("WEBARENA_JUDGE_BASE_URL", "https://inference-api.nvidia.com/v1").rstrip("/")
    timeout = float(os.environ.get("WEBARENA_JUDGE_TIMEOUT", "120"))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
        "top_p": 1.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    max_retries = 3
    retry_sleep = 5
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            try:
                resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                break
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                if not _is_retriable_judge_error(exc) or attempt == max_retries:
                    raise
                time.sleep(retry_sleep)
    content = data["choices"][0]["message"].get("content")
    if not isinstance(content, str):
        finish_reason = data["choices"][0].get("finish_reason")
        usage = data.get("usage")
        raise RuntimeError(
            "WebVoyager judge returned no text content "
            f"(content={content!r}, finish_reason={finish_reason!r}, usage={usage})"
        )
    return content


def _is_retriable_judge_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or 500 <= status_code < 600
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _parse_verdict(response: str) -> dict[str, str]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"WebVoyager judge did not return JSON: {response}")
        parsed = json.loads(match.group(0))

    verdict = str(parsed.get("verdict", "")).upper()
    if verdict not in {"SUCCESS", "FAILURE"}:
        raise RuntimeError(f"Unexpected WebVoyager judge verdict: {parsed}")
    return {
        "thought": str(parsed.get("thought", "")),
        "verdict": verdict,
    }


def _task_instruction(task_config: dict) -> str:
    start_urls = task_config.get("start_urls") or []
    if start_urls:
        return f"On {start_urls[0]}: {task_config['intent']}"
    return task_config["intent"]


def _build_messages(task: str, answer: Any, screenshots: list[Path]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Task: {task}\n\n"
                f"Result Response: {answer if answer is not None else 'No answer'}\n\n"
                f"{len(screenshots)} screenshots recorded during the task:"
            ),
        }
    ]
    for idx, screenshot in enumerate(screenshots, start=1):
        user_content.append({"type": "text", "text": f"Step {idx}"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_encode_image(screenshot)}"},
        })
    return [
        {"role": "system", "content": WEBVOYAGER_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_extended_messages(task: str, answer: Any, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Task: {task}\n\n"
                f"Result Response: {answer if answer is not None else 'No answer'}\n\n"
                f"Evidence timeline contains {len(evidence)} item(s):"
            ),
        }
    ]
    for item in evidence:
        step = item.get("step", "?")
        if item["kind"] == "text":
            user_content.append({
                "type": "text",
                "text": f"Step {step} text-tool evidence:\n\n{item['text']}",
            })
        elif item["kind"] == "screenshot":
            user_content.append({"type": "text", "text": f"Step {step} screenshot:"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image(item['path'])}"},
            })
    return [
        {"role": "system", "content": WEBVOYAGER_EXTEND_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def evaluate_webvoyager_task(
    task_config: dict,
    agent_result: dict,
    result_dir: Path,
    judge_log_path: Path | None = None,
    screenshots: list[Path] | None = None,
) -> tuple[float, str]:
    screenshots = screenshots if screenshots is not None else collect_webvoyager_screenshots(result_dir)
    if not screenshots:
        return 0.0, "webvoyager_judge: score=0.0 error=no screenshots found"

    task = _task_instruction(task_config)
    answer = agent_result.get("answer")
    messages = _build_messages(task, answer, screenshots)
    response = _judge_chat(messages)
    verdict = _parse_verdict(response)

    score = 1.0 if verdict["verdict"] == "SUCCESS" else 0.0
    eval_msg = f"webvoyager_judge: score={score} verdict={verdict['verdict']}"

    log_path = judge_log_path or (result_dir / "webvoyager_judge_response.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_payload = {
        "judge_type": "webvoyager",
        "question": task,
        "prediction": answer,
        "screenshots": [str(path) for path in screenshots],
        "response": response,
        "parsed": verdict,
        "score": score,
    }
    if log_path.suffix == ".jsonl":
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload, ensure_ascii=True) + "\n")
    else:
        log_path.write_text(json.dumps(log_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    return score, eval_msg


def evaluate_webvoyager_extend_task(
    task_config: dict,
    agent_result: dict,
    result_dir: Path,
    judge_log_path: Path | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> tuple[float, str]:
    evidence = evidence if evidence is not None else collect_webvoyager_extended_evidence(result_dir)
    if not evidence:
        return 0.0, "webvoyager_extend_judge: score=0.0 error=no evidence found"

    task = _task_instruction(task_config)
    answer = agent_result.get("answer")
    messages = _build_extended_messages(task, answer, evidence)
    response = _judge_chat(messages)
    verdict = _parse_verdict(response)

    score = 1.0 if verdict["verdict"] == "SUCCESS" else 0.0
    eval_msg = f"webvoyager_extend_judge: score={score} verdict={verdict['verdict']}"

    log_path = judge_log_path or (result_dir / "webvoyager_extend_judge_response.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_payload = {
        "judge_type": "webvoyager-extend",
        "question": task,
        "prediction": answer,
        "evidence": [
            {
                **{key: value for key, value in item.items() if key != "path"},
                **({"path": str(item["path"])} if item.get("path") else {}),
            }
            for item in evidence
        ],
        "response": response,
        "parsed": verdict,
        "score": score,
    }
    if log_path.suffix == ".jsonl":
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_payload, ensure_ascii=True) + "\n")
    else:
        log_path.write_text(json.dumps(log_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    return score, eval_msg
