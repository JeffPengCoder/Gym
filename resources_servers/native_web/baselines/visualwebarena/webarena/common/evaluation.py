"""WebArena-Verified evaluation helpers."""

import json
import logging
import os
from pathlib import Path

from .config import DEFAULT_CREDENTIALS, detect_task_type

logger = logging.getLogger(__name__)


def import_eval_deps():
    """Import webarena_verified evaluation dependencies (heavy, do lazily)."""
    from webarena_verified.api import WebArenaVerifiedDataReader, WebArenaVerifiedEvaluator
    from webarena_verified.types import FinalAgentResponse, WebArenaVerifiedTask
    from webarena_verified.types.config import (
        EnvironmentConfig,
        WebArenaSite,
        WebArenaVerifiedConfig,
    )
    from webarena_verified.types.eval import TaskEvalContext
    from webarena_verified.types.tracing import NetworkTrace
    return {
        "WebArenaVerifiedDataReader": WebArenaVerifiedDataReader,
        "WebArenaVerifiedEvaluator": WebArenaVerifiedEvaluator,
        "FinalAgentResponse": FinalAgentResponse,
        "WebArenaVerifiedTask": WebArenaVerifiedTask,
        "EnvironmentConfig": EnvironmentConfig,
        "WebArenaSite": WebArenaSite,
        "WebArenaVerifiedConfig": WebArenaVerifiedConfig,
        "TaskEvalContext": TaskEvalContext,
        "NetworkTrace": NetworkTrace,
    }


def make_evaluator(urls: dict[str, str], deps: dict | None = None):
    """Create a WebArenaVerifiedEvaluator instance."""
    if deps is None:
        deps = import_eval_deps()

    config = deps["WebArenaVerifiedConfig"](
        environments={
            **{
                site: deps["EnvironmentConfig"](
                    urls=[url],
                    credentials=DEFAULT_CREDENTIALS.get(site),
                )
                for site, url in urls.items()
            },
            deps["WebArenaSite"].HOMEPAGE: deps["EnvironmentConfig"](
                urls=[os.environ.get("WA_HOMEPAGE", "http://homepage-not-used")],
            ),
        },
    )
    reader = deps["WebArenaVerifiedDataReader"](config)
    return deps["WebArenaVerifiedEvaluator"](config=config, reader=reader)


def build_agent_response(task_config: dict, agent_result: dict, deps: dict | None = None) -> str:
    """Build a FinalAgentResponse JSON string from the agent result."""
    if deps is None:
        deps = import_eval_deps()

    task_type = detect_task_type(task_config).upper()
    agent_status = agent_result["status"]
    status = "SUCCESS" if agent_status == "done" else "UNKNOWN_ERROR"

    retrieved_data = None
    if task_type == "RETRIEVE" and status == "SUCCESS" and agent_result.get("answer"):
        raw_answer = agent_result["answer"]
        try:
            parsed = json.loads(raw_answer)
            if isinstance(parsed, list):
                retrieved_data = parsed
            else:
                retrieved_data = [parsed]
        except (json.JSONDecodeError, TypeError):
            retrieved_data = [raw_answer]

    response = deps["FinalAgentResponse"](
        task_type=task_type,
        status=status,
        retrieved_data=retrieved_data,
    )
    return response.model_dump_json()


def evaluate(
    evaluator,
    task_config: dict,
    answer_json: str,
    har_path: Path,
    deps: dict | None = None,
) -> float:
    """Run evaluation and print per-evaluator results. Returns score."""
    if deps is None:
        deps = import_eval_deps()

    task = deps["WebArenaVerifiedTask"].model_validate(task_config)
    context = deps["TaskEvalContext"](
        task=task,
        agent_response_raw=answer_json,
        network_trace=deps["NetworkTrace"].from_content(har_path),
        config=evaluator.config,
    )

    logger.info(f"Evaluating task {task.task_id}")
    results = evaluator.evaluate_task(context=context)

    for result in results.evaluators_results:
        status_str = f"score={result.score}"
        if result.error_msg:
            status_str += f", error={result.error_msg}"
        print(f"  {result.evaluator_name}: {status_str}")
        if result.score < 1.0 and result.assertions:
            for assertion in result.assertions:
                print(f"    -> {assertion.assertion_name}: {assertion.assertion_msgs}")

    return results.score


def evaluate_task(evaluator, task_config, answer_json, har_path, deps):
    """Run evaluation, return (score, message_string). For parallel runner."""
    try:
        task = deps["WebArenaVerifiedTask"].model_validate(task_config)
        context = deps["TaskEvalContext"](
            task=task,
            agent_response_raw=answer_json,
            network_trace=deps["NetworkTrace"].from_content(har_path),
            config=evaluator.config,
        )
        results = evaluator.evaluate_task(context=context)
        messages = []
        for r in results.evaluators_results:
            msg = f"{r.evaluator_name}: score={r.score}"
            if r.error_msg:
                msg += f" error={r.error_msg}"
            messages.append(msg)
        return results.score, "; ".join(messages)
    except Exception as e:
        return 0.0, f"Evaluation error: {e}"
