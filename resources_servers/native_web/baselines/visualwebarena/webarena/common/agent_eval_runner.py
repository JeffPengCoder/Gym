#!/usr/bin/env python3
"""Shared WebArena evaluation runners for desktop-control agents."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import importlib
import json
import logging
import multiprocessing
import queue
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.browser import CHROME_ARGS, setup_browser_and_login, setup_task_context
from common.cloudflare_handler import is_retryable_navigation_error, resolve_challenge
from common.config import (
    augment_intent_with_schema,
    detect_task_type,
    filter_tasks,
    find_completed_tasks,
    get_urls,
    is_classic_webarena_task,
    is_visualwebarena_task,
    is_webvoyager_benchmark_path,
    is_webvoyager_task,
    load_tasks,
    split_tasks,
)
from common.evaluation import (
    build_agent_response,
    evaluate,
    evaluate_task,
    import_eval_deps,
    make_evaluator,
)
from common.eval_collision import build_collision_plan, has_collision_mitigation
from common.eval_snapshots import (
    build_snapshot_context,
    collect_browser_snapshots,
    collect_snapshots,
    merge_snapshots,
)
from common.pyautogui_utils import init_pyautogui
from common.xvfb import start_xvfb, stop_xvfb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[97m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"


@dataclass(frozen=True)
class AgentEvalSpec:
    """Provider-specific configuration for a shared eval runner."""

    agent_name: str
    agent_module: str
    agent_class: str
    logger_prefix: str
    default_result_dir: str
    default_model: str | None = None
    single_logger_name: str | None = None
    model_required: bool = False
    supports_thinking: bool = False
    supports_max_image_history: bool = False
    supports_max_model_len: bool = False
    supports_tokenizer_model: bool = False
    supports_single_temperature: bool = False
    supports_parallel_temperature: bool = False
    supports_reasoning_effort: bool = False
    supports_expanded_browser_tools: bool = False
    default_expanded_browser_tools: bool = False
    default_thinking: bool = True
    default_max_image_history: int = 3
    default_max_model_len: int = 131072
    default_temperature: float = 1.0
    default_reasoning_effort: str = "medium"
    default_viewport_width: int = 1920
    default_viewport_height: int = 1080
    single_timeout_default: int | None = 2000
    parallel_timeout_default: int | None = None
    production_style_title: bool = False


def _get(source: argparse.Namespace | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _load_agent_class(spec: AgentEvalSpec):
    module = importlib.import_module(spec.agent_module)
    return getattr(module, spec.agent_class)


def _agent_accepts_kwarg(agent_cls, kwarg: str) -> bool:
    try:
        signature = inspect.signature(agent_cls.__init__)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return kwarg in signature.parameters


def _callable_accepts_kwarg(func, kwarg: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return kwarg in signature.parameters


def _resolve_task_input_images(
    task_config: dict[str, Any],
    benchmark_path: str | None,
    logger: logging.Logger,
) -> list[Path]:
    image_value = task_config.get("image")
    if not image_value:
        return []
    raw_images = image_value if isinstance(image_value, list) else [image_value]

    repo_root = Path(__file__).resolve().parents[2]
    benchmark = Path(benchmark_path) if benchmark_path else None
    if benchmark is not None and not benchmark.is_absolute():
        benchmark = repo_root / benchmark

    candidate_roots = [repo_root]
    if benchmark is not None:
        candidate_roots.extend([benchmark.parent, benchmark.parent.parent])
    webarena_root = Path(__file__).resolve().parent.parent
    candidate_roots.extend([webarena_root, webarena_root / "benchmarks"])

    resolved: list[Path] = []
    for raw_image in raw_images:
        path = Path(str(raw_image))
        candidates = [path] if path.is_absolute() else [root / path for root in candidate_roots]
        match = next((candidate for candidate in candidates if candidate.exists()), None)
        if match is None:
            logger.warning(
                "VisualWebArena task image not found for task_%s: %s",
                task_config.get("task_id"),
                raw_image,
            )
            continue
        resolved.append(match)
    return resolved


def _build_agent_run_kwargs(agent, task_input_images: list[Path]) -> dict[str, Any]:
    if not task_input_images or not _callable_accepts_kwarg(agent.run, "task_input_images"):
        return {}
    return {"task_input_images": task_input_images}


def _build_agent_kwargs(
    spec: AgentEvalSpec,
    args_source: argparse.Namespace | dict[str, Any],
    logger_name: str | None = None,
    agent_cls=None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": _get(args_source, "model"),
        "max_steps": _get(args_source, "max_steps"),
        "screen_width": _get(args_source, "viewport_width"),
        "screen_height": _get(args_source, "viewport_height"),
    }
    if spec.supports_max_image_history and _get(args_source, "max_image_history") is not None:
        kwargs["max_image_history"] = _get(args_source, "max_image_history")
    if spec.supports_max_model_len and _get(args_source, "max_model_len") is not None:
        kwargs["max_model_len"] = _get(args_source, "max_model_len")
    if spec.supports_tokenizer_model and _get(args_source, "tokenizer_model") is not None:
        kwargs["tokenizer_model"] = _get(args_source, "tokenizer_model")
    if _get(args_source, "temperature") is not None:
        kwargs["temperature"] = _get(args_source, "temperature")
    if spec.supports_thinking and _get(args_source, "thinking") is not None:
        kwargs["thinking"] = _get(args_source, "thinking")
    if spec.supports_reasoning_effort and _get(args_source, "reasoning_effort") is not None:
        kwargs["reasoning_effort"] = _get(args_source, "reasoning_effort")
    if spec.supports_expanded_browser_tools and _get(args_source, "expanded_browser_tool") is not None:
        kwargs["expanded_browser_tools"] = _get(args_source, "expanded_browser_tool")
    if logger_name:
        kwargs["logger_name"] = logger_name
    agent_cls = agent_cls or _load_agent_class(spec)
    if _agent_accepts_kwarg(agent_cls, "captcha_solver"):
        kwargs["captcha_solver"] = resolve_challenge
    return kwargs


def _judge_modes(args_source: argparse.Namespace | dict[str, Any], task_config: dict) -> tuple[bool, bool, bool]:
    judge = _get(args_source, "judge", "auto")
    is_classic = judge == "classic" or (
        judge == "auto" and is_classic_webarena_task(task_config)
    )
    is_visualwebarena = judge == "visualwebarena" or (
        judge == "auto" and is_visualwebarena_task(task_config)
    )
    is_webvoyager = judge in {"webvoyager", "webvoyager-extend"} or (
        judge == "auto" and is_webvoyager_task(task_config)
    )
    return is_classic, is_visualwebarena, is_webvoyager


def _parallel_judge_modes(
    args_source: argparse.Namespace | dict[str, Any],
    task_config: dict,
) -> tuple[bool, bool, bool]:
    """Match the historical parallel runner's benchmark-format based dispatch."""
    judge = _get(args_source, "judge", "auto")
    if judge == "skip":
        return False, False, False
    is_classic = is_classic_webarena_task(task_config)
    is_visualwebarena = is_visualwebarena_task(task_config)
    is_webvoyager = judge in {"webvoyager", "webvoyager-extend"} or (
        judge == "auto" and is_webvoyager_task(task_config)
    )
    return is_classic, is_visualwebarena, is_webvoyager


def _skip_judge(args_source: argparse.Namespace | dict[str, Any]) -> bool:
    return _get(args_source, "judge", "auto") == "skip"


def _parallel_imports_verified_deps(args_source: argparse.Namespace | dict[str, Any]) -> bool:
    if _skip_judge(args_source):
        return False
    benchmark_format = _get(args_source, "benchmark_format")
    return benchmark_format not in ("classic_webarena", "visualwebarena", "webvoyager")


def _urls_required(args_source: argparse.Namespace | dict[str, Any]) -> bool:
    judge = _get(args_source, "judge", "auto")
    benchmark_path = _get(args_source, "benchmark_path")
    return not (
        judge in {"webvoyager", "webvoyager-extend"}
        or (judge == "skip" and is_webvoyager_benchmark_path(benchmark_path))
        or (judge == "auto" and is_webvoyager_benchmark_path(benchmark_path))
    )


def _uses_verified_judge(
    args_source: argparse.Namespace | dict[str, Any],
    task_config: dict | None = None,
    benchmark_format: str | None = None,
) -> bool:
    judge = _get(args_source, "judge", "auto")
    if judge == "verified":
        return True
    if judge != "auto":
        return False
    if task_config is not None:
        return (
            not is_classic_webarena_task(task_config)
            and not is_visualwebarena_task(task_config)
            and not is_webvoyager_task(task_config)
        )
    return benchmark_format not in ("classic_webarena", "visualwebarena", "webvoyager")


def _make_result(
    task_config: dict,
    task_type: str,
    score: float,
    eval_msg: str,
    agent_result: dict,
) -> dict[str, Any]:
    return {
        "task_id": task_config["task_id"],
        "task_type": task_type,
        "sites": task_config["sites"],
        "instruction": task_config["intent"],
        "passed": score >= 1.0,
        "score": score,
        "agent_status": agent_result["status"],
        "agent_answer": agent_result.get("answer"),
        "steps": agent_result["steps"],
        "errors": agent_result.get("errors", []),
        "benchmark_format": task_config.get("benchmark_format", "webarena_verified"),
        "eval_message": eval_msg,
    }


def _make_crash_result(task_config: dict, task_type: str, exc: Exception) -> dict[str, Any]:
    return {
        "task_id": task_config["task_id"],
        "task_type": task_type,
        "sites": task_config.get("sites", []),
        "instruction": task_config.get("intent", ""),
        "passed": False,
        "score": 0.0,
        "eval_message": f"Crashed: {exc}",
        "agent_status": "error",
        "agent_answer": None,
        "steps": -1,
        "errors": [str(exc)],
        "benchmark_format": task_config.get("benchmark_format", "webarena_verified"),
    }


def _make_skip_result(task_config: dict, task_type: str, agent_result: dict) -> dict[str, Any]:
    return {
        "task_id": task_config["task_id"],
        "task_type": task_type,
        "sites": task_config["sites"],
        "instruction": task_config["intent"],
        "passed": False,
        "score": 0.0,
        "agent_status": agent_result["status"],
        "agent_answer": agent_result.get("answer"),
        "steps": agent_result["steps"],
        "errors": agent_result.get("errors", []),
        "benchmark_format": task_config.get("benchmark_format", "webarena_verified"),
        "eval_message": "Judge skipped",
        "judge_skipped": True,
    }


def _write_result_files(task_dir: Path, result: dict[str, Any]) -> None:
    with open(task_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    with open(task_dir / "result.txt", "w", encoding="utf-8") as handle:
        handle.write(f"{result['score']}\n")


def _write_skip_result_files(task_dir: Path, result: dict[str, Any]) -> None:
    with open(task_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    with open(task_dir / "result.txt", "w", encoding="utf-8"):
        pass


def _write_agent_result(task_dir: Path, agent_result: dict) -> None:
    with open(task_dir / "agent_result.json", "w", encoding="utf-8") as handle:
        json.dump(agent_result, handle, indent=2, default=str)


def _drain_result_queue(result_queue: multiprocessing.Queue, results: list[dict[str, Any]]) -> None:
    """Drain all currently available worker results without relying on Queue.empty()."""
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break
        except Exception:
            break


def _merge_result_files(run_dir: Path, results: list[dict[str, Any]], task_ids: set[int]) -> None:
    """Recover completed task results if multiprocessing queue shutdown dropped an item."""
    result_ids = {result["task_id"] for result in results if "task_id" in result}
    for task_id in sorted(task_ids - result_ids):
        result_path = run_dir / f"task_{task_id}" / "result.json"
        if not result_path.exists():
            continue
        try:
            with open(result_path, encoding="utf-8") as handle:
                results.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue


def _close_context(loop, context, logger_obj, timeout: int) -> None:
    try:
        loop.run_until_complete(asyncio.wait_for(context.close(), timeout=timeout))
        logger_obj.info("Context cleanup OK")
    except asyncio.TimeoutError:
        logger_obj.warning("Context close timed out after %ss", timeout)
    except Exception as exc:
        logger_obj.warning("Context close failed: %s", exc)


def _shutdown_playwright(loop, context, browser, pw, logger_obj) -> None:
    for label, obj in [
        ("context.close", context),
        ("browser.close", browser),
        ("pw.stop", pw),
    ]:
        if obj is None:
            continue
        try:
            coro = obj.close() if label != "pw.stop" else obj.stop()
            loop.run_until_complete(coro)
        except Exception as exc:
            logger_obj.warning("Playwright cleanup error in %s: %s", label, exc)
    try:
        loop.close()
    except Exception:
        pass


def _is_transient_setup_error(exc: Exception) -> bool:
    """Transient navigation failures during setup should be retried by a later resume."""
    message = str(exc)
    is_timeout = (
        "Page.goto:" in message
        and "Timeout" in exc.__class__.__name__
        and "Timeout" in message
    )
    return is_timeout or is_retryable_navigation_error(exc)


def _is_transient_judge_error(exc: Exception) -> bool:
    try:
        import httpx
    except ModuleNotFoundError:
        return False

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or 500 <= status_code < 600
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def run_single_task(
    spec: AgentEvalSpec,
    task_config: dict,
    urls: dict[str, str],
    evaluator,
    args: argparse.Namespace,
) -> float:
    """Run one task end-to-end: Xvfb -> browser -> agent -> evaluate."""
    logger = logging.getLogger(__name__)
    task_id = task_config["task_id"]
    task_type = detect_task_type(task_config).upper()
    instruction = augment_intent_with_schema(task_config)
    is_classic, is_visualwebarena, is_webvoyager = _judge_modes(args, task_config)
    skip_judge = _skip_judge(args)

    print(f"\n{'=' * 60}")
    print(f"Task ID:    {task_id}")
    print(f"Sites:      {', '.join(task_config['sites'])}")
    print(f"Task type:  {task_type}")
    print(f"{'=' * 60}")
    print(f"\n  {instruction}\n")
    print(f"{'=' * 60}")

    result_dir = Path(args.result_dir) / f"task_{task_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "instruction.txt").write_text(instruction, encoding="utf-8")

    task_handler = logging.FileHandler(result_dir / "worker.log", encoding="utf-8")
    task_handler.setLevel(logging.DEBUG)
    task_handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s %(levelname)s %(name)s/%(lineno)d] %(message)s"
    ))
    agent_logger = logging.getLogger(spec.logger_prefix)
    agent_logger.addHandler(task_handler)

    har_path = None if (is_classic or is_visualwebarena or is_webvoyager or skip_judge) else result_dir / "network.har"
    xvfb_proc = None
    display = None
    loop = None
    pw = None
    browser = None
    context = None

    try:
        xvfb_proc, display = start_xvfb(
            width=args.viewport_width,
            height=args.viewport_height,
        )
        init_pyautogui()

        from playwright.async_api import async_playwright

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pw = loop.run_until_complete(async_playwright().start())
        browser = loop.run_until_complete(pw.chromium.launch(
            headless=False,
            args=CHROME_ARGS,
        ))
        context, page = loop.run_until_complete(
            setup_browser_and_login(
                task_config,
                urls,
                pw,
                browser,
                args.viewport_width,
                args.viewport_height,
                har_path,
            )
        )

        time.sleep(2)

        collision_plan = build_collision_plan(task_config)
        snapshot_before: dict[str, Any] = {}
        snapshot_context: dict[str, Any] | None = None
        if collision_plan.get("snapshot_adapters"):
            snapshot_before = merge_snapshots(
                collect_snapshots(collision_plan),
                loop.run_until_complete(collect_browser_snapshots(page, collision_plan)),
            )

        print(f"Running {spec.agent_name} agent...")
        agent_cls = _load_agent_class(spec)
        agent = agent_cls(**_build_agent_kwargs(
            spec,
            args,
            logger_name=spec.single_logger_name,
            agent_cls=agent_cls,
        ))
        task_input_images = _resolve_task_input_images(task_config, args.benchmark_path, logger)
        agent_run_kwargs = _build_agent_run_kwargs(agent, task_input_images)

        try:
            agent_result = agent.run(
                instruction,
                task_dir=result_dir,
                page=page,
                loop=loop,
                **agent_run_kwargs,
            )
        except Exception as exc:
            logger.error("Agent crashed: %s", exc, exc_info=True)
            agent_result = {
                "status": "error",
                "answer": None,
                "steps": -1,
                "errors": [str(exc)],
            }

        print(
            f"Agent finished: status={agent_result['status']}, "
            f"steps={agent_result['steps']}, answer={agent_result.get('answer')}"
        )

        answer_json = None
        if not is_classic and not is_visualwebarena and not is_webvoyager and not skip_judge:
            answer_json = build_agent_response(task_config, agent_result)
            print(f"Agent response: {answer_json}")

        _write_agent_result(result_dir, agent_result)

        if collision_plan.get("snapshot_adapters"):
            snapshot_after = merge_snapshots(
                collect_snapshots(collision_plan),
                loop.run_until_complete(collect_browser_snapshots(page, collision_plan)),
            )
            snapshot_context = build_snapshot_context(
                collision_plan,
                snapshot_before,
                snapshot_after,
            )

        score = 0.0
        eval_msg = ""
        agent_errored = agent_result["status"] == "error"

        if is_classic and not agent_errored:
            from common.classic_evaluation import evaluate_classic_task

            print("\nRunning classic WebArena evaluation...")
            score, eval_msg = loop.run_until_complete(
                evaluate_classic_task(
                    task_config,
                    agent_result,
                    page,
                    judge_log_path=result_dir / "judge_responses.jsonl",
                    eval_context=snapshot_context,
                )
            )
        elif is_visualwebarena and not agent_errored:
            from common.visualwebarena_evaluation import evaluate_visualwebarena_task

            print("\nRunning VisualWebArena evaluation...")
            score, eval_msg = loop.run_until_complete(
                evaluate_visualwebarena_task(
                    task_config,
                    agent_result,
                    page,
                    judge_log_path=result_dir / "judge_responses.jsonl",
                    eval_context=snapshot_context,
                )
            )

        _shutdown_playwright(loop, context, browser, pw, logger)
        loop = None
        context = None
        browser = None
        pw = None

        if agent_errored:
            print("\nAgent errored out - skipping evaluation.")
            print(f"\n{'=' * 60}")
            print("  STATUS: ERROR")
            print(f"{'=' * 60}")
            return 0.0

        if skip_judge:
            result = _make_skip_result(task_config, task_type, agent_result)
            _write_skip_result_files(result_dir, result)
            print("\nJudge skipped; wrote empty result.txt completion marker.")
            print(f"\n{'=' * 60}")
            print("  STATUS: JUDGE SKIPPED")
            print(f"{'=' * 60}")
            return 0.0

        if is_webvoyager:
            from common.webvoyager_evaluation import (
                evaluate_webvoyager_extend_task,
                evaluate_webvoyager_task,
            )

            print("\nRunning WebVoyager LLM judge...")
            try:
                if args.judge == "webvoyager-extend":
                    score, eval_msg = evaluate_webvoyager_extend_task(
                        task_config,
                        agent_result,
                        result_dir,
                        judge_log_path=result_dir / "webvoyager_extend_judge_response.json",
                    )
                else:
                    score, eval_msg = evaluate_webvoyager_task(
                        task_config,
                        agent_result,
                        result_dir,
                        judge_log_path=result_dir / "webvoyager_judge_response.json",
                    )
            except Exception as exc:
                logger.error("WebVoyager evaluation failed: %s", exc, exc_info=True)
                score = 0.0
                eval_msg = f"WebVoyager evaluation failed: {exc}"
        elif not is_classic and not is_visualwebarena:
            print("\nRunning evaluation...")
            try:
                score = evaluate(evaluator, task_config, answer_json, har_path)
            except Exception as exc:
                logger.error("Evaluation failed: %s", exc, exc_info=True)
                score = 0.0

        result = _make_result(task_config, task_type, score, eval_msg, agent_result)
        _write_result_files(result_dir, result)

        print(f"\n{'=' * 60}")
        print(f"  SCORE: {score}")
        print(f"{'=' * 60}")
        return score
    finally:
        if loop is not None:
            _shutdown_playwright(loop, context, browser, pw, logger)
        agent_logger.removeHandler(task_handler)
        task_handler.close()
        stop_xvfb(xvfb_proc, display)


def worker_process(
    spec: AgentEvalSpec,
    worker_id: int,
    task_queue: multiprocessing.Queue,
    urls: dict[str, str],
    run_dir_str: str,
    args_dict: dict,
    result_queue: multiprocessing.Queue,
    init_lock: multiprocessing.Lock,
    worker_done_flags: Any | None = None,
) -> None:
    """Worker process: owns its Xvfb, Chrome, and task loop."""
    run_dir = Path(run_dir_str)
    tag = f"[W{worker_id:02d}]"
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    wlog_path = run_dir / f"worker_{worker_id:02d}.log"
    wlog_handler = logging.FileHandler(wlog_path, encoding="utf-8")
    wlog_handler.setLevel(logging.DEBUG)
    wlog_handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s %(levelname)s W%(name)s/%(lineno)d] %(message)s"
    ))
    wlogger = logging.getLogger(f"{spec.logger_prefix}.worker.{worker_id}")
    wlogger.addHandler(wlog_handler)
    wlogger.setLevel(logging.DEBUG)

    xvfb_proc = None
    display = None
    loop = None
    pw = None
    browser = None

    try:
        with init_lock:
            wlogger.info("Worker starting (acquired init lock)")
            xvfb_proc, display = start_xvfb(
                width=args_dict["viewport_width"],
                height=args_dict["viewport_height"],
            )
            wlogger.info("Xvfb started on %s", display)

            deps = None
            if _parallel_imports_verified_deps(args_dict):
                deps = import_eval_deps()

            from playwright.async_api import async_playwright

            agent_cls = _load_agent_class(spec)
            init_pyautogui()
            wlogger.info("pyautogui initialized")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            pw = loop.run_until_complete(async_playwright().start())
            browser = loop.run_until_complete(pw.chromium.launch(
                headless=False,
                args=CHROME_ARGS,
            ))
            wlogger.info("Playwright + browser started, releasing init lock")

        evaluator = make_evaluator(urls, deps) if deps is not None else None
        wlogger.info("Evaluator ready")
        print(f"  {tag} ready on display {display}", flush=True)

        startup_stagger_s = worker_id * 30
        wlogger.info("Startup stagger: sleeping %ss before first task setup", startup_stagger_s)
        time.sleep(startup_stagger_s)

        tasks_completed = 0
        while True:
            try:
                task_config = task_queue.get_nowait()
            except queue.Empty:
                break

            task_id = task_config["task_id"]
            task_type = "UNKNOWN"
            task_dir = None
            task_logger = None
            task_handler = None
            context = None
            browser_setup_complete = False
            webvoyager_judge_started = False
            snapshot_before: dict[str, Any] = {}
            snapshot_context: dict[str, Any] | None = None

            try:
                task_type = detect_task_type(task_config).upper()
                is_classic, is_visualwebarena, is_webvoyager = _parallel_judge_modes(args_dict, task_config)
                skip_judge = _skip_judge(args_dict)
                task_dir = run_dir / f"task_{task_id}"
                task_dir.mkdir(parents=True, exist_ok=True)

                instruction = augment_intent_with_schema(task_config)
                (task_dir / "instruction.txt").write_text(instruction, encoding="utf-8")

                har_path = None if (is_classic or is_visualwebarena or is_webvoyager or skip_judge) else task_dir / "network.har"

                task_logger_name = f"{spec.logger_prefix}.worker{worker_id}.task{task_id}"
                task_logger = logging.getLogger(task_logger_name)
                task_handler = logging.FileHandler(task_dir / "worker.log", encoding="utf-8")
                task_handler.setLevel(logging.DEBUG)
                task_handler.setFormatter(logging.Formatter(
                    fmt="[%(asctime)s %(levelname)s %(name)s/%(lineno)d] %(message)s"
                ))
                task_logger.addHandler(task_handler)
                task_logger.setLevel(logging.DEBUG)

                wlogger.info("Starting task_%s (%s)", task_id, task_type)
                task_logger.info("Starting task_%s (%s)", task_id, task_type)

                context, page = loop.run_until_complete(
                    setup_task_context(
                        browser,
                        task_config,
                        urls,
                        args_dict["viewport_width"],
                        args_dict["viewport_height"],
                        har_path,
                    )
                )
                browser_setup_complete = True
                time.sleep(2)

                collision_plan = task_config.get("_collision_plan") or {}
                if has_collision_mitigation(collision_plan):
                    task_logger.info("Collision plan: %s", collision_plan)
                if collision_plan.get("snapshot_adapters"):
                    snapshot_before = merge_snapshots(
                        collect_snapshots(collision_plan),
                        loop.run_until_complete(collect_browser_snapshots(page, collision_plan)),
                    )
                    task_logger.info(
                        "Collected before snapshots: %s",
                        sorted(snapshot_before.keys()),
                    )

                agent = agent_cls(
                    **_build_agent_kwargs(
                        spec,
                        args_dict,
                        logger_name=task_logger_name,
                        agent_cls=agent_cls,
                    )
                )
                task_input_images = _resolve_task_input_images(
                    task_config,
                    args_dict.get("benchmark_path"),
                    task_logger,
                )
                agent_run_kwargs = _build_agent_run_kwargs(agent, task_input_images)

                start_time = time.time()
                try:
                    agent_result = agent.run(
                        instruction,
                        task_dir=task_dir,
                        page=page,
                        loop=loop,
                        **agent_run_kwargs,
                    )
                except Exception as exc:
                    task_logger.error("Agent crashed: %s", exc, exc_info=True)
                    agent_result = {
                        "status": "error",
                        "answer": None,
                        "steps": -1,
                        "errors": [str(exc)],
                    }
                elapsed = time.time() - start_time

                _write_agent_result(task_dir, agent_result)
                task_logger.info(
                    "Saved agent_result.json (status=%s, elapsed=%.0fs)",
                    agent_result["status"],
                    elapsed,
                )

                if collision_plan.get("snapshot_adapters"):
                    snapshot_after = merge_snapshots(
                        collect_snapshots(collision_plan),
                        loop.run_until_complete(collect_browser_snapshots(page, collision_plan)),
                    )
                    snapshot_context = build_snapshot_context(
                        collision_plan,
                        snapshot_before,
                        snapshot_after,
                    )
                    task_logger.info(
                        "Collected after snapshots and built deltas: %s",
                        sorted(snapshot_context.get("deltas", {}).keys()),
                    )

                if agent_result["status"] == "error":
                    _close_context(loop, context, task_logger, timeout=30)
                    context = None
                    task_logger.warning("Agent errored out, skipping evaluation")
                    print(
                        f"  {tag} task_{task_id:>3d} {task_type:8s} "
                        f"{BG_RED}{WHITE}{BOLD} ERR  {RESET} "
                        f"{DIM}agent error{RESET}",
                        flush=True,
                    )
                    result = {
                        "task_id": task_id,
                        "task_type": task_type,
                        "sites": task_config.get("sites", []),
                        "instruction": task_config.get("intent", ""),
                        "passed": False,
                        "score": 0.0,
                        "agent_status": "error",
                        "agent_answer": None,
                        "steps": agent_result.get("steps", -1),
                        "errors": agent_result.get("errors", []),
                        "benchmark_format": task_config.get("benchmark_format", "webarena_verified"),
                        "eval_message": "Agent errored out; judge skipped",
                    }
                    with open(task_dir / "result.json", "w", encoding="utf-8") as handle:
                        json.dump(result, handle, indent=2, default=str)
                    result_queue.put(result)
                    tasks_completed += 1
                    continue

                if skip_judge:
                    _close_context(loop, context, task_logger, timeout=30)
                    context = None
                    result = _make_skip_result(task_config, task_type, agent_result)
                    _write_skip_result_files(task_dir, result)
                    task_logger.info("Judge skipped; wrote empty result.txt completion marker")
                    print(
                        f"  {tag} task_{task_id:>3d} {task_type:8s} "
                        f"{YELLOW}{BOLD}SKIP {RESET} "
                        f"{DIM}{agent_result['steps']} steps, judge skipped{RESET}",
                        flush=True,
                    )
                    result_queue.put(result)
                    tasks_completed += 1
                    continue

                task_logger.info("Running evaluation")
                if is_classic:
                    from common.classic_evaluation import evaluate_classic_task

                    score, eval_msg = loop.run_until_complete(
                        evaluate_classic_task(
                            task_config,
                            agent_result,
                            page,
                            judge_log_path=task_dir / "judge_responses.jsonl",
                            eval_context=snapshot_context,
                        )
                    )
                elif is_visualwebarena:
                    from common.visualwebarena_evaluation import evaluate_visualwebarena_task

                    score, eval_msg = loop.run_until_complete(
                        evaluate_visualwebarena_task(
                            task_config,
                            agent_result,
                            page,
                            judge_log_path=task_dir / "judge_responses.jsonl",
                            eval_context=snapshot_context,
                        )
                    )
                elif is_webvoyager:
                    from common.webvoyager_evaluation import (
                        evaluate_webvoyager_extend_task,
                        evaluate_webvoyager_task,
                    )

                    _close_context(loop, context, task_logger, timeout=30)
                    context = None
                    webvoyager_judge_started = True
                    if args_dict.get("judge") == "webvoyager-extend":
                        score, eval_msg = evaluate_webvoyager_extend_task(
                            task_config,
                            agent_result,
                            task_dir,
                            judge_log_path=task_dir / "webvoyager_extend_judge_response.json",
                        )
                    else:
                        score, eval_msg = evaluate_webvoyager_task(
                            task_config,
                            agent_result,
                            task_dir,
                            judge_log_path=task_dir / "webvoyager_judge_response.json",
                        )
                    webvoyager_judge_started = False
                else:
                    answer_json = build_agent_response(task_config, agent_result, deps)
                    _close_context(loop, context, task_logger, timeout=30)
                    context = None
                    score, eval_msg = evaluate_task(evaluator, task_config, answer_json, har_path, deps)

                if (is_classic or is_visualwebarena) and context:
                    _close_context(loop, context, task_logger, timeout=30)
                    context = None

                result = _make_result(task_config, task_type, score, eval_msg, agent_result)
                _write_result_files(task_dir, result)
                task_logger.info(
                    "Evaluation done: score=%s, passed=%s",
                    result["score"],
                    result["passed"],
                )

                status_badge = (
                    f"{BG_GREEN}{WHITE}{BOLD} PASS {RESET}"
                    if result["passed"]
                    else f"{BG_RED}{WHITE}{BOLD} FAIL {RESET}"
                )
                print(
                    f"  {tag} task_{task_id:>3d} {task_type:8s} "
                    f"{status_badge} {DIM}{agent_result['steps']} steps{RESET}",
                    flush=True,
                )

                result_queue.put(result)
                tasks_completed += 1

            except Exception as exc:
                err_logger = task_logger or wlogger
                err_logger.error("Task %s crashed: %s", task_id, exc, exc_info=True)
                if task_logger:
                    wlogger.error("Task %s crashed: %s", task_id, exc, exc_info=True)
                if not browser_setup_complete and _is_transient_setup_error(exc):
                    if task_dir:
                        task_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"  {tag} task_{task_id:>3d} {task_type:8s} "
                        f"{BG_RED}{WHITE}{BOLD} ERR  {RESET} "
                        f"{DIM}transient setup error; will retry on resume: {exc}{RESET}",
                        flush=True,
                    )
                    continue
                if webvoyager_judge_started and _is_transient_judge_error(exc):
                    if task_dir:
                        task_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"  {tag} task_{task_id:>3d} {task_type:8s} "
                        f"{BG_RED}{WHITE}{BOLD} ERR  {RESET} "
                        f"{DIM}transient judge error; will retry on resume: {exc}{RESET}",
                        flush=True,
                    )
                    continue
                result = _make_crash_result(task_config, task_type, exc)
                if task_dir:
                    task_dir.mkdir(parents=True, exist_ok=True)
                    _write_result_files(task_dir, result)
                print(
                    f"  {tag} task_{task_id:>3d} {task_type:8s} "
                    f"{BG_RED}{WHITE}{BOLD} ERR  {RESET} {DIM}{exc}{RESET}",
                    flush=True,
                )
                try:
                    result_queue.put(result)
                except Exception:
                    wlogger.error("Failed to put error result into queue")
                tasks_completed += 1

            finally:
                if context:
                    _close_context(loop, context, task_logger or wlogger, timeout=15)
                    context = None
                if task_handler and task_logger:
                    task_logger.removeHandler(task_handler)
                    task_handler.close()

        wlogger.info("Worker finished normally, completed %s tasks", tasks_completed)

    except Exception as exc:
        wlogger.error("Worker failed: %s", exc, exc_info=True)
        print(f"  {tag} {RED}failed: {exc}{RESET}", flush=True)
    finally:
        shutdown_timeout = 30
        if browser and loop:
            try:
                loop.run_until_complete(asyncio.wait_for(browser.close(), timeout=shutdown_timeout))
            except Exception as exc:
                wlogger.warning("Shutdown browser failed: %s", exc)
        if pw and loop:
            try:
                loop.run_until_complete(asyncio.wait_for(pw.stop(), timeout=shutdown_timeout))
            except Exception as exc:
                wlogger.warning("Shutdown playwright failed: %s", exc)
        if loop:
            try:
                loop.close()
            except Exception:
                pass
        stop_xvfb(xvfb_proc, display)
        wlogger.info("Worker shut down")
        wlog_handler.close()
        if worker_done_flags is not None:
            try:
                worker_done_flags[worker_id] = 1
            except Exception:
                pass
        try:
            result_queue.cancel_join_thread()
        except Exception:
            pass
        print(f"  {tag} {DIM}shut down{RESET}", flush=True)


def run_parallel(spec: AgentEvalSpec, args: argparse.Namespace) -> None:
    logger = logging.getLogger(f"{spec.logger_prefix}.parallel")
    urls = get_urls(required=_urls_required(args))
    all_tasks = load_tasks(urls, benchmark_path=args.benchmark_path)

    tasks = filter_tasks(all_tasks, task_ids=args.task_ids, task_type=args.task_type, sites=args.sites)
    if not tasks:
        print(f"{RED}No tasks matched filters.{RESET}")
        sys.exit(1)

    if args.split_idx is not None and args.split_total is not None:
        tasks = split_tasks(tasks, args.split_idx, args.split_total)
        print(f"{CYAN}Split {args.split_idx}/{args.split_total}: {len(tasks)} tasks{RESET}")
        if not tasks:
            print(f"{YELLOW}No tasks in this split.{RESET}")
            return

    run_dir = Path(args.resume) if args.resume else Path(args.result_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    task_ids_in_scope = {task["task_id"] for task in tasks}
    completed = find_completed_tasks(run_dir, task_ids=task_ids_in_scope)
    if completed:
        tasks = [task for task in tasks if task["task_id"] not in completed]
        if not tasks:
            print(f"{GREEN}All tasks already completed in {run_dir}{RESET}")
            return
        print(f"{YELLOW}Resuming: {len(completed)} done, {len(tasks)} remaining{RESET}")

    tasks_with_plans = []
    mitigated_count = 0
    for task in tasks:
        task_with_plan = dict(task)
        collision_plan = build_collision_plan(task_with_plan)
        if has_collision_mitigation(collision_plan):
            mitigated_count += 1
        task_with_plan["_collision_plan"] = collision_plan
        tasks_with_plans.append(task_with_plan)
    tasks = tasks_with_plans

    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    num_workers = min(args.workers, len(tasks))
    title = f"{spec.agent_name} Agent Parallel Evaluation"
    if spec.production_style_title:
        title += " (Production-Style)"

    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"  {DIM}Model:{RESET}     {BOLD}{args.model}{RESET}")
    print(f"  {DIM}Tasks:{RESET}     {BOLD}{len(tasks)}{RESET}")
    print(f"  {DIM}Workers:{RESET}   {BOLD}{num_workers}{RESET}")
    print(f"  {DIM}Max steps:{RESET} {BOLD}{args.max_steps}{RESET}")
    if spec.production_style_title:
        print(f"  {DIM}Timeout:{RESET}   {BOLD}disabled (max_steps only){RESET}")
    if spec.supports_max_image_history and hasattr(args, "max_image_history"):
        print(f"  {DIM}Images:{RESET}    {BOLD}{args.max_image_history}{RESET}")
    if spec.supports_parallel_temperature and hasattr(args, "temperature"):
        print(f"  {DIM}Temp:{RESET}      {BOLD}{args.temperature}{RESET}")
    print(f"  {DIM}Judge:{RESET}     {BOLD}{args.judge}{RESET}")
    print(f"  {DIM}Eval snapshots:{RESET} {BOLD}{mitigated_count} task(s) with snapshot mitigation{RESET}")
    if spec.supports_thinking:
        print(f"  {DIM}Thinking:{RESET}  {BOLD}{'on' if args.thinking else 'off'}{RESET}")
    print(f"  {DIM}Screen:{RESET}    {BOLD}{args.viewport_width}x{args.viewport_height}{RESET}")
    print(f"  {DIM}Displays:{RESET}  {BOLD}auto (starting from :99){RESET}")
    print(f"  {DIM}Output:{RESET}    {run_dir}")
    print(f"{CYAN}{'-' * 60}{RESET}\n")

    random.Random(42).shuffle(tasks)

    task_queue = multiprocessing.Queue()
    for task in tasks:
        task_queue.put(task)

    result_queue = multiprocessing.Queue()
    args_dict = vars(args).copy()
    args_dict["benchmark_format"] = tasks[0].get("benchmark_format", "webarena_verified")

    init_lock = multiprocessing.Lock()
    worker_done_flags = multiprocessing.Array("b", num_workers)
    processes: list[multiprocessing.Process] = []
    for i in range(num_workers):
        process = multiprocessing.Process(
            target=worker_process,
            args=(
                spec,
                i,
                task_queue,
                urls,
                str(run_dir),
                args_dict,
                result_queue,
                init_lock,
                worker_done_flags,
            ),
            daemon=True,
        )
        processes.append(process)

    for process in processes:
        process.start()

    results = []
    try:
        while any(process.is_alive() for process in processes):
            _drain_result_queue(result_queue, results)
            if all(worker_done_flags[i] for i in range(num_workers)):
                logger.warning("All workers reported shutdown; joining lingering processes")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Shutting down workers...{RESET}")
        for process in processes:
            if process.is_alive():
                process.terminate()

    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            logger.warning("Process %s still alive after join timeout, killing", process.pid)
            process.kill()
            process.join(timeout=5)

    _drain_result_queue(result_queue, results)

    for i, process in enumerate(processes):
        code = process.exitcode
        if code is not None and code != 0:
            sig_msg = ""
            if code < 0:
                import signal as _sig

                try:
                    sig_msg = f" ({_sig.Signals(-code).name})"
                except (ValueError, AttributeError):
                    sig_msg = f" (signal {-code})"
            print(f"  {YELLOW}[W{i:02d}] exited abnormally: code={code}{sig_msg}{RESET}")

    remaining = 0
    while not task_queue.empty():
        try:
            task_queue.get_nowait()
            remaining += 1
        except Exception:
            break
    if remaining:
        print(f"  {YELLOW}{remaining} tasks were still in queue (will be picked up on resume){RESET}")

    _merge_result_files(run_dir, results, task_ids_in_scope)

    for queue_obj in (task_queue, result_queue):
        try:
            queue_obj.cancel_join_thread()
        except Exception:
            pass
        try:
            queue_obj.close()
        except Exception:
            pass

    if not results:
        print(f"{RED}No tasks completed.{RESET}")
        return

    results.sort(key=lambda result: result["task_id"])

    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    pct = round(passed / total * 100, 1) if total else 0

    by_type: dict[str, dict] = {}
    for result in results:
        task_type = result.get("task_type", "UNKNOWN")
        by_type.setdefault(task_type, {"total": 0, "passed": 0})
        by_type[task_type]["total"] += 1
        if result.get("passed"):
            by_type[task_type]["passed"] += 1

    by_site: dict[str, dict] = {}
    for result in results:
        for site in result.get("sites", []):
            by_site.setdefault(site, {"total": 0, "passed": 0})
            by_site[site]["total"] += 1
            if result.get("passed"):
                by_site[site]["passed"] += 1

    aggregate = {
        "model": args.model,
        "benchmark_path": args.benchmark_path,
        "benchmark_format": tasks[0].get("benchmark_format", "webarena_verified") if tasks else None,
        "total": total,
        "passed": passed,
        "pass_rate": pct,
        "by_type": by_type,
        "by_site": by_site,
        "tasks": results,
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, default=str)

    pct_color = GREEN if pct >= 30 else RED
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"  {BOLD}Results: {pct_color}{passed}/{total} passed ({pct}%){RESET}")
    print()
    for task_type, info in sorted(by_type.items()):
        p = info["passed"]
        t = info["total"]
        color = GREEN if p > 0 else RED
        print(f"    {task_type:10s} {color}{p}/{t}{RESET}")
    print()
    for site, info in sorted(by_site.items()):
        p = info["passed"]
        t = info["total"]
        color = GREEN if p > 0 else RED
        print(f"    {site:18s} {color}{p}/{t}{RESET}")
    print()
    print(f"  {DIM}Output:{RESET} {run_dir}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")


def add_common_arguments(parser: argparse.ArgumentParser, spec: AgentEvalSpec, *, parallel: bool) -> None:
    model_kwargs: dict[str, Any] = {
        "type": str,
        "required": spec.model_required,
    }
    if spec.default_model is not None:
        model_kwargs["default"] = spec.default_model
    parser.add_argument("--model", **model_kwargs)
    if parallel:
        parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--max_steps", type=int, default=100)
    timeout_default = spec.parallel_timeout_default if parallel else spec.single_timeout_default
    parser.add_argument("--timeout", type=int, default=timeout_default, help="Deprecated no-op")
    if spec.supports_max_image_history:
        parser.add_argument("--max_image_history", type=int, default=spec.default_max_image_history)
    if spec.supports_max_model_len:
        parser.add_argument("--max-model-len", dest="max_model_len", type=int, default=spec.default_max_model_len)
    if spec.supports_tokenizer_model:
        parser.add_argument(
            "--tokenizer-model",
            dest="tokenizer_model",
            type=str,
            default=None,
            help="Tokenizer model/path for request text budgeting. Defaults to --model.",
        )
    if (parallel and spec.supports_parallel_temperature) or (
        not parallel and spec.supports_single_temperature
    ):
        parser.add_argument("--temperature", type=float, default=spec.default_temperature)
    if spec.supports_reasoning_effort:
        parser.add_argument("--reasoning_effort", type=str, default=spec.default_reasoning_effort)
    if spec.supports_expanded_browser_tools:
        parser.add_argument(
            "--expanded-browser-tool",
            dest="expanded_browser_tool",
            action="store_true",
            default=spec.default_expanded_browser_tools,
            help="Enable Navigator n1.5 expanded browser tools.",
        )
    if spec.supports_thinking:
        parser.add_argument("--thinking", action="store_true", default=spec.default_thinking)
        parser.add_argument("--no_thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--viewport_width", type=int, default=spec.default_viewport_width)
    parser.add_argument("--viewport_height", type=int, default=spec.default_viewport_height)
    parser.add_argument("--result_dir", type=str, default=spec.default_result_dir)
    if parallel:
        parser.add_argument("--resume", type=str, default=None, help="Resume a partial run from this directory")
        parser.add_argument(
            "--task_ids",
            type=str,
            default=None,
            help="Task IDs: single (410), range (0-50), or comma-separated (0,5,10)",
        )
        parser.add_argument(
            "--task_type",
            type=str,
            default=None,
            choices=["retrieve", "mutate", "navigate", "webvoyager"],
            help="Filter by task type",
        )
        parser.add_argument("--sites", type=str, default=None, help="Filter by site (comma-separated)")
    else:
        parser.add_argument("--task_id", type=int, required=True, help="Task ID")
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default="webarena/benchmarks/webarena.jsonl",
        help="Optional benchmark path (.jsonl for classic/WebVoyager, .json for verified)",
    )
    parser.add_argument(
        "--judge",
        choices=["auto", "classic", "visualwebarena", "webvoyager", "webvoyager-extend", "verified", "skip"],
        default="auto",
        help="Evaluation backend to use. 'skip' writes an empty result.txt completion marker without judging.",
    )
    if parallel:
        parser.add_argument("--split_idx", type=int, default=None, help="Index of this split (0-based)")
        parser.add_argument("--split_total", type=int, default=None, help="Total number of splits")


def finalize_args(args: argparse.Namespace, *, parallel: bool) -> None:
    if getattr(args, "no_thinking", False):
        args.thinking = False
    if parallel and (args.split_idx is None) != (args.split_total is None):
        print("Error: --split_idx and --split_total must be used together.")
        sys.exit(1)


def main_single(spec: AgentEvalSpec, description: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=description or f"{spec.agent_name} agent evaluation on WebArena",
    )
    add_common_arguments(parser, spec, parallel=False)
    args = parser.parse_args()
    finalize_args(args, parallel=False)

    urls = get_urls(required=_urls_required(args))
    all_tasks = load_tasks(urls, benchmark_path=args.benchmark_path)
    task_config = next((task for task in all_tasks if task["task_id"] == args.task_id), None)
    if task_config is None:
        print(f"Error: task_id {args.task_id} not found")
        sys.exit(1)

    evaluator = make_evaluator(urls) if _uses_verified_judge(args, task_config=task_config) else None
    run_single_task(spec, task_config, urls, evaluator, args)


def main_parallel(spec: AgentEvalSpec, description: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=description or f"Parallel {spec.agent_name} agent evaluation on WebArena/WebVoyager",
    )
    add_common_arguments(parser, spec, parallel=True)
    args = parser.parse_args()
    finalize_args(args, parallel=True)
    run_parallel(spec, args)
