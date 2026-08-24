# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mingjie-recipe-compatible headed Chromium driver for Gym web sessions."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from collections import deque
from typing import Any
from urllib.parse import unquote, urlparse

from nemo_gym.web.actions import parse_native_tool_calls
from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.models import (
    BROWSER_TARGET_CLOSED_STATUS,
    CAPTCHA_BUDGET_EXHAUSTED_STATUS,
    WebAction,
    WebArtifactRef,
    WebBenchmark,
    WebObservation,
    WebStepResult,
    WebTab,
    WebTask,
    WebVerifierResult,
)
from resources_servers.native_web.captcha import (
    BROWSER_PROXY_CONFIG_ATTR,
    CAPTCHA_INTERCEPT_SCRIPT,
    captcha_solver_from_environment,
)
from resources_servers.native_web.config import NativeWebResourcesServerConfig


LOG = logging.getLogger("nemo_gym.resources_servers.native_web")


class BrowserTargetClosedDuringCaptcha(RuntimeError):
    """The browser target disappeared while CAPTCHA handling inspected it."""


def _is_playwright_target_closed_error(exc: BaseException) -> bool:
    """Recognize the pinned Playwright target-closed type without a private import."""

    error_type = type(exc)
    return error_type.__name__ == "TargetClosedError" and error_type.__module__.startswith("playwright.")


CHROME_ARGS = [
    "--window-position=0,0",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-breakpad",
    "--disable-component-extensions-with-background-pages",
    "--disable-dev-shm-usage",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
    "--disable-renderer-backgrounding",
    "--force-color-profile=srgb",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--mute-audio",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-quic",
    "--disable-http2",
]
PROXY_START_URL_HOSTS = frozenset({"www.allrecipes.com", "www.amazon.com", "dictionary.cambridge.org"})
PROXY_START_URL_PREFIXES = ("https://www.google.com/maps/",)
PRINT_INTERCEPT_SCRIPT = """(() => {
    if (window.__webvoyagerPrintHookInstalled) return;
    Object.defineProperty(window, "__webvoyagerPrintHookInstalled", {value: true});
    Object.defineProperty(window, "__webvoyagerPrintCalled", {value: false, writable: true});
    Object.defineProperty(window, "__webvoyagerPrintCalls", {value: [], writable: true});
    window.print = function() {
        window.__webvoyagerPrintCalled = true;
        window.__webvoyagerPrintCalls.push({url: window.location.href, timestamp: Date.now()});
    };
})();"""
SPECIAL_TEXT_KEYS = {"\n": "enter", "\t": "tab"}
SHIFT_TEXT_KEYS = {"<": ","}
# The pinned native runner settles the initial start URL on domcontentloaded but
# waits for `load` on every navigation the policy requests, so a tool-driven page
# transition is screenshotted after its subresources land.
RESET_WAIT_UNTIL = "domcontentloaded"
NAVIGATION_WAIT_UNTIL = "load"
# Transport-level navigation faults the reference runner retries in place. A
# Playwright timeout is deliberately absent: it is a slow page, not a dropped
# connection, and retrying it would multiply the wait.
RETRYABLE_NAVIGATION_ERRORS = (
    "net::ERR_EMPTY_RESPONSE",
    "net::ERR_PROXY_CONNECTION_FAILED",
    "net::ERR_TUNNEL_CONNECTION_FAILED",
    "net::ERR_CONNECTION_CLOSED",
    "net::ERR_CONNECTION_RESET",
    "net::ERR_TIMED_OUT",
)
NAVIGATION_RETRY_DELAYS_S = (4, 4, 4, 8)


def _is_retryable_navigation_error(exc: Exception) -> bool:
    message = str(exc)
    return "Page.goto:" in message and any(marker in message for marker in RETRYABLE_NAVIGATION_ERRORS)


def _url_origin(url: str) -> str:
    """Return a log-safe origin without URL paths, queries, or credentials."""

    parsed = urlparse(url)
    if not parsed.hostname:
        return "unknown"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme or 'unknown'}://{parsed.hostname}{port}"


def _stop_clipboard_owner(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        process.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=0.5)


def _paste_unicode(pyautogui: Any, text: str) -> None:
    xclip = shutil.which("xclip")
    if xclip is None:
        raise RuntimeError("xclip is required for Unicode browser text input")
    process = subprocess.Popen(
        [xclip, "-selection", "clipboard", "-in"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if process.stdin is None:
        raise RuntimeError("xclip stdin pipe was not created")
    try:
        process.stdin.write(text.encode("utf-8"))
        process.stdin.close()
        time.sleep(0.1)
        if process.poll() not in {None, 0}:
            raise RuntimeError(f"xclip exited before paste with code {process.returncode}")
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.5)
    finally:
        _stop_clipboard_owner(process)


def _type_browser_text(pyautogui: Any, text: str) -> None:
    """Type text without losing Unicode or interpreting newlines as glyphs."""

    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        chunk = "".join(buffer)
        if chunk.isascii():
            pyautogui.write(chunk, interval=0.01)
        else:
            _paste_unicode(pyautogui, chunk)
        buffer.clear()

    for character in text:
        if character in SPECIAL_TEXT_KEYS:
            flush()
            pyautogui.press(SPECIAL_TEXT_KEYS[character])
        elif character in SHIFT_TEXT_KEYS:
            flush()
            pyautogui.hotkey("shift", SHIFT_TEXT_KEYS[character])
        else:
            buffer.append(character)
    flush()


class NativeWebDriver:
    """One thread-affine Playwright context with visible PyAutoGUI actions."""

    def __init__(
        self,
        config: NativeWebResourcesServerConfig,
        session_id: str,
        artifacts: WebArtifactStore,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.artifacts = artifacts
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._task: WebTask | None = None
        self._observation: WebObservation | None = None
        self._step = 0
        self._started_at = 0.0
        self._last_action = ""
        self._last_error = ""
        self._last_captcha_failure_step: int | None = None
        self._captcha_failures = 0
        self._captcha_budget_exhausted = False
        self._browser_target_closed = False
        self._evidence: deque[WebArtifactRef] = deque(maxlen=config.max_evidence_screenshots)
        self._captcha_solver = captcha_solver_from_environment()

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]:
        started = time.monotonic()
        LOG.info(
            "event=native_browser_reset_start session=%s benchmark=%s task=%s start_origin=%s "
            "display=%s viewport=%dx%d",
            self.session_id,
            task.benchmark.value,
            task.task_id,
            _url_origin(task.start_urls[0]) if task.start_urls else "none",
            os.environ.get("DISPLAY", "unset"),
            self.config.viewport_width,
            self.config.viewport_height,
        )
        self.close()
        if task.benchmark != WebBenchmark.WEBVOYAGER:
            raise ValueError("native_web currently supports WebVoyager only")
        if not os.environ.get("DISPLAY"):
            raise ValueError("DISPLAY is required; run the native resource server under Xvfb")

        from playwright.sync_api import sync_playwright

        self._configure_pyautogui()

        self._playwright = sync_playwright().start()
        launch: dict[str, Any] = {
            "headless": False,
            "args": CHROME_ARGS,
        }
        if self.config.browser_channel:
            launch["channel"] = self.config.browser_channel
        proxy = self._proxy_for_task(task)
        LOG.info(
            "event=native_browser_launch session=%s task=%s proxy_enabled=%s proxy_origin=%s "
            "captcha_enabled=%s browser_channel=%s",
            self.session_id,
            task.task_id,
            bool(proxy),
            _url_origin(proxy) if proxy else "none",
            bool(self.config.captcha_solver()),
            self.config.browser_channel or "bundled",
        )
        self._browser = self._playwright.chromium.launch(**launch)
        context_kwargs: dict[str, Any] = {
            "viewport": {"width": self.config.viewport_width, "height": self.config.viewport_height}
        }
        if proxy:
            context_kwargs["proxy"] = self._playwright_proxy(proxy)
        self._context = self._browser.new_context(**context_kwargs)
        # One context-wide deadline, as the reference runner sets, instead of a
        # per-navigation override. Every Playwright operation is then bounded.
        self._context.set_default_timeout(self.config.default_timeout_ms)
        self._context.set_default_navigation_timeout(self.config.default_timeout_ms)
        if proxy:
            try:
                setattr(self._context, BROWSER_PROXY_CONFIG_ATTR, dict(context_kwargs["proxy"]))
            except Exception:
                LOG.warning(
                    "event=native_browser_proxy_metadata_failed session=%s task=%s",
                    self.session_id,
                    task.task_id,
                )
        self._context.add_init_script(CAPTCHA_INTERCEPT_SCRIPT)
        self._context.add_init_script(PRINT_INTERCEPT_SCRIPT)
        self._context.on("page", self._configure_page)
        self._page = self._context.new_page()
        self._task = task
        self._step = 0
        self._started_at = time.monotonic()
        self._last_action = ""
        self._last_error = ""
        self._last_captcha_failure_step = None
        self._captcha_failures = 0
        self._captcha_budget_exhausted = False
        self._browser_target_closed = False
        self._evidence.clear()
        if task.start_urls:
            self._goto(self._page, task.start_urls[0], wait_until=RESET_WAIT_UNTIL)
        self._page.bring_to_front()
        time.sleep(self.config.action_delay_seconds)
        self._maybe_solve_captcha("initial")
        self._observation = self._capture()
        LOG.info(
            "event=native_browser_reset_complete session=%s task=%s origin=%s tabs=%d elapsed_seconds=%.3f",
            self.session_id,
            task.task_id,
            _url_origin(self._observation.url),
            len(self._observation.tabs),
            time.monotonic() - started,
        )
        return self._observation, {
            "runtime_profile": "native_visual",
            "driver": "playwright_context_pyautogui_actions",
            "viewport": [self.config.viewport_width, self.config.viewport_height],
            "proxy_enabled": bool(proxy),
            "captcha_enabled": bool(self.config.captcha_solver()),
        }

    def observe(self) -> WebObservation:
        if self._observation is None:
            raise RuntimeError("native browser has not been reset")
        return self._observation

    def step(self, action: WebAction) -> WebStepResult:
        if self._page is None:
            raise RuntimeError("native browser has not been reset")
        self._last_action = action.raw_model_output or action.name
        self._last_error = ""
        execution_ok = True
        started = time.monotonic()
        captcha_failure_step = self._step
        calls = action.arguments.get("calls", [])
        call_names = [str(call.get("name", "unknown")) for call in calls if isinstance(call, dict)]
        LOG.info(
            "event=native_browser_step_start session=%s task=%s step=%d action=%s calls=%s terminal=%s",
            self.session_id,
            self._task.task_id if self._task is not None else "unknown",
            self._step,
            action.name,
            ",".join(call_names) or "none",
            action.terminal,
        )
        try:
            validated_action = parse_native_tool_calls(
                [
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": call.get("name"),
                        "arguments": call.get("arguments"),
                    }
                    for call in calls
                    if isinstance(call, dict)
                ],
                max_computer_actions=self.config.max_computer_actions,
            )
            if validated_action.terminal != action.terminal:
                raise ValueError("native action terminal flag does not match its tool calls")
            calls = validated_action.arguments["calls"]
            for call in calls:
                call_started = time.monotonic()
                self._execute_call(call["name"], call.get("arguments") or {})
                LOG.info(
                    "event=native_browser_tool_complete session=%s task=%s step=%d tool=%s elapsed_seconds=%.3f",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    self._step,
                    call["name"],
                    time.monotonic() - call_started,
                )
                if call["name"] != "terminate":
                    self._maybe_solve_captcha(f"after {call['name']}", failure_step=captcha_failure_step)
        except Exception as exc:  # A malformed/failed UI operation is policy-visible.
            execution_ok = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            LOG.exception(
                "event=native_browser_step_failed session=%s task=%s step=%d action=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                action.name,
            )
        self._step += 1
        if action.terminal:
            # The reference trajectory does not add a duplicate screenshot for
            # terminate; return the most recent observation unchanged.
            if self._observation is None:
                raise RuntimeError("terminal action has no prior observation")
        else:
            if not self._browser_target_closed:
                time.sleep(self.config.action_delay_seconds)
            if not self._captcha_budget_exhausted and not self._browser_target_closed:
                try:
                    self._maybe_solve_captcha(
                        "before post-action screenshot",
                        failure_step=captcha_failure_step,
                    )
                except Exception as exc:
                    # This lifecycle check runs after the tool-execution try
                    # block above.  An exhausted CAPTCHA budget must still be
                    # returned as a terminal task status; allowing the
                    # exception to cross the HTTP boundary turns the intended
                    # masked rollout into a resource-server 500 with no row.
                    execution_ok = False
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    LOG.exception(
                        "event=native_browser_post_action_captcha_failed session=%s task=%s step=%d",
                        self.session_id,
                        self._task.task_id if self._task is not None else "unknown",
                        self._step,
                    )
            if not self._browser_target_closed:
                self._observation = self._capture()
        # The pinned native runner treats an exhausted CAPTCHA budget as a
        # task-level terminal error, even when ordinary action errors remain
        # policy-visible.  Do not give the agent more correction turns that
        # could create additional paid solver tasks.
        terminated = (
            action.terminal
            or self._captcha_budget_exhausted
            or self._browser_target_closed
            or (not execution_ok and self.config.terminate_on_action_error)
        )
        LOG.info(
            "event=native_browser_step_complete session=%s task=%s step=%d execution_ok=%s "
            "terminated=%s origin=%s elapsed_seconds=%.3f",
            self.session_id,
            self._task.task_id if self._task is not None else "unknown",
            self._step,
            execution_ok,
            terminated,
            _url_origin(self._observation.url) if self._observation is not None else "none",
            time.monotonic() - started,
        )
        # An exhausted CAPTCHA budget is a site-access failure, not a policy
        # outcome. The reference runner drops such a task from its scored set and
        # re-runs it later, so report it under its own status instead of folding
        # it into a normal action error that would still be judged.
        if self._browser_target_closed:
            info = {"action_error": self._last_error, "native_status": BROWSER_TARGET_CLOSED_STATUS}
        elif self._captcha_budget_exhausted:
            info = {"action_error": self._last_error, "native_status": CAPTCHA_BUDGET_EXHAUSTED_STATUS}
        elif self._last_error:
            info = {"action_error": self._last_error, "native_status": "error"}
        else:
            info = {"native_status": "done" if action.terminal else "running"}
        return WebStepResult(
            observation=self._observation,
            execution_ok=execution_ok,
            terminated=terminated,
            info=info,
        )

    def evaluate(self, final_answer: str | None = None) -> WebVerifierResult:
        LOG.info(
            "event=native_browser_evaluate session=%s task=%s screenshots=%d final_answer_present=%s",
            self.session_id,
            self._task.task_id if self._task is not None else "unknown",
            len(self._evidence),
            bool(final_answer),
        )
        return WebVerifierResult(
            valid_sample=False,
            failure_kind="external_judge_required",
            evidence=list(self._evidence),
            verifier_version="native-webvoyager-gemini-v1",
            metadata={"final_answer": final_answer or "", "screenshots": len(self._evidence)},
        )

    def close(self) -> None:
        had_runtime = any(owner is not None for owner in (self._context, self._browser, self._playwright))
        task_id = self._task.task_id if self._task is not None else "unknown"
        started = time.monotonic()
        for owner in (self._context, self._browser):
            if owner is not None:
                try:
                    owner.close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._playwright = self._browser = self._context = self._page = None
        self._task = None
        self._observation = None
        if had_runtime:
            LOG.info(
                "event=native_browser_close session=%s task=%s elapsed_seconds=%.3f",
                self.session_id,
                task_id,
                time.monotonic() - started,
            )

    def _proxy_for_task(self, task: WebTask) -> str:
        if self.config.proxy_mode == "disabled":
            return ""
        proxy = self.config.browser_proxy()
        if self.config.proxy_mode == "always":
            return proxy
        for start_url in task.start_urls:
            parsed = urlparse(start_url)
            if parsed.scheme == "https" and parsed.netloc.lower() in PROXY_START_URL_HOSTS:
                return proxy
            if any(start_url.startswith(prefix) for prefix in PROXY_START_URL_PREFIXES):
                return proxy
        return ""

    def _maybe_solve_captcha(self, phase: str, *, failure_step: int | None = None) -> bool:
        if self.config.require_captcha_solver and not self.config.captcha_solver():
            LOG.error(
                "event=captcha_precondition_failed session=%s task=%s phase=%s missing_env=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                phase,
                self.config.captcha_solver_env,
            )
            raise RuntimeError(f"captcha solver is required but {self.config.captcha_solver_env} is unset")
        try:
            solved = self._captcha_solver.maybe_solve(self._page, phase=phase)
        except Exception as exc:
            if _is_playwright_target_closed_error(exc):
                self._browser_target_closed = True
                LOG.error(
                    "event=captcha_browser_target_closed session=%s task=%s step=%d phase=%s "
                    "failure_budget_counted=false error_type=%s",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    self._step,
                    phase,
                    type(exc).__name__,
                )
                raise BrowserTargetClosedDuringCaptcha(
                    f"browser target closed during CAPTCHA handling at phase {phase!r}"
                ) from None
            # Match the native runner's failure boundary: a transient solver
            # timeout or a challenge that remains visible is not a resource
            # server crash. Keep the live page available to the agent and let
            # the normal task/judge path determine the outcome.
            LOG.warning(
                "event=captcha_solver_deferred session=%s task=%s step=%d phase=%s origin=%s error_type=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                phase,
                _url_origin(self._page.url if self._page is not None else ""),
                type(exc).__name__,
            )
            if failure_step is not None and self._last_captcha_failure_step != failure_step:
                self._last_captcha_failure_step = failure_step
                self._captcha_failures += 1
                try:
                    max_failures = int(os.environ.get("WA_MAX_CAPTCHA_FAILURES", "3"))
                except ValueError:
                    max_failures = 3
                    LOG.warning(
                        "event=captcha_failure_budget_invalid value_present=true fallback=%d",
                        max_failures,
                    )
                max_failures = max(0, max_failures)
                LOG.warning(
                    "event=captcha_failure_counted session=%s task=%s step=%d failures=%d max_failures=%d",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    failure_step,
                    self._captcha_failures,
                    max_failures,
                )
                if self._captcha_failures > max_failures:
                    self._captcha_budget_exhausted = True
                    LOG.error(
                        "event=captcha_failure_budget_exhausted session=%s task=%s step=%d "
                        "failures=%d max_failures=%d",
                        self.session_id,
                        self._task.task_id if self._task is not None else "unknown",
                        failure_step,
                        self._captcha_failures,
                        max_failures,
                    )
                    raise RuntimeError(
                        f"Captcha solver failed more than {max_failures} times "
                        f"after VLM inference; aborting task at step {failure_step}"
                    ) from None
            return False
        if solved:
            LOG.info(
                "event=captcha_applied session=%s task=%s step=%d phase=%s origin=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                phase,
                _url_origin(self._page.url if self._page is not None else ""),
            )
        return solved

    @staticmethod
    def _playwright_proxy(proxy: str) -> dict[str, str]:
        parsed = urlparse(proxy if "://" in proxy else f"http://{proxy}")
        if not parsed.hostname:
            raise ValueError("WA_BROWSER_PROXY_SERVER is not a valid proxy URL")
        server = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            server += f":{parsed.port}"
        config = {"server": server}
        if parsed.username:
            config["username"] = unquote(parsed.username)
        if parsed.password:
            config["password"] = unquote(parsed.password)
        return config

    def _capture(self) -> WebObservation:
        from PIL import ImageGrab

        image = ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"))
        screenshot = self.artifacts.save_screenshot(self.session_id, self._step, image)
        if screenshot.artifact is not None:
            self._evidence.append(screenshot.artifact)
        pages = list(self._context.pages)
        active = pages.index(self._page) if self._page in pages else 0
        tabs = [
            WebTab(index=index, url=page.url, title=self._safe_title(page), active=index == active)
            for index, page in enumerate(pages)
        ]
        if screenshot.artifact is not None:
            LOG.info(
                "event=native_browser_screenshot session=%s task=%s step=%d origin=%s bytes=%d sha256=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                _url_origin(self._page.url if self._page is not None else ""),
                screenshot.artifact.size_bytes,
                screenshot.artifact.sha256[:12],
            )
        return WebObservation(
            goal=[{"type": "text", "text": self._task.intent if self._task else ""}],
            screenshot=screenshot,
            url=self._page.url if self._page is not None else "",
            tabs=tabs,
            active_tab_index=active,
            last_action=self._last_action,
            last_action_error=self._last_error,
            elapsed_time=max(0.0, time.monotonic() - self._started_at),
            metadata={"step": self._step, "runtime": "native_visual"},
        )

    @staticmethod
    def _safe_title(page: Any) -> str:
        try:
            return page.title()
        except Exception:
            return ""

    @staticmethod
    def _configure_page(page: Any) -> None:
        page.on("dialog", lambda dialog: dialog.accept())

    def _goto(self, page: Any, url: str, *, wait_until: str) -> Any:
        """Navigate, retrying the transport faults the reference runner retries.

        Only ``page.goto`` is retried. The reference runner leaves history
        navigation to a single attempt, and a retried ``go_back`` would move
        through history twice.
        """

        attempts = len(NAVIGATION_RETRY_DELAYS_S) + 1
        for attempt in range(1, attempts + 1):
            try:
                return page.goto(url, wait_until=wait_until)
            except Exception as exc:
                if attempt >= attempts or not _is_retryable_navigation_error(exc):
                    raise
                delay_seconds = NAVIGATION_RETRY_DELAYS_S[attempt - 1]
                LOG.warning(
                    "event=native_browser_navigation_retry session=%s task=%s step=%d origin=%s "
                    "attempt=%d/%d error_type=%s sleep_seconds=%d",
                    self.session_id,
                    self._task.task_id if self._task is not None else "unknown",
                    self._step,
                    _url_origin(url),
                    attempt,
                    attempts,
                    type(exc).__name__,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
        raise RuntimeError("navigation retry loop exited without a result")

    def _execute_call(self, name: str, arguments: dict[str, Any]) -> None:
        if name == "computer":
            for action in arguments["actions"]:
                self._execute_computer(action)
            return
        if name == "navigate":
            self._select_page(arguments.get("tab_id"))
            url = arguments["url"]
            if url == "back":
                self._page.go_back(wait_until=NAVIGATION_WAIT_UNTIL)
            elif url == "forward":
                self._page.go_forward(wait_until=NAVIGATION_WAIT_UNTIL)
            else:
                self._goto(self._page, url, wait_until=NAVIGATION_WAIT_UNTIL)
            self._page.bring_to_front()
            return
        if name == "tabs_create":
            self._page = self._context.new_page()
            url = arguments.get("url", "about:blank")
            if url != "about:blank":
                self._goto(self._page, url, wait_until=NAVIGATION_WAIT_UNTIL)
            self._page.bring_to_front()
            return
        if name == "tabs_focus":
            self._select_page(arguments["tab_id"])
            self._page.bring_to_front()
            return
        if name == "terminate":
            return
        raise ValueError(f"unsupported native tool: {name}")

    def _select_page(self, tab_id: int | None) -> None:
        if tab_id is None:
            return
        pages = list(self._context.pages)
        if not 0 <= tab_id < len(pages):
            raise ValueError(f"unknown tab_id: {tab_id}")
        self._page = pages[tab_id]

    def _execute_computer(self, spec: dict[str, Any]) -> None:
        import pyautogui

        name = spec["action"]
        coordinate = spec.get("coordinate")
        point = self._pixel(coordinate) if coordinate is not None else None
        if name in {"left_click", "middle_click", "right_click", "double_click", "triple_click"}:
            if point is None:
                raise ValueError(f"{name} requires coordinate")
            button = {"middle_click": "middle", "right_click": "right"}.get(name, "left")
            clicks = {"double_click": 2, "triple_click": 3}.get(name, 1)
            pyautogui.click(*point, clicks=clicks, button=button, interval=0.1)
        elif name == "mouse_move":
            pyautogui.moveTo(*point)
        elif name == "type":
            text = str(spec.get("text", ""))
            LOG.info(
                "event=native_browser_type session=%s task=%s step=%d characters=%d unicode=%s",
                self.session_id,
                self._task.task_id if self._task is not None else "unknown",
                self._step,
                len(text),
                not text.isascii(),
            )
            _type_browser_text(pyautogui, text)
        elif name == "key_press":
            keys = [str(key).lower() for key in spec.get("keys") or []]
            if not keys:
                raise ValueError("key_press requires keys")
            normalized = [self._normalize_key(key) for key in keys]
            if len(normalized) == 1:
                pyautogui.press(normalized[0])
            else:
                pyautogui.hotkey(*normalized)
        elif name == "wait":
            time.sleep(float(spec.get("duration") or self.config.action_delay_seconds))
        elif name == "scroll":
            params = spec.get("scroll_parameters") or {}
            amount = int(params.get("scroll_amount", 1))
            direction = params.get("scroll_direction", "down")
            if point is not None:
                pyautogui.moveTo(*point)
            if direction in {"up", "down"}:
                pyautogui.scroll(amount if direction == "up" else -amount)
            else:
                pyautogui.hscroll(amount if direction == "right" else -amount)
        elif name == "left_click_drag":
            start = self._pixel(spec.get("start_coordinate"))
            end = self._pixel(spec.get("coordinate"))
            pyautogui.moveTo(*start)
            pyautogui.dragTo(*end, duration=0.5, button="left")
        else:
            raise ValueError(f"unsupported computer action: {name}")
        time.sleep(0.3)

    def _pixel(self, coordinate: Any) -> tuple[int, int]:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError("coordinate must contain normalized x and y")
        x, y = float(coordinate[0]), float(coordinate[1])
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("coordinate values must be in [0, 1]")
        return (
            max(0, min(self.config.viewport_width - 1, round(x * self.config.viewport_width))),
            max(0, min(self.config.viewport_height - 1, round(y * self.config.viewport_height))),
        )

    @staticmethod
    def _normalize_key(key: str) -> str:
        aliases = {
            "cmd": "ctrl",
            "command": "ctrl",
            "control": "ctrl",
            "return": "enter",
            "escape": "esc",
            "option": "alt",
        }
        return aliases.get(key.lower(), key.lower())

    @staticmethod
    def _configure_pyautogui() -> None:
        os.environ.pop("WAYLAND_DISPLAY", None)
        import pyautogui

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0


def native_backend_factory(config, session_id: str, artifacts: WebArtifactStore) -> NativeWebDriver:
    return NativeWebDriver(config, session_id, artifacts)
