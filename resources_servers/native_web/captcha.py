# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CapSolver integration boundary for native public-site browser sessions.

The browser driver calls this hook at the same lifecycle points as the native
runner. The default implementation deliberately supports explicit page hooks
and fails closed when a challenge is detected but no reviewed solver is
installed; provider-specific challenge code stays replaceable.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import time
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

import httpx


LOG = logging.getLogger("nemo_gym.resources_servers.native_web.captcha")


BROWSER_PROXY_CONFIG_ATTR = "_nemo_gym_browser_proxy_config"
CHALLENGE_TITLE_MARKERS = (
    "just a moment",
    "attention required",
    "checking your browser",
    "verify you are human",
    "human verification",
    "security check",
    "captcha",
)
CHALLENGE_TEXT_MARKERS = (
    "verify you are human",
    "performing security verification",
    "checking if the site connection is secure",
    "complete the security check",
    "complete the captcha",
    "please verify that you are not a robot",
    "i am not a robot",
    "protected by cloudflare",
)
CAPTCHA_FRAME_URL_MARKERS = (
    "challenges.cloudflare.com",
    "turnstile",
    "google.com/recaptcha",
    "recaptcha.net/recaptcha",
)
CAPTCHA_INTERCEPT_SCRIPT = """(() => {
    if (window.__nemoGymTurnstileHookInstalled) return;
    window.__nemoGymTurnstileHookInstalled = true;
    window.__nemoGymTurnstileParams = null;
    window.__nemoGymTurnstileCallback = null;
    const capture = (params) => {
        if (!params || !params.sitekey) return;
        window.__nemoGymTurnstileParams = {
            websiteKey: params.sitekey,
            action: params.action || null,
            cdata: params.cData || params.cdata || null,
        };
        if (typeof params.callback === 'function') {
            window.__nemoGymTurnstileCallback = params.callback;
        }
    };
    const wrap = (turnstile) => {
        if (!turnstile || turnstile.__nemoGymWrapped || typeof turnstile.render !== 'function') {
            return turnstile;
        }
        const render = turnstile.render.bind(turnstile);
        turnstile.render = (container, params = {}) => {
            capture(params);
            return render(container, params);
        };
        turnstile.__nemoGymWrapped = true;
        return turnstile;
    };
    let value = window.turnstile;
    Object.defineProperty(window, 'turnstile', {
        configurable: true,
        get() { return value; },
        set(next) { value = wrap(next); },
    });
    if (value) value = wrap(value);
})();"""


def _origin(url: str) -> str:
    """Return a log-safe origin without query parameters or credentials."""

    parsed = urlparse(url)
    if not parsed.hostname:
        return "unknown"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme or 'unknown'}://{parsed.hostname}{port}"


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class CaptchaSolver(Protocol):
    def maybe_solve(self, page: Any, *, phase: str) -> bool: ...


class NoopCaptchaSolver:
    def maybe_solve(self, page: Any, *, phase: str) -> bool:
        LOG.debug(
            "event=captcha_skipped provider=none phase=%s origin=%s",
            phase,
            _origin(getattr(page, "url", "")),
        )
        return False


class ModuleCaptchaSolver:
    """Load an operator-reviewed solver without coupling Gym to its secrets."""

    def __init__(self, spec: str) -> None:
        module_name, separator, attribute = spec.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("WA_CAPTCHA_SOLVER must use module.path:factory format")
        factory = getattr(importlib.import_module(module_name), attribute)
        self._solver = factory()
        if not hasattr(self._solver, "maybe_solve"):
            raise TypeError("captcha solver factory must return an object with maybe_solve(page, phase=...)")

    def maybe_solve(self, page: Any, *, phase: str) -> bool:
        started = time.monotonic()
        try:
            solved = bool(self._solver.maybe_solve(page, phase=phase))
        except Exception:
            LOG.exception(
                "event=captcha_solver_failed provider=custom phase=%s origin=%s elapsed_seconds=%.3f",
                phase,
                _origin(getattr(page, "url", "")),
                time.monotonic() - started,
            )
            raise
        LOG.info(
            "event=captcha_solver_complete provider=custom phase=%s origin=%s solved=%s elapsed_seconds=%.3f",
            phase,
            _origin(getattr(page, "url", "")),
            solved,
            time.monotonic() - started,
        )
        return solved


class CapSolverBrowserSolver:
    """Solve visible Turnstile/reCAPTCHA v2 widgets and inject the token."""

    CREATE_URL = "https://api.capsolver.com/createTask"
    RESULT_URL = "https://api.capsolver.com/getTaskResult"

    def __init__(self, api_key: str, *, timeout: float = 45.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._completed_challenges: set[tuple[str, str, str]] = set()

    def maybe_solve(self, page: Any, *, phase: str) -> bool:
        started = time.monotonic()
        origin = _origin(getattr(page, "url", ""))
        blocking_challenge = self._is_challenge_page(page)
        challenge = self._challenge(page)
        if challenge is None:
            if blocking_challenge:
                LOG.error(
                    "event=captcha_unresolved provider=capsolver phase=%s origin=%s "
                    "reason=site_key_missing",
                    phase,
                    origin,
                )
                raise RuntimeError("CAPTCHA challenge detected but no supported site key was found")
            LOG.debug(
                "event=captcha_scan provider=capsolver phase=%s origin=%s challenge=none",
                phase,
                origin,
            )
            return False
        kind, site_key = challenge
        identity = (origin, kind, _fingerprint(site_key))
        if identity in self._completed_challenges:
            LOG.error(
                "event=captcha_unresolved provider=capsolver phase=%s origin=%s challenge=%s "
                "reason=repeated_after_solution site_key_sha256=%s",
                phase,
                origin,
                kind,
                identity[2],
            )
            raise RuntimeError("CAPTCHA challenge remained after an accepted solver response")
        task = self._build_task(page, kind, site_key)
        task_type = str(task["type"])
        LOG.info(
            "event=captcha_detected provider=capsolver phase=%s origin=%s challenge=%s "
            "site_key_sha256=%s task_type=%s",
            phase,
            origin,
            kind,
            _fingerprint(site_key),
            task_type,
        )
        try:
            with httpx.Client(timeout=min(30.0, self._timeout)) as client:
                response = client.post(
                    self.CREATE_URL,
                    json={
                        "clientKey": self._api_key,
                        "task": task,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                task_id = payload.get("taskId")
                if not task_id:
                    raise RuntimeError(f"CapSolver createTask failed: {payload.get('errorDescription', 'unknown error')}")
                task_fingerprint = _fingerprint(str(task_id))
                LOG.info(
                    "event=captcha_task_created provider=capsolver phase=%s origin=%s "
                    "challenge=%s provider_task_sha256=%s",
                    phase,
                    origin,
                    kind,
                    task_fingerprint,
                )
                deadline = time.monotonic() + self._timeout
                polls = 0
                while time.monotonic() < deadline:
                    time.sleep(1.0)
                    polls += 1
                    result = client.post(self.RESULT_URL, json={"clientKey": self._api_key, "taskId": task_id})
                    result.raise_for_status()
                    result_payload = result.json()
                    if result_payload.get("status") == "processing":
                        if polls == 1 or polls % 10 == 0:
                            LOG.debug(
                                "event=captcha_poll provider=capsolver phase=%s origin=%s "
                                "provider_task_sha256=%s polls=%d elapsed_seconds=%.3f status=processing",
                                phase,
                                origin,
                                task_fingerprint,
                                polls,
                                time.monotonic() - started,
                            )
                        continue
                    if result_payload.get("status") != "ready":
                        raise RuntimeError(
                            f"CapSolver task failed: {result_payload.get('errorDescription', 'unknown error')}"
                        )
                    solution = result_payload.get("solution") or {}
                    token = solution.get("token") or solution.get("gRecaptchaResponse")
                    if not token:
                        raise RuntimeError("CapSolver returned no browser token")
                    injection = self._inject(page, kind, str(token))
                    self._completed_challenges.add(identity)
                    if blocking_challenge and not self._wait_for_challenge_clear(page):
                        raise RuntimeError("CAPTCHA solution was injected but the challenge page did not clear")
                    LOG.info(
                        "event=captcha_solved provider=capsolver phase=%s origin=%s challenge=%s "
                        "provider_task_sha256=%s polls=%d fields=%d callbacks=%d elapsed_seconds=%.3f",
                        phase,
                        origin,
                        kind,
                        task_fingerprint,
                        polls,
                        int(injection.get("fieldCount", 0)),
                        int(injection.get("callbacksCalled", 0)),
                        time.monotonic() - started,
                    )
                    return True
        except Exception:
            LOG.exception(
                "event=captcha_solver_failed provider=capsolver phase=%s origin=%s "
                "challenge=%s elapsed_seconds=%.3f",
                phase,
                origin,
                kind,
                time.monotonic() - started,
            )
            raise
        LOG.error(
            "event=captcha_solver_timeout provider=capsolver phase=%s origin=%s challenge=%s "
            "elapsed_seconds=%.3f timeout_seconds=%.1f",
            phase,
            origin,
            kind,
            time.monotonic() - started,
            self._timeout,
        )
        raise TimeoutError(f"CapSolver did not finish within {self._timeout:.1f}s")

    @staticmethod
    def _challenge(page: Any) -> tuple[str, str] | None:
        candidates: list[tuple[str, str]] = []
        for selector, kind in ((".cf-turnstile[data-sitekey]", "turnstile"), ("[data-sitekey]", "recaptcha")):
            locator = page.locator(selector)
            if locator.count():
                value = locator.first.get_attribute("data-sitekey")
                if value:
                    candidates.append((kind, value))
        for frame in page.frames:
            parsed = urlparse(frame.url)
            query = parse_qs(parsed.query)
            value = (query.get("k") or query.get("sitekey") or [None])[0]
            if not value:
                continue
            kind = "turnstile" if "cloudflare" in parsed.netloc or "turnstile" in frame.url else "recaptcha"
            candidates.append((kind, value))
        try:
            captured = page.evaluate(
                """() => {
                    const params = window.__nemoGymTurnstileParams;
                    return params && params.websiteKey ? params.websiteKey : null;
                }"""
            )
            if captured:
                candidates.insert(0, ("turnstile", str(captured)))
        except Exception:
            pass
        return candidates[0] if candidates else None

    @staticmethod
    def _is_challenge_page(page: Any) -> bool:
        try:
            title = str(page.title() or "").lower()
            if any(marker in title for marker in CHALLENGE_TITLE_MARKERS):
                return True
        except Exception:
            pass
        try:
            body = str(page.locator("body").inner_text(timeout=1_000) or "").lower()
            if any(marker in body for marker in CHALLENGE_TEXT_MARKERS):
                return True
        except Exception:
            pass
        for frame in getattr(page, "frames", []):
            frame_url = str(getattr(frame, "url", "")).lower()
            if any(marker in frame_url for marker in CAPTCHA_FRAME_URL_MARKERS):
                return True
        return False

    @staticmethod
    def _proxy_fields(page: Any) -> dict[str, Any]:
        try:
            config = getattr(page.context, BROWSER_PROXY_CONFIG_ATTR, None)
        except Exception:
            config = None
        if not config:
            return {}
        parsed = urlparse(str(config.get("server", "")))
        try:
            port = parsed.port
        except ValueError:
            port = None
        if parsed.scheme.lower() not in {"http", "https", "socks5"} or not parsed.hostname or not port:
            return {}
        fields: dict[str, Any] = {
            "proxyType": parsed.scheme.lower(),
            "proxyAddress": parsed.hostname,
            "proxyPort": port,
        }
        if config.get("username"):
            fields["proxyLogin"] = config["username"]
        if config.get("password"):
            fields["proxyPassword"] = config["password"]
        return fields

    @classmethod
    def _build_task(cls, page: Any, kind: str, site_key: str) -> dict[str, Any]:
        proxy_fields = cls._proxy_fields(page)
        if kind == "turnstile":
            task_type = "AntiTurnstileTask" if proxy_fields else "AntiTurnstileTaskProxyLess"
        else:
            task_type = "ReCaptchaV2Task" if proxy_fields else "ReCaptchaV2TaskProxyLess"
        return {
            "type": task_type,
            "websiteURL": page.url,
            "websiteKey": site_key,
            **proxy_fields,
        }

    @classmethod
    def _wait_for_challenge_clear(cls, page: Any, *, timeout: float = 12.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not cls._is_challenge_page(page):
                return True
            try:
                page.wait_for_timeout(1_000)
            except Exception:
                time.sleep(1.0)
        return not cls._is_challenge_page(page)

    @staticmethod
    def _inject(page: Any, kind: str, token: str) -> dict[str, int]:
        field_name = "cf-turnstile-response" if kind == "turnstile" else "g-recaptcha-response"
        result = page.evaluate(
            """([name, token]) => {
                let fields = Array.from(document.querySelectorAll(`textarea[name="${name}"], input[name="${name}"]`));
                if (!fields.length) {
                    const field = document.createElement('textarea');
                    field.name = name;
                    field.style.display = 'none';
                    document.body.appendChild(field);
                    fields = [field];
                }
                for (const field of fields) {
                    field.value = token;
                    field.dispatchEvent(new Event('input', {bubbles: true}));
                    field.dispatchEvent(new Event('change', {bubbles: true}));
                }
                let callbacksCalled = 0;
                for (const callback of Object.values(window)) {
                    if (typeof callback === 'function' && /captcha|turnstile/i.test(callback.name || '')) {
                        try { callback(token); callbacksCalled += 1; } catch (_) {}
                    }
                }
                if (typeof window.__nemoGymTurnstileCallback === 'function') {
                    try {
                        window.__nemoGymTurnstileCallback(token);
                        callbacksCalled += 1;
                    } catch (_) {}
                }
                return {fieldCount: fields.length, callbacksCalled};
            }""",
            [field_name, token],
        )
        if not isinstance(result, dict):
            return {"fieldCount": 0, "callbacksCalled": 0}
        return {
            "fieldCount": int(result.get("fieldCount", 0)),
            "callbacksCalled": int(result.get("callbacksCalled", 0)),
        }


def captcha_solver_from_environment() -> CaptchaSolver:
    """Resolve the approved solver implementation for a run.

    The built-in provider is selected by a CapSolver key. An explicit module
    remains available when an operator needs a reviewed browser-specific
    implementation.
    """

    spec = os.environ.get("WA_CAPTCHA_SOLVER", "").strip()
    if spec:
        LOG.info("event=captcha_solver_configured provider=custom spec=%s", spec)
        return ModuleCaptchaSolver(spec)
    api_key = os.environ.get("CAPSOLVER_API_KEY", "").strip()
    if api_key and os.environ.get("WA_CAPTCHA_PROVIDER", "capsolver").lower() == "capsolver":
        timeout = float(os.environ.get("WA_CAPTCHA_TIMEOUT", "45"))
        LOG.info(
            "event=captcha_solver_configured provider=capsolver key_present=true timeout_seconds=%.1f",
            timeout,
        )
        return CapSolverBrowserSolver(api_key, timeout=timeout)
    LOG.info("event=captcha_solver_configured provider=none key_present=%s", bool(api_key))
    return NoopCaptchaSolver()
