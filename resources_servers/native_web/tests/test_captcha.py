# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from resources_servers.native_web import captcha


class _Locator:
    def __init__(self, site_key: str | None) -> None:
        self._site_key = site_key
        self.first = self

    def count(self) -> int:
        return int(self._site_key is not None)

    def get_attribute(self, name: str) -> str | None:
        assert name == "data-sitekey"
        return self._site_key


class _Page:
    url = "https://example.test/form?private=query"
    frames: list = []

    def __init__(self, site_key: str | None = "public-site-key") -> None:
        self._site_key = site_key
        self.injected_token: str | None = None
        self.context = type("Context", (), {})()

    def locator(self, selector: str) -> _Locator:
        if selector.startswith(".cf-turnstile"):
            return _Locator(self._site_key)
        return _Locator(None)

    def evaluate(self, _script: str, arguments: list[str]):
        _field_name, token = arguments
        self.injected_token = token
        return {"fieldCount": 1, "callbacksCalled": 1}


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout
        self.requests: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def post(self, url: str, *, json: dict) -> _Response:
        self.requests.append((url, json))
        if url.endswith("createTask"):
            return _Response({"taskId": "provider-task-secret"})
        return _Response(
            {
                "status": "ready",
                "solution": {"token": "captcha-solution-secret"},
            }
        )


def test_capsolver_success_logs_lifecycle_without_secrets(monkeypatch, caplog) -> None:
    client = _Client(timeout=30.0)
    monkeypatch.setattr(captcha.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(captcha.time, "sleep", lambda _seconds: None)
    page = _Page()
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.resources_servers.native_web.captcha"):
        assert solver.maybe_solve(page, phase="initial") is True

    assert page.injected_token == "captcha-solution-secret"
    assert [url.rsplit("/", 1)[-1] for url, _payload in client.requests] == ["createTask", "getTaskResult"]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_detected" in messages
    assert "event=captcha_task_created" in messages
    assert "event=captcha_solved" in messages
    assert "origin=https://example.test" in messages
    assert "fields=1" in messages
    assert "callbacks=1" in messages
    for secret in (
        "CAP-private-key",
        "captcha-solution-secret",
        "provider-task-secret",
        "public-site-key",
        "private=query",
    ):
        assert secret not in messages


def test_capsolver_no_challenge_emits_debug_scan(caplog) -> None:
    page = _Page(site_key=None)
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.DEBUG, logger="nemo_gym.resources_servers.native_web.captcha"):
        assert solver.maybe_solve(page, phase="after wait") is False

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_scan" in messages
    assert "challenge=none" in messages
    assert "CAP-private-key" not in messages


def test_capsolver_environment_selection_logs_presence_not_value(monkeypatch, caplog) -> None:
    monkeypatch.setenv("CAPSOLVER_API_KEY", "CAP-private-key")
    monkeypatch.setenv("WA_CAPTCHA_PROVIDER", "capsolver")

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.native_web.captcha"):
        solver = captcha.captcha_solver_from_environment()

    assert isinstance(solver, captcha.CapSolverBrowserSolver)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "provider=capsolver" in messages
    assert "key_present=true" in messages
    assert "CAP-private-key" not in messages


def test_capsolver_task_uses_the_browser_proxy_without_logging_credentials() -> None:
    page = _Page()
    setattr(
        page.context,
        captcha.BROWSER_PROXY_CONFIG_ATTR,
        {
            "server": "http://proxy.example:19407",
            "username": "proxy-user",
            "password": "proxy-password",
        },
    )

    task = captcha.CapSolverBrowserSolver._build_task(page, "turnstile", "public-site-key")

    assert task == {
        "type": "AntiTurnstileTask",
        "websiteURL": page.url,
        "websiteKey": "public-site-key",
        "proxyType": "http",
        "proxyAddress": "proxy.example",
        "proxyPort": 19407,
        "proxyLogin": "proxy-user",
        "proxyPassword": "proxy-password",
    }


class _ChallengeBody:
    def inner_text(self, *, timeout: int) -> str:
        assert timeout == 1_000
        return "Checking if the site connection is secure"


class _ChallengePage(_Page):
    def __init__(self) -> None:
        super().__init__(site_key=None)

    def title(self) -> str:
        return "Just a moment..."

    def locator(self, selector: str):
        if selector == "body":
            return _ChallengeBody()
        return _Locator(None)


def test_capsolver_fails_closed_for_blocking_challenge_without_site_key(caplog) -> None:
    solver = captcha.CapSolverBrowserSolver("CAP-private-key", timeout=5)

    with caplog.at_level(logging.INFO, logger="nemo_gym.resources_servers.native_web.captcha"):
        with pytest.raises(RuntimeError, match="no supported site key"):
            solver.maybe_solve(_ChallengePage(), phase="initial")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=captcha_unresolved" in messages
    assert "reason=site_key_missing" in messages
    assert "CAP-private-key" not in messages
