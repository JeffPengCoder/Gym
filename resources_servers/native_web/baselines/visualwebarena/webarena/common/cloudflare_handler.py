"""Automatic Cloudflare / Turnstile challenge detection and resolution for eval harness."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger(__name__)

CHALLENGE_TITLE_MARKERS = (
    "just a moment",
    "attention required",
    "checking your browser",
    "verify you are human",
    "human verification",
    "security check",
    "captcha",
)
CAPTCHA_FRAME_URL_MARKERS = (
    "challenges.cloudflare.com",
    "turnstile",
    "google.com/recaptcha",
    "recaptcha.net/recaptcha",
    "hcaptcha.com",
    "arkoselabs.com",
    "funcaptcha",
    "datadome.co",
)
TURNSTILE_FRAME_URL_MARKERS = ("challenges.cloudflare.com", "turnstile")
RECAPTCHA_FRAME_URL_MARKERS = ("google.com/recaptcha", "recaptcha.net/recaptcha")
CAPTCHA_FRAME_SELECTORS = (
    'iframe[src*="challenges.cloudflare.com"]',
    'iframe[src*="turnstile"]',
    'iframe[src*="google.com/recaptcha"]',
    'iframe[src*="recaptcha.net/recaptcha"]',
    'iframe[src*="hcaptcha.com"]',
    'iframe[src*="arkoselabs.com"]',
    'iframe[src*="funcaptcha"]',
    'iframe[src*="datadome.co"]',
    'iframe[title*="Cloudflare"]',
    'iframe[title*="captcha"]',
    'iframe[title*="CAPTCHA"]',
)
TURNSTILE_FRAME_SELECTORS = (
    'iframe[src*="challenges.cloudflare.com"]',
    'iframe[src*="turnstile"]',
    'iframe[title*="Cloudflare"]',
)
CHALLENGE_TEXT_MARKERS = (
    "verify you are human",
    "performing security verification",
    "checking if the site connection is secure",
    "complete the security check",
    "complete the captcha",
    "solve this captcha",
    "please verify that you are not a robot",
    "i am not a robot",
    "protected by cloudflare",
    "cf-challenge",
)
RETRYABLE_NAVIGATION_ERRORS = (
    "net::ERR_EMPTY_RESPONSE",
    "net::ERR_PROXY_CONNECTION_FAILED",
    "net::ERR_TUNNEL_CONNECTION_FAILED",
    "net::ERR_CONNECTION_CLOSED",
    "net::ERR_CONNECTION_RESET",
    "net::ERR_TIMED_OUT",
)
TRANSIENT_EVALUATE_ERRORS = (
    "Execution context was destroyed",
    "Cannot find context with specified id",
    "Target page, context or browser has been closed",
)
NAVIGATION_RETRY_DELAYS_S = (4, 4, 4, 8)
CAPSOLVER_CREATE_TASK_URL = "https://api.capsolver.com/createTask"
CAPSOLVER_GET_TASK_RESULT_URL = "https://api.capsolver.com/getTaskResult"
CONTEXT_PROXY_CONFIG_ATTR = "_webarena_browser_proxy_config"
TURNSTILE_TEST_SITEKEYS = {
    "1x00000000000000000000AA",
    "2x00000000000000000000AB",
    "3x00000000000000000000FF",
}
TURNSTILE_SITEKEY_RE = re.compile(r"0x4A{5,}[0-9A-Za-z_-]{10,}")
RECAPTCHA_SITEKEY_RE = re.compile(r"6[0-9A-Za-z_-]{30,}")


def auto_solve_enabled() -> bool:
    return os.environ.get("WA_AUTO_SOLVE_CAPTCHA", "1").lower() not in ("0", "false", "no")


def auto_dismiss_obstructions_enabled() -> bool:
    return os.environ.get("WA_AUTO_DISMISS_OBSTRUCTIONS", "1").lower() not in ("0", "false", "no")


def capsolver_enabled() -> bool:
    provider = os.environ.get("WA_CAPTCHA_PROVIDER", "capsolver").lower()
    if provider not in ("auto", "capsolver"):
        return False
    return bool(os.environ.get("CAPSOLVER_API_KEY"))


def _redact_api_key(text: str) -> str:
    api_key = os.environ.get("CAPSOLVER_API_KEY")
    if api_key:
        return text.replace(api_key, "<CAPSOLVER_API_KEY>")
    return text


def browser_proxy_config_from_server(proxy_server: str | None) -> dict[str, str] | None:
    """Build a Playwright proxy config while preserving credentials for solvers."""
    if not proxy_server:
        return None
    normalized = proxy_server if "://" in proxy_server else f"http://{proxy_server}"
    parsed = urlparse(normalized)
    if not parsed.hostname:
        logger.warning("Ignoring invalid browser proxy server: %r", proxy_server)
        return None
    try:
        proxy_port = parsed.port
    except ValueError:
        logger.warning("Ignoring browser proxy with invalid port: %r", proxy_server)
        return None

    server = f"{parsed.scheme}://{parsed.hostname}"
    if proxy_port:
        server = f"{server}:{proxy_port}"

    proxy_config = {"server": server}
    if parsed.username:
        proxy_config["username"] = unquote(parsed.username)
    if parsed.password:
        proxy_config["password"] = unquote(parsed.password)
    return proxy_config


def attach_browser_proxy_metadata(context, proxy_config: dict[str, str] | None) -> None:
    """Mark contexts that actually run through a browser proxy."""
    if not proxy_config:
        return
    try:
        setattr(context, CONTEXT_PROXY_CONFIG_ATTR, dict(proxy_config))
    except Exception as exc:
        logger.debug("Failed to attach proxy metadata to browser context: %s", exc)


def _browser_proxy_config_for_page(page) -> dict[str, str] | None:
    try:
        proxy_config = getattr(page.context, CONTEXT_PROXY_CONFIG_ATTR, None)
    except Exception:
        return None
    return dict(proxy_config) if proxy_config else None


def _capsolver_proxy_fields(proxy_config: dict[str, str] | None) -> dict[str, Any] | None:
    fields = _capsolver_proxy_connection_fields(proxy_config)
    if not fields:
        return None
    return {"type": "AntiTurnstileTask", **fields}


def _capsolver_proxy_connection_fields(proxy_config: dict[str, str] | None) -> dict[str, Any] | None:
    if not proxy_config:
        return None
    parsed = urlparse(proxy_config["server"])
    proxy_type = parsed.scheme.lower()
    if proxy_type not in ("http", "https", "socks5"):
        logger.info("CapSolver proxy skipped: unsupported proxy type %r", proxy_type)
        return None
    try:
        proxy_port = parsed.port
    except ValueError:
        proxy_port = None
    if not parsed.hostname or not proxy_port:
        logger.info("CapSolver proxy skipped: proxy host or port missing")
        return None

    fields: dict[str, Any] = {
        "proxyType": proxy_type,
        "proxyAddress": parsed.hostname,
        "proxyPort": proxy_port,
    }
    if proxy_config.get("username"):
        fields["proxyLogin"] = proxy_config["username"]
    if proxy_config.get("password"):
        fields["proxyPassword"] = proxy_config["password"]
    return fields


def _capsolver_proxy_string(proxy_config: dict[str, str] | None) -> str | None:
    fields = _capsolver_proxy_connection_fields(proxy_config)
    if not fields:
        return None
    proxy = f"{fields['proxyType']}:{fields['proxyAddress']}:{fields['proxyPort']}"
    username = fields.get("proxyLogin")
    password = fields.get("proxyPassword")
    if username and password:
        proxy = f"{proxy}:{username}:{password}"
    return proxy


def _capsolver_cloudflare_proxy(proxy_config: dict[str, str] | None) -> str | None:
    if not proxy_config:
        return None
    parsed = urlparse(proxy_config["server"])
    try:
        proxy_port = parsed.port
    except ValueError:
        proxy_port = None
    if not parsed.hostname or not proxy_port:
        return None
    proxy = f"{parsed.hostname}:{proxy_port}"
    username = proxy_config.get("username")
    password = proxy_config.get("password")
    if username and password:
        proxy = f"{proxy}:{username}:{password}"
    return proxy


def _is_likely_turnstile_sitekey(value: str | None) -> bool:
    return bool(value and (value in TURNSTILE_TEST_SITEKEYS or TURNSTILE_SITEKEY_RE.fullmatch(value)))


async def install_captcha_hooks(target) -> None:
    """Capture Turnstile render params before challenge scripts execute."""
    try:
        await target.add_init_script(
            """(() => {
                if (window.__webarenaTurnstileHookInstalled) {
                    return;
                }
                window.__webarenaTurnstileHookInstalled = true;
                window.__webarenaTurnstileParams = null;
                window.__webarenaTurnstileCallback = null;

                const capture = (container, params) => {
                    if (!params || !params.sitekey) {
                        return;
                    }
                    window.__webarenaTurnstileParams = {
                        websiteURL: location.href,
                        websiteKey: params.sitekey,
                        action: params.action || null,
                        cdata: params.cData || params.cdata || null,
                    };
                    if (typeof params.callback === 'function') {
                        window.__webarenaTurnstileCallback = params.callback;
                    }
                };

                const wrapTurnstile = (turnstile) => {
                    if (!turnstile || turnstile.__webarenaWrapped || typeof turnstile.render !== 'function') {
                        return turnstile;
                    }
                    const originalRender = turnstile.render.bind(turnstile);
                    turnstile.render = (container, params = {}) => {
                        capture(container, params);
                        return originalRender(container, params);
                    };
                    turnstile.__webarenaWrapped = true;
                    return turnstile;
                };

                let turnstileValue = window.turnstile;
                Object.defineProperty(window, 'turnstile', {
                    configurable: true,
                    get() {
                        return turnstileValue;
                    },
                    set(value) {
                        turnstileValue = wrapTurnstile(value);
                    },
                });
                if (turnstileValue) {
                    turnstileValue = wrapTurnstile(turnstileValue);
                }
            })();"""
        )
    except Exception as exc:
        logger.info("Failed to install captcha hooks: %s", exc)


def is_retryable_navigation_error(exc: Exception) -> bool:
    message = str(exc)
    return "Page.goto:" in message and any(marker in message for marker in RETRYABLE_NAVIGATION_ERRORS)


def _is_transient_evaluate_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in TRANSIENT_EVALUATE_ERRORS)


def storage_state_path() -> str | None:
    path = os.environ.get("PW_STORAGE_STATE")
    if path and Path(path).exists():
        return path
    return None


def save_storage_state_path() -> str | None:
    return os.environ.get("PW_SAVE_STORAGE_STATE")


def apply_storage_state(context_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Load persisted cookies/localStorage when available."""
    state = storage_state_path()
    if state:
        context_kwargs["storage_state"] = state
        logger.info("Using Playwright storage state from %s", state)
    return context_kwargs


async def _page_has_captcha_frame(page) -> bool:
    for selector in CAPTCHA_FRAME_SELECTORS:
        try:
            locator = page.locator(selector)
            count = await locator.count()
            for idx in range(min(count, 5)):
                box = await locator.nth(idx).bounding_box()
                if box and box.get("width", 0) > 20 and box.get("height", 0) > 20:
                    return True
        except Exception:
            pass
    return False


async def _page_has_turnstile_frame(page) -> bool:
    for selector in TURNSTILE_FRAME_SELECTORS:
        try:
            locator = page.locator(selector)
            count = await locator.count()
            for idx in range(min(count, 5)):
                box = await locator.nth(idx).bounding_box()
                if box and box.get("width", 0) > 20 and box.get("height", 0) > 20:
                    return True
        except Exception:
            pass
    return False


async def _page_has_recaptcha_frame(page) -> bool:
    for selector in (
        'iframe[src*="google.com/recaptcha"]',
        'iframe[src*="recaptcha.net/recaptcha"]',
        'iframe[title*="reCAPTCHA"]',
        'iframe[title*="recaptcha"]',
    ):
        try:
            locator = page.locator(selector)
            count = await locator.count()
            for idx in range(min(count, 5)):
                box = await locator.nth(idx).bounding_box()
                if box and box.get("width", 0) > 20 and box.get("height", 0) > 20:
                    return True
        except Exception:
            pass
    return False


async def _page_has_challenge_copy(page) -> bool:
    for text in CHALLENGE_TEXT_MARKERS:
        try:
            if await page.get_by_text(text, exact=False).count() > 0:
                return True
        except Exception:
            pass
    return False


def _turnstile_params_from_url(candidate_url: str, website_url: str) -> dict[str, Any] | None:
    parsed = urlparse(candidate_url)
    query = parse_qs(parsed.query)
    website_key = None
    for key in ("sitekey", "siteKey", "websiteKey", "k"):
        values = query.get(key)
        if values and _is_likely_turnstile_sitekey(values[0]):
            website_key = values[0]
            break
    if not website_key:
        match = TURNSTILE_SITEKEY_RE.search(candidate_url)
        if match:
            website_key = match.group(0)
    if not website_key:
        return None
    return {
        "websiteURL": website_url,
        "websiteKey": website_key,
        "action": (query.get("action") or [None])[0],
        "cdata": (query.get("cData") or query.get("cdata") or [None])[0],
    }


def _turnstile_params_from_frame_urls(page) -> dict[str, Any] | None:
    for frame in page.frames:
        frame_url = frame.url or ""
        lowered = frame_url.lower()
        if not any(marker in lowered for marker in TURNSTILE_FRAME_URL_MARKERS):
            continue
        params = _turnstile_params_from_url(frame_url, page.url)
        if params:
            return params
    return None


async def _extract_turnstile_params(page) -> dict[str, Any] | None:
    frame_params = _turnstile_params_from_frame_urls(page)
    if frame_params:
        return frame_params
    try:
        params = await page.evaluate(
            """() => {
                if (window.__webarenaTurnstileParams && window.__webarenaTurnstileParams.websiteKey) {
                    return window.__webarenaTurnstileParams;
                }

                const result = {
                    websiteURL: location.href,
                    websiteKey: null,
                    action: null,
                    cdata: null,
                };
                const turnstileTestSitekeys = new Set([
                    '1x00000000000000000000AA',
                    '2x00000000000000000000AB',
                    '3x00000000000000000000FF',
                ]);
                const isLikelyTurnstileSitekey = (value) => (
                    typeof value === 'string' && (
                        turnstileTestSitekeys.has(value) ||
                        /^0x4A{5,}[0-9A-Za-z_-]{10,}$/.test(value)
                    )
                );

                const attr = (el, name) => el && (
                    el.getAttribute(name) ||
                    el.getAttribute(`data-${name}`) ||
                    (el.dataset && el.dataset[name])
                );

                const elements = Array.from(document.querySelectorAll(
                    '.cf-turnstile, [data-sitekey], [name="cf-turnstile-response"]'
                ));
                for (const el of elements) {
                    const key = attr(el, 'sitekey');
                    if (isLikelyTurnstileSitekey(key)) {
                        result.websiteKey = key;
                        result.action = attr(el, 'action') || result.action;
                        result.cdata = attr(el, 'cdata') || result.cdata;
                        break;
                    }
                }

                if (!result.websiteKey) {
                    const html = document.documentElement && document.documentElement.innerHTML || '';
                    const sitekeyMatch = (
                        html.match(/["']?(?:sitekey|siteKey|websiteKey)["']?\\s*[:=]\\s*["']([^"']+)["']/i) ||
                        html.match(/data-sitekey=["']([^"']+)["']/i) ||
                        html.match(/0x4A{5,}[0-9A-Za-z_-]{10,}/)
                    );
                    const candidate = sitekeyMatch && (sitekeyMatch[1] || sitekeyMatch[0]);
                    if (isLikelyTurnstileSitekey(candidate)) {
                        result.websiteKey = candidate;
                    }
                    const actionMatch = html.match(/["']?action["']?\\s*[:=]\\s*["']([^"']+)["']/i);
                    if (actionMatch) {
                        result.action = actionMatch[1];
                    }
                    const cdataMatch = html.match(/["']?cData["']?\\s*[:=]\\s*["']([^"']+)["']/i);
                    if (cdataMatch) {
                        result.cdata = cdataMatch[1];
                    }
                }

                if (!result.websiteKey) {
                    const urls = Array.from(document.querySelectorAll('iframe[src], script[src]'))
                        .map((el) => el.getAttribute('src') || '')
                        .filter(Boolean);
                    for (const rawUrl of urls) {
                        try {
                            const parsed = new URL(rawUrl, location.href);
                            const key = (
                                parsed.searchParams.get('sitekey') ||
                                parsed.searchParams.get('siteKey') ||
                                parsed.searchParams.get('websiteKey') ||
                                parsed.searchParams.get('k')
                            );
                            if (isLikelyTurnstileSitekey(key)) {
                                result.websiteKey = key;
                                result.action = parsed.searchParams.get('action') || result.action;
                                result.cdata = parsed.searchParams.get('cData') || parsed.searchParams.get('cdata') || result.cdata;
                                break;
                            }
                            const match = parsed.href.match(/0x4A{5,}[0-9A-Za-z_-]{10,}/);
                            if (match) {
                                result.websiteKey = match[0];
                                result.action = parsed.searchParams.get('action') || result.action;
                                result.cdata = parsed.searchParams.get('cData') || parsed.searchParams.get('cdata') || result.cdata;
                                break;
                            }
                        } catch (_) {}
                    }
                }

                return result.websiteKey ? result : null;
            }"""
        )
    except Exception as exc:
        if _is_transient_evaluate_error(exc):
            logger.debug("Turnstile parameter extraction interrupted by page navigation: %s", exc)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass
            return None
        logger.info("Failed to extract Turnstile parameters: %s", exc)
        return None
    if not params or not _is_likely_turnstile_sitekey(params.get("websiteKey")):
        return None
    return params


async def _wait_for_turnstile_params(page, timeout_s: float) -> dict[str, Any] | None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if not await is_challenge_page(page):
            return None
        params = await _extract_turnstile_params(page)
        if params:
            return params
        await asyncio.sleep(1)
    return await _extract_turnstile_params(page)


def _recaptcha_params_from_url(candidate_url: str, website_url: str) -> dict[str, Any] | None:
    parsed = urlparse(candidate_url)
    query = parse_qs(parsed.query)
    website_key = None
    for key in ("k", "sitekey", "siteKey", "websiteKey"):
        values = query.get(key)
        if values and RECAPTCHA_SITEKEY_RE.fullmatch(values[0]):
            website_key = values[0]
            break
    if not website_key:
        match = RECAPTCHA_SITEKEY_RE.search(candidate_url)
        if match:
            website_key = match.group(0)
    if not website_key:
        return None

    params: dict[str, Any] = {
        "websiteURL": website_url,
        "websiteKey": website_key,
        "apiDomain": "recaptcha.net" if "recaptcha.net" in parsed.netloc else "google.com",
        "pageAction": (query.get("sa") or [None])[0],
        "recaptchaDataSValue": (query.get("s") or [None])[0],
        "isInvisible": (query.get("size") or [None])[0] == "invisible",
    }
    return params


def _recaptcha_params_from_frame_urls(page) -> dict[str, Any] | None:
    for frame in page.frames:
        frame_url = frame.url or ""
        lowered = frame_url.lower()
        if not any(marker in lowered for marker in RECAPTCHA_FRAME_URL_MARKERS):
            continue
        params = _recaptcha_params_from_url(frame_url, page.url)
        if params:
            return params
    return None


async def _extract_recaptcha_v2_params(page) -> dict[str, Any] | None:
    frame_params = _recaptcha_params_from_frame_urls(page)
    if frame_params:
        return frame_params
    if not await _page_has_recaptcha_frame(page):
        return None
    try:
        params = await page.evaluate(
            """() => {
                const result = {
                    websiteURL: location.href,
                    websiteKey: null,
                    apiDomain: 'google.com',
                    pageAction: null,
                    recaptchaDataSValue: null,
                    isInvisible: false,
                };

                const sitekeyRe = /^6[0-9A-Za-z_-]{30,}$/;
                const attr = (el, name) => el && (
                    el.getAttribute(name) ||
                    el.getAttribute(`data-${name}`) ||
                    (el.dataset && el.dataset[name])
                );

                for (const el of Array.from(document.querySelectorAll('.g-recaptcha, [data-sitekey]'))) {
                    const key = attr(el, 'sitekey');
                    if (sitekeyRe.test(key || '')) {
                        result.websiteKey = key;
                        result.pageAction = attr(el, 'action') || result.pageAction;
                        result.recaptchaDataSValue = attr(el, 's') || result.recaptchaDataSValue;
                        result.isInvisible = attr(el, 'size') === 'invisible';
                        break;
                    }
                }

                const urls = Array.from(document.querySelectorAll('iframe[src], script[src]'))
                    .map((el) => el.getAttribute('src') || '')
                    .filter(Boolean);
                for (const rawUrl of urls) {
                    try {
                        const parsed = new URL(rawUrl, location.href);
                        if (!parsed.href.includes('/recaptcha/')) {
                            continue;
                        }
                        result.apiDomain = parsed.hostname.includes('recaptcha.net') ? 'recaptcha.net' : 'google.com';
                        const key = (
                            parsed.searchParams.get('k') ||
                            parsed.searchParams.get('sitekey') ||
                            parsed.searchParams.get('siteKey') ||
                            parsed.searchParams.get('websiteKey')
                        );
                        if (sitekeyRe.test(key || '')) {
                            result.websiteKey = key;
                            result.pageAction = parsed.searchParams.get('sa') || result.pageAction;
                            result.recaptchaDataSValue = parsed.searchParams.get('s') || result.recaptchaDataSValue;
                            result.isInvisible = parsed.searchParams.get('size') === 'invisible' || result.isInvisible;
                            break;
                        }
                    } catch (_) {}
                }

                if (!result.websiteKey) {
                    const html = document.documentElement && document.documentElement.innerHTML || '';
                    const match = html.match(/6[0-9A-Za-z_-]{30,}/);
                    if (match) {
                        result.websiteKey = match[0];
                    }
                }

                return result.websiteKey ? result : null;
            }"""
        )
    except Exception as exc:
        if _is_transient_evaluate_error(exc):
            logger.debug("reCAPTCHA parameter extraction interrupted by page navigation: %s", exc)
            return None
        logger.info("Failed to extract reCAPTCHA parameters: %s", exc)
        return None
    if not params or not RECAPTCHA_SITEKEY_RE.fullmatch(str(params.get("websiteKey", ""))):
        return None
    return params


async def is_challenge_page(page) -> bool:
    """Return True while an active captcha/challenge appears to be showing."""
    try:
        if page.is_closed():
            return False
    except Exception:
        pass
    try:
        title = (await page.title()).lower()
    except Exception:
        return False

    if any(marker in title for marker in CHALLENGE_TITLE_MARKERS):
        return True
    if await _page_has_captcha_frame(page):
        return True
    if await _page_has_challenge_copy(page):
        return True
    return False


async def _safe_page_title(page) -> str:
    try:
        return await page.title()
    except Exception as exc:
        return f"<unavailable: {exc}>"


async def _capsolver_create_turnstile_task(
    params: dict[str, Any],
    timeout_s: float,
    proxy_config: dict[str, str] | None = None,
) -> str | None:
    api_key = os.environ.get("CAPSOLVER_API_KEY")
    if not api_key:
        return None
    try:
        import httpx
    except ImportError:
        logger.info("CapSolver unavailable: httpx is not installed")
        return None

    proxy_fields = _capsolver_proxy_fields(proxy_config)
    task: dict[str, Any] = {
        "type": "AntiTurnstileTaskProxyLess",
        "websiteURL": params["websiteURL"],
        "websiteKey": params["websiteKey"],
    }
    if proxy_fields:
        task.update(proxy_fields)
    metadata = {
        key: params[key]
        for key in ("action", "cdata")
        if params.get(key)
    }
    if metadata:
        task["metadata"] = metadata

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                CAPSOLVER_CREATE_TASK_URL,
                json={"clientKey": api_key, "task": task},
            )
            if response.status_code >= 400:
                logger.info(
                    "CapSolver createTask HTTP %s: %s",
                    response.status_code,
                    _redact_api_key(response.text),
                )
                return None
            payload = response.json()
    except Exception as exc:
        logger.info("CapSolver createTask failed: %s", exc)
        return None

    if payload.get("errorId"):
        logger.info(
            "CapSolver createTask error: %s",
            payload.get("errorDescription") or payload.get("errorCode"),
        )
        return None
    task_id = payload.get("taskId")
    if not task_id:
        logger.info("CapSolver createTask returned no taskId")
        return None
    logger.info(
        "CapSolver task created for Turnstile sitekey prefix %r",
        str(params["websiteKey"])[:12],
    )
    return str(task_id)


async def _capsolver_poll_result(task_id: str, timeout_s: float) -> str | None:
    api_key = os.environ.get("CAPSOLVER_API_KEY")
    if not api_key:
        return None
    try:
        import httpx
    except ImportError:
        logger.info("CapSolver unavailable: httpx is not installed")
        return None

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    async with httpx.AsyncClient(timeout=min(15.0, timeout_s)) as client:
        while loop.time() < deadline:
            await asyncio.sleep(2)
            try:
                response = await client.post(
                    CAPSOLVER_GET_TASK_RESULT_URL,
                    json={"clientKey": api_key, "taskId": task_id},
                )
                if response.status_code >= 400:
                    logger.info(
                        "CapSolver getTaskResult HTTP %s: %s",
                        response.status_code,
                        _redact_api_key(response.text),
                    )
                    return None
                payload = response.json()
            except Exception as exc:
                logger.info("CapSolver getTaskResult failed: %s", exc)
                return None

            if payload.get("errorId"):
                logger.info(
                    "CapSolver getTaskResult error: %s",
                    payload.get("errorDescription") or payload.get("errorCode"),
                )
                return None
            status = payload.get("status")
            if status == "ready":
                solution = payload.get("solution") or {}
                return solution.get("token") or solution.get("gRecaptchaResponse")
            if status not in ("idle", "processing"):
                logger.info("CapSolver returned unexpected status: %r", status)
                return None
    logger.info("CapSolver task %s timed out after %.0fs", task_id, timeout_s)
    return None


async def _capsolver_poll_solution(task_id: str, timeout_s: float) -> dict[str, Any] | None:
    api_key = os.environ.get("CAPSOLVER_API_KEY")
    if not api_key:
        return None
    try:
        import httpx
    except ImportError:
        logger.info("CapSolver unavailable: httpx is not installed")
        return None

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    async with httpx.AsyncClient(timeout=min(15.0, timeout_s)) as client:
        while loop.time() < deadline:
            await asyncio.sleep(2)
            try:
                response = await client.post(
                    CAPSOLVER_GET_TASK_RESULT_URL,
                    json={"clientKey": api_key, "taskId": task_id},
                )
                if response.status_code >= 400:
                    logger.info(
                        "CapSolver getTaskResult HTTP %s: %s",
                        response.status_code,
                        _redact_api_key(response.text),
                    )
                    return None
                payload = response.json()
            except Exception as exc:
                logger.info("CapSolver getTaskResult failed: %s", exc)
                return None

            if payload.get("errorId"):
                logger.info(
                    "CapSolver getTaskResult error: %s",
                    payload.get("errorDescription") or payload.get("errorCode"),
                )
                return None
            status = payload.get("status")
            if status == "ready":
                solution = payload.get("solution") or {}
                return solution if isinstance(solution, dict) else None
            if status not in ("idle", "processing"):
                logger.info("CapSolver returned unexpected status: %r", status)
                return None
    logger.info("CapSolver task %s timed out after %.0fs", task_id, timeout_s)
    return None


async def _capsolver_create_cloudflare_task(
    page,
    proxy_config: dict[str, str],
    timeout_s: float,
) -> str | None:
    api_key = os.environ.get("CAPSOLVER_API_KEY")
    if not api_key:
        return None
    proxy = _capsolver_cloudflare_proxy(proxy_config)
    if not proxy:
        logger.info("CapSolver Cloudflare skipped: browser proxy is missing host/port")
        return None
    try:
        import httpx
    except ImportError:
        logger.info("CapSolver unavailable: httpx is not installed")
        return None

    task: dict[str, Any] = {
        "type": "AntiCloudflareTask",
        "websiteURL": page.url,
        "proxy": proxy,
    }
    try:
        user_agent = await page.evaluate("() => navigator.userAgent")
        if user_agent:
            task["userAgent"] = user_agent
    except Exception:
        pass
    try:
        html = await page.content()
        if html:
            task["html"] = html
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                CAPSOLVER_CREATE_TASK_URL,
                json={"clientKey": api_key, "task": task},
            )
            if response.status_code >= 400:
                logger.info(
                    "CapSolver Cloudflare createTask HTTP %s: %s",
                    response.status_code,
                    _redact_api_key(response.text),
                )
                return None
            payload = response.json()
    except Exception as exc:
        logger.info("CapSolver Cloudflare createTask failed: %s", exc)
        return None

    if payload.get("errorId"):
        logger.info(
            "CapSolver Cloudflare createTask error: %s",
            payload.get("errorDescription") or payload.get("errorCode"),
        )
        return None
    task_id = payload.get("taskId")
    if not task_id:
        logger.info("CapSolver Cloudflare createTask returned no taskId")
        return None
    logger.info("CapSolver Cloudflare task created")
    return str(task_id)


async def _capsolver_create_recaptcha_v2_task(
    params: dict[str, Any],
    timeout_s: float,
    proxy_config: dict[str, str] | None = None,
) -> str | None:
    api_key = os.environ.get("CAPSOLVER_API_KEY")
    if not api_key:
        return None
    try:
        import httpx
    except ImportError:
        logger.info("CapSolver unavailable: httpx is not installed")
        return None

    proxy = _capsolver_proxy_string(proxy_config)
    task: dict[str, Any] = {
        "type": "ReCaptchaV2Task" if proxy else "ReCaptchaV2TaskProxyLess",
        "websiteURL": params["websiteURL"],
        "websiteKey": params["websiteKey"],
    }
    if proxy:
        task["proxy"] = proxy
    for source_key, task_key in (
        ("pageAction", "pageAction"),
        ("recaptchaDataSValue", "recaptchaDataSValue"),
        ("isInvisible", "isInvisible"),
    ):
        value = params.get(source_key)
        if value:
            task[task_key] = value
    api_domain = params.get("apiDomain")
    if api_domain:
        task["apiDomain"] = api_domain

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                CAPSOLVER_CREATE_TASK_URL,
                json={"clientKey": api_key, "task": task},
            )
            if response.status_code >= 400:
                logger.info(
                    "CapSolver reCAPTCHA createTask HTTP %s: %s",
                    response.status_code,
                    _redact_api_key(response.text),
                )
                return None
            payload = response.json()
    except Exception as exc:
        logger.info("CapSolver reCAPTCHA createTask failed: %s", exc)
        return None

    if payload.get("errorId"):
        logger.info(
            "CapSolver reCAPTCHA createTask error: %s",
            payload.get("errorDescription") or payload.get("errorCode"),
        )
        return None
    task_id = payload.get("taskId")
    if not task_id:
        logger.info("CapSolver reCAPTCHA createTask returned no taskId")
        return None
    logger.info(
        "CapSolver task created for reCAPTCHA sitekey prefix %r (%s)",
        str(params["websiteKey"])[:12],
        "browser proxy" if proxy else "proxyless",
    )
    return str(task_id)


async def _apply_cloudflare_solution(page, solution: dict[str, Any]) -> bool:
    cookies = solution.get("cookies") or {}
    token = solution.get("token")
    if token and "cf_clearance" not in cookies:
        cookies["cf_clearance"] = token
    if not cookies:
        logger.info("CapSolver Cloudflare solution contained no cookies")
        return False

    parsed = urlparse(page.url)
    origin = f"{parsed.scheme}://{parsed.netloc}/"
    try:
        await page.context.add_cookies([
            {"name": name, "value": str(value), "url": origin}
            for name, value in cookies.items()
            if value
        ])
    except Exception as exc:
        logger.info("Failed to apply CapSolver Cloudflare cookies: %s", exc)
        return False

    solution_user_agent = solution.get("userAgent")
    if solution_user_agent:
        try:
            page_user_agent = await page.evaluate("() => navigator.userAgent")
            if page_user_agent and page_user_agent != solution_user_agent:
                logger.warning("CapSolver Cloudflare userAgent differs from browser userAgent")
        except Exception:
            pass
    return True


async def _solve_cloudflare_with_capsolver(page, timeout_s: float) -> bool:
    if not capsolver_enabled():
        return False
    proxy_config = _browser_proxy_config_for_page(page)
    if not proxy_config:
        logger.info("CapSolver Cloudflare skipped: browser proxy is not enabled for this task")
        return False

    logger.info("Submitting Cloudflare managed challenge to CapSolver with browser proxy")
    task_timeout_s = max(5.0, min(timeout_s, float(os.environ.get("CAPSOLVER_TIMEOUT", "35"))))
    task_id = await _capsolver_create_cloudflare_task(page, proxy_config, min(15.0, task_timeout_s))
    if not task_id:
        return False

    solution = await _capsolver_poll_solution(task_id, task_timeout_s)
    if not solution:
        return False
    if not await _apply_cloudflare_solution(page, solution):
        return False

    logger.info("Applied CapSolver Cloudflare cookies; reloading challenge page")
    try:
        await page.reload(wait_until="domcontentloaded")
    except Exception:
        pass
    return await _wait_for_challenge_clear(page, min(10.0, timeout_s))


async def _inject_recaptcha_token(page, token: str) -> bool:
    try:
        return bool(await page.evaluate(
            """(token) => {
                let injected = false;
                const ensureResponseField = () => {
                    let field = document.querySelector('textarea[name="g-recaptcha-response"]');
                    if (!field) {
                        field = document.createElement('textarea');
                        field.name = 'g-recaptcha-response';
                        field.style.display = 'none';
                        document.body.appendChild(field);
                    }
                    return field;
                };

                for (const selector of [
                    'textarea[name="g-recaptcha-response"]',
                    'input[name="g-recaptcha-response"]',
                    'textarea[name^="g-recaptcha-response"]',
                    'input[name^="g-recaptcha-response"]',
                ]) {
                    for (const el of document.querySelectorAll(selector)) {
                        el.value = token;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        injected = true;
                    }
                }

                const field = ensureResponseField();
                field.value = token;
                field.dispatchEvent(new Event('input', { bubbles: true }));
                field.dispatchEvent(new Event('change', { bubbles: true }));
                injected = true;

                for (const el of document.querySelectorAll('[data-callback]')) {
                    const callbackName = el.getAttribute('data-callback');
                    const callback = callbackName && window[callbackName];
                    if (typeof callback === 'function') {
                        callback(token);
                        injected = true;
                    }
                }

                const callbacks = [];
                const seen = new Set();
                const scan = (obj, depth = 0) => {
                    if (!obj || depth > 8 || seen.has(obj)) {
                        return;
                    }
                    if (typeof obj !== 'object' && typeof obj !== 'function') {
                        return;
                    }
                    seen.add(obj);
                    for (const key of Object.keys(obj)) {
                        let value;
                        try {
                            value = obj[key];
                        } catch (_) {
                            continue;
                        }
                        if ((key === 'callback' || key === 'promise-callback') && typeof value === 'function') {
                            callbacks.push(value);
                        } else if (value && (typeof value === 'object' || typeof value === 'function')) {
                            scan(value, depth + 1);
                        }
                    }
                };
                if (window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients) {
                    scan(window.___grecaptcha_cfg.clients);
                }
                for (const callback of callbacks) {
                    try {
                        callback(token);
                        injected = true;
                    } catch (_) {}
                }

                return injected;
            }""",
            token,
        ))
    except Exception as exc:
        logger.info("Failed to inject reCAPTCHA token: %s", exc)
        return False


async def _solve_recaptcha_v2_with_capsolver(page, timeout_s: float) -> bool:
    if not capsolver_enabled():
        return False
    params = await _extract_recaptcha_v2_params(page)
    if not params:
        return False

    proxy_config = _browser_proxy_config_for_page(page)
    if proxy_config:
        logger.info("Submitting reCAPTCHA v2 challenge to CapSolver with browser proxy for %s", params["websiteURL"])
    else:
        logger.info("Submitting reCAPTCHA v2 challenge to CapSolver without browser proxy for %s", params["websiteURL"])
    task_timeout_s = max(5.0, min(timeout_s, float(os.environ.get("CAPSOLVER_TIMEOUT", "35"))))
    task_id = await _capsolver_create_recaptcha_v2_task(
        params,
        min(15.0, task_timeout_s),
        proxy_config,
    )
    if not task_id:
        return False

    solution = await _capsolver_poll_solution(task_id, task_timeout_s)
    if not solution:
        return False
    token = solution.get("gRecaptchaResponse") or solution.get("token")
    if not token:
        logger.info("CapSolver reCAPTCHA solution contained no token")
        return False

    if not await _inject_recaptcha_token(page, token):
        logger.info("CapSolver reCAPTCHA token received, but no response target was found")
        return False

    logger.info("Injected CapSolver reCAPTCHA token; waiting for challenge to clear")
    return await _wait_for_challenge_clear(page, min(10.0, timeout_s))


async def _inject_turnstile_token(page, token: str) -> bool:
    try:
        return bool(await page.evaluate(
            """(token) => {
                let injected = false;
                const selectors = [
                    'input[name="cf-turnstile-response"]',
                    'textarea[name="cf-turnstile-response"]',
                    'input[name="g-recaptcha-response"]',
                    'textarea[name="g-recaptcha-response"]',
                ];

                for (const selector of selectors) {
                    for (const el of document.querySelectorAll(selector)) {
                        el.value = token;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        injected = true;
                    }
                }

                for (const el of document.querySelectorAll('[data-callback]')) {
                    const callbackName = el.getAttribute('data-callback');
                    const callback = callbackName && window[callbackName];
                    if (typeof callback === 'function') {
                        callback(token);
                        injected = true;
                    }
                }

                if (typeof window.__webarenaTurnstileCallback === 'function') {
                    window.__webarenaTurnstileCallback(token);
                    injected = true;
                }

                return injected;
            }""",
            token,
        ))
    except Exception as exc:
        logger.info("Failed to inject Turnstile token: %s", exc)
        return False


async def _solve_turnstile_with_capsolver(page, timeout_s: float) -> bool:
    if not capsolver_enabled():
        return False

    params = await _wait_for_turnstile_params(page, min(8.0, max(1.0, timeout_s)))
    if not params:
        if not await is_challenge_page(page):
            logger.info("Challenge cleared before CapSolver sitekey was available")
            await _maybe_save_storage_state(page)
            return True
        logger.info("CapSolver skipped: no Turnstile sitekey found")
        return await _solve_cloudflare_with_capsolver(page, timeout_s)

    proxy_config = _browser_proxy_config_for_page(page)
    if proxy_config:
        logger.info("Submitting Turnstile challenge to CapSolver with browser proxy for %s", params["websiteURL"])
    else:
        logger.info("Submitting Turnstile challenge to CapSolver without browser proxy for %s", params["websiteURL"])
    task_timeout_s = max(5.0, min(timeout_s, float(os.environ.get("CAPSOLVER_TIMEOUT", "35"))))
    task_id = await _capsolver_create_turnstile_task(params, min(15.0, task_timeout_s), proxy_config)
    if not task_id:
        return False

    token = await _capsolver_poll_result(task_id, task_timeout_s)
    if not token:
        return False

    if not await _inject_turnstile_token(page, token):
        logger.info("CapSolver token received, but no Turnstile response target was found")
        return False

    logger.info("Injected CapSolver Turnstile token; waiting for challenge to clear")
    return await _wait_for_challenge_clear(page, min(10.0, timeout_s))


async def _viewport_screen_offset(page) -> tuple[float, float]:
    """Map Playwright viewport coordinates to X11/pyautogui screen coordinates."""
    try:
        offset = await page.evaluate(
            """() => ({
                x: window.screenX + (window.outerWidth - window.innerWidth) / 2,
                y: window.screenY + (window.outerHeight - window.innerHeight)
            })"""
        )
        return float(offset["x"]), float(offset["y"])
    except Exception:
        return 0.0, 0.0


async def _physical_click(page, x: float, y: float) -> None:
    """Prefer real X11 clicks; Playwright synthetic clicks are often ignored by Turnstile."""
    try:
        from common.pyautogui_utils import init_pyautogui

        pag = init_pyautogui()
        pag.click(int(round(x)), int(round(y)))
        logger.info("Physical click at (%.0f, %.0f)", x, y)
        return
    except Exception as exc:
        logger.debug("Physical click failed, falling back to Playwright mouse: %s", exc)

    await page.mouse.click(x, y)
    logger.info("Playwright click at (%.0f, %.0f)", x, y)


async def _click_turnstile(page) -> bool:
    """Try clicking the Turnstile widget via iframe/frame interactions."""
    offset_x, offset_y = await _viewport_screen_offset(page)
    has_turnstile_frame = await _page_has_turnstile_frame(page)

    for selector in TURNSTILE_FRAME_SELECTORS:
        try:
            iframe = page.locator(selector).first
            if await iframe.count() == 0:
                continue
            box = await iframe.bounding_box()
            if not box:
                continue
            x = offset_x + box["x"] + min(box["width"] * 0.12, 24)
            y = offset_y + box["y"] + box["height"] / 2
            await _physical_click(page, x, y)
            return True
        except Exception as exc:
            logger.info("Turnstile physical click via %s failed: %s", selector, exc)

    for frame in page.frames:
        frame_url = (frame.url or "").lower()
        if not any(marker in frame_url for marker in TURNSTILE_FRAME_URL_MARKERS):
            continue
        for target in ("input[type=checkbox]", "label", ".ctp-checkbox-label", "body"):
            try:
                await frame.click(target, timeout=2000)
                logger.info("Clicked Turnstile frame target %s", target)
                return True
            except Exception:
                continue

    if has_turnstile_frame:
        try:
            iframe = page.locator("iframe").first
            if await iframe.count() > 0:
                box = await iframe.bounding_box()
                if box:
                    x = offset_x + box["x"] + min(box["width"] * 0.12, 24)
                    y = offset_y + box["y"] + box["height"] / 2
                    await _physical_click(page, x, y)
                    return True
        except Exception as exc:
            logger.info("Generic iframe click failed: %s", exc)

        try:
            frame_locator = page.frame_locator("iframe").first
            for target in ("input[type=checkbox]", "label", "body"):
                try:
                    await frame_locator.locator(target).first.click(timeout=2000)
                    logger.info("Clicked iframe locator target %s", target)
                    return True
                except Exception:
                    continue
        except Exception as exc:
            logger.info("Frame locator click failed: %s", exc)

    for text in ("Verify you are human", "Verify"):
        try:
            locator = page.get_by_text(text, exact=False)
            if await locator.count() > 0:
                box = await locator.first.bounding_box()
                if box:
                    x = offset_x + box["x"] + 16
                    y = offset_y + box["y"] + box["height"] / 2
                    await _physical_click(page, x, y)
                    return True
        except Exception:
            pass

    logger.info("No Turnstile click target found")
    return False


async def _wait_for_challenge_clear(page, timeout_s: float) -> bool:
    try:
        await page.wait_for_function(
            """() => {
                const title = (document.title || '').toLowerCase();
                if (
                    title.includes('just a moment') ||
                    title.includes('checking your browser') ||
                    title.includes('verify you are human') ||
                    title.includes('human verification') ||
                    title.includes('captcha')
                ) {
                    return false;
                }
                const text = ((document.body && document.body.innerText) || '').toLowerCase();
                if (
                    text.includes('verify you are human') ||
                    text.includes('performing security verification') ||
                    text.includes('checking if the site connection is secure') ||
                    text.includes('complete the security check') ||
                    text.includes('complete the captcha') ||
                    text.includes('please verify that you are not a robot')
                ) {
                    return false;
                }
                return true;
            }""",
            timeout=int(timeout_s * 1000),
        )
    except Exception:
        pass
    return not await is_challenge_page(page)


async def _maybe_save_storage_state(page) -> None:
    path = save_storage_state_path()
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        await page.context.storage_state(path=path)
        logger.info("Saved browser storage state to %s", path)
    except Exception as exc:
        logger.warning("Failed to save storage state: %s", exc)


async def _page_has_google_vignette(page) -> bool:
    try:
        if "google_vignette" in (page.url or ""):
            return True
    except Exception:
        pass
    try:
        return bool(await page.evaluate(
            """() => {
                const hasVignetteHash = location.hash.includes('google_vignette');
                if (hasVignetteHash) {
                    return true;
                }
                const selectors = ['[id*="google_vignette" i]', '[class*="google_vignette" i]', '[class*="google-vignette" i]'];
                const viewportArea = window.innerWidth * window.innerHeight;
                for (const el of document.querySelectorAll(selectors.join(','))) {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    if (
                        rect.width > 100 &&
                        rect.height > 100 &&
                        rect.bottom > 0 &&
                        rect.right > 0 &&
                        rect.top < window.innerHeight &&
                        rect.left < window.innerWidth &&
                        style.visibility !== 'hidden' &&
                        style.display !== 'none'
                    ) {
                        if (rect.width * rect.height > viewportArea * 0.08) {
                            return true;
                        }
                    }
                }
                for (const el of document.body.querySelectorAll('*')) {
                    const style = window.getComputedStyle(el);
                    if (style.position !== 'fixed' && style.position !== 'sticky') {
                        continue;
                    }
                    const rect = el.getBoundingClientRect();
                    const zIndex = Number.parseInt(style.zIndex || '0', 10);
                    if (
                        rect.width * rect.height > viewportArea * 0.50 &&
                        zIndex >= 100 &&
                        style.visibility !== 'hidden' &&
                        style.display !== 'none' &&
                        style.pointerEvents !== 'none'
                    ) {
                        const text = (el.innerText || el.textContent || '').toLowerCase();
                        if (text.includes('advertisement') || el.querySelector('iframe[id^="google_ads_iframe"], iframe[src*="googlesyndication.com"], iframe[src*="doubleclick.net"]')) {
                            return true;
                        }
                    }
                }
                return false;
            }"""
        ))
    except Exception:
        return False


async def _click_common_overlay_close_targets(page) -> bool:
    selectors = (
        '[aria-label="Close"]',
        '[aria-label="close"]',
        '[aria-label*="Close" i]',
        'button[title*="Close" i]',
        '[role="button"][aria-label*="Close" i]',
        '.close',
        '.modal-close',
        '.close-button',
    )
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() == 0:
                continue
            box = await locator.bounding_box()
            if not box or box.get("width", 0) <= 0 or box.get("height", 0) <= 0:
                continue
            await locator.click(timeout=1000, force=True)
            await page.wait_for_timeout(500)
            if not await _page_has_google_vignette(page):
                return True
        except Exception:
            continue
    return False


async def _click_google_vignette_corners(page) -> bool:
    try:
        boxes = await page.evaluate(
            """() => {
                const candidates = [];
                const selectors = [
                    '[id*="google_vignette" i]',
                    '[class*="google_vignette" i]',
                    '[class*="google-vignette" i]',
                    'iframe[id^="google_ads_iframe"]',
                    'iframe[src*="googlesyndication.com"]',
                    'iframe[src*="doubleclick.net"]',
                    'iframe[src*="googleads"]',
                ];
                for (const el of document.querySelectorAll(selectors.join(','))) {
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    if (
                        rect.width > 100 &&
                        rect.height > 100 &&
                        rect.bottom > 0 &&
                        rect.right > 0 &&
                        rect.top < window.innerHeight &&
                        rect.left < window.innerWidth &&
                        style.visibility !== 'hidden' &&
                        style.display !== 'none'
                    ) {
                        candidates.push({
                            x: Math.max(0, rect.left),
                            y: Math.max(0, rect.top),
                            width: Math.min(rect.width, window.innerWidth - Math.max(0, rect.left)),
                            height: Math.min(rect.height, window.innerHeight - Math.max(0, rect.top)),
                            area: rect.width * rect.height,
                        });
                    }
                }
                candidates.sort((a, b) => b.area - a.area);
                return candidates.slice(0, 3);
            }"""
        )
    except Exception:
        boxes = []

    for box in boxes or []:
        try:
            x = float(box["x"]) + float(box["width"]) - 12
            y = float(box["y"]) + 12
            await page.mouse.click(x, y)
            await page.wait_for_timeout(500)
            if not await _page_has_google_vignette(page):
                return True
        except Exception:
            continue
    try:
        viewport = page.viewport_size or {}
        width = float(viewport.get("width") or 0)
        if width:
            await page.mouse.click(width - 20, 20)
            await page.wait_for_timeout(500)
            return not await _page_has_google_vignette(page)
    except Exception:
        pass
    return False


async def _remove_google_vignette_dom(page) -> bool:
    try:
        removed = int(await page.evaluate(
            """() => {
                let removed = 0;
                const remove = (el) => {
                    if (el && el.parentNode) {
                        el.parentNode.removeChild(el);
                        removed += 1;
                    }
                };
                const selectors = [
                    '[id*="google_vignette" i]',
                    '[class*="google_vignette" i]',
                    '[class*="google-vignette" i]',
                    'iframe[id^="google_ads_iframe"]',
                    'iframe[src*="googlesyndication.com"]',
                    'iframe[src*="doubleclick.net"]',
                    'iframe[src*="googleads"]',
                    'ins.adsbygoogle',
                ];
                for (const el of Array.from(document.querySelectorAll(selectors.join(',')))) {
                    const rect = el.getBoundingClientRect();
                    if (
                        location.hash.includes('google_vignette') ||
                        (rect.width > 100 && rect.height > 100)
                    ) {
                        remove(el);
                    }
                }

                const viewportArea = window.innerWidth * window.innerHeight;
                for (const el of Array.from(document.body.querySelectorAll('*'))) {
                    const style = window.getComputedStyle(el);
                    if (style.position !== 'fixed' && style.position !== 'sticky') {
                        continue;
                    }
                    const rect = el.getBoundingClientRect();
                    const zIndex = Number.parseInt(style.zIndex || '0', 10);
                    if (
                        rect.width * rect.height > viewportArea * 0.30 &&
                        rect.width > window.innerWidth * 0.40 &&
                        rect.height > window.innerHeight * 0.40 &&
                        (zIndex >= 100 || location.hash.includes('google_vignette'))
                    ) {
                        remove(el);
                    }
                }

                document.documentElement.style.overflow = 'auto';
                document.body.style.overflow = 'auto';
                if (location.hash.includes('google_vignette')) {
                    history.replaceState(null, document.title, location.pathname + location.search);
                }
                return removed;
            }"""
        ))
    except Exception as exc:
        logger.info("Failed to remove Google vignette DOM: %s", exc)
        return False
    if removed:
        logger.info("Removed %s Google vignette/ad overlay element(s)", removed)
        await page.wait_for_timeout(500)
    return removed > 0 and not await _page_has_google_vignette(page)


async def _navigate_without_google_vignette(page) -> bool:
    try:
        clean_url = await page.evaluate(
            """() => {
                if (!location.hash.includes('google_vignette')) {
                    return null;
                }
                return location.origin + location.pathname + location.search;
            }"""
        )
    except Exception:
        clean_url = None

    if not clean_url:
        try:
            url = page.url or ""
            clean_url = url.replace("#google_vignette", "")
        except Exception:
            return False

    if not clean_url:
        return False

    try:
        logger.info("Reloading page without Google vignette fragment: %s", clean_url)
        await page.goto(clean_url, wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(1000)
        return not await _page_has_google_vignette(page)
    except Exception as exc:
        logger.info("Failed to reload without Google vignette fragment: %s", exc)
        return False


async def maybe_dismiss_page_obstructions(page) -> bool:
    """Dismiss known non-captcha overlays that block the page."""
    if not auto_dismiss_obstructions_enabled():
        return False
    if not await _page_has_google_vignette(page):
        return False

    logger.info("Google vignette/ad overlay detected at %s; attempting dismiss", page.url)
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)
        if not await _page_has_google_vignette(page):
            logger.info("Google vignette dismissed with Escape")
            return True
    except Exception:
        pass

    if await _click_common_overlay_close_targets(page):
        logger.info("Google vignette dismissed with close control")
        return True
    if await _click_google_vignette_corners(page):
        logger.info("Google vignette dismissed with corner click")
        return True
    if await _navigate_without_google_vignette(page):
        logger.info("Google vignette dismissed by reloading clean URL")
        return True
    if await _remove_google_vignette_dom(page):
        logger.info("Google vignette dismissed by removing overlay DOM")
        return True

    logger.info("Google vignette/ad overlay still appears present after dismiss attempts")
    return False


async def resolve_challenge(page, timeout_s: float | None = None) -> bool:
    """Wait for or interact with a challenge page until it clears or times out."""
    await maybe_dismiss_page_obstructions(page)

    if not auto_solve_enabled():
        return not await is_challenge_page(page)

    if not await is_challenge_page(page):
        return True

    timeout_s = timeout_s or float(os.environ.get("WA_CAPTCHA_TIMEOUT", "45"))
    logger.info("Captcha/challenge detected at %s; attempting auto-resolve", page.url)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    attempt = 0
    last_click_at = 0.0
    capsolver_attempted = False

    while loop.time() < deadline:
        if not await is_challenge_page(page):
            logger.info("Challenge cleared; now at %s (%s)", page.url, await _safe_page_title(page))
            await _maybe_save_storage_state(page)
            return True

        remaining = max(0.0, deadline - loop.time())
        attempt += 1
        logger.info(
            "Captcha resolve attempt %s (%.0fs remaining, title=%r)",
            attempt,
            remaining,
            await _safe_page_title(page),
        )

        await asyncio.sleep(2)
        if not await is_challenge_page(page):
            logger.info("Challenge auto-cleared")
            await _maybe_save_storage_state(page)
            return True

        now = loop.time()
        remaining = max(0.0, deadline - now)
        if not capsolver_attempted and remaining >= 5:
            capsolver_attempted = True
            if await _solve_recaptcha_v2_with_capsolver(page, remaining):
                logger.info("Challenge cleared by CapSolver reCAPTCHA; now at %s (%s)", page.url, await _safe_page_title(page))
                await _maybe_save_storage_state(page)
                return True
            if await _solve_turnstile_with_capsolver(page, remaining):
                logger.info("Challenge cleared by CapSolver; now at %s (%s)", page.url, await _safe_page_title(page))
                await _maybe_save_storage_state(page)
                return True

        if now - last_click_at >= 5:
            if await _click_turnstile(page):
                last_click_at = now
                cleared = await _wait_for_challenge_clear(page, min(15.0, remaining))
                if cleared:
                    logger.info("Challenge cleared after click; now at %s (%s)", page.url, await _safe_page_title(page))
                    await _maybe_save_storage_state(page)
                    return True
            await asyncio.sleep(2)

    if await is_challenge_page(page):
        logger.warning(
            "Failed to resolve captcha within %.0fs (still blocked: title=%r url=%s)",
            timeout_s,
            await _safe_page_title(page),
            page.url,
        )
        return False

    await _maybe_save_storage_state(page)
    return True


async def maybe_resolve_after_navigation(page) -> bool:
    """Resolve page obstructions and challenges after navigation, if present."""
    dismissed = await maybe_dismiss_page_obstructions(page)
    if not await is_challenge_page(page):
        return True
    return await resolve_challenge(page) or dismissed


async def goto(page, url: str, **kwargs):
    """Navigate and auto-resolve any captcha/challenge on the landing page."""
    response = None
    max_attempts = len(NAVIGATION_RETRY_DELAYS_S) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = await page.goto(url, **kwargs)
            break
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_navigation_error(exc):
                raise
            delay_s = NAVIGATION_RETRY_DELAYS_S[attempt - 1]
            logger.warning(
                "Retryable navigation error on attempt %s/%s for %s: %s; retrying in %ss",
                attempt,
                max_attempts,
                url,
                exc,
                delay_s,
            )
            await asyncio.sleep(delay_s)
    await maybe_resolve_after_navigation(page)
    return response

async def go_back(page, **kwargs):
    response = await page.go_back(**kwargs)
    await maybe_resolve_after_navigation(page)
    return response


async def go_forward(page, **kwargs):
    response = await page.go_forward(**kwargs)
    await maybe_resolve_after_navigation(page)
    return response


async def reload_page(page, **kwargs):
    response = await page.reload(**kwargs)
    await maybe_resolve_after_navigation(page)
    return response


def resolve_after_navigation_sync(page) -> bool:
    """Sync Playwright API wrapper for captcha resolution."""
    return asyncio.run(maybe_resolve_after_navigation(page))


def goto_sync(page, url: str, **kwargs):
    response = page.goto(url, **kwargs)
    resolve_after_navigation_sync(page)
    return response
