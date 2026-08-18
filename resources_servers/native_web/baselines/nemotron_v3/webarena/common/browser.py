"""Playwright browser management: Chrome launch, site login, context setup."""

import asyncio
import json
import logging
import os
from urllib.parse import urlparse

from .cloudflare_handler import (
    apply_storage_state,
    attach_browser_proxy_metadata,
    browser_proxy_config_from_server,
    goto,
    install_captcha_hooks,
)
from .config import DEFAULT_CREDENTIALS, is_webvoyager_task

logger = logging.getLogger(__name__)


def _chrome_args() -> list[str]:
    args = [
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
    return args


CHROME_ARGS = _chrome_args()
PROXY_START_URL_HOSTS = (
    "www.allrecipes.com",
    "www.amazon.com",
    "dictionary.cambridge.org",
)
PROXY_START_URL_PREFIXES = (
    "https://www.google.com/maps/",
)


async def _dismiss_dialog(dialog):
    try:
        await dialog.accept()
    except Exception:
        pass


async def _goto_and_clear_challenge(page, url: str) -> None:
    await goto(page, url, wait_until="domcontentloaded")
    await asyncio.sleep(1)
    await page.bring_to_front()


async def _install_webvoyager_print_hook(context, task_config) -> None:
    if not is_webvoyager_task(task_config):
        return
    if os.environ.get("WA_WEBVOYAGER_INTERCEPT_PRINT", "1") == "0":
        return
    await context.add_init_script(
        """(() => {
            if (window.__webvoyagerPrintHookInstalled) {
                return;
            }
            Object.defineProperty(window, "__webvoyagerPrintHookInstalled", {
                value: true,
                configurable: true,
            });
            Object.defineProperty(window, "__webvoyagerPrintCalled", {
                value: false,
                writable: true,
                configurable: true,
            });
            Object.defineProperty(window, "__webvoyagerPrintCalls", {
                value: [],
                writable: true,
                configurable: true,
            });
            window.print = function() {
                window.__webvoyagerPrintCalled = true;
                window.__webvoyagerPrintCalls.push({
                    url: window.location.href,
                    timestamp: Date.now(),
                });
            };
        })();"""
    )


def _task_uses_browser_proxy(task_config) -> bool:
    if not task_config:
        return False
    candidate_urls = []
    for key in ("start_urls", "start_url", "web"):
        value = task_config.get(key)
        if not value:
            continue
        if isinstance(value, str):
            candidate_urls.extend(part.strip() for part in value.split(" |AND| ") if part.strip())
        else:
            candidate_urls.extend(value)
    for url in candidate_urls:
        if not isinstance(url, str):
            continue
        parsed = urlparse(url)
        if parsed.scheme == "https" and parsed.netloc.lower() in PROXY_START_URL_HOSTS:
            return True
        if any(url.startswith(prefix) for prefix in PROXY_START_URL_PREFIXES):
            return True
    return False


def _browser_proxy_server() -> str | None:
    return os.environ.get("WA_BROWSER_PROXY_SERVER")


def _build_context_kwargs(viewport_width: int, viewport_height: int, har_path, task_config=None) -> dict:
    context_kwargs = {"viewport": {"width": viewport_width, "height": viewport_height}}
    if har_path is not None:
        context_kwargs["record_har_path"] = str(har_path)
    proxy_server = _browser_proxy_server()
    if proxy_server and _task_uses_browser_proxy(task_config):
        proxy_config = browser_proxy_config_from_server(proxy_server)
        if not proxy_config:
            return apply_storage_state(context_kwargs)
        logger.info("Using browser proxy for task context from WA_BROWSER_PROXY_SERVER")
        context_kwargs["proxy"] = proxy_config
    elif proxy_server:
        logger.info("Browser proxy configured but not selected for this task")
    return apply_storage_state(context_kwargs)


async def login_site(site: str, url: str, context):
    """Log into a WebArena site using default credentials."""
    creds = DEFAULT_CREDENTIALS.get(site)
    if not creds:
        return
    login_page = await context.new_page()
    try:
        if site == "reddit":
            await goto(login_page, url)
            await login_page.get_by_role("link", name="Log in").click()
            await login_page.get_by_label("Username").fill(creds["username"])
            await login_page.get_by_label("Password").fill(creds["password"])
            await login_page.get_by_role("button", name="Log in").click()
        elif site == "gitlab":
            await goto(login_page, f"{url}/users/sign_in")
            await login_page.get_by_label("Username or email").fill(creds["username"])
            await login_page.get_by_label("Password").fill(creds["password"])
            await login_page.get_by_role("button", name="Sign in").click()
        elif site == "shopping":
            await goto(login_page, f"{url}/customer/account/login/")
            await login_page.get_by_label("Email", exact=True).fill(creds["username"])
            await login_page.get_by_label("Password", exact=True).fill(creds["password"])
            await login_page.get_by_role("button", name="Sign In").click()
        elif site == "shopping_admin":
            await goto(login_page, url)
            await login_page.get_by_label("Username").fill(creds["username"])
            await login_page.get_by_label("Password").fill(creds["password"])
            await login_page.get_by_role("button", name="Sign in").click()
        elif site == "classifieds":
            await goto(login_page, f"{url}/index.php?page=login")
            await login_page.locator("#email").fill(creds["username"])
            await login_page.locator("#password").fill(creds["password"])
            await login_page.get_by_role("button", name="Log in").click()
        elif site in ("wikipedia", "map"):
            await goto(login_page, url)
        await asyncio.sleep(2)
    finally:
        await login_page.close()


async def login_sites(sites: list[str], urls: dict[str, str], context):
    for site in sites:
        if site not in urls:
            logger.info("No configured WebArena URL for site %r; skipping login", site)
            continue
        for attempt in range(3):
            try:
                await login_site(site, urls[site], context)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Login to {site} failed (attempt {attempt+1}/3): {e}")
                await asyncio.sleep(2 ** attempt)


async def setup_browser_and_login(task_config, urls, pw, browser, viewport_width, viewport_height, har_path):
    """Create a browser context, login to sites, and navigate to start URLs.

    Returns (context, page). The caller is responsible for closing them.
    """
    context_kwargs = _build_context_kwargs(viewport_width, viewport_height, har_path, task_config)
    context = await browser.new_context(**context_kwargs)
    attach_browser_proxy_metadata(context, context_kwargs.get("proxy"))
    timeout_ms = int(os.environ.get("PW_DEFAULT_TIMEOUT_MS", "45000"))
    context.set_default_timeout(timeout_ms)
    context.set_default_navigation_timeout(timeout_ms)
    await install_captcha_hooks(context)
    await _install_webvoyager_print_hook(context, task_config)

    if os.environ.get("PW_EXTRA_HEADERS"):
        try:
            with open(os.environ["PW_EXTRA_HEADERS"]) as f:
                extra_headers = json.load(f)
            await context.set_extra_http_headers(extra_headers)
        except Exception as e:
            logger.warning(f"Failed to load extra headers: {e}")

    page = await context.new_page()
    page.on("dialog", lambda d: asyncio.ensure_future(_dismiss_dialog(d)))

    await login_sites(task_config["sites"], urls, context)

    if start_urls := task_config.get("start_urls"):
        for idx, url in enumerate(start_urls):
            if idx > 0:
                page = await context.new_page()
                page.on("dialog", lambda d: asyncio.ensure_future(_dismiss_dialog(d)))
            for attempt in range(3):
                try:
                    await _goto_and_clear_challenge(page, url)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
        await asyncio.sleep(2)

    return context, page


async def setup_task_context(browser, task_config, urls, viewport_width, viewport_height, har_path):
    """Create a fresh browser context for a single task (used by parallel runner).

    Returns (context, page). Cleans up context on failure.
    """
    context = None
    try:
        context_kwargs = _build_context_kwargs(viewport_width, viewport_height, har_path, task_config)
        context = await browser.new_context(**context_kwargs)
        attach_browser_proxy_metadata(context, context_kwargs.get("proxy"))
        timeout_ms = int(os.environ.get("PW_DEFAULT_TIMEOUT_MS", "45000"))
        context.set_default_timeout(timeout_ms)
        context.set_default_navigation_timeout(timeout_ms)
        await install_captcha_hooks(context)
        await _install_webvoyager_print_hook(context, task_config)

        if os.environ.get("PW_EXTRA_HEADERS"):
            try:
                with open(os.environ["PW_EXTRA_HEADERS"]) as f:
                    extra_headers = json.load(f)
                await context.set_extra_http_headers(extra_headers)
            except Exception:
                pass

        page = await context.new_page()
        page.on("dialog", lambda d: asyncio.ensure_future(_dismiss_dialog(d)))

        await login_sites(task_config["sites"], urls, context)

        if start_urls := task_config.get("start_urls"):
            for idx, url in enumerate(start_urls):
                if idx > 0:
                    page = await context.new_page()
                    page.on("dialog", lambda d: asyncio.ensure_future(_dismiss_dialog(d)))
                for attempt in range(3):
                    try:
                        await _goto_and_clear_challenge(page, url)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 ** attempt)
            await asyncio.sleep(2)

        return context, page

    except Exception:
        if context:
            try:
                await context.close()
            except Exception:
                pass
        raise
