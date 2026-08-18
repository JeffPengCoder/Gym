"""Playwright DOM extraction and indexed element actions for DOM agents."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_DOM_MAX_CHARS = 40_000


def _truncate(text: str, max_chars: int = DEFAULT_DOM_MAX_CHARS) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 120)] + f"\n... [truncated to {max_chars} of {len(text)} characters]"


def _friendly_tag_name(tag: Any) -> str:
    tag_text = str(tag or "element").lower()
    return {
        "a": "link",
        "button": "button",
        "input": "input",
        "textarea": "text area",
        "select": "dropdown",
        "option": "option",
        "label": "label",
        "summary": "expandable section",
        "img": "image",
        "svg": "icon",
        "div": "element",
        "span": "element",
    }.get(tag_text, tag_text or "element")


async def browser_use_read_page(page: Any, *, max_chars: int = DEFAULT_DOM_MAX_CHARS) -> str:
    """Return a compact visible DOM listing with stable per-observation refs."""

    try:
        await page.bring_to_front()
        try:
            await page.wait_for_selector("body", timeout=5000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass

        payload = await page.evaluate(
            """
() => {
  const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1;
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 1;
  const doc = document.documentElement;
  const body = document.body || doc;
  const scrollY = window.scrollY || doc.scrollTop || body.scrollTop || 0;
  const pageHeight = Math.max(doc.scrollHeight || 0, body.scrollHeight || 0, viewportHeight);
  const skipped = new Set(["html", "head", "body", "script", "style", "meta", "link", "title", "noscript", "template"]);
  const interactiveRoles = new Set(["button", "checkbox", "combobox", "link", "listbox", "menuitem", "option", "radio", "searchbox", "slider", "spinbutton", "switch", "tab", "textbox"]);
  const interactiveTags = new Set(["a", "button", "input", "select", "textarea", "summary"]);

  function compact(value) {
    return String(value || "").replace(/\\s+/g, " ").trim();
  }
  function attr(el, name) {
    return compact(el.getAttribute(name) || "");
  }
  function roleFor(el) {
    const explicit = attr(el, "role");
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "input") {
      const type = attr(el, "type") || "text";
      if (["button", "submit", "reset", "file"].includes(type)) return "button";
      if (type === "checkbox") return "checkbox";
      if (type === "radio") return "radio";
      if (type === "range") return "slider";
      if (type === "search") return "searchbox";
      return "textbox";
    }
    return tag;
  }
  function nameFor(el) {
    const labelledBy = attr(el, "aria-labelledby");
    if (labelledBy) {
      const text = labelledBy.split(/\\s+/).map((id) => document.getElementById(id)?.textContent || "").join(" ");
      if (compact(text)) return compact(text);
    }
    for (const key of ["aria-label", "placeholder", "title", "alt", "value"]) {
      const value = attr(el, key);
      if (value) return value;
    }
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label && compact(label.textContent)) return compact(label.textContent);
    }
    return compact(el.innerText || el.textContent || "");
  }
  function isVisible(el) {
    if (!(el instanceof Element)) return false;
    if (el.hasAttribute("hidden") || attr(el, "aria-hidden") === "true") return false;
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 1 && rect.height > 1 && rect.bottom > 0 && rect.right > 0 && rect.top < viewportHeight && rect.left < viewportWidth;
  }
  function isInteractive(el, role) {
    const tag = el.tagName.toLowerCase();
    return interactiveTags.has(tag) || interactiveRoles.has(role) || el.hasAttribute("onclick") || el.hasAttribute("tabindex") || attr(el, "contenteditable") === "true";
  }
  function attrsFor(el, role) {
    const attrs = [];
    for (const key of ["id", "type", "name", "placeholder", "aria-label", "title", "href", "value", "checked", "disabled"]) {
      const value = attr(el, key);
      if (value) attrs.push(`${key}="${value.slice(0, 100).replace(/"/g, "'")}"`);
    }
    if (role && role !== el.tagName.toLowerCase()) attrs.push(`role="${role}"`);
    return attrs.join(" ");
  }
  function shouldInclude(el, role, name) {
    if (!isVisible(el)) return false;
    if (isInteractive(el, role)) return true;
    const tag = el.tagName.toLowerCase();
    if (["iframe", "frame", "img", "label", "option", "summary", "h1", "h2", "h3", "h4"].includes(tag)) return true;
    return Boolean(name) && !["div", "span", "li", "ul", "ol", "section"].includes(tag);
  }

  const entries = [];
  window.__domAgentElements = [];
  for (const el of document.querySelectorAll("*")) {
    const tag = el.tagName.toLowerCase();
    if (skipped.has(tag)) continue;
    const role = roleFor(el);
    const name = nameFor(el);
    if (!shouldInclude(el, role, name)) continue;
    const rect = el.getBoundingClientRect();
    const index = window.__domAgentElements.length;
    window.__domAgentElements.push(el);
    const prefix = `[${index}]<${tag}`;
    const attrs = attrsFor(el, role);
    const label = name ? ` ${name.slice(0, 140)}` : "";
    entries.push(`${prefix}${attrs ? " " + attrs : ""}/>${label}`);
  }

  const pagesAbove = Math.max(0, scrollY) / Math.max(viewportHeight, 1);
  const pagesBelow = Math.max(0, pageHeight - scrollY - viewportHeight) / Math.max(viewportHeight, 1);
  return {
    url: location.href,
    title: document.title || "",
    pagesAbove,
    pagesBelow,
    entries,
  };
}
"""
        )
        entries = payload.get("entries") or []
        body = "\n".join(entries) if entries else "empty page"
        text = "\n".join(
            [
                f"URL: {payload.get('url', page.url)}",
                f"Title: {payload.get('title', '')}",
                f"<page_info>{payload.get('pagesAbove', 0):.1f} pages above, {payload.get('pagesBelow', 0):.1f} pages below</page_info>",
                "Page elements (use [index] values for click and fill_form):",
                body,
            ]
        )
        return _truncate(text, max_chars=max_chars)
    except Exception as exc:
        return f"[ERROR] browser_use_read_page failed: {type(exc).__name__}: {exc}"


def find_browser_use_dom_lines(dom: str, query: str, max_results: int = 20) -> str:
    query = " ".join(str(query or "").lower().split())
    if not query:
        return "[ERROR] find query must be non-empty."
    limit = max(1, int(max_results or 20))
    matches: list[str] = []
    for line in dom.splitlines():
        searchable = " ".join(line.lower().split())
        if query in searchable:
            matches.append(line)
            if len(matches) >= limit:
                break
    if not matches:
        return (
            f"No current visible/interactable DOM lines matched {query!r}. "
            "If searching for information, use find_page_text because it searches the whole page."
        )
    return f"Found {len(matches)} current DOM match(es) for {query!r}:\n" + "\n".join(matches)


async def _resolve_index(page: Any, index: int, *, scroll_into_view: bool = False) -> dict[str, Any]:
    result = await page.evaluate(
        """
({index, scrollIntoView}) => {
  const elements = window.__domAgentElements || [];
  const el = elements[Number(index)];
  if (!el || !el.isConnected) return {ok: false, error: `No live element for index ${index}. Scroll the target into view if needed, observe again, then retry with a fresh index.`};
  if (scrollIntoView && typeof el.scrollIntoView === "function") {
    el.scrollIntoView({block: "center", inline: "center", behavior: "instant"});
  }
  const rect = el.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0) return {ok: false, error: "Element has no visible bounding box. Scroll the target into view, observe again, then retry with a fresh index."};
  if (rect.bottom <= 0 || rect.right <= 0 || rect.top >= window.innerHeight || rect.left >= window.innerWidth) {
    return {ok: false, error: "Element is not currently visible. Scroll it into view, observe again, then retry with a fresh index."};
  }
  const x = Math.max(0, Math.min(window.innerWidth - 1, rect.left + rect.width / 2));
  const y = Math.max(0, Math.min(window.innerHeight - 1, rect.top + rect.height / 2));
  return {ok: true, x, y, tag: el.tagName.toLowerCase(), text: (el.innerText || el.getAttribute("aria-label") || "").trim().replace(/\\s+/g, " ").slice(0, 120)};
}
""",
        {"index": int(index), "scrollIntoView": bool(scroll_into_view)},
    )
    if not result.get("ok"):
        raise ValueError(str(result.get("error", "Could not resolve DOM index")))
    return result


async def browser_use_click(page: Any, *, index: int, button: str = "left", clicks: int = 1) -> str:
    try:
        button = str(button or "left").lower()
        if button not in {"left", "middle", "right"}:
            raise ValueError("button must be left, middle, or right")
        clicks = max(1, min(int(clicks), 3))
        await page.bring_to_front()
        info = await _resolve_index(page, int(index), scroll_into_view=True)
        await page.mouse.click(float(info["x"]), float(info["y"]), button=button, click_count=clicks)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        return f"Clicked element [{index}] ({_friendly_tag_name(info.get('tag'))})."
    except Exception as exc:
        return (
            f"[ERROR] browser_use_click failed for index {index}: {type(exc).__name__}: {exc} "
            "Scroll the target into view, observe again, then retry with a fresh index."
        )


async def browser_use_fill_form(page: Any, *, index: int, value: str | bool) -> str:
    try:
        await page.bring_to_front()
        await _resolve_index(page, int(index), scroll_into_view=True)
        result = await page.evaluate(
            """
({index, value}) => {
  const el = (window.__domAgentElements || [])[Number(index)];
  if (!el || !el.isConnected) return {ok: false, error: `No live element for index ${index}.`};
  const tag = el.tagName.toLowerCase();
  const type = (el.getAttribute("type") || "").toLowerCase();
  const fire = (name) => el.dispatchEvent(new Event(name, {bubbles: true}));
  if (el.disabled || el.getAttribute("aria-disabled") === "true") return {ok: false, error: "Element is disabled."};
  if (typeof el.focus === "function") el.focus();
  if (tag === "select") {
    const text = String(value ?? "");
    const option = Array.from(el.options || []).find((item) => item.value === text || item.textContent.trim() === text);
    if (!option) return {ok: false, error: `No select option matching "${text}".`};
    el.value = option.value; fire("input"); fire("change");
    return {ok: true, message: `Selected option "${option.textContent.trim()}".`};
  }
  if (type === "checkbox" || type === "radio") {
    el.checked = typeof value === "boolean" ? value : !["false", "0", "off", "no", "unchecked"].includes(String(value).toLowerCase());
    fire("input"); fire("change");
    return {ok: true, message: `Set checked=${el.checked}.`};
  }
  if (el.isContentEditable) {
    el.textContent = String(value ?? ""); fire("input"); fire("change");
    return {ok: true, message: "Set editable text."};
  }
  if ("value" in el) {
    el.value = String(value ?? ""); fire("input"); fire("change");
    return {ok: true, message: `Set value to "${String(value ?? "").slice(0, 120)}".`};
  }
  return {ok: false, error: `Element <${tag}> does not support form input.`};
}
""",
            {"index": int(index), "value": value},
        )
        if not result.get("ok"):
            raise ValueError(str(result.get("error", "fill_form failed")))
        return f"Set element [{index}]: {result.get('message', 'OK')}"
    except Exception as exc:
        return (
            f"[ERROR] browser_use_fill_form failed for index {index}: {type(exc).__name__}: {exc} "
            "Scroll the target into view, observe again, then retry with a fresh index."
        )


async def browser_use_element_center(
    page: Any,
    *,
    index: int,
    scroll_into_view: bool = False,
) -> tuple[int, int]:
    """Resolve a latest-observation visible DOM ref index to viewport pixel center."""

    await page.bring_to_front()
    info = await _resolve_index(page, int(index), scroll_into_view=scroll_into_view)
    return int(round(float(info["x"]))), int(round(float(info["y"])))


def normalize_index(value: Any) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def normalize_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return "about:blank"
    if text == "about:blank" or re.search(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        return text
    return f"https://{text}"


__all__ = [
    "DEFAULT_DOM_MAX_CHARS",
    "browser_use_click",
    "browser_use_element_center",
    "browser_use_fill_form",
    "browser_use_read_page",
    "find_browser_use_dom_lines",
    "normalize_index",
    "normalize_url",
]
