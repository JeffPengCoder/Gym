"""Full-page text reading and search tools for Playwright pages."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


DEFAULT_PART_CHARS = 6000
DEFAULT_MAX_RESULTS = 5
DEFAULT_SEARCH_TARGET_CHARS = 1400


@dataclass(frozen=True)
class PageTextDocument:
    url: str
    title: str
    extractor: str
    markdown: str
    content_hash: str


_PAGE_CACHE: dict[int, PageTextDocument] = {}


def _error(message: str) -> str:
    return f"[ERROR] {message}"


def _normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in markdown.split("\n")]
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines).strip())


async def _body_inner_text(page: Any) -> str:
    try:
        return await page.locator("body").inner_text(timeout=5000)
    except Exception:
        return await page.evaluate("document.body ? document.body.innerText : ''")


async def _extract_markdown(page: Any) -> PageTextDocument:
    title = await page.title()
    url = page.url
    html = await page.content()
    content_hash = hashlib.sha1(f"{url}\0{html}".encode("utf-8", errors="ignore")).hexdigest()[:12]

    try:
        import trafilatura  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        text = await _body_inner_text(page)
        return PageTextDocument(
            url=url,
            title=title,
            extractor="document.body.innerText fallback (trafilatura not installed)",
            markdown=_normalize_markdown(text),
            content_hash=content_hash,
        )

    markdown = trafilatura.extract(html, url=url, favor_precision=True, output_format="markdown")
    if markdown:
        return PageTextDocument(
            url=url,
            title=title,
            extractor="trafilatura",
            markdown=_normalize_markdown(markdown),
            content_hash=content_hash,
        )

    text = await _body_inner_text(page)
    return PageTextDocument(
        url=url,
        title=title,
        extractor="document.body.innerText fallback",
        markdown=_normalize_markdown(text),
        content_hash=content_hash,
    )


async def get_page_text_document(page: Any) -> PageTextDocument:
    cache_key = id(page)
    cached = _PAGE_CACHE.get(cache_key)
    if cached is not None and cached.url == page.url:
        return cached
    document = await _extract_markdown(page)
    _PAGE_CACHE[cache_key] = document
    return document


def _boundary_score(text: str, index: int) -> int:
    if text.startswith("\n# ", index) or text.startswith("\n##", index):
        return 5
    if text.startswith("\n\n", index):
        return 4
    if text.startswith("\n", index):
        return 3
    if text.startswith(". ", index) or text.startswith("? ", index) or text.startswith("! ", index):
        return 2
    return 0


def _best_split_index(text: str, start: int, target_end: int, max_end: int) -> int:
    lower = max(start + 1, start + int((target_end - start) * 0.6))
    upper = min(len(text), max_end)
    if upper <= lower:
        return min(len(text), max(start + 1, target_end))

    best_index = min(target_end, upper)
    best_tuple = (-1, -abs(best_index - target_end))
    for index in range(lower, upper):
        score = _boundary_score(text, index)
        if score <= 0:
            continue
        candidate = (score, -abs(index - target_end))
        if candidate > best_tuple:
            best_tuple = candidate
            best_index = index
    return best_index


def split_markdown_parts(markdown: str, max_chars: int = DEFAULT_PART_CHARS) -> list[str]:
    markdown = markdown.strip()
    if not markdown:
        return [""]
    max_chars = max(500, int(max_chars))
    if len(markdown) <= max_chars:
        return [markdown]

    part_count = max(1, math.ceil(len(markdown) / max_chars))
    target_size = max(500, math.ceil(len(markdown) / part_count))
    parts: list[str] = []
    start = 0
    while start < len(markdown):
        remaining = len(markdown) - start
        if remaining <= max_chars:
            parts.append(markdown[start:].strip())
            break
        target_end = min(len(markdown), start + target_size)
        max_end = min(len(markdown), start + max_chars)
        end = _best_split_index(markdown, start, target_end, max_end)
        if end <= start:
            end = max_end
        parts.append(markdown[start:end].strip())
        start = end
        while start < len(markdown) and markdown[start].isspace():
            start += 1
    return parts or [markdown]


def _header(document: PageTextDocument, *, extra_lines: list[str]) -> str:
    return "\n".join([
        f"URL: {document.url}",
        f"Title: {document.title or '(untitled)'}",
        f"Extractor: {document.extractor}",
        *extra_lines,
        "",
    ])


async def read_page_text(page: Any, *, part: int = 0, max_chars: int = DEFAULT_PART_CHARS) -> str:
    try:
        if part < 0:
            raise ValueError("part must be >= 0")
        if max_chars < 500:
            raise ValueError("max_chars must be >= 500")
        document = await get_page_text_document(page)
        parts = split_markdown_parts(document.markdown, max_chars=max_chars)
        if part >= len(parts):
            return _error(f"part {part} is out of range; available parts: 0-{len(parts) - 1}")
        body = parts[part]
        return _header(
            document,
            extra_lines=[
                f"Part: {part}/{len(parts) - 1}",
                f"Chars: {len(body):,}",
                f"Available parts: 0-{len(parts) - 1}",
            ],
        ) + (body or "[No text extracted]")
    except Exception as exc:
        return _error(f"read_page_text failed: {type(exc).__name__}: {exc}")


def _query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[A-Za-z0-9]+", query.lower()) if term]


def _markdown_search_blocks(markdown: str, target_chars: int = DEFAULT_SEARCH_TARGET_CHARS) -> list[dict[str, Any]]:
    paragraphs: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n", markdown):
        stripped = markdown[start:match.start()].strip()
        if stripped:
            paragraphs.append((start, match.start(), stripped))
        start = match.end()
    tail = markdown[start:].strip()
    if tail:
        paragraphs.append((start, len(markdown), tail))

    blocks: list[dict[str, Any]] = []
    current: list[str] = []
    current_start: int | None = None
    current_end = 0

    def flush() -> None:
        nonlocal current, current_start, current_end
        if current_start is None or not current:
            return
        blocks.append({
            "id": str(len(blocks)),
            "text": "\n\n".join(current),
            "start": current_start,
            "end": current_end,
        })
        current = []
        current_start = None
        current_end = 0

    for paragraph_start, paragraph_end, paragraph in paragraphs:
        current_len = sum(len(item) for item in current) + max(0, len(current) - 1) * 2
        if current and current_len + len(paragraph) + 2 > target_chars:
            flush()
        if current_start is None:
            current_start = paragraph_start
        current.append(paragraph)
        current_end = paragraph_end
    flush()
    return blocks


def _bm25_results(documents: list[dict[str, Any]], query: str, max_results: int) -> list[dict[str, Any]]:
    query_terms = list(dict.fromkeys(_query_terms(query)))
    if not documents or not query_terms:
        return []

    tokenized_documents = [_query_terms(str(document.get("text", ""))) for document in documents]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    average_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    if average_length <= 0:
        return []

    term_frequencies = [Counter(tokens) for tokens in tokenized_documents]
    document_frequencies = {
        term: sum(1 for counts in term_frequencies if counts.get(term, 0) > 0)
        for term in query_terms
    }
    document_count = len(documents)
    scored: list[dict[str, Any]] = []
    for document, counts, document_length in zip(documents, term_frequencies, document_lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            idf = math.log(1 + (document_count - document_frequencies[term] + 0.5) / (document_frequencies[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * (document_length / average_length))
            score += idf * ((frequency * 2.5) / denominator)
        text = str(document.get("text", ""))
        if query.strip().lower() in text.lower():
            score += 1.0
        if score > 0:
            scored.append({**document, "score": score})
    return sorted(scored, key=lambda item: (-float(item["score"]), int(item.get("start") or 0)))[:max_results]


async def search_page_text(
    page: Any,
    *,
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    target_chars: int = DEFAULT_SEARCH_TARGET_CHARS,
) -> str:
    try:
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        if target_chars < 200:
            raise ValueError("target_chars must be >= 200")

        document = await get_page_text_document(page)
        blocks = _markdown_search_blocks(document.markdown, target_chars=target_chars)
        selected = _bm25_results(blocks, query, max_results)
        header = _header(
            document,
            extra_lines=[f'Query: "{query}"', f"Excerpts shown: {len(selected)}"],
        )
        if not selected:
            return header + "No matches found."
        return header + "\n\n".join(
            f"{idx}.\n{str(result.get('text', '')).strip()}"
            for idx, result in enumerate(selected, start=1)
        )
    except Exception as exc:
        return _error(f"search_page_text failed: {type(exc).__name__}: {exc}")


__all__ = [
    "PageTextDocument",
    "get_page_text_document",
    "read_page_text",
    "search_page_text",
    "split_markdown_parts",
]
