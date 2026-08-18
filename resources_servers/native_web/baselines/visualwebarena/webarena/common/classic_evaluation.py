"""Classic WebArena JSONL evaluation helpers.

This mirrors the original WebArena evaluator behavior for the benchmark format
with eval_types such as string_match, url_match, and program_html.
"""

from __future__ import annotations

import collections
import asyncio
import html
import inspect
import logging
import os
import re
import urllib.parse
import json
from pathlib import Path
from typing import Any

import httpx

from .cloudflare_handler import goto, resolve_after_navigation_sync
from .config import DEFAULT_CREDENTIALS
from .eval_collision import extract_helper_calls, snapshot_target_key

logger = logging.getLogger(__name__)


def _clean_answer(answer: Any) -> str:
    text = "" if answer is None else str(answer)
    text = text.strip()
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        text = text[1:-1]
    return text.lower()


def _exact_match(ref: str, pred: str) -> float:
    return float(_clean_answer(pred) == _clean_answer(ref))


def _must_include(ref: str, pred: str, tokenize: bool = False) -> float:
    clean_ref = _clean_answer(ref)
    clean_pred = _clean_answer(pred)
    if (
        tokenize
        and len(clean_ref) == 1
        and len(_word_tokenize_like_webarena(clean_ref)) == 1
    ):
        return float(clean_ref in _word_tokenize_like_webarena(clean_pred))
    return float(clean_ref in clean_pred)


def _reference_alternatives(ref: Any) -> list[str]:
    if isinstance(ref, list):
        return [str(item) for item in ref]
    return [str(ref)]


def _format_required_group(alternatives: list[str]) -> str:
    if len(alternatives) == 1:
        return alternatives[0]
    return "one of: " + " or ".join(alternatives)


def _preview_text(text: Any, max_len: int = 300) -> str:
    preview = str(text).replace("\n", "\\n")
    if len(preview) > max_len:
        return preview[:max_len] + "..."
    return preview


def _word_tokenize_like_webarena(text: str) -> list[str]:
    """Small tokenizer for the original single-character must_include guard."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _append_judge_log(
    judge_log_path: Path | None,
    *,
    judge_type: str,
    question: str,
    reference: str,
    prediction: str,
    messages: list[dict[str, str]],
    response: str,
) -> None:
    if judge_log_path is None:
        return
    judge_log_path.parent.mkdir(parents=True, exist_ok=True)
    with judge_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "judge_type": judge_type,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "messages": messages,
            "response": response,
        }, ensure_ascii=True))
        f.write("\n")


def _judge_chat(messages: list[dict[str, str]]) -> str:
    api_key = os.environ.get("WEBARENA_JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError("WEBARENA_JUDGE_API_KEY is required for fuzzy_match evaluation")

    model = os.environ.get("WEBARENA_JUDGE_MODEL", "gpt-4-1106-preview")
    base_url = os.environ.get("WEBARENA_JUDGE_BASE_URL", "https://inference-api.nvidia.com/v1").rstrip("/")
    timeout = float(os.environ.get("WEBARENA_JUDGE_TIMEOUT", "120"))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 768,
        "top_p": 1.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def _llm_fuzzy_match(
    pred: str,
    reference: str,
    question: str,
    judge_log_path: Path | None = None,
    judge_type: str = "fuzzy_match",
) -> float:
    message = (
        "Help a teacher to grade the answer of a student given a question. "
        "Keep in mind that the student may use different phrasing or wording "
        "to answer the question. The goal is to evaluate whether the answer is "
        "semantically equivalent to the reference answer.\n"
        f"question: {question}\n"
        f"reference answer: {reference}\n"
        "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
        f"student answer: {pred}\n"
        "Conclude the judgement by correct/incorrect/partially correct."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]
    response_raw = _judge_chat(messages)
    _append_judge_log(
        judge_log_path,
        judge_type=judge_type,
        question=question,
        reference=reference,
        prediction=pred,
        messages=messages,
        response=response_raw,
    )
    response = response_raw.lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    if "correct" not in response:
        raise RuntimeError(f"Unexpected fuzzy judge response: {response}")
    return 1.0


def _llm_must_include(
    pred: str,
    reference: str,
    question: str,
    judge_log_path: Path | None = None,
) -> float:
    message = (
        "Help a teacher to grade whether a student's answer is correct for a web "
        "task. The student may use different phrasing or wording. The answer is "
        "correct if it semantically contains every required item below. For a "
        "required line that starts with 'one of:', the answer only needs to "
        "contain one option from that line.\n"
        f"question: {question}\n"
        f"required information:\n{reference}\n"
        f"student answer: {pred}\n"
        "Conclude the judgement by correct/incorrect/partially correct. "
        "Use correct only if every required line is satisfied."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]
    response_raw = _judge_chat(messages)
    _append_judge_log(
        judge_log_path,
        judge_type="must_include_fallback",
        question=question,
        reference=reference,
        prediction=pred,
        messages=messages,
        response=response_raw,
    )
    response = response_raw.lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    if "correct" not in response:
        raise RuntimeError(f"Unexpected must_include judge response: {response}")
    return 1.0


def _llm_ua_match(
    pred: str,
    reference: str,
    question: str,
    judge_log_path: Path | None = None,
) -> float:
    message = (
        f"task: {question}\n"
        f"actual unachievable reason: {reference}\n"
        f"reported unachievable reason: {pred}\n"
        "The task described above is inherently unachievable due to the reason "
        "specified under 'actual unachievable reason'. An individual previously "
        "attempted this task and was unable to complete it. They provided a "
        "reason for their failure, which is listed under 'reported unachievable "
        "reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if "
        "implicitly. If the stated reason is in line with the actual reason, "
        "respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]
    response_raw = _judge_chat(messages)
    _append_judge_log(
        judge_log_path,
        judge_type="unachievable_match",
        question=question,
        reference=reference,
        prediction=pred,
        messages=messages,
        response=response_raw,
    )
    response = response_raw.lower()
    if "different" in response:
        return 0.0
    if "same" not in response:
        raise RuntimeError(f"Unexpected unachievable judge response: {response}")
    return 1.0


def _string_match(task_config: dict, answer: Any, judge_log_path: Path | None = None) -> float:
    pred = _clean_answer(answer)
    refs = task_config["eval"].get("reference_answers") or {}
    score = 1.0
    for approach, value in refs.items():
        if approach == "exact_match":
            alternatives = _reference_alternatives(value)
            cur_score = max(_exact_match(ref=alt, pred=pred) for alt in alternatives)
            if cur_score != 1.0:
                cur_score = _llm_fuzzy_match(
                    pred=pred,
                    reference=_format_required_group(alternatives),
                    question=task_config["intent"],
                    judge_log_path=judge_log_path,
                    judge_type="exact_match_fallback",
                )
            score *= cur_score
        elif approach == "must_include":
            required_groups = []
            rule_score = 1.0
            for must_value in value:
                alternatives = _reference_alternatives(must_value)
                required_groups.append(alternatives)
                rule_score *= max(
                    _must_include(ref=alt, pred=pred, tokenize=(len(value) == 1))
                    for alt in alternatives
                )
            if rule_score == 1.0:
                score *= rule_score
            else:
                reference = "\n".join(
                    f"{idx}. {_format_required_group(alternatives)}"
                    for idx, alternatives in enumerate(required_groups, start=1)
                )
                score *= _llm_must_include(
                    pred=pred,
                    reference=reference,
                    question=task_config["intent"],
                    judge_log_path=judge_log_path,
                )
        elif approach == "fuzzy_match":
            intent = task_config["intent"]
            if value == "N/A":
                score *= _exact_match(ref=value, pred=pred)
                if score != 1:
                    score = _llm_ua_match(
                        pred=pred,
                        reference=task_config["eval"].get("string_note", ""),
                        question=intent,
                        judge_log_path=judge_log_path,
                    )
            else:
                for reference in value:
                    score *= _llm_fuzzy_match(
                        pred=pred,
                        reference=reference,
                        question=intent,
                        judge_log_path=judge_log_path,
                    )
        else:
            raise ValueError(f"Unknown string_match approach: {approach}")
    return score


def _clean_url(url: str) -> str:
    return str(url).rstrip("/")


def _parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
    parsed_url = urllib.parse.urlparse(url)
    base_path = parsed_url.netloc + parsed_url.path
    query = urllib.parse.parse_qs(parsed_url.query)
    return base_path, query


def _parse_urls(urls: list[str]) -> tuple[list[str], dict[str, set[str]]]:
    base_paths = []
    queries = collections.defaultdict(set)
    for url in urls:
        base_path, query = _parse_url(url)
        base_paths.append(base_path)
        for key, values in query.items():
            queries[key].update(values)
    return base_paths, queries


async def _get_live_page_url(page) -> str:
    try:
        live_url = await page.evaluate("window.location.href")
        return live_url or page.url
    except Exception as e:
        logger.debug("Failed to read window.location.href: %s", e)
        return page.url


def _get_live_page_url_sync(page) -> str:
    try:
        live_url = page.evaluate("window.location.href")
        return live_url or page.url
    except Exception as e:
        logger.debug("Failed to read window.location.href: %s", e)
        return page.url


def _url_match(task_config: dict, current_url: str) -> float:
    pred = _clean_url(current_url)
    reference_url = task_config["eval"].get("reference_url") or ""
    ref_urls = [_clean_url(url) for url in reference_url.split(" |OR| ") if url]
    if not ref_urls:
        return 0.0

    matching_rule = task_config["eval"].get("url_note", "GOLD in PRED")
    if matching_rule != "GOLD in PRED":
        raise ValueError(f"Unknown URL matching rule: {matching_rule}")

    ref_base_paths, ref_queries = _parse_urls(ref_urls)
    pred_base_path, pred_query = _parse_url(pred)
    base_score = float(any(ref_base_path in pred_base_path for ref_base_path in ref_base_paths))

    query_score = 1.0
    for key, possible_values in ref_queries.items():
        key_score = float(
            any(possible_ref_value in pred_query.get(key, []) for possible_ref_value in possible_values)
        )
        query_score *= key_score
    score = base_score * query_score
    return score


def _site_url(site: str) -> str:
    env_name = {
        "shopping": "WA_SHOPPING",
        "shopping_admin": "WA_SHOPPING_ADMIN",
        "reddit": "WA_REDDIT",
        "gitlab": "WA_GITLAB",
        "wikipedia": "WA_WIKIPEDIA",
        "map": "WA_MAP",
    }[site]
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is required for classic WebArena helper evaluation")
    return value.rstrip("/")


def _shopping_get_auth_token() -> str:
    creds = DEFAULT_CREDENTIALS["shopping_admin"]
    response = httpx.post(
        f"{_site_url('shopping')}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        json={"username": creds["username"], "password": creds["password"]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _shopping_admin_api_base_url() -> str:
    admin_url = _site_url("shopping_admin")
    return admin_url.removesuffix("/admin")


def _shopping_admin_get_auth_token() -> str:
    creds = DEFAULT_CREDENTIALS["shopping_admin"]
    response = httpx.post(
        f"{_shopping_admin_api_base_url()}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        json={"username": creds["username"], "password": creds["password"]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _normalize_number_string(value: Any) -> str:
    text = str(value or "")
    if "." in text:
        return text.rstrip("0").rstrip(".")
    return text


def shopping_get_latest_order_url() -> str:
    """Get the latest order URL from the shopping website."""
    headers = {
        "Authorization": f"Bearer {_shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }
    response = httpx.get(
        f"{_site_url('shopping')}/rest/V1/orders",
        params=params,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    order_id = int(response.json()["items"][0]["increment_id"])
    return f"{_site_url('shopping')}/sales/order/view/order_id/{order_id}/"


def shopping_admin_get_cart_price_rule(rule_name: str) -> str:
    """Return normalized cart price rule fields for a saved Magento sales rule."""
    headers = {
        "Authorization": f"Bearer {_shopping_admin_get_auth_token()}",
        "Content-Type": "application/json",
    }
    params = {
        "searchCriteria[filter_groups][0][filters][0][field]": "name",
        "searchCriteria[filter_groups][0][filters][0][value]": rule_name,
        "searchCriteria[filter_groups][0][filters][0][condition_type]": "eq",
        "searchCriteria[pageSize]": "10",
    }
    response = httpx.get(
        f"{_shopping_admin_api_base_url()}/rest/V1/salesRules/search",
        params=params,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    if not items:
        response = httpx.get(
            f"{_shopping_admin_api_base_url()}/rest/V1/salesRules/search",
            params={"searchCriteria[pageSize]": "100"},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
    if not items:
        logger.info("shopping_admin_get_cart_price_rule: no cart price rules found")
        return ""
    rule = next(
        (item for item in items if str(item.get("name", "")).lower() == rule_name.lower()),
        None,
    )
    if rule is None:
        logger.info(
            "shopping_admin_get_cart_price_rule: no rule named %r; available rules=%s",
            rule_name,
            [item.get("name") for item in items],
        )
        return ""
    normalized = {
        "name": rule.get("name"),
        "customer_group_ids": rule.get("customer_group_ids"),
        "simple_action": rule.get("simple_action"),
        "discount_amount": _normalize_number_string(rule.get("discount_amount")),
    }
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True)


def shopping_get_sku_latest_review_author(sku: str) -> str:
    headers = {
        "Authorization": f"Bearer {_shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = httpx.get(
        f"{_site_url('shopping')}/rest/V1/products/{sku}/reviews",
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    reviews = response.json()
    if not reviews:
        return ""
    return str(reviews[-1]["nickname"])


def shopping_get_sku_latest_review_rating(sku: str) -> str:
    headers = {
        "Authorization": f"Bearer {_shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = httpx.get(
        f"{_site_url('shopping')}/rest/V1/products/{sku}/reviews",
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    reviews = response.json()
    if not reviews:
        return ""
    return str(reviews[-1]["ratings"][0]["percent"])


def reddit_get_post_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    tok_url = parsed.path.split("/")
    if len(tok_url) < 4 or tok_url[1] != "f":
        return url
    subreddit = tok_url[2]
    post_id = tok_url[3]
    return f"{parsed.scheme}://{parsed.netloc}/f/{subreddit}/{post_id}/"


async def gitlab_get_project_memeber_role(page, account_name: str) -> str:
    try:
        account_idx = await page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )
        return str(await page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        ))
    except Exception:
        return ""


def gitlab_get_project_memeber_role_sync(page, account_name: str) -> str:
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )
        return str(page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        ))
    except Exception:
        return ""


async def _eval_helper_expression(expr: str, page) -> Any:
    helper_expr = expr.split("func:", 1)[1] if expr.startswith("func:") else expr
    helper_expr = helper_expr.replace("__last_url__", page.url)
    allowed = {
        "shopping_get_latest_order_url": shopping_get_latest_order_url,
        "shopping_admin_get_cart_price_rule": shopping_admin_get_cart_price_rule,
        "shopping_get_sku_latest_review_author": shopping_get_sku_latest_review_author,
        "shopping_get_sku_latest_review_rating": shopping_get_sku_latest_review_rating,
        "reddit_get_post_url": reddit_get_post_url,
        "gitlab_get_project_memeber_role": gitlab_get_project_memeber_role,
        "__page__": page,
    }
    try:
        result = eval(helper_expr, {"__builtins__": {}}, allowed)
        if inspect.isawaitable(result):
            return await result
        return result
    except Exception as e:
        raise RuntimeError(f"program_html helper failed for {expr}: {e}") from e


def _eval_helper_expression_sync(expr: str, page) -> Any:
    helper_expr = expr.split("func:", 1)[1] if expr.startswith("func:") else expr
    helper_expr = helper_expr.replace("__last_url__", page.url)
    allowed = {
        "shopping_get_latest_order_url": shopping_get_latest_order_url,
        "shopping_admin_get_cart_price_rule": shopping_admin_get_cart_price_rule,
        "shopping_get_sku_latest_review_author": shopping_get_sku_latest_review_author,
        "shopping_get_sku_latest_review_rating": shopping_get_sku_latest_review_rating,
        "reddit_get_post_url": reddit_get_post_url,
        "gitlab_get_project_memeber_role": gitlab_get_project_memeber_role_sync,
        "__page__": page,
    }
    try:
        return eval(helper_expr, {"__builtins__": {}}, allowed)
    except Exception as e:
        raise RuntimeError(f"program_html helper failed for {expr}: {e}") from e


def _score_program_html_required(required: dict[str, Any], selected_element: Any) -> float:
        score = 1.0
        selected_element = html.unescape(str(selected_element))
        if "exact_match" in required:
            score *= _exact_match(ref=required["exact_match"], pred=selected_element)
        elif "must_include" in required:
            for content in required["must_include"]:
                content_or = content.split(" |OR| ")
                score *= float(
                    any(_must_include(ref=part, pred=selected_element) for part in content_or)
                )
        else:
            raise ValueError(f"Unknown required_contents: {required.keys()}")
        return score


def _order_delta_urls(eval_context: dict[str, Any] | None) -> list[str]:
    if not eval_context:
        return []
    order_delta = (eval_context.get("deltas") or {}).get("shopping_orders") or {}
    urls = []
    for bucket in ("added", "changed"):
        for record in order_delta.get(bucket, []):
            url = record.get("url")
            if url and url not in urls:
                urls.append(str(url))
    return urls


def _review_delta_values(
    expr: Any,
    eval_context: dict[str, Any] | None,
) -> list[str]:
    if not eval_context:
        return []
    calls = extract_helper_calls(expr)
    if not calls:
        return []
    call = calls[0]
    helper_name = call["name"]
    if helper_name not in {
        "shopping_get_sku_latest_review_author",
        "shopping_get_sku_latest_review_rating",
    }:
        return []
    args = call.get("args") or []
    if not args or args[0] is None:
        return []
    sku = str(args[0])
    sku_delta = ((eval_context.get("deltas") or {}).get("shopping_reviews") or {}).get(sku, {})
    values = []
    for bucket in ("added", "changed"):
        for review in sku_delta.get(bucket, []):
            if helper_name == "shopping_get_sku_latest_review_author":
                value = review.get("nickname")
            else:
                value = review.get("rating_percent")
            if value not in (None, ""):
                values.append(str(value))
    return values


def _program_html_snapshot_values(
    target: dict[str, Any],
    eval_context: dict[str, Any] | None,
) -> list[str]:
    if not eval_context:
        return []
    key = snapshot_target_key(target)
    program_delta = (eval_context.get("deltas") or {}).get("program_html") or {}
    values = []
    for bucket in ("added", "changed"):
        for record in program_delta.get(bucket, []):
            if record.get("key") != key:
                continue
            delta_value = str(record.get("delta_value") or "").strip()
            full_value = str(record.get("value") or "")
            if delta_value:
                values.append(delta_value)
            elif full_value:
                values.append(full_value)
    return values


async def _program_html_target(target: dict, page, eval_context: dict[str, Any] | None = None) -> float:
        score = 1.0
        target_url = target["url"]
        original_target_url = target_url
        logger.info("program_html target start: url=%s page_url=%s locator=%s",
                    original_target_url, page.url, target.get("locator"))
        snapshot_values = _program_html_snapshot_values(target, eval_context)
        if snapshot_values:
            target_score = max(
                _score_program_html_required(target["required_contents"], value)
                for value in snapshot_values
            )
            logger.info("program_html snapshot-delta score=%s", target_score)
            return target_score

        if any(call["name"] == "shopping_get_latest_order_url" for call in extract_helper_calls(target_url)):
            candidate_urls = _order_delta_urls(eval_context)
            if candidate_urls:
                target_score = 0.0
                for candidate_url in candidate_urls:
                    candidate_target = {**target, "url": candidate_url}
                    target_score = max(
                        target_score,
                        await _program_html_target(candidate_target, page, None),
                    )
                logger.info("program_html order-delta candidate score=%s", target_score)
                return target_score

        review_values = _review_delta_values(target.get("locator"), eval_context)
        if review_values:
            return max(
                _score_program_html_required(target["required_contents"], value)
                for value in review_values
            )

        if isinstance(target_url, str) and target_url.startswith("func:"):
            target_url = await _eval_helper_expression(target_url, page)
            logger.info("program_html resolved func url: %s -> %s", original_target_url, target_url)
        if target_url != "last":
            try:
                await goto(page, target_url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                raise RuntimeError(f"program_html navigation failed for {target_url}: {e}") from e
            await asyncio.sleep(3)
            logger.info("program_html navigated: page_url=%s", page.url)

        locator = target["locator"]
        if not locator.strip():
            selected_element = await page.content()
        elif locator.startswith("document.") or locator.startswith("[...document."):
            if "prep_actions" in target:
                try:
                    for prep_action in target["prep_actions"]:
                        await page.evaluate(f"() => {prep_action}")
                except Exception:
                    logger.debug("Classic eval prep_actions failed", exc_info=True)
            try:
                selected_element = str(await page.evaluate(f"() => {locator}") or "")
            except Exception:
                selected_element = ""
        elif locator.startswith("func:"):
            selected_element = str(await _eval_helper_expression(locator, page))
        else:
            raise ValueError(f"Unknown program_html locator: {locator}")

        selected_element = html.unescape(selected_element)
        required = target["required_contents"]
        logger.info("program_html extracted: page_url=%s locator=%s value=%s required=%s",
                    page.url, locator, _preview_text(selected_element), required)
        score *= _score_program_html_required(required, selected_element)
        logger.info("program_html target score: url=%s locator=%s score=%s",
                    original_target_url, locator, score)
        return score


async def _program_html(task_config: dict, page, eval_context: dict[str, Any] | None = None) -> float:
    score = 1.0
    for target in task_config["eval"].get("program_html") or []:
        if target["url"] == "last":
            pages = list(page.context.pages)
            if not pages:
                pages = [page]
            target_score = 0.0
            logger.info("program_html checking %d open tab(s) for locator=%s",
                        len(pages), target.get("locator"))
            for idx, candidate in enumerate(pages):
                try:
                    candidate_score = await _program_html_target(target, candidate, eval_context)
                    logger.info("program_html tab %d score=%s url=%s",
                                idx, candidate_score, candidate.url)
                    target_score = max(target_score, candidate_score)
                except Exception:
                    logger.info("program_html tab %d check failed url=%s",
                                idx, candidate.url, exc_info=True)
            score *= target_score
        else:
            score *= await _program_html_target(target, page, eval_context)
    return score


def _program_html_target_sync(target: dict, page, eval_context: dict[str, Any] | None = None) -> float:
        score = 1.0
        target_url = target["url"]
        original_target_url = target_url
        logger.info("program_html target start: url=%s page_url=%s locator=%s",
                    original_target_url, page.url, target.get("locator"))
        snapshot_values = _program_html_snapshot_values(target, eval_context)
        if snapshot_values:
            target_score = max(
                _score_program_html_required(target["required_contents"], value)
                for value in snapshot_values
            )
            logger.info("program_html snapshot-delta score=%s", target_score)
            return target_score

        if any(call["name"] == "shopping_get_latest_order_url" for call in extract_helper_calls(target_url)):
            candidate_urls = _order_delta_urls(eval_context)
            if candidate_urls:
                target_score = 0.0
                for candidate_url in candidate_urls:
                    candidate_target = {**target, "url": candidate_url}
                    target_score = max(
                        target_score,
                        _program_html_target_sync(candidate_target, page, None),
                    )
                logger.info("program_html order-delta candidate score=%s", target_score)
                return target_score

        review_values = _review_delta_values(target.get("locator"), eval_context)
        if review_values:
            return max(
                _score_program_html_required(target["required_contents"], value)
                for value in review_values
            )

        if isinstance(target_url, str) and target_url.startswith("func:"):
            target_url = _eval_helper_expression_sync(target_url, page)
            logger.info("program_html resolved func url: %s -> %s", original_target_url, target_url)
        if target_url != "last":
            try:
                page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                resolve_after_navigation_sync(page)
            except Exception as e:
                raise RuntimeError(f"program_html navigation failed for {target_url}: {e}") from e
            import time
            time.sleep(3)
            logger.info("program_html navigated: page_url=%s", page.url)

        locator = target["locator"]
        if not locator.strip():
            selected_element = page.content()
        elif locator.startswith("document.") or locator.startswith("[...document."):
            if "prep_actions" in target:
                try:
                    for prep_action in target["prep_actions"]:
                        page.evaluate(f"() => {prep_action}")
                except Exception:
                    logger.debug("Classic eval prep_actions failed", exc_info=True)
            try:
                selected_element = str(page.evaluate(f"() => {locator}") or "")
            except Exception:
                selected_element = ""
        elif locator.startswith("func:"):
            selected_element = str(_eval_helper_expression_sync(locator, page))
        else:
            raise ValueError(f"Unknown program_html locator: {locator}")

        selected_element = html.unescape(selected_element)
        required = target["required_contents"]
        logger.info("program_html extracted: page_url=%s locator=%s value=%s required=%s",
                    page.url, locator, _preview_text(selected_element), required)
        score *= _score_program_html_required(required, selected_element)
        logger.info("program_html target score: url=%s locator=%s score=%s",
                    original_target_url, locator, score)
        return score


def _program_html_sync(task_config: dict, page, eval_context: dict[str, Any] | None = None) -> float:
    score = 1.0
    for target in task_config["eval"].get("program_html") or []:
        if target["url"] == "last":
            pages = list(page.context.pages)
            if not pages:
                pages = [page]
            target_score = 0.0
            logger.info("program_html checking %d open tab(s) for locator=%s",
                        len(pages), target.get("locator"))
            for idx, candidate in enumerate(pages):
                try:
                    candidate_score = _program_html_target_sync(target, candidate, eval_context)
                    logger.info("program_html tab %d score=%s url=%s",
                                idx, candidate_score, candidate.url)
                    target_score = max(target_score, candidate_score)
                except Exception:
                    logger.info("program_html tab %d check failed url=%s",
                                idx, candidate.url, exc_info=True)
            score *= target_score
        else:
            score *= _program_html_target_sync(target, page, eval_context)
    return score


async def evaluate_classic_task(
    task_config: dict,
    agent_result: dict,
    page,
    judge_log_path: Path | None = None,
    eval_context: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Evaluate a classic WebArena JSONL task against the live final page."""
    try:
        eval_types = task_config["eval"]["eval_types"]
        messages = []
        score = 1.0
        for eval_type in eval_types:
            if eval_type == "string_match":
                cur_score = _string_match(task_config, agent_result.get("answer"), judge_log_path)
            elif eval_type == "url_match":
                cur_score = _url_match(task_config, await _get_live_page_url(page))
            elif eval_type == "program_html":
                cur_score = await _program_html(task_config, page, eval_context)
            else:
                raise ValueError(f"eval_type {eval_type} is not supported")
            score *= cur_score
            messages.append(f"{eval_type}: score={cur_score}")
        return score, "; ".join(messages)
    except Exception as e:
        return 0.0, f"Classic evaluation error: {e}"


def evaluate_classic_task_sync(
    task_config: dict,
    agent_result: dict,
    page,
    judge_log_path: Path | None = None,
    eval_context: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Synchronous variant for the human runner's sync Playwright page."""
    try:
        eval_types = task_config["eval"]["eval_types"]
        messages = []
        score = 1.0
        for eval_type in eval_types:
            if eval_type == "string_match":
                cur_score = _string_match(task_config, agent_result.get("answer"), judge_log_path)
            elif eval_type == "url_match":
                cur_score = _url_match(task_config, _get_live_page_url_sync(page))
            elif eval_type == "program_html":
                cur_score = _program_html_sync(task_config, page, eval_context)
            else:
                raise ValueError(f"eval_type {eval_type} is not supported")
            score *= cur_score
            messages.append(f"{eval_type}: score={cur_score}")
        return score, "; ".join(messages)
    except Exception as e:
        return 0.0, f"Classic evaluation error: {e}"
