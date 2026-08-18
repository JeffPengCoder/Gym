"""WebArena site configuration, task loading, and task utilities."""

import importlib.resources
import json
import logging
import os
import re
import sys
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

URL_PLACEHOLDERS = {
    "__GITLAB__": "gitlab",
    "__REDDIT__": "reddit",
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__WIKIPEDIA__": "wikipedia",
    "__MAP__": "map",
    "__CLASSIFIEDS__": "classifieds",
}

DERIVED_URL_PLACEHOLDERS = {
    "__GITLAB_SSH__": "gitlab_ssh",
}

REQUIRED_ENV_VARS = {
    "shopping": "WA_SHOPPING",
    "shopping_admin": "WA_SHOPPING_ADMIN",
    "reddit": "WA_REDDIT",
    "gitlab": "WA_GITLAB",
    "wikipedia": "WA_WIKIPEDIA",
    "map": "WA_MAP",
}

OPTIONAL_ENV_VARS = {
    "classifieds": "WA_CLASSIFIEDS",
}

DEFAULT_CREDENTIALS = {
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "classifieds": {"username": "blake.sullivan@gmail.com", "password": "Password.123"},
}


def get_urls(required: bool = True) -> dict[str, str]:
    urls = {}
    missing = []
    for site, var in REQUIRED_ENV_VARS.items():
        val = os.environ.get(var)
        if not val:
            missing.append(var)
        else:
            urls[site] = val
    for site, var in OPTIONAL_ENV_VARS.items():
        val = os.environ.get(var)
        if val:
            urls[site] = val
    if "gitlab" in urls:
        urls["gitlab_ssh"] = _gitlab_ssh_authority(urls["gitlab"])
    if missing and required:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        for var in missing:
            print(f'  export {var}="http://<your-host>:<port>"')
        sys.exit(1)
    return urls


def _gitlab_ssh_authority(gitlab_url: str) -> str:
    parsed = urllib.parse.urlparse(gitlab_url)
    if not parsed.netloc:
        parsed = urllib.parse.urlparse(f"//{gitlab_url}")
    host = parsed.hostname or gitlab_url.split("/", 1)[0].split(":", 1)[0]
    return f"{host}:2222"


def load_all_tasks(urls: dict[str, str], dataset_dir: Path | None = None) -> list[dict]:
    """Load and template-substitute all WebArena-Verified tasks.

    Args:
        urls: Site URL mapping.
        dataset_dir: Directory containing webarena-verified.json.
                     Defaults to webarena/ project root.
    """
    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parent.parent

    dataset_path = dataset_dir / "webarena-verified.json"
    if dataset_path.exists():
        all_configs_str = dataset_path.read_text()
    else:
        logger.warning(f"Local dataset not found at {dataset_path}, falling back to webarena_verified package")
        all_configs_str = (
            importlib.resources.files("webarena_verified")
            .joinpath("assets/dataset/webarena-verified.json")
            .read_text()
        )
    all_configs_str = _substitute_url_placeholders(all_configs_str, urls)
    tasks = json.loads(all_configs_str)
    for task in tasks:
        task.setdefault("benchmark_format", "webarena_verified")
    return tasks


def _substitute_url_placeholders(raw: str, urls: dict[str, str]) -> str:
    if "__GITLAB_SSH__" in raw and "gitlab_ssh" not in urls and "gitlab" in urls:
        urls = {**urls, "gitlab_ssh": _gitlab_ssh_authority(urls["gitlab"])}
    for pattern, url_key in DERIVED_URL_PLACEHOLDERS.items():
        if url_key in urls:
            raw = raw.replace(pattern, urls[url_key])
    for pattern, url_key in URL_PLACEHOLDERS.items():
        if url_key in urls:
            raw = raw.replace(pattern, urls[url_key])
    return raw


def _parse_webarena_id(raw_id: str, fallback_idx: int) -> int:
    if isinstance(raw_id, str) and raw_id.startswith("webarena-"):
        return int(raw_id.split("-", 1)[1])
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return fallback_idx


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _split_and_urls(values) -> list[str]:
    urls: list[str] = []
    for value in _as_list(values):
        if isinstance(value, str):
            urls.extend(part.strip() for part in value.split(" |AND| ") if part.strip())
        else:
            urls.append(value)
    return urls


def load_classic_webarena_tasks(path: Path, urls: dict[str, str]) -> list[dict]:
    """Load WebArena/WebVoyager-style JSONL tasks and normalize common fields."""
    tasks: list[dict] = []
    raw = _substitute_url_placeholders(path.read_text(encoding="utf-8"), urls)
    for idx, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        task_id = _parse_webarena_id(item.get("id"), idx)
        benchmark_format = "classic_webarena" if "eval" in item else "webvoyager"
        tasks.append({
            **item,
            "benchmark_format": benchmark_format,
            "task_id": task_id,
            "sites": _as_list(item.get("web_name")),
            "intent": item["ques"],
            "start_urls": _as_list(item.get("web")),
        })
    return tasks


def _looks_like_visualwebarena_task(item: dict) -> bool:
    return (
        "eval" in item
        and (
            str(item.get("id", "")).startswith("visualwebarena-")
            or item.get("benchmark_format") == "visualwebarena"
            or "visual_difficulty" in item
            or "overall_difficulty" in item
            or "intent_template_id" in item
            or "page_image_query" in item.get("eval", {})
        )
    )


def load_visualwebarena_tasks(path: Path, urls: dict[str, str]) -> list[dict]:
    """Load VisualWebArena JSON arrays or JSONL records and normalize runner fields."""
    raw = _substitute_url_placeholders(path.read_text(encoding="utf-8"), urls)
    if path.suffix == ".jsonl":
        items = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    else:
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError(f"Expected VisualWebArena task list in {path}")

    tasks: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Unexpected VisualWebArena task at index {idx}: {item!r}")
        start_urls = _split_and_urls(item.get("start_url") or item.get("start_urls"))
        if not start_urls:
            start_urls = _split_and_urls(item.get("web"))
        tasks.append({
            **item,
            "benchmark_format": "visualwebarena",
            "task_id": item.get("task_id", idx),
            "sites": _as_list(item.get("sites") or item.get("web_name")),
            "intent": item.get("intent") or item["ques"],
            "start_urls": start_urls,
        })
    return tasks


def resolve_benchmark_path(benchmark_path: str | Path) -> Path:
    path = Path(benchmark_path)
    if not path.exists() and not path.is_absolute():
        webarena_dir = Path(__file__).resolve().parent.parent
        candidates = [webarena_dir / path]
        if path.parts and path.parts[0] == "webarena":
            candidates.append(webarena_dir / Path(*path.parts[1:]))
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    return path


def is_webvoyager_benchmark_path(benchmark_path: str | Path | None) -> bool:
    if benchmark_path is None:
        return False
    path = resolve_benchmark_path(benchmark_path)
    if path.suffix != ".jsonl" or not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        return "eval" not in item
    return False


def detect_benchmark_format(path: Path | None) -> str:
    if path is None:
        return "webarena_verified"
    if path.suffix == ".jsonl":
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    break
                if isinstance(item, dict) and _looks_like_visualwebarena_task(item):
                    return "visualwebarena"
                break
        return "classic_webarena"
    if path.suffix == ".json" and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "webarena_verified"
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if _looks_like_visualwebarena_task(data[0]):
                return "visualwebarena"
    return "webarena_verified"


def load_tasks(
    urls: dict[str, str],
    benchmark_path: str | Path | None = None,
    dataset_dir: Path | None = None,
) -> list[dict]:
    """Load tasks from either WebArena-Verified JSON or classic WebArena JSONL."""
    if benchmark_path is None:
        return load_all_tasks(urls, dataset_dir=dataset_dir)

    path = resolve_benchmark_path(benchmark_path)
    benchmark_format = detect_benchmark_format(path)
    if benchmark_format == "classic_webarena":
        return load_classic_webarena_tasks(path, urls)
    if benchmark_format == "visualwebarena":
        return load_visualwebarena_tasks(path, urls)

    raw = _substitute_url_placeholders(path.read_text(encoding="utf-8"), urls)
    tasks = json.loads(raw)
    for task in tasks:
        task.setdefault("benchmark_format", "webarena_verified")
    return tasks


def is_classic_webarena_task(config: dict) -> bool:
    return config.get("benchmark_format") == "classic_webarena"


def is_visualwebarena_task(config: dict) -> bool:
    return config.get("benchmark_format") == "visualwebarena"


def is_webvoyager_task(config: dict) -> bool:
    return config.get("benchmark_format") == "webvoyager"


STATE_CHANGE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(create|add|change|set|update|edit|modify|delete|remove|cancel|approve|disable|enable|mark|assign|invite|reply|post|comment|subscribe|unsubscribe|upvote|downvote|like|dislike|star|unstar|follow|unfollow|fork|promote|buy|checkout|rate|notify|submit|reopen|rename|move|make|start|reduce|increase)\b",
        r"\bopen an? (issue|merge request)\b",
        r"\b(place|submit|create|make) an? order\b",
        r"\bnew (issue|forum|repository|repo|project|group|marketing|cart price rule)\b",
        r"\bto my (wish ?list|cart)\b",
        r"\bmy address is\b",
    )
)


READ_ONLY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(tell me|what is|what are|which|who|when|where|how many|how much|get the|get me|find|list|show me|show the|open the|open my|go to|navigate to|view|browse|pull up)\b",
        r"\b(show|view) (on )?the (map|route|path|report|list)\b",
    )
)


STATE_CHANGE_OVERRIDES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bopen an? issue\b",
        r"\bcreate\b",
        r"\bpost my question\b",
        r"\bcreate a new post\b",
        r"\bcreate a post\b",
        r"\bupdate\b",
        r"\badd .* to my (wish ?list|cart)\b",
    )
)


VISUAL_STATE_CHANGE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\badd\b",
        r"\b(purchase|prepend|append|update|edit|modify|delete|remove|cancel|repost)\b",
        r"\b(empty|clear)\b.{0,80}\bcart\b",
        r"\b(put|place)\b.{0,120}\b(cart|wish\s*list|order)\b",
        r"(?:^|[.!?]\s*|\b(can|could|would) you\s+|\b(and|then|please)\s+)order\b",
        r"\b(create|make|submit|publish)\b(?:\s+\S+){0,8}\s+post\b",
        r"\b(write|find)\b.{0,120}\band post\b",
        r"(?:^|[.!?]\s*|\b(can|could|would) you\s+|\b(and|then|please)\s+)post\b",
        r"\b(leave|write|make)\b.{0,80}\b(comment|review|rating)\b",
        r"(?:^|[.!?]\s*|\b(can|could|would) you\s+|\b(and|then|please)\s+)comment\b",
        r"\bcomment with\b",
        r"\b(send (a )?(direct )?message|send\b.{0,40}\b(dm|message)|message (the|a|all|user|users|poster|seller))\b",
        r"\b(subscribe|unsubscribe|upvote|downvote|block|unblock|follow|unfollow)\b",
        r"\brate\b",
        r"\bchange\b.{0,100}\b(address|price|description|status|profile|phone|listing|username)\b",
        r"\bset\b.{0,100}\baddress\b",
    )
)


VISUAL_NON_ACTION_BUY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bshould i buy\b",
        r"\bi can buy\b",
        r"\bwhere i can buy\b",
    )
)


def _is_state_change_instruction(intent: str) -> bool:
    if any(pattern.search(intent) for pattern in STATE_CHANGE_OVERRIDES):
        return True
    if any(pattern.search(intent) for pattern in READ_ONLY_PATTERNS):
        return False
    return any(pattern.search(intent) for pattern in STATE_CHANGE_PATTERNS)


def _is_visual_state_change_instruction(intent: str) -> bool:
    if any(pattern.search(intent) for pattern in VISUAL_STATE_CHANGE_PATTERNS):
        return True
    if re.search(r"\bbuy\b", intent, re.IGNORECASE):
        return not any(pattern.search(intent) for pattern in VISUAL_NON_ACTION_BUY_PATTERNS)
    return False


def detect_task_type(config: dict) -> str:
    if is_webvoyager_task(config):
        return "webvoyager"
    if is_visualwebarena_task(config):
        intent = config.get("intent") or config.get("ques") or ""
        return "state_change" if _is_visual_state_change_instruction(intent) else "non_state_change"
    if is_classic_webarena_task(config):
        intent = config.get("intent") or config.get("ques") or ""
        return "state_change" if _is_state_change_instruction(intent) else "non_state_change"
    for ev in config.get("eval", []):
        if ev.get("evaluator") == "AgentResponseEvaluator":
            return ev["expected"].get("task_type", "mutate")
    return "mutate"


def _schema_to_format_hint(schema: dict) -> str:
    """Convert a results_schema JSON schema into a human-readable format hint."""
    stype = schema.get("type", "")

    if stype == "array":
        items = schema.get("items", {})
        itype = items.get("type", "")
        if itype == "object":
            props = items.get("properties", {})
            if props:
                key_descs = []
                for k, v in props.items():
                    vtype = v.get("type", "")
                    if vtype and vtype != "string":
                        key_descs.append(f'"{k}" ({vtype})')
                    else:
                        key_descs.append(f'"{k}"')
                keys_str = ", ".join(key_descs[:-1]) + f" and {key_descs[-1]}" if len(key_descs) > 1 else key_descs[0]
                return f"Return a list of objects with keys {keys_str}."
            return "Return a list of objects."
        if itype == "number":
            return "Return the value as a number (e.g., 10.99) only, without any additional details."
        if itype == "string":
            return "Return the value as a string only, without any additional details."
        return "Return a list of values."

    if stype == "object":
        props = schema.get("properties", {})
        if props:
            key_descs = []
            for k, v in props.items():
                vtype = v.get("type", "")
                if vtype and vtype != "string":
                    key_descs.append(f'"{k}" ({vtype})')
                else:
                    key_descs.append(f'"{k}"')
            keys_str = ", ".join(key_descs[:-1]) + f" and {key_descs[-1]}" if len(key_descs) > 1 else key_descs[0]
            return f"Return an object with keys {keys_str}."

    if stype == "number":
        return "Return the value as a number (e.g., 10.99) only, without any additional details."
    if stype == "string":
        return "Return the value as a string only, without any additional details."

    return ""


def augment_intent_with_schema(task_config: dict) -> str:
    """Return the task intent, appending a format hint from results_schema if needed."""
    intent = task_config["intent"]

    if is_classic_webarena_task(task_config) or is_visualwebarena_task(task_config):
        return intent

    if detect_task_type(task_config) != "retrieve":
        return intent

    inst_dict = task_config.get("instantiation_dict", {})
    if inst_dict.get("retrieved_data_format_spec", "").strip():
        return intent

    for ev in task_config.get("eval", []):
        schema = ev.get("results_schema")
        if schema:
            hint = _schema_to_format_hint(schema)
            if hint:
                stripped = intent.rstrip()
                if stripped.endswith((".", "?", "!")):
                    return f"{stripped} {hint}"
                return f"{stripped}. {hint}"
            break

    return intent


def split_tasks(tasks: list[dict], split_idx: int, split_total: int) -> list[dict]:
    """Round-robin split of tasks for multi-machine evaluation.

    Each split deterministically owns the same tasks regardless of how many
    are completed, so resume works correctly across restarts.
    """
    assert 0 <= split_idx < split_total, f"split_idx={split_idx} must be in [0, {split_total})"
    return [t for i, t in enumerate(tasks) if i % split_total == split_idx]


def filter_tasks(
    all_tasks: list[dict],
    task_ids: str | None = None,
    task_type: str | None = None,
    sites: str | None = None,
) -> list[dict]:
    tasks = all_tasks

    if task_ids:
        selected_ids: set[int] = set()
        for part in task_ids.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                selected_ids.update(range(int(lo), int(hi) + 1))
            else:
                selected_ids.add(int(part))
        tasks = [t for t in tasks if t["task_id"] in selected_ids]

    if task_type:
        tasks = [
            t for t in tasks
            if detect_task_type(t).lower() == task_type.lower()
        ]

    if sites:
        site_set = {s.strip() for s in sites.split(",")}
        tasks = [t for t in tasks if set(t["sites"]) & site_set]

    return tasks


def find_completed_tasks(run_dir: Path, task_ids: set[int] | None = None) -> set[int]:
    """Scan for completed tasks. Clean up incomplete dirs for selected tasks only."""
    completed = set()
    allowed_task_ids = task_ids
    if not run_dir.exists():
        return completed
    for task_dir in run_dir.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        try:
            task_id = int(task_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        if (task_dir / "result.txt").exists():
            completed.add(task_id)
        else:
            for f in task_dir.iterdir():
                if f.is_file():
                    # Concurrent split jobs share run_dir; another process may delete first.
                    f.unlink(missing_ok=True)
            logger.info(f"Cleaned up incomplete task dir: {task_dir.name}")
    return completed
