"""WebArena site configuration, task loading, and task utilities."""

import importlib.resources
import json
import logging
import os
import sys
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
    if missing and required:
        print(f"Error: missing environment variables: {', '.join(missing)}")
        for var in missing:
            print(f'  export {var}="http://<your-host>:<port>"')
        sys.exit(1)
    return urls


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
    for pattern, url_key in URL_PLACEHOLDERS.items():
        all_configs_str = all_configs_str.replace(pattern, urls[url_key])
    tasks = json.loads(all_configs_str)
    for task in tasks:
        task.setdefault("benchmark_format", "webarena_verified")
    return tasks


def _substitute_url_placeholders(raw: str, urls: dict[str, str]) -> str:
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


def detect_task_type(config: dict) -> str:
    if is_webvoyager_task(config):
        return "webvoyager"
    if is_classic_webarena_task(config):
        eval_types = config.get("eval", {}).get("eval_types", [])
        if "string_match" in eval_types:
            return "retrieve"
        if "url_match" in eval_types or "program_html" in eval_types:
            return "navigate"
        return "classic"
    if is_visualwebarena_task(config):
        eval_types = config.get("eval", {}).get("eval_types", [])
        if "string_match" in eval_types:
            return "retrieve"
        if "url_match" in eval_types or "program_html" in eval_types or "page_image_query" in eval_types:
            return "navigate"
        return "visualwebarena"
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
        classic_tasks = [t for t in tasks if is_classic_webarena_task(t)]
        verified_tasks = [t for t in tasks if not is_classic_webarena_task(t)]
        if classic_tasks and not verified_tasks:
            logger.warning("--task_type is only used for WebArena-Verified tasks; ignoring it for classic WebArena JSONL.")
        elif verified_tasks:
            tasks = classic_tasks + [
                t for t in verified_tasks
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
