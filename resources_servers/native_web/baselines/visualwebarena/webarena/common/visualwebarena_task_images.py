"""VisualWebArena task input image prompt helpers."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

TASK_INPUT_IMAGE_MARKER = "_webarena_task_input_image"


def image_mime_type(path: Path, data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def load_task_input_image_parts(
    image_paths: list[str | Path] | None,
    *,
    mark_images: bool = False,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    paths = image_paths or []
    total = len(paths)
    for idx, image_path in enumerate(paths, start=1):
        path = Path(image_path)
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode()
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime_type(path, data)};base64,{encoded}"},
        }
        if mark_images:
            image_part[TASK_INPUT_IMAGE_MARKER] = True
        parts.append({"type": "text", "text": f"Task image {idx} of {total}:"})
        parts.append(image_part)
    return parts
