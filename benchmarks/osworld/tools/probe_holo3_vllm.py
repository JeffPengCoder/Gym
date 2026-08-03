#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe a Holo3 endpoint with its exact three-image structured agent contract."""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from responses_api_agents.osworld_agent.holo3_agent import Holo3Agent, Holo3Step  # noqa: E402


def make_solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a deterministic RGB PNG without a Pillow dependency."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)

    scanline = b"\x00" + bytes(rgb) * width
    raw = scanline * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def request_json(url: str, api_key: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _observation(agent: Holo3Agent, png: bytes) -> dict:
    return agent._image_observation(png)


def build_probe_payload(
    *,
    model: str,
    width: int = 1920,
    height: int = 1080,
    image_count: int = 3,
) -> tuple[dict, Holo3Agent]:
    """Build a request matching the largest Holo3 history used by OSWorld."""

    if image_count not in {1, 3}:
        raise ValueError("image_count must be 1 or 3")
    agent = Holo3Agent(model=model, screen_size=(width, height), temperature=0.0)
    messages = [
        {"role": "system", "content": agent._system_prompt()},
        {"role": "user", "content": "Task: Inspect the desktop colors, then choose one valid next action."},
    ]
    colors = ((40, 180, 80), (50, 100, 210), (220, 130, 40))
    for index in range(image_count):
        png = make_solid_png(width, height, colors[index])
        messages.append(_observation(agent, png))
        if index < image_count - 1:
            prior = Holo3Step.model_validate(
                {
                    "note": f"Screenshot {index + 1} was retained for the endpoint probe.",
                    "thought": "Wait for the next observation.",
                    "tool_call": {"tool_name": "wait", "duration": 1},
                }
            )
            messages.append({"role": "assistant", "content": prior.model_dump_json()})
            messages.append(
                {"role": "user", "content": '<tool_output tool="wait">\nWaited 1 seconds.\n</tool_output>'}
            )

    payload = agent._payload(messages, step=image_count, parse_attempt=1)
    direct_payload = {key: value for key, value in payload.items() if not key.startswith("_") and key != "extra_body"}
    direct_payload.update(payload["extra_body"])
    return direct_payload, agent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--model", default="Hcompany/Holo3-35B-A3B")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--image-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    models = request_json(f"{base_url}/models", args.api_key)
    model_ids = [item.get("id") for item in models.get("data", [])]
    print(json.dumps({"endpoint": base_url, "models": model_ids}, ensure_ascii=False))
    if args.model not in model_ids:
        raise RuntimeError(f"Configured model {args.model!r} is not served by endpoint: {model_ids}")
    if args.models_only:
        return 0

    payload, _agent = build_probe_payload(
        model=args.model,
        width=args.width,
        height=args.height,
        image_count=args.image_count,
    )
    result = request_json(f"{base_url}/chat/completions", args.api_key, payload)
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    parsed = Holo3Step.model_validate_json(content)
    if not reasoning:
        raise RuntimeError("Holo3 probe returned no separate reasoning stream")
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        raise RuntimeError(f"Holo3 probe did not finish cleanly: {choice.get('finish_reason')!r}")

    print(
        json.dumps(
            {
                "probe": "ok",
                "finish_reason": choice.get("finish_reason"),
                "image_count": args.image_count,
                "image_size": [args.width, args.height],
                "reasoning_chars": len(reasoning),
                "tool_name": parsed.tool_call.tool_name,
                "usage": result.get("usage"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
