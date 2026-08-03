#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe vLLM with Yi's exact two-image Sagent structured-note request."""

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

from responses_api_agents.osworld_agent.sagent_holo3_agent import SagentHolo3Agent  # noqa: E402


def make_solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a deterministic RGB PNG without Pillow."""

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
        with urllib.request.urlopen(request, timeout=1200) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def build_probe_payload(
    *,
    model: str,
    width: int = 1920,
    height: int = 1080,
    enable_thinking: bool | None = None,
) -> tuple[dict, SagentHolo3Agent]:
    agent = SagentHolo3Agent(
        model=model,
        max_steps=200,
        max_tokens=4096,
        temperature=0.8,
        top_p=0.95,
        max_image_history_length=2,
        enable_thinking=enable_thinking,
        transport_retries=1,
        transport_retry_sleep_s=0,
    )
    first_png = make_solid_png(width, height, (40, 180, 80))
    second_png = make_solid_png(width, height, (50, 100, 210))
    prior_step = {
        "note": "The first synthetic desktop was green.",
        "thought": "Wait for the next observation before choosing a desktop action.",
        "tool_call": {"tool_name": "wait_desktop", "seconds": 1.0},
    }
    agent.history.append(
        {
            "screenshot": first_png,
            "step": prior_step,
            "tool_name": "wait_desktop",
            "tool_result": "Waited.",
        }
    )
    messages = agent._agent._build_messages(  # noqa: SLF001 - exact contract probe.
        "Inspect the two synthetic desktops and choose one valid next action.",
        second_png,
    )
    payload = agent._agent._build_payload(messages)  # noqa: SLF001 - exact contract probe.
    payload["top_p"] = 0.95
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return payload, agent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--model", default="vllm_local")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_const", const=True, dest="enable_thinking")
    thinking.add_argument("--disable-thinking", action="store_const", const=False, dest="enable_thinking")
    parser.set_defaults(enable_thinking=None)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    models = request_json(f"{base_url}/models", args.api_key)
    model_ids = [item.get("id") for item in models.get("data", [])]
    print(json.dumps({"endpoint": base_url, "models": model_ids}, ensure_ascii=False))
    if args.model not in model_ids:
        raise RuntimeError(f"Configured model {args.model!r} is not served by endpoint: {model_ids}")
    if args.models_only:
        return 0

    payload, agent = build_probe_payload(
        model=args.model,
        width=args.width,
        height=args.height,
        enable_thinking=args.enable_thinking,
    )
    result = request_json(f"{base_url}/chat/completions", args.api_key, payload)
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    parsed = agent._agent._parse_response(message)  # noqa: SLF001 - exact parser preflight.
    actions, _tool_result = agent._agent._tool_to_pyautogui(parsed["tool_call"])  # noqa: SLF001
    if choice.get("finish_reason") not in {"stop", "tool_calls"}:
        raise RuntimeError(f"Sagent probe did not finish cleanly: {choice.get('finish_reason')!r}")

    print(
        json.dumps(
            {
                "probe": "ok",
                "finish_reason": choice.get("finish_reason"),
                "image_count": 2,
                "image_size": [args.width, args.height],
                "content_chars": len(message.get("content") or ""),
                "reasoning_chars": len(message.get("reasoning_content") or message.get("reasoning") or ""),
                "tool_name": parsed["tool_call"]["tool_name"],
                "actions": actions,
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
