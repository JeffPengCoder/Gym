#!/usr/bin/env python3
"""Single-task evaluation runner for the Nemotron agent."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR))

from common.agent_eval_runner import AgentEvalSpec, main_single


SPEC = AgentEvalSpec(
    agent_name="Nemotron",
    agent_module="nemotron_agent",
    agent_class="NemotronAgent",
    logger_prefix="nemotron_eval",
    default_result_dir="webarena/nvidia/results",
    model_required=True,
    supports_thinking=True,
    supports_parallel_temperature=True,
    default_temperature=1.0,
    production_style_title=True,
)

TOOLCALL_SPEC = AgentEvalSpec(
    agent_name="Nemotron ToolCall",
    agent_module="nemotron_toolcall_agent",
    agent_class="NemotronToolCallAgent",
    logger_prefix="nemotron_toolcall_eval",
    default_result_dir="webarena/nvidia/results_toolcall",
    model_required=True,
    supports_thinking=True,
    supports_max_image_history=True,
    supports_max_model_len=True,
    supports_tokenizer_model=True,
    supports_parallel_temperature=True,
    supports_expanded_browser_tools=True,
    default_expanded_browser_tools=True,
    default_temperature=1.0,
    production_style_title=True,
)


def select_spec() -> AgentEvalSpec:
    if "--tool-call" not in sys.argv:
        return SPEC
    sys.argv.remove("--tool-call")
    return TOOLCALL_SPEC


if __name__ == "__main__":
    spec = select_spec()
    main_single(spec, description=f"{spec.agent_name} evaluation (production-style) on WebArena-Verified")
