#!/usr/bin/env bash
# Run Holo3 against an OpenAI-compatible vLLM endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export HOLO3_VLLM_BASE_URL="${HOLO3_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export HOLO3_VLLM_API_KEY="${HOLO3_VLLM_API_KEY:-local-vllm}"
export HOLO3_VLLM_MODEL="${HOLO3_VLLM_MODEL:-Hcompany/Holo3-35B-A3B}"
export HOLO3_REASONING_EFFORT="${HOLO3_REASONING_EFFORT:-medium}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-${HOLO3_VLLM_MODEL}}"
export OSWORLD_POLICY_MODEL_NAME="${OSWORLD_POLICY_MODEL_NAME:-${POLICY_MODEL_NAME}}"

vllm_host="${HOLO3_VLLM_BASE_URL#*://}"
vllm_host="${vllm_host%%[:/]*}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${vllm_host}"
export no_proxy="${no_proxy:+${no_proxy},}${vllm_host}"

PREFLIGHT="${PREFLIGHT:-1}"
HOLO3_PREFLIGHT_IMAGE_COUNT="${HOLO3_PREFLIGHT_IMAGE_COUNT:-3}"
if [[ "${PREFLIGHT}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/probe_holo3_vllm.py" \
    --base-url "${HOLO3_VLLM_BASE_URL}" \
    --api-key "${HOLO3_VLLM_API_KEY}" \
    --model "${HOLO3_VLLM_MODEL}" \
    --image-count "${HOLO3_PREFLIGHT_IMAGE_COUNT}"
fi

export GYM_ROOT
export RUNNER_NAME="${RUNNER_NAME:-holo3_agent}"
export INPUT_JSONL="${INPUT_JSONL:-benchmarks/osworld/data/example.jsonl}"
export LIMIT="${LIMIT:-5}"
export NUM_ENVS="${NUM_ENVS:-1}"
export NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-${NUM_ENVS}}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-0}"
export MAX_STEPS="${MAX_STEPS:-100}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,benchmarks/osworld/configs/osworld_agent_holo3.yaml,benchmarks/osworld/configs/vllm_model_holo3.yaml}"

exec bash "${SCRIPT_DIR}/run_multienv_osworld_agent.sh"
