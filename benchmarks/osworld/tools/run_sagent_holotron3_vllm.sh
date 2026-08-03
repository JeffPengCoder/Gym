#!/usr/bin/env bash
# Run Yi's Sagent/Holotron-3-Nano recipe through Gym and one external vLLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

export SAGENT_HOLO3_VLLM_BASE_URL="${SAGENT_HOLO3_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export SAGENT_HOLO3_VLLM_API_KEY="${SAGENT_HOLO3_VLLM_API_KEY:-local-vllm}"
export SAGENT_HOLO3_SERVED_MODEL="${SAGENT_HOLO3_SERVED_MODEL:-vllm_local}"
export POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-${SAGENT_HOLO3_SERVED_MODEL}}"
export OSWORLD_POLICY_MODEL_NAME="${OSWORLD_POLICY_MODEL_NAME:-${POLICY_MODEL_NAME}}"
# Match the official AMI assumptions used by Yi's headline run.
export OSWORLD_GUEST_SUDO_NORMALIZE="${OSWORLD_GUEST_SUDO_NORMALIZE:-1}"

vllm_host="${SAGENT_HOLO3_VLLM_BASE_URL#*://}"
vllm_host="${vllm_host%%[:/]*}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${vllm_host}"
export no_proxy="${no_proxy:+${no_proxy},}${vllm_host}"

PREFLIGHT="${PREFLIGHT:-1}"
if [[ "${PREFLIGHT}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  probe_args=(
    --base-url "${SAGENT_HOLO3_VLLM_BASE_URL}"
    --api-key "${SAGENT_HOLO3_VLLM_API_KEY}"
    --model "${SAGENT_HOLO3_SERVED_MODEL}"
  )
  case "${SAGENT_ENABLE_THINKING:-vendor-default}" in
    1|true) probe_args+=(--enable-thinking) ;;
    0|false) probe_args+=(--disable-thinking) ;;
    vendor-default|"") ;;
    *) echo "SAGENT_ENABLE_THINKING must be 1, 0, or unset" >&2; exit 2 ;;
  esac
  "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/probe_sagent_holotron3_vllm.py" "${probe_args[@]}"
fi

export GYM_ROOT
export RUNNER_NAME="${RUNNER_NAME:-sagent_holo3_agent}"
export INPUT_JSONL="${INPUT_JSONL:-benchmarks/osworld/data/example.jsonl}"
export LIMIT="${LIMIT:-5}"
# One reservation pair; VM concurrency within that ComputeLab worker remains configurable.
export NUM_ENVS="${NUM_ENVS:-4}"
export NUM_SAMPLES_IN_PARALLEL="${NUM_SAMPLES_IN_PARALLEL:-${NUM_ENVS}}"
export RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-0}"
export MAX_STEPS="${MAX_STEPS:-200}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
export TEMPERATURE="${TEMPERATURE:-0.8}"
export RECORD_VIDEO="${RECORD_VIDEO:-0}"
export CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/osworld_agent/configs/osworld_agent.yaml,benchmarks/osworld/configs/osworld_agent_sagent_holotron3.yaml,benchmarks/osworld/configs/vllm_model_sagent_holotron3.yaml}"

exec bash "${SCRIPT_DIR}/run_multienv_osworld_agent.sh"
