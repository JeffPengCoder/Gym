#!/bin/bash
# Launch multiple split sbatch jobs for parallel WebVoyager evaluation.
#
# Each split gets its own SLURM job running launch_nemotron_webvoyager.sh
# with a different --webvoyager-split-idx. After all splits finish, a cleanup
# job runs the full task list without splits to retry failures.
#
# Usage:
#   WEBARENA_JUDGE_API_KEY=... ./launch_nemotron_webvoyager_parallel.sh \
#       --splits 4 \
#       --model-path /path/to/model \
#       --webvoyager-benchmark-path /path/to/webvoyager.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_nemotron_webvoyager.sh"

# ----------------------------------------------------------
# Input Configuration
# ----------------------------------------------------------
MODEL_PATH=""
NUM_SPLITS="2"
WEBVOYAGER_RESULT_DIR=""
WEBVOYAGER_BENCHMARK_PATH="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/osworld_internal/webarena/benchmarks/webvoyager.jsonl"

# ----------------------------------------------------------
# Default Configuration - vLLM Server
# ----------------------------------------------------------
SERVE_BIN="/lustre/fsw/portfolios/llmservice/users/kchumachenko/nano_v3_vllm/vllm/serve_wrapper.py"
VLLM_CHAT_TMPL="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/mingjiel/workspace/output/nemotron_v3.chat_template.keep_history.jinja"
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/nano-tokenizer"
REASONING_PARSER_PLUGIN="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/NVIDIA-Nemotron-Nano-12B-v2/nano_v3_reasoning_parser.py"
SERVED_MODEL_NAME="vllm_local"
PORT="8000"
MAX_MODEL_LEN=128000
TENSOR_PARALLEL_SIZE=8
DATA_PARALLEL_SIZE=1
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/pytorch25.06-moe-avlm-eval-1217-vllm-gpu.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/mingjiel/root:/root,/dev/shm:/dev/shm"

# ----------------------------------------------------------
# Default Configuration - WebVoyager Evaluation
# ----------------------------------------------------------
WEBVOYAGER_DIR="$SCRIPT_DIR"
WEBVOYAGER_EVAL_SCRIPT="webarena/nvidia/run_eval_parallel.py"
WEBVOYAGER_CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/webarena.sqsh"
WEBVOYAGER_CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/mingjiel/root:/root"
WEBVOYAGER_TEMPERATURE="0.1"
WEBVOYAGER_WORKERS="16"
WEBVOYAGER_MAX_STEPS="100"
# This is used to determine auto history compaction. Use 0 to disable auto history compaction.
WEBVOYAGER_MAX_MODEL_LEN="0"
WEBVOYAGER_VIEWPORT_WIDTH="1920"
WEBVOYAGER_VIEWPORT_HEIGHT="1080"
WEBVOYAGER_TIMEOUT="4000"
WEBVOYAGER_TASK_IDS=""
WEBVOYAGER_SITES=""

# ----------------------------------------------------------
# Default Configuration - WebArena Judge
# ----------------------------------------------------------
WEBARENA_JUDGE_MODEL="${WEBARENA_JUDGE_MODEL:-gcp/google/gemini-3-flash-preview}"
WEBARENA_JUDGE_BASE_URL="${WEBARENA_JUDGE_BASE_URL:-https://inference-api.nvidia.com}"
WEBARENA_JUDGE_TIMEOUT="${WEBARENA_JUDGE_TIMEOUT:-120}"
WA_BROWSER_PROXY_SERVER="${WA_BROWSER_PROXY_SERVER:-}"

# ----------------------------------------------------------
# Parse Command Line Arguments
# ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --splits)
            NUM_SPLITS="$2"; shift 2;;
        --launch-script)
            LAUNCH_SCRIPT="$2"; shift 2;;
        --model-path)
            MODEL_PATH="$2"; shift 2;;
        --serve-bin)
            SERVE_BIN="$2"; shift 2;;
        --chat-template)
            VLLM_CHAT_TMPL="$2"; shift 2;;
        --tokenizer)
            TOKENIZER_MODEL="$2"; shift 2;;
        --reasoning-parser-plugin)
            REASONING_PARSER_PLUGIN="$2"; shift 2;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"; shift 2;;
        --port)
            PORT="$2"; shift 2;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift 2;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"; shift 2;;
        --data-parallel-size)
            DATA_PARALLEL_SIZE="$2"; shift 2;;
        --container-image)
            CONTAINER_IMAGE="$2"; shift 2;;
        --container-mounts)
            CONTAINER_MOUNTS="$2"; shift 2;;
        --webvoyager-dir)
            WEBVOYAGER_DIR="$2"; shift 2;;
        --webvoyager-eval-script)
            WEBVOYAGER_EVAL_SCRIPT="$2"; shift 2;;
        --webvoyager-container-image)
            WEBVOYAGER_CONTAINER_IMAGE="$2"; shift 2;;
        --webvoyager-container-mounts)
            WEBVOYAGER_CONTAINER_MOUNTS="$2"; shift 2;;
        --webvoyager-result-dir)
            WEBVOYAGER_RESULT_DIR="$2"; shift 2;;
        --webvoyager-benchmark-path)
            WEBVOYAGER_BENCHMARK_PATH="$2"; shift 2;;
        --webvoyager-temperature)
            WEBVOYAGER_TEMPERATURE="$2"; shift 2;;
        --webvoyager-workers)
            WEBVOYAGER_WORKERS="$2"; shift 2;;
        --webvoyager-max-steps)
            WEBVOYAGER_MAX_STEPS="$2"; shift 2;;
        --webvoyager-max-model-len)
            WEBVOYAGER_MAX_MODEL_LEN="$2"; shift 2;;
        --webvoyager-viewport-width)
            WEBVOYAGER_VIEWPORT_WIDTH="$2"; shift 2;;
        --webvoyager-viewport-height)
            WEBVOYAGER_VIEWPORT_HEIGHT="$2"; shift 2;;
        --webvoyager-timeout)
            WEBVOYAGER_TIMEOUT="$2"; shift 2;;
        --webvoyager-task-ids)
            WEBVOYAGER_TASK_IDS="$2"; shift 2;;
        --webvoyager-sites)
            WEBVOYAGER_SITES="$2"; shift 2;;
        --webarena-judge-model)
            WEBARENA_JUDGE_MODEL="$2"; shift 2;;
        --webarena-judge-base-url)
            WEBARENA_JUDGE_BASE_URL="$2"; shift 2;;
        --webarena-judge-timeout)
            WEBARENA_JUDGE_TIMEOUT="$2"; shift 2;;
        -h|--help)
            cat << EOF
Usage: WEBARENA_JUDGE_API_KEY=... $0 --splits N --model-path PATH --webvoyager-benchmark-path PATH [OPTIONS]

Required Arguments:
  --splits N                          Number of parallel split jobs to launch
  --model-path PATH                   Path to model checkpoint
  --webvoyager-benchmark-path PATH    WebVoyager JSONL benchmark path

Required Environment:
  WEBARENA_JUDGE_API_KEY              API key for the WebVoyager LLM judge

Optional Arguments (Parallel):
  --launch-script PATH                Path to launch_nemotron_webvoyager.sh

Optional Arguments (vLLM Server):
  --serve-bin PATH                    Path to serve_wrapper.py
  --chat-template PATH                Path to chat template file
  --tokenizer PATH                    Path to tokenizer model
  --reasoning-parser-plugin PATH      Path to reasoning parser plugin
  --served-model-name NAME            Name exposed via the API (default: vllm_local)
  --port PORT                         vLLM server port (default: 8000)
  --max-model-len N                   Maximum model context length (default: 64000)
  --tensor-parallel-size N            Tensor parallelism (default: 8)
  --data-parallel-size N              Data parallelism (default: 1)
  --container-image PATH              Container .sqsh image for vLLM
  --container-mounts MOUNTS           Container bind mounts

Optional Arguments (WebVoyager Evaluation):
  --webvoyager-dir PATH               Path to repo root
  --webvoyager-eval-script PATH       Relative path to eval script
  --webvoyager-container-image PATH   Container image for WebVoyager eval
  --webvoyager-container-mounts MOUNTS Container bind mounts
  --webvoyager-result-dir PATH        Result directory
  --webvoyager-temperature TEMP       Sampling temperature (default: 0.1)
  --webvoyager-workers N              Parallel eval workers per split job (default: 16)
  --webvoyager-max-steps N            Max agent steps per task (default: 100)
  --webvoyager-max-model-len N        Eval context budget length; 0 disables eval-side compaction (default: 0)
  --webvoyager-viewport-width N       Browser viewport width (default: 1280)
  --webvoyager-viewport-height N      Browser viewport height (default: 720)
  --webvoyager-timeout N              Deprecated no-op; max_steps bounds task runtime
  --webvoyager-task-ids IDS           Task IDs to run (e.g. "0-50" or "0,5,10")
  --webvoyager-sites SITES            Filter by site/web_name (comma-separated)

Optional Arguments (WebArena Judge):
  --webarena-judge-model NAME         Judge model
  --webarena-judge-base-url URL       Judge OpenAI-compatible base URL
  --webarena-judge-timeout SECONDS    Judge request timeout

Example:
  WEBARENA_JUDGE_API_KEY=... $0 --splits 4 \\
      --model-path /path/to/model \\
      --webvoyager-benchmark-path /path/to/webvoyager.jsonl
EOF
            exit 0;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use -h or --help for usage information"
            exit 1;;
    esac
done

# ----------------------------------------------------------
# Validate Parameters
# ----------------------------------------------------------
if [[ -z "$NUM_SPLITS" ]]; then
    echo "ERROR: --splits is required" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

if ! [[ "$NUM_SPLITS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --splits must be a positive integer (got: $NUM_SPLITS)" >&2
    exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model-path is required" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

if [[ -z "$WEBVOYAGER_BENCHMARK_PATH" ]]; then
    echo "ERROR: --webvoyager-benchmark-path is required" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

if [[ -z "${WEBARENA_JUDGE_API_KEY:-}" ]]; then
    echo "ERROR: WEBARENA_JUDGE_API_KEY must be set for WebVoyager evaluation" >&2
    exit 1
fi

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "ERROR: Launch script not found: $LAUNCH_SCRIPT" >&2
    exit 1
fi

[[ -z "$WEBVOYAGER_RESULT_DIR" ]] && WEBVOYAGER_RESULT_DIR="$(dirname "$MODEL_PATH")/webvoyager_results"

mkdir -p logs

webarena_judge_key_fingerprint() {
    if [[ -z "${WEBARENA_JUDGE_API_KEY:-}" ]]; then
        printf "<unset>"
        return
    fi

    local digest
    digest=$(printf "%s" "$WEBARENA_JUDGE_API_KEY" | sha256sum | cut -c1-16)
    printf "sha256:%s len:%d" "$digest" "${#WEBARENA_JUDGE_API_KEY}"
}

# ----------------------------------------------------------
# Build forwarded arguments
# ----------------------------------------------------------
FORWARD_ARGS=(
    --model-path "$MODEL_PATH"
    --serve-bin "$SERVE_BIN"
    --chat-template "$VLLM_CHAT_TMPL"
    --tokenizer "$TOKENIZER_MODEL"
    --reasoning-parser-plugin "$REASONING_PARSER_PLUGIN"
    --served-model-name "$SERVED_MODEL_NAME"
    --port "$PORT"
    --max-model-len "$MAX_MODEL_LEN"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --container-image "$CONTAINER_IMAGE"
    --container-mounts "$CONTAINER_MOUNTS"
    --webvoyager-dir "$WEBVOYAGER_DIR"
    --webvoyager-eval-script "$WEBVOYAGER_EVAL_SCRIPT"
    --webvoyager-container-image "$WEBVOYAGER_CONTAINER_IMAGE"
    --webvoyager-container-mounts "$WEBVOYAGER_CONTAINER_MOUNTS"
    --webvoyager-result-dir "$WEBVOYAGER_RESULT_DIR"
    --webvoyager-benchmark-path "$WEBVOYAGER_BENCHMARK_PATH"
    --webvoyager-temperature "$WEBVOYAGER_TEMPERATURE"
    --webvoyager-workers "$WEBVOYAGER_WORKERS"
    --webvoyager-max-steps "$WEBVOYAGER_MAX_STEPS"
    --webvoyager-max-model-len "$WEBVOYAGER_MAX_MODEL_LEN"
    --webvoyager-viewport-width "$WEBVOYAGER_VIEWPORT_WIDTH"
    --webvoyager-viewport-height "$WEBVOYAGER_VIEWPORT_HEIGHT"
    --webvoyager-timeout "$WEBVOYAGER_TIMEOUT"
    --webarena-judge-model "$WEBARENA_JUDGE_MODEL"
    --webarena-judge-base-url "$WEBARENA_JUDGE_BASE_URL"
    --webarena-judge-timeout "$WEBARENA_JUDGE_TIMEOUT"
)

[[ -n "$WEBVOYAGER_TASK_IDS" ]] && FORWARD_ARGS+=(--webvoyager-task-ids "$WEBVOYAGER_TASK_IDS")
[[ -n "$WEBVOYAGER_SITES" ]] && FORWARD_ARGS+=(--webvoyager-sites "$WEBVOYAGER_SITES")

# ----------------------------------------------------------
# Launch split jobs
# ----------------------------------------------------------
echo "=============================================="
echo "Launching $NUM_SPLITS parallel WebVoyager split jobs"
echo "=============================================="
echo "  Launch script:     $LAUNCH_SCRIPT"
echo "  Model path:        $MODEL_PATH"
echo "  Benchmark path:    $WEBVOYAGER_BENCHMARK_PATH"
echo "  Result dir:        $WEBVOYAGER_RESULT_DIR"
echo "  Workers/split:     $WEBVOYAGER_WORKERS"
echo "  Max steps:         $WEBVOYAGER_MAX_STEPS"
echo "  Eval max model len: $WEBVOYAGER_MAX_MODEL_LEN"
echo "  Viewport:          ${WEBVOYAGER_VIEWPORT_WIDTH}x${WEBVOYAGER_VIEWPORT_HEIGHT}"
echo "  Judge model:       $WEBARENA_JUDGE_MODEL"
echo "  Judge API key:     $(webarena_judge_key_fingerprint)"
echo "  Browser proxy:     $(if [[ -n "$WA_BROWSER_PROXY_SERVER" ]]; then echo enabled; else echo disabled; fi)"
echo "=============================================="
echo ""

JOB_IDS=()
for ((i = 0; i < NUM_SPLITS; i++)); do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,WEBARENA_JUDGE_API_KEY,WA_BROWSER_PROXY_SERVER \
        --job-name="llmservice_fm_vision:nemotron-webvoyager-split-${i}-of-${NUM_SPLITS}" \
        "$LAUNCH_SCRIPT" \
        "${FORWARD_ARGS[@]}" \
        --webvoyager-split-idx "$i" \
        --webvoyager-split-total "$NUM_SPLITS")
    JOB_IDS+=("$JOB_ID")
    echo "  Split $i/$NUM_SPLITS -> Job $JOB_ID"
done

echo ""

# ----------------------------------------------------------
# Launch cleanup job (runs after all splits finish)
# ----------------------------------------------------------
DEPENDENCY_STR=$(IFS=:; echo "${JOB_IDS[*]}")
CLEANUP_JOB_ID=$(sbatch --parsable \
    --export=ALL,WEBARENA_JUDGE_API_KEY,WA_BROWSER_PROXY_SERVER \
    --job-name="llmservice_fm_vision:nemotron-webvoyager-cleanup" \
    --dependency="afterany:${DEPENDENCY_STR}" \
    "$LAUNCH_SCRIPT" \
    "${FORWARD_ARGS[@]}")
JOB_IDS+=("$CLEANUP_JOB_ID")
echo "  Cleanup job   -> Job $CLEANUP_JOB_ID (runs after all splits)"

echo ""
echo "=============================================="
echo "All jobs submitted ($NUM_SPLITS splits + 1 cleanup):"
printf '  %s\n' "${JOB_IDS[@]}"
echo ""
echo "Monitor with:  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "=============================================="
