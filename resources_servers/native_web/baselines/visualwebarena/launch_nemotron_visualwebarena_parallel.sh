#!/bin/bash
# Launch multiple split sbatch jobs for parallel VisualWebArena evaluation.
#
# Each split gets its own SLURM job running launch_nemotron_visualwebarena.sh
# with a different --visualwebarena-split-idx. After all splits finish, a cleanup
# job runs the full task list without splits to retry failures.
#
# Usage:
#   WEBARENA_JUDGE_API_KEY=... ./launch_nemotron_visualwebarena_parallel.sh \
#       --splits 4 \
#       --model-path /path/to/model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_nemotron_visualwebarena.sh"

# ----------------------------------------------------------
# Input Configuration
# ----------------------------------------------------------
MODEL_PATH=""
NUM_SPLITS="2"
VISUALWEBARENA_RESULT_DIR=""
VISUALWEBARENA_BENCHMARK_PATH="webarena/benchmarks/visualwebarena.jsonl"
VISUALWEBARENA_JUDGE="visualwebarena"

# ----------------------------------------------------------
# Default Configuration - vLLM Server
# ----------------------------------------------------------
SERVE_BIN="/lustre/fsw/portfolios/llmservice/users/kchumachenko/nano_v3_vllm/vllm/serve_wrapper.py"
VLLM_CHAT_TMPL="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/mingjiel/workspace/output/nemotron_v3.chat_template.keep_history.jinja"
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/nano-tokenizer"
REASONING_PARSER_PLUGIN="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/NVIDIA-Nemotron-Nano-12B-v2/nano_v3_reasoning_parser.py"
SERVED_MODEL_NAME="vllm_local"
PORT="8000"
MAX_MODEL_LEN=64000
TENSOR_PARALLEL_SIZE=8
DATA_PARALLEL_SIZE=1
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/pytorch25.06-moe-avlm-eval-1217-vllm-gpu.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/mingjiel/root:/root,/dev/shm:/dev/shm"

# ----------------------------------------------------------
# Default Configuration - VisualWebArena Evaluation
# ----------------------------------------------------------
VISUALWEBARENA_DIR="$SCRIPT_DIR"
VISUALWEBARENA_EVAL_SCRIPT="webarena/nvidia/run_eval_parallel.py"
VISUALWEBARENA_CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/webarena.sqsh"
VISUALWEBARENA_CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/mingjiel/root:/root"
VISUALWEBARENA_TEMPERATURE="0.1"
VISUALWEBARENA_WORKERS="16"
VISUALWEBARENA_MAX_STEPS="100"
VISUALWEBARENA_MAX_MODEL_LEN="49152"
VISUALWEBARENA_VIEWPORT_WIDTH="1920"
VISUALWEBARENA_VIEWPORT_HEIGHT="1080"
VISUALWEBARENA_TIMEOUT="4000"
VISUALWEBARENA_TASK_IDS=""
VISUALWEBARENA_SITES=""

# ----------------------------------------------------------
# Default Configuration - WebArena Site URLs
# ----------------------------------------------------------
WA_HOSTNAME="3.137.180.238"
WA_SHOPPING="http://${WA_HOSTNAME}:7770"
WA_SHOPPING_ADMIN="http://${WA_HOSTNAME}:7780/admin"
WA_REDDIT="http://${WA_HOSTNAME}:9999"
WA_GITLAB="http://${WA_HOSTNAME}:8023"
WA_WIKIPEDIA="http://${WA_HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
WA_MAP="http://${WA_HOSTNAME}:3000"
WA_CLASSIFIEDS="http://${WA_HOSTNAME}:9980"

# ----------------------------------------------------------
# Default Configuration - WebArena Judge
# ----------------------------------------------------------
WEBARENA_JUDGE_MODEL="${WEBARENA_JUDGE_MODEL:-us/azure/openai/gpt-4.1}"
WEBARENA_JUDGE_BASE_URL="${WEBARENA_JUDGE_BASE_URL:-https://inference-api.nvidia.com}"
WEBARENA_JUDGE_TIMEOUT="${WEBARENA_JUDGE_TIMEOUT:-120}"

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
        --visualwebarena-dir)
            VISUALWEBARENA_DIR="$2"; shift 2;;
        --visualwebarena-eval-script)
            VISUALWEBARENA_EVAL_SCRIPT="$2"; shift 2;;
        --visualwebarena-container-image)
            VISUALWEBARENA_CONTAINER_IMAGE="$2"; shift 2;;
        --visualwebarena-container-mounts)
            VISUALWEBARENA_CONTAINER_MOUNTS="$2"; shift 2;;
        --visualwebarena-result-dir)
            VISUALWEBARENA_RESULT_DIR="$2"; shift 2;;
        --visualwebarena-benchmark-path)
            VISUALWEBARENA_BENCHMARK_PATH="$2"; shift 2;;
        --visualwebarena-judge)
            VISUALWEBARENA_JUDGE="$2"; shift 2;;
        --visualwebarena-temperature)
            VISUALWEBARENA_TEMPERATURE="$2"; shift 2;;
        --visualwebarena-workers)
            VISUALWEBARENA_WORKERS="$2"; shift 2;;
        --visualwebarena-max-steps)
            VISUALWEBARENA_MAX_STEPS="$2"; shift 2;;
        --visualwebarena-max-model-len)
            VISUALWEBARENA_MAX_MODEL_LEN="$2"; shift 2;;
        --visualwebarena-viewport-width)
            VISUALWEBARENA_VIEWPORT_WIDTH="$2"; shift 2;;
        --visualwebarena-viewport-height)
            VISUALWEBARENA_VIEWPORT_HEIGHT="$2"; shift 2;;
        --visualwebarena-timeout)
            VISUALWEBARENA_TIMEOUT="$2"; shift 2;;
        --visualwebarena-task-ids)
            VISUALWEBARENA_TASK_IDS="$2"; shift 2;;
        --visualwebarena-sites)
            VISUALWEBARENA_SITES="$2"; shift 2;;
        --webarena-judge-model)
            WEBARENA_JUDGE_MODEL="$2"; shift 2;;
        --webarena-judge-base-url)
            WEBARENA_JUDGE_BASE_URL="$2"; shift 2;;
        --webarena-judge-timeout)
            WEBARENA_JUDGE_TIMEOUT="$2"; shift 2;;
        --wa-shopping)
            WA_SHOPPING="$2"; shift 2;;
        --wa-shopping-admin)
            WA_SHOPPING_ADMIN="$2"; shift 2;;
        --wa-reddit)
            WA_REDDIT="$2"; shift 2;;
        --wa-gitlab)
            WA_GITLAB="$2"; shift 2;;
        --wa-wikipedia)
            WA_WIKIPEDIA="$2"; shift 2;;
        --wa-map)
            WA_MAP="$2"; shift 2;;
        --wa-classifieds)
            WA_CLASSIFIEDS="$2"; shift 2;;
        -h|--help)
            cat << EOF
Usage: WEBARENA_JUDGE_API_KEY=... $0 --splits N --model-path PATH [OPTIONS]

Required Arguments:
  --splits N                              Number of parallel split jobs to launch
  --model-path PATH                       Path to model checkpoint

Required Environment:
  WEBARENA_JUDGE_API_KEY                  API key for VisualWebArena LLM judging

Optional Arguments (Parallel):
  --launch-script PATH                    Path to launch_nemotron_visualwebarena.sh

Optional Arguments (vLLM Server):
  --serve-bin PATH                        Path to serve_wrapper.py
  --chat-template PATH                    Path to chat template file
  --tokenizer PATH                        Path to tokenizer model
  --reasoning-parser-plugin PATH          Path to reasoning parser plugin
  --served-model-name NAME                Name exposed via the API (default: vllm_local)
  --port PORT                             vLLM server port (default: 8000)
  --max-model-len N                       Maximum model context length (default: 64000)
  --tensor-parallel-size N                Tensor parallelism (default: 8)
  --data-parallel-size N                  Data parallelism (default: 1)
  --container-image PATH                  Container .sqsh image for vLLM
  --container-mounts MOUNTS               Container bind mounts

Optional Arguments (VisualWebArena Evaluation):
  --visualwebarena-dir PATH               Path to repo root
  --visualwebarena-eval-script PATH       Relative path to eval script
  --visualwebarena-container-image PATH   Container image for VisualWebArena eval
  --visualwebarena-container-mounts MOUNTS Container bind mounts
  --visualwebarena-result-dir PATH        Result directory
  --visualwebarena-benchmark-path PATH    VisualWebArena JSONL benchmark path
  --visualwebarena-judge MODE             Judge mode (default: visualwebarena)
  --visualwebarena-temperature TEMP       Sampling temperature (default: 0.1)
  --visualwebarena-workers N              Parallel eval workers per split job (default: 16)
  --visualwebarena-max-steps N            Max agent steps per task (default: 100)
  --visualwebarena-max-model-len N        Eval context budget length (default: 49152)
  --visualwebarena-viewport-width N       Browser viewport width (default: 1920)
  --visualwebarena-viewport-height N      Browser viewport height (default: 1080)
  --visualwebarena-timeout N              Deprecated no-op; max_steps bounds task runtime
  --visualwebarena-task-ids IDS           Task IDs to run (e.g. "0-50" or "0,5,10")
  --visualwebarena-sites SITES            Filter by site/web_name (comma-separated)

Optional Arguments (WebArena Judge):
  --webarena-judge-model NAME             Judge model
  --webarena-judge-base-url URL           Judge OpenAI-compatible base URL
  --webarena-judge-timeout SECONDS        Judge request timeout

Optional Arguments (WebArena Site URLs):
  --wa-shopping URL                       Shopping site URL
  --wa-shopping-admin URL                 Shopping admin site URL
  --wa-reddit URL                         Reddit site URL
  --wa-gitlab URL                         GitLab site URL
  --wa-wikipedia URL                      Wikipedia site URL
  --wa-map URL                            Map site URL
  --wa-classifieds URL                    Classifieds site URL

Example:
  WEBARENA_JUDGE_API_KEY=... $0 --splits 4 --model-path /path/to/model
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

if [[ -z "$VISUALWEBARENA_BENCHMARK_PATH" ]]; then
    echo "ERROR: --visualwebarena-benchmark-path is required" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

if [[ -z "${WEBARENA_JUDGE_API_KEY:-}" ]]; then
    echo "ERROR: WEBARENA_JUDGE_API_KEY must be set for VisualWebArena evaluation" >&2
    exit 1
fi

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "ERROR: Launch script not found: $LAUNCH_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$VISUALWEBARENA_DIR" ]]; then
    echo "ERROR: VisualWebArena repo directory not found: $VISUALWEBARENA_DIR" >&2
    exit 1
fi

if [[ "$VISUALWEBARENA_BENCHMARK_PATH" = /* ]]; then
    BENCHMARK_FILE="$VISUALWEBARENA_BENCHMARK_PATH"
else
    BENCHMARK_FILE="$VISUALWEBARENA_DIR/$VISUALWEBARENA_BENCHMARK_PATH"
fi

if [[ ! -f "$BENCHMARK_FILE" ]]; then
    echo "ERROR: VisualWebArena benchmark not found: $BENCHMARK_FILE" >&2
    exit 1
fi

[[ -z "$VISUALWEBARENA_RESULT_DIR" ]] && VISUALWEBARENA_RESULT_DIR="$(dirname "$MODEL_PATH")/visualwebarena_results"

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
    --visualwebarena-dir "$VISUALWEBARENA_DIR"
    --visualwebarena-eval-script "$VISUALWEBARENA_EVAL_SCRIPT"
    --visualwebarena-container-image "$VISUALWEBARENA_CONTAINER_IMAGE"
    --visualwebarena-container-mounts "$VISUALWEBARENA_CONTAINER_MOUNTS"
    --visualwebarena-result-dir "$VISUALWEBARENA_RESULT_DIR"
    --visualwebarena-benchmark-path "$VISUALWEBARENA_BENCHMARK_PATH"
    --visualwebarena-judge "$VISUALWEBARENA_JUDGE"
    --visualwebarena-temperature "$VISUALWEBARENA_TEMPERATURE"
    --visualwebarena-workers "$VISUALWEBARENA_WORKERS"
    --visualwebarena-max-steps "$VISUALWEBARENA_MAX_STEPS"
    --visualwebarena-max-model-len "$VISUALWEBARENA_MAX_MODEL_LEN"
    --visualwebarena-viewport-width "$VISUALWEBARENA_VIEWPORT_WIDTH"
    --visualwebarena-viewport-height "$VISUALWEBARENA_VIEWPORT_HEIGHT"
    --visualwebarena-timeout "$VISUALWEBARENA_TIMEOUT"
    --webarena-judge-model "$WEBARENA_JUDGE_MODEL"
    --webarena-judge-base-url "$WEBARENA_JUDGE_BASE_URL"
    --webarena-judge-timeout "$WEBARENA_JUDGE_TIMEOUT"
    --wa-shopping "$WA_SHOPPING"
    --wa-shopping-admin "$WA_SHOPPING_ADMIN"
    --wa-reddit "$WA_REDDIT"
    --wa-gitlab "$WA_GITLAB"
    --wa-wikipedia "$WA_WIKIPEDIA"
    --wa-map "$WA_MAP"
    --wa-classifieds "$WA_CLASSIFIEDS"
)

[[ -n "$VISUALWEBARENA_TASK_IDS" ]] && FORWARD_ARGS+=(--visualwebarena-task-ids "$VISUALWEBARENA_TASK_IDS")
[[ -n "$VISUALWEBARENA_SITES" ]] && FORWARD_ARGS+=(--visualwebarena-sites "$VISUALWEBARENA_SITES")

# ----------------------------------------------------------
# Launch split jobs
# ----------------------------------------------------------
echo "=============================================="
echo "Launching $NUM_SPLITS parallel VisualWebArena split jobs"
echo "=============================================="
echo "  Launch script:      $LAUNCH_SCRIPT"
echo "  Model path:         $MODEL_PATH"
echo "  Benchmark path:     $VISUALWEBARENA_BENCHMARK_PATH"
echo "  Judge:              $VISUALWEBARENA_JUDGE"
echo "  Result dir:         $VISUALWEBARENA_RESULT_DIR"
echo "  Workers/split:      $VISUALWEBARENA_WORKERS"
echo "  Max steps:          $VISUALWEBARENA_MAX_STEPS"
echo "  Eval max model len: $VISUALWEBARENA_MAX_MODEL_LEN"
echo "  Viewport:           ${VISUALWEBARENA_VIEWPORT_WIDTH}x${VISUALWEBARENA_VIEWPORT_HEIGHT}"
echo "  Judge model:        $WEBARENA_JUDGE_MODEL"
echo "  Judge API key:      $(webarena_judge_key_fingerprint)"
echo "=============================================="
echo ""

JOB_IDS=()
for ((i = 0; i < NUM_SPLITS; i++)); do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,WEBARENA_JUDGE_API_KEY \
        --job-name="nemotron_omni_vision:nemotron-vwa-split-${i}-of-${NUM_SPLITS}" \
        "$LAUNCH_SCRIPT" \
        "${FORWARD_ARGS[@]}" \
        --visualwebarena-split-idx "$i" \
        --visualwebarena-split-total "$NUM_SPLITS")
    JOB_IDS+=("$JOB_ID")
    echo "  Split $i/$NUM_SPLITS -> Job $JOB_ID"
done

echo ""

# ----------------------------------------------------------
# Launch cleanup job (runs after all splits finish)
# ----------------------------------------------------------
DEPENDENCY_STR=$(IFS=:; echo "${JOB_IDS[*]}")
CLEANUP_JOB_ID=$(sbatch --parsable \
    --export=ALL,WEBARENA_JUDGE_API_KEY \
    --job-name="nemotron_omni_vision:nemotron-vwa-cleanup" \
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
