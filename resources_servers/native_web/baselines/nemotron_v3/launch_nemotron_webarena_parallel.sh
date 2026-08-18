#!/bin/bash
# Launch multiple split sbatch jobs for parallel WebArena evaluation.
#
# Each split gets its own SLURM job running launch_nemotron_webarena.sh
# with a different --webarena-split-idx. By default, jobs are submitted in
# two waves: non-state-changing tasks first, then all remaining tasks.
#
# Usage:
#   ./launch_nemotron_webarena_parallel.sh --splits 4 --model-path /path/to/model [OPTIONS]
#
# Example:
#   ./launch_nemotron_webarena_parallel.sh --splits 4 \
#       --model-path /path/to/model \
#       --webarena-result-dir /path/to/results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_nemotron_webarena.sh"
#ROOT_CACHE="${NEMOTRON_EVAL_ROOT_CACHE:-$SCRIPT_DIR/tmp/container_root_nemotron_eval}"
ROOT_CACHE="${NEMOTRON_EVAL_ROOT_CACHE:-/lustre/fsw/portfolios/nvr/users/mingjiel/root}"
PLAYWRIGHT_SOURCE="/lustre/fsw/portfolios/nvr/users/mingjiel/root/.cache/ms-playwright/chromium-1200"

# ----------------------------------------------------------
# Input Configuration (No need to change)
# ----------------------------------------------------------
WEBARENA_RESULT_DIR=""
MODEL_PATH=""
NUM_SPLITS="2"
SUBMIT_CLEANUP="1"
SLURM_CPUS_PER_TASK="128"
SLURM_MEM="512G"

# ----------------------------------------------------------
# Default Configuration — vLLM Server
# ----------------------------------------------------------
SERVE_BIN="/lustre/fsw/portfolios/llmservice/users/kchumachenko/nano_v3_vllm/vllm/serve_wrapper.py"
VLLM_CHAT_TMPL="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/mingjiel/workspace/output/nemotron_v3.chate_template_new.jinja"
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/nano-tokenizer"
REASONING_PARSER_PLUGIN="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/NVIDIA-Nemotron-Nano-12B-v2/nano_v3_reasoning_parser.py"
SERVED_MODEL_NAME="vllm_local"
PORT="8000"
MAX_MODEL_LEN=128000
TENSOR_PARALLEL_SIZE=8
DATA_PARALLEL_SIZE=1
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/pytorch25.06-moe-avlm-eval-1217-vllm-gpu.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,$ROOT_CACHE:/root,/dev/shm:/dev/shm"

# ----------------------------------------------------------
# Default Configuration — WebArena Evaluation
# ----------------------------------------------------------
WEBARENA_DIR=$SCRIPT_DIR
WEBARENA_CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/webarena.sqsh"
WEBARENA_CONTAINER_MOUNTS="/lustre:/lustre,$ROOT_CACHE:/root,/dev/shm:/dev/shm"
WEBARENA_BENCHMARK_PATH="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/osworld_internal/webarena/benchmarks/webarena.jsonl"
WEBARENA_TEMPERATURE="0.1"
WEBARENA_WORKERS="32"
WEBARENA_MAX_STEPS="100"
WEBARENA_TIMEOUT="4000"
WEBARENA_TASK_IDS=""
WEBARENA_TASK_TYPE=""
WEBARENA_SITES=""

# Default WebArena site URLs
WA_HOSTNAME="18.116.12.228"
WA_SHOPPING="http://${WA_HOSTNAME}:7770"
WA_SHOPPING_ADMIN="http://${WA_HOSTNAME}:7780/admin"
WA_REDDIT="http://${WA_HOSTNAME}:9999"
WA_GITLAB="http://${WA_HOSTNAME}:8023"
WA_WIKIPEDIA="http://${WA_HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
WA_MAP="http://${WA_HOSTNAME}:3000"

# ----------------------------------------------------------
# Parse Command Line Arguments
# ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --splits)
            NUM_SPLITS="$2"; shift 2;;
        --no-cleanup)
            SUBMIT_CLEANUP="0"; shift;;
        --slurm-cpus-per-task|--cpus-per-task)
            SLURM_CPUS_PER_TASK="$2"; shift 2;;
        --slurm-mem|--mem)
            SLURM_MEM="$2"; shift 2;;
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
        --webarena-dir)
            WEBARENA_DIR="$2"; shift 2;;
        --webarena-container-image)
            WEBARENA_CONTAINER_IMAGE="$2"; shift 2;;
        --webarena-container-mounts)
            WEBARENA_CONTAINER_MOUNTS="$2"; shift 2;;
        --webarena-result-dir)
            WEBARENA_RESULT_DIR="$2"; shift 2;;
        --webarena-benchmark-path)
            WEBARENA_BENCHMARK_PATH="$2"; shift 2;;
        --webarena-temperature)
            WEBARENA_TEMPERATURE="$2"; shift 2;;
        --webarena-workers)
            WEBARENA_WORKERS="$2"; shift 2;;
        --webarena-max-steps)
            WEBARENA_MAX_STEPS="$2"; shift 2;;
        --webarena-timeout)
            WEBARENA_TIMEOUT="$2"; shift 2;;
        --webarena-task-ids)
            WEBARENA_TASK_IDS="$2"; shift 2;;
        --webarena-task-type)
            WEBARENA_TASK_TYPE="$2"; shift 2;;
        --webarena-sites)
            WEBARENA_SITES="$2"; shift 2;;
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
        -h|--help)
            cat << EOF
Usage: $0 --splits N --model-path PATH [OPTIONS]

Required Arguments:
  --splits N                       Number of parallel split jobs to launch
  --no-cleanup                     Do not submit cleanup job after split jobs
  --model-path PATH                Path to model checkpoint

Optional Arguments (Parallel):
  --launch-script PATH             Path to launch_nemotron_webarena.sh (auto-detected)
  --slurm-cpus-per-task N          CPUs per Slurm worker job (default: 128)
  --slurm-mem MEM                  Memory per Slurm worker job (default: 512G)

Optional Arguments (vLLM Server):
  --serve-bin PATH                 Path to serve_wrapper.py
  --chat-template PATH             Path to chat template file
  --tokenizer PATH                 Path to tokenizer model
  --reasoning-parser-plugin PATH   Path to reasoning parser plugin
  --served-model-name NAME         Name exposed via the API (default: vllm_local)
  --port PORT                      vLLM server port (default: 8000)
  --max-model-len N                Maximum model context length (default: 64000)
  --tensor-parallel-size N         Tensor parallelism (default: 8)
  --data-parallel-size N           Data parallelism (default: 1)
  --container-image PATH           Container .sqsh image for vLLM
  --container-mounts MOUNTS        Container bind mounts

Optional Arguments (WebArena Evaluation):
  --webarena-dir PATH              Path to repo root (containing webarena/ dir)
  --webarena-container-image PATH  Container image for WebArena eval
  --webarena-container-mounts MOUNTS Container bind mounts
  --webarena-result-dir PATH       Result directory (default: parent of model-path)
  --webarena-benchmark-path PATH   Benchmark JSONL path (default: webarea/webarena.jsonl)
  --webarena-temperature TEMP      Sampling temperature (default: 1.0)
  --webarena-workers N             Number of parallel eval workers per job (default: 4)
  --webarena-max-steps N           Max agent steps per task (default: 100)
  --webarena-timeout N             Deprecated no-op; task runtime is bounded by --max-steps
  --webarena-task-ids IDS          Task IDs to run (e.g. "0-50" or "0,5,10")
  --webarena-task-type TYPE        Filter by task type; disables default two-wave ordering
  --webarena-sites SITES           Filter by site (comma-separated)

Optional Arguments (WebArena Site URLs):
  --wa-shopping URL                Shopping site URL
  --wa-shopping-admin URL          Shopping admin URL
  --wa-reddit URL                  Reddit site URL
  --wa-gitlab URL                  GitLab site URL
  --wa-wikipedia URL               Wikipedia site URL
  --wa-map URL                     Map site URL

Example:
  $0 --splits 4 --model-path /path/to/model --webarena-workers 8
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

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: --model-path is required" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "ERROR: Launch script not found: $LAUNCH_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$VLLM_CHAT_TMPL" ]]; then
    echo "ERROR: Chat template not found: $VLLM_CHAT_TMPL" >&2
    exit 1
fi

[[ -z "$WEBARENA_RESULT_DIR" ]] && WEBARENA_RESULT_DIR="$(dirname "$MODEL_PATH")/webarena_results"

mkdir -p logs
mkdir -p \
    "$ROOT_CACHE/.cache/huggingface" \
    "$ROOT_CACHE/.cache/matplotlib" \
    "$ROOT_CACHE/.cache/ms-playwright" \
    "$ROOT_CACHE/.config"

if [[ ! -e "$ROOT_CACHE/.cache/ms-playwright/chromium-1200" && -e "$PLAYWRIGHT_SOURCE" ]]; then
    ln -s "$PLAYWRIGHT_SOURCE" "$ROOT_CACHE/.cache/ms-playwright/chromium-1200"
fi

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
    --webarena-dir "$WEBARENA_DIR"
    --webarena-container-image "$WEBARENA_CONTAINER_IMAGE"
    --webarena-container-mounts "$WEBARENA_CONTAINER_MOUNTS"
    --webarena-result-dir "$WEBARENA_RESULT_DIR"
    --webarena-benchmark-path "$WEBARENA_BENCHMARK_PATH"
    --webarena-temperature "$WEBARENA_TEMPERATURE"
    --webarena-workers "$WEBARENA_WORKERS"
    --webarena-max-steps "$WEBARENA_MAX_STEPS"
    --wa-shopping "$WA_SHOPPING"
    --wa-shopping-admin "$WA_SHOPPING_ADMIN"
    --wa-reddit "$WA_REDDIT"
    --wa-gitlab "$WA_GITLAB"
    --wa-wikipedia "$WA_WIKIPEDIA"
    --wa-map "$WA_MAP"
)

[[ -n "$WEBARENA_TASK_IDS" ]] && FORWARD_ARGS+=(--webarena-task-ids "$WEBARENA_TASK_IDS")
[[ -n "$WEBARENA_TASK_TYPE" ]] && FORWARD_ARGS+=(--webarena-task-type "$WEBARENA_TASK_TYPE")
[[ -n "$WEBARENA_SITES" ]] && FORWARD_ARGS+=(--webarena-sites "$WEBARENA_SITES")

SBATCH_ARGS=(--parsable --export=ALL)
[[ -n "$SLURM_CPUS_PER_TASK" ]] && SBATCH_ARGS+=(--cpus-per-task "$SLURM_CPUS_PER_TASK")
[[ -n "$SLURM_MEM" ]] && SBATCH_ARGS+=(--mem "$SLURM_MEM")

# ----------------------------------------------------------
# Launch split jobs
# ----------------------------------------------------------
echo "=============================================="
echo "Launching $NUM_SPLITS parallel WebArena split jobs"
echo "=============================================="
echo "  Launch script:     $LAUNCH_SCRIPT"
echo "  Model path:        $MODEL_PATH"
echo "  Result dir:        $WEBARENA_RESULT_DIR"
echo "  Benchmark:         $WEBARENA_BENCHMARK_PATH"
echo "  Workers/split:     $WEBARENA_WORKERS"
echo "  Max steps:         $WEBARENA_MAX_STEPS"
echo "  Slurm CPUs/task:   $SLURM_CPUS_PER_TASK"
echo "  Slurm memory:      $SLURM_MEM"
echo "  Root cache:        $ROOT_CACHE"
echo "  Timeout:           disabled (max_steps only)"
if [[ -n "$WEBARENA_TASK_TYPE" ]]; then
    echo "  Task type:         $WEBARENA_TASK_TYPE (single filtered wave)"
else
    echo "  Task order:        non_state_change wave, then all remaining tasks"
fi
echo "=============================================="
echo ""

JOB_IDS=()
FIRST_WAVE_JOB_IDS=()
SECOND_WAVE_JOB_IDS=()

if [[ -n "$WEBARENA_TASK_TYPE" ]]; then
    for ((i = 0; i < NUM_SPLITS; i++)); do
        JOB_ID=$(sbatch \
            "${SBATCH_ARGS[@]}" \
            --job-name="llmservice_fm_vision:nemotron-webarena-split-${i}-of-${NUM_SPLITS}" \
            "$LAUNCH_SCRIPT" \
            "${FORWARD_ARGS[@]}" \
            --webarena-split-idx "$i" \
            --webarena-split-total "$NUM_SPLITS")
        JOB_IDS+=("$JOB_ID")
        SECOND_WAVE_JOB_IDS+=("$JOB_ID")
        echo "  Split $i/$NUM_SPLITS -> Job $JOB_ID"
    done
else
    echo "Submitting wave 1: non_state_change splits (2 hour time limit)"
    for ((i = 0; i < NUM_SPLITS; i++)); do
        JOB_ID=$(sbatch \
            "${SBATCH_ARGS[@]}" \
            --job-name="llmservice_fm_vision:nemotron-webarena-nonstate-${i}-of-${NUM_SPLITS}" \
            --time=02:00:00 \
            "$LAUNCH_SCRIPT" \
            "${FORWARD_ARGS[@]}" \
            --webarena-task-type non_state_change \
            --webarena-split-idx "$i" \
            --webarena-split-total "$NUM_SPLITS")
        JOB_IDS+=("$JOB_ID")
        FIRST_WAVE_JOB_IDS+=("$JOB_ID")
        echo "  Wave 1 split $i/$NUM_SPLITS -> Job $JOB_ID"
    done

    FIRST_WAVE_DEPENDENCY=$(IFS=:; echo "${FIRST_WAVE_JOB_IDS[*]}")
    echo ""
    echo "Submitting wave 2: all remaining splits after wave 1 (2 hour time limit)"
    for ((i = 0; i < NUM_SPLITS; i++)); do
        JOB_ID=$(sbatch \
            "${SBATCH_ARGS[@]}" \
            --job-name="llmservice_fm_vision:nemotron-webarena-remaining-${i}-of-${NUM_SPLITS}" \
            --dependency="afterany:${FIRST_WAVE_DEPENDENCY}" \
            --time=02:00:00 \
            "$LAUNCH_SCRIPT" \
            "${FORWARD_ARGS[@]}" \
            --webarena-split-idx "$i" \
            --webarena-split-total "$NUM_SPLITS")
        JOB_IDS+=("$JOB_ID")
        SECOND_WAVE_JOB_IDS+=("$JOB_ID")
        echo "  Wave 2 split $i/$NUM_SPLITS -> Job $JOB_ID"
    done
fi

echo ""

# ----------------------------------------------------------
if [[ "$SUBMIT_CLEANUP" == "1" ]]; then
    # Launch cleanup job (runs after the final wave finishes).
    # Runs the full task list without splits to retry any failures.
    # ----------------------------------------------------------
    DEPENDENCY_STR=$(IFS=:; echo "${SECOND_WAVE_JOB_IDS[*]}")
    CLEANUP_JOB_ID=$(sbatch \
        "${SBATCH_ARGS[@]}" \
        --job-name="llmservice_fm_vision:nemotron-webarena-cleanup" \
        --dependency="afterany:${DEPENDENCY_STR}" \
        "$LAUNCH_SCRIPT" \
        "${FORWARD_ARGS[@]}")
    JOB_IDS+=("$CLEANUP_JOB_ID")
    echo "  Cleanup job   -> Job $CLEANUP_JOB_ID (runs after final wave)"
fi

echo ""
echo "=============================================="
if [[ -n "$WEBARENA_TASK_TYPE" ]]; then
    if [[ "$SUBMIT_CLEANUP" == "1" ]]; then
        echo "All jobs submitted ($NUM_SPLITS filtered splits + 1 cleanup):"
    else
        echo "All jobs submitted ($NUM_SPLITS filtered splits, no cleanup):"
    fi
else
    if [[ "$SUBMIT_CLEANUP" == "1" ]]; then
        echo "All jobs submitted (2 waves x $NUM_SPLITS splits + 1 cleanup):"
    else
        echo "All jobs submitted (2 waves x $NUM_SPLITS splits, no cleanup):"
    fi
fi
printf '  %s\n' "${JOB_IDS[@]}"
echo ""
echo "Monitor with:  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "=============================================="
