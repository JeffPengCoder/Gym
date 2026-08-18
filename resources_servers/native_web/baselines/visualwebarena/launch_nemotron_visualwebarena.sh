#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH --job-name=nemotron_omni_vision:nemotron-vwa-evals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --partition=batch_block1,interactive
#SBATCH --output=logs/nemotron_visualwebarena_%j.out
#SBATCH --error=logs/nemotron_visualwebarena_%j.err

# ----------------------------------------------------------
# Nemotron vLLM Server + VisualWebArena Evaluation (Single Node)
# ----------------------------------------------------------

set -euo pipefail

# ----------------------------------------------------------
# Configuration - defaults are managed by the parallel wrapper.
# ----------------------------------------------------------
SERVE_BIN=""
VLLM_CHAT_TMPL=""
TOKENIZER_MODEL=""
REASONING_PARSER_PLUGIN=""
MODEL_PATH=""
SERVED_MODEL_NAME=""
PORT=""
MAX_MODEL_LEN=""
TENSOR_PARALLEL_SIZE=""
DATA_PARALLEL_SIZE=""
CONTAINER_IMAGE=""
CONTAINER_MOUNTS=""

VISUALWEBARENA_DIR=""
VISUALWEBARENA_EVAL_SCRIPT=""
VISUALWEBARENA_CONTAINER_IMAGE=""
VISUALWEBARENA_CONTAINER_MOUNTS=""
VISUALWEBARENA_RESULT_DIR=""
VISUALWEBARENA_BENCHMARK_PATH=""
VISUALWEBARENA_JUDGE=""
VISUALWEBARENA_TEMPERATURE=""
VISUALWEBARENA_WORKERS=""
VISUALWEBARENA_MAX_STEPS=""
VISUALWEBARENA_MAX_MODEL_LEN=""
VISUALWEBARENA_VIEWPORT_WIDTH=""
VISUALWEBARENA_VIEWPORT_HEIGHT=""
VISUALWEBARENA_TIMEOUT=""
VISUALWEBARENA_TASK_IDS=""
VISUALWEBARENA_SITES=""
VISUALWEBARENA_SPLIT_IDX=""
VISUALWEBARENA_SPLIT_TOTAL=""

WEBARENA_JUDGE_API_KEY="${WEBARENA_JUDGE_API_KEY:-}"
WEBARENA_JUDGE_MODEL="${WEBARENA_JUDGE_MODEL:-us/azure/openai/gpt-4.1}"
WEBARENA_JUDGE_BASE_URL="${WEBARENA_JUDGE_BASE_URL:-https://inference-api.nvidia.com}"
WEBARENA_JUDGE_TIMEOUT="${WEBARENA_JUDGE_TIMEOUT:-120}"

WA_SHOPPING=""
WA_SHOPPING_ADMIN=""
WA_REDDIT=""
WA_GITLAB=""
WA_WIKIPEDIA=""
WA_MAP=""
WA_CLASSIFIEDS=""

# ----------------------------------------------------------
# Parse Command Line Arguments
# ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --visualwebarena-split-idx)
            VISUALWEBARENA_SPLIT_IDX="$2"; shift 2;;
        --visualwebarena-split-total)
            VISUALWEBARENA_SPLIT_TOTAL="$2"; shift 2;;
        --webarena-judge-api-key)
            WEBARENA_JUDGE_API_KEY="$2"; shift 2;;
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
Usage: sbatch $0 [OPTIONS]

This script expects configuration to be passed via CLI arguments.
Use launch_nemotron_visualwebarena_parallel.sh as the main entry point.

Arguments (vLLM Server):
  --model-path PATH                        Path to model checkpoint
  --serve-bin PATH                         Path to serve_wrapper.py
  --chat-template PATH                     Path to chat template file
  --tokenizer PATH                         Path to tokenizer model
  --reasoning-parser-plugin PATH           Path to reasoning parser plugin
  --served-model-name NAME                 Name exposed via the API
  --port PORT                              vLLM server port
  --max-model-len N                        Maximum model context length
  --tensor-parallel-size N                 Tensor parallelism
  --data-parallel-size N                   Data parallelism
  --container-image PATH                   Container .sqsh image for vLLM
  --container-mounts MOUNTS                Container bind mounts

Arguments (VisualWebArena Evaluation):
  --visualwebarena-dir PATH                Path to repo root
  --visualwebarena-eval-script PATH        Relative path to eval script
  --visualwebarena-container-image PATH    Container image for VisualWebArena eval
  --visualwebarena-container-mounts MOUNTS Container bind mounts
  --visualwebarena-result-dir PATH         Result directory
  --visualwebarena-benchmark-path PATH     VisualWebArena JSONL benchmark path
  --visualwebarena-judge MODE              Judge mode
  --visualwebarena-temperature TEMP        Sampling temperature
  --visualwebarena-workers N               Parallel eval workers per job
  --visualwebarena-max-steps N             Max agent steps per task
  --visualwebarena-max-model-len N         Eval context budget length
  --visualwebarena-viewport-width N        Browser viewport width
  --visualwebarena-viewport-height N       Browser viewport height
  --visualwebarena-timeout N               Deprecated no-op; max_steps bounds task runtime
  --visualwebarena-task-ids IDS            Task IDs to run (e.g. "0-50" or "0,5,10")
  --visualwebarena-sites SITES             Filter by site/web_name (comma-separated)
  --visualwebarena-split-idx IDX           Split index (0-based)
  --visualwebarena-split-total N           Total number of splits

Arguments (WebArena Judge):
  --webarena-judge-api-key KEY             Judge API key (prefer WEBARENA_JUDGE_API_KEY env)
  --webarena-judge-model NAME              Judge model
  --webarena-judge-base-url URL            Judge OpenAI-compatible base URL
  --webarena-judge-timeout SECONDS         Judge request timeout

Arguments (WebArena Site URLs):
  --wa-shopping URL                        Shopping site URL
  --wa-shopping-admin URL                  Shopping admin site URL
  --wa-reddit URL                          Reddit site URL
  --wa-gitlab URL                          GitLab site URL
  --wa-wikipedia URL                       Wikipedia site URL
  --wa-map URL                             Map site URL
  --wa-classifieds URL                     Classifieds site URL
EOF
            exit 0;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use -h or --help for usage information"
            exit 1;;
    esac
done

# ----------------------------------------------------------
# Validate Required Parameters
# ----------------------------------------------------------
MISSING=()
[[ -z "$MODEL_PATH" ]] && MISSING+=("--model-path")
[[ -z "$SERVE_BIN" ]] && MISSING+=("--serve-bin")
[[ -z "$VLLM_CHAT_TMPL" ]] && MISSING+=("--chat-template")
[[ -z "$TOKENIZER_MODEL" ]] && MISSING+=("--tokenizer")
[[ -z "$REASONING_PARSER_PLUGIN" ]] && MISSING+=("--reasoning-parser-plugin")
[[ -z "$SERVED_MODEL_NAME" ]] && MISSING+=("--served-model-name")
[[ -z "$PORT" ]] && MISSING+=("--port")
[[ -z "$MAX_MODEL_LEN" ]] && MISSING+=("--max-model-len")
[[ -z "$TENSOR_PARALLEL_SIZE" ]] && MISSING+=("--tensor-parallel-size")
[[ -z "$DATA_PARALLEL_SIZE" ]] && MISSING+=("--data-parallel-size")
[[ -z "$CONTAINER_IMAGE" ]] && MISSING+=("--container-image")
[[ -z "$CONTAINER_MOUNTS" ]] && MISSING+=("--container-mounts")
[[ -z "$VISUALWEBARENA_DIR" ]] && MISSING+=("--visualwebarena-dir")
[[ -z "$VISUALWEBARENA_EVAL_SCRIPT" ]] && MISSING+=("--visualwebarena-eval-script")
[[ -z "$VISUALWEBARENA_CONTAINER_IMAGE" ]] && MISSING+=("--visualwebarena-container-image")
[[ -z "$VISUALWEBARENA_CONTAINER_MOUNTS" ]] && MISSING+=("--visualwebarena-container-mounts")
[[ -z "$VISUALWEBARENA_BENCHMARK_PATH" ]] && MISSING+=("--visualwebarena-benchmark-path")
[[ -z "$VISUALWEBARENA_JUDGE" ]] && MISSING+=("--visualwebarena-judge")
[[ -z "$WEBARENA_JUDGE_API_KEY" ]] && MISSING+=("WEBARENA_JUDGE_API_KEY")
[[ -z "$WA_SHOPPING" ]] && MISSING+=("--wa-shopping")
[[ -z "$WA_SHOPPING_ADMIN" ]] && MISSING+=("--wa-shopping-admin")
[[ -z "$WA_REDDIT" ]] && MISSING+=("--wa-reddit")
[[ -z "$WA_GITLAB" ]] && MISSING+=("--wa-gitlab")
[[ -z "$WA_WIKIPEDIA" ]] && MISSING+=("--wa-wikipedia")
[[ -z "$WA_MAP" ]] && MISSING+=("--wa-map")
[[ -z "$WA_CLASSIFIEDS" ]] && MISSING+=("--wa-classifieds")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing required arguments/env: ${MISSING[*]}" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

[[ -z "$VISUALWEBARENA_RESULT_DIR" ]] && VISUALWEBARENA_RESULT_DIR="$(dirname "$MODEL_PATH")/visualwebarena_results"
[[ -z "$VISUALWEBARENA_WORKERS" ]] && VISUALWEBARENA_WORKERS="16"
[[ -z "$VISUALWEBARENA_MAX_STEPS" ]] && VISUALWEBARENA_MAX_STEPS="100"
[[ -z "$VISUALWEBARENA_MAX_MODEL_LEN" ]] && VISUALWEBARENA_MAX_MODEL_LEN="49152"
[[ -z "$VISUALWEBARENA_VIEWPORT_WIDTH" ]] && VISUALWEBARENA_VIEWPORT_WIDTH="1920"
[[ -z "$VISUALWEBARENA_VIEWPORT_HEIGHT" ]] && VISUALWEBARENA_VIEWPORT_HEIGHT="1080"
[[ -z "$VISUALWEBARENA_TIMEOUT" ]] && VISUALWEBARENA_TIMEOUT="4000"
[[ -z "$VISUALWEBARENA_TEMPERATURE" ]] && VISUALWEBARENA_TEMPERATURE="0.1"

if [[ ! -d "$VISUALWEBARENA_DIR" ]]; then
    echo "ERROR: VisualWebArena repo directory not found: $VISUALWEBARENA_DIR" >&2
    exit 1
fi

if [[ ! -f "$VISUALWEBARENA_DIR/$VISUALWEBARENA_EVAL_SCRIPT" ]]; then
    echo "ERROR: VisualWebArena eval script not found: $VISUALWEBARENA_DIR/$VISUALWEBARENA_EVAL_SCRIPT" >&2
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

# ----------------------------------------------------------
# Setup Environment
# ----------------------------------------------------------
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

echo "==========================================="
echo "Nemotron vLLM Server + VisualWebArena Evaluation"
echo "==========================================="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Nodes: ${SLURM_NODELIST:-unknown}"
echo "Date: $(date)"
echo ""
echo "vLLM Configuration:"
echo "  Model Path:          $MODEL_PATH"
echo "  Served Model Name:   $SERVED_MODEL_NAME"
echo "  Server Port:         $PORT"
echo "  Tensor Parallel:     $TENSOR_PARALLEL_SIZE"
echo "  Data Parallel:       $DATA_PARALLEL_SIZE"
echo "  Max Model Len:       $MAX_MODEL_LEN"
echo "  Serve Binary:        $SERVE_BIN"
echo "  Chat Template:       $VLLM_CHAT_TMPL"
echo "  Tokenizer:           $TOKENIZER_MODEL"
echo "  Container Image:     ${CONTAINER_IMAGE:-<bare metal>}"
echo ""
echo "VisualWebArena Configuration:"
echo "  Repo Dir:            $VISUALWEBARENA_DIR"
echo "  Eval Script:         $VISUALWEBARENA_EVAL_SCRIPT"
echo "  Benchmark Path:      $VISUALWEBARENA_BENCHMARK_PATH"
echo "  Judge:               $VISUALWEBARENA_JUDGE"
echo "  Container Image:     $VISUALWEBARENA_CONTAINER_IMAGE"
echo "  Result Dir:          $VISUALWEBARENA_RESULT_DIR"
echo "  Workers:             $VISUALWEBARENA_WORKERS"
echo "  Max Steps:           $VISUALWEBARENA_MAX_STEPS"
echo "  Eval Max Model Len:  $VISUALWEBARENA_MAX_MODEL_LEN"
echo "  Viewport:            ${VISUALWEBARENA_VIEWPORT_WIDTH}x${VISUALWEBARENA_VIEWPORT_HEIGHT}"
echo "  Timeout:             disabled (max_steps only)"
echo "  Temperature:         $VISUALWEBARENA_TEMPERATURE"
echo "  Judge Model:         $WEBARENA_JUDGE_MODEL"
echo "  Judge Base URL:      $WEBARENA_JUDGE_BASE_URL"
echo "  Judge API Key:       $(webarena_judge_key_fingerprint)"
[[ -n "$VISUALWEBARENA_SPLIT_IDX" ]] && echo "  Split:               $VISUALWEBARENA_SPLIT_IDX / $VISUALWEBARENA_SPLIT_TOTAL"
echo ""
echo "WebArena Sites:"
echo "  Shopping:            $WA_SHOPPING"
echo "  Shopping Admin:      $WA_SHOPPING_ADMIN"
echo "  Reddit:              $WA_REDDIT"
echo "  GitLab:              $WA_GITLAB"
echo "  Wikipedia:           $WA_WIKIPEDIA"
echo "  Map:                 $WA_MAP"
echo "  Classifieds:         $WA_CLASSIFIEDS"
echo "==========================================="
echo ""

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1

# ----------------------------------------------------------
# Resolve Node Address
# ----------------------------------------------------------
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${NODES[0]}"
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Node: $HEAD_NODE ($HEAD_ADDR)"
echo ""

CONTAINER_ARGS=""
if [[ -n "$CONTAINER_IMAGE" ]]; then
    CONTAINER_ARGS="--container-image=$CONTAINER_IMAGE --container-mounts=$CONTAINER_MOUNTS"
fi

# ----------------------------------------------------------
# Cleanup
# ----------------------------------------------------------
PIDS=()
cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping PID $pid ..."
            kill "$pid"
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# ----------------------------------------------------------
# Launch vLLM Server
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM server ..."

VLLM_LOG="logs/vllm_nemotron_visualwebarena_${SLURM_JOB_ID}.log"

# The evaluation runs as a second Slurm step while vLLM stays alive.
# Mark this long-running step as overlappable so Slurm does not reserve all
# node memory exclusively for the server step.
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap $CONTAINER_ARGS \
    bash -c "python $SERVE_BIN \
        --model $MODEL_PATH \
        --trust-remote-code \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --served-model-name $SERVED_MODEL_NAME \
        --tokenizer $TOKENIZER_MODEL \
        --chat-template $VLLM_CHAT_TMPL \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.9 \
        --max-num-seqs 32 \
        --host 0.0.0.0 \
        --port $PORT \
        --swap-space 8 \
        --allowed-local-media-path / \
        --data-parallel-size $DATA_PARALLEL_SIZE \
        --reasoning-parser-plugin $REASONING_PARSER_PLUGIN \
        --reasoning-parser nano_v3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
PIDS+=($VLLM_PID)
echo "  vLLM PID: $VLLM_PID"
echo "  vLLM log: $VLLM_LOG"
echo ""

# ----------------------------------------------------------
# Wait for Server to be Ready
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for vLLM server to be ready..."
MAX_WAIT=2000
WAIT_INTERVAL=30
ELAPSED=0

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready!"
        echo ""
        echo "  Endpoint: http://${HEAD_ADDR}:${PORT}/v1"
        echo "  Model:    $SERVED_MODEL_NAME"
        echo ""
        break
    fi

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process died unexpectedly!" >&2
        echo "Check logs at: $VLLM_LOG" >&2
        exit 1
    fi

    echo "  Waiting... (${ELAPSED}s elapsed)"
    sleep "$WAIT_INTERVAL"
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo "ERROR: Server did not become ready within ${MAX_WAIT}s!" >&2
    echo "Check logs at: $VLLM_LOG" >&2
    exit 1
fi

# ----------------------------------------------------------
# Run VisualWebArena Evaluation
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting VisualWebArena evaluation..."
echo ""

VISUALWEBARENA_LOG="logs/visualwebarena_nemotron_${SLURM_JOB_ID}.log"

EVAL_ARGS="--tool-call"
EVAL_ARGS+=" --model $SERVED_MODEL_NAME"
EVAL_ARGS+=" --result_dir $VISUALWEBARENA_RESULT_DIR"
EVAL_ARGS+=" --workers $VISUALWEBARENA_WORKERS"
EVAL_ARGS+=" --max_steps $VISUALWEBARENA_MAX_STEPS"
EVAL_ARGS+=" --max-model-len $VISUALWEBARENA_MAX_MODEL_LEN"
EVAL_ARGS+=" --tokenizer-model $TOKENIZER_MODEL"
EVAL_ARGS+=" --viewport_width $VISUALWEBARENA_VIEWPORT_WIDTH"
EVAL_ARGS+=" --viewport_height $VISUALWEBARENA_VIEWPORT_HEIGHT"
EVAL_ARGS+=" --temperature $VISUALWEBARENA_TEMPERATURE"
EVAL_ARGS+=" --benchmark_path $VISUALWEBARENA_BENCHMARK_PATH"
EVAL_ARGS+=" --judge $VISUALWEBARENA_JUDGE"
EVAL_ARGS+=" --thinking"
[[ -n "$VISUALWEBARENA_TASK_IDS" ]] && EVAL_ARGS+=" --task_ids $VISUALWEBARENA_TASK_IDS"
[[ -n "$VISUALWEBARENA_SITES" ]] && EVAL_ARGS+=" --sites $VISUALWEBARENA_SITES"
[[ -n "$VISUALWEBARENA_SPLIT_IDX" ]] && EVAL_ARGS+=" --split_idx $VISUALWEBARENA_SPLIT_IDX"
[[ -n "$VISUALWEBARENA_SPLIT_TOTAL" ]] && EVAL_ARGS+=" --split_total $VISUALWEBARENA_SPLIT_TOTAL"

export WEBARENA_JUDGE_API_KEY
export WEBARENA_JUDGE_MODEL
export WEBARENA_JUDGE_BASE_URL
export WEBARENA_JUDGE_TIMEOUT
export WA_SHOPPING
export WA_SHOPPING_ADMIN
export WA_REDDIT
export WA_GITLAB
export WA_WIKIPEDIA
export WA_MAP
export WA_CLASSIFIEDS

srun --jobid=$SLURM_JOB_ID \
     --nodelist="$HEAD_NODE" \
     --container-image="$VISUALWEBARENA_CONTAINER_IMAGE" \
     --container-mounts="$VISUALWEBARENA_CONTAINER_MOUNTS" \
     --overlap \
     bash -c "
         export VLLM_API_ENDPOINT='http://${HEAD_ADDR}:${PORT}/v1/chat/completions'
         export VLLM_API_KEY='EMPTY'
         cd '$VISUALWEBARENA_DIR'
         python $VISUALWEBARENA_EVAL_SCRIPT $EVAL_ARGS
     " 2>&1 | tee "$VISUALWEBARENA_LOG"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] VisualWebArena evaluation completed successfully!"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] VisualWebArena evaluation failed with exit code: $EVAL_EXIT_CODE" >&2
fi

# ----------------------------------------------------------
# Job Summary
# ----------------------------------------------------------
echo ""
echo "==========================================="
echo "Job Summary"
echo "==========================================="
echo "  vLLM log:             $VLLM_LOG"
echo "  VisualWebArena log:   $VISUALWEBARENA_LOG"
echo "  Results dir:          $VISUALWEBARENA_RESULT_DIR"
echo "  Job completed at:     $(date)"
echo "==========================================="

exit $EVAL_EXIT_CODE
