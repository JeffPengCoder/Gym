#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH --job-name=nemotron_omni_vision:nemotron-webvoyager-evals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --partition=batch_block1,interactive
#SBATCH --output=logs/nemotron_webvoyager_%j.out
#SBATCH --error=logs/nemotron_webvoyager_%j.err

# ----------------------------------------------------------
# Nemotron vLLM Server + WebVoyager Evaluation (Single Node)
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

WEBVOYAGER_DIR=""
WEBVOYAGER_EVAL_SCRIPT=""
WEBVOYAGER_CONTAINER_IMAGE=""
WEBVOYAGER_CONTAINER_MOUNTS=""
WEBVOYAGER_RESULT_DIR=""
WEBVOYAGER_BENCHMARK_PATH=""
WEBVOYAGER_TEMPERATURE=""
WEBVOYAGER_WORKERS=""
WEBVOYAGER_MAX_STEPS=""
WEBVOYAGER_MAX_MODEL_LEN=""
WEBVOYAGER_VIEWPORT_WIDTH=""
WEBVOYAGER_VIEWPORT_HEIGHT=""
WEBVOYAGER_TIMEOUT=""
WEBVOYAGER_TASK_IDS=""
WEBVOYAGER_SITES=""
WEBVOYAGER_SPLIT_IDX=""
WEBVOYAGER_SPLIT_TOTAL=""

WEBARENA_JUDGE_API_KEY="${WEBARENA_JUDGE_API_KEY:-}"
WEBARENA_JUDGE_MODEL="${WEBARENA_JUDGE_MODEL:-gcp/google/gemini-3-flash-preview}"
WEBARENA_JUDGE_BASE_URL="${WEBARENA_JUDGE_BASE_URL:-https://inference-api.nvidia.com}"
WEBARENA_JUDGE_TIMEOUT="${WEBARENA_JUDGE_TIMEOUT:-120}"
WA_BROWSER_PROXY_SERVER="${WA_BROWSER_PROXY_SERVER:-}"

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
        --webvoyager-split-idx)
            WEBVOYAGER_SPLIT_IDX="$2"; shift 2;;
        --webvoyager-split-total)
            WEBVOYAGER_SPLIT_TOTAL="$2"; shift 2;;
        --webarena-judge-api-key)
            WEBARENA_JUDGE_API_KEY="$2"; shift 2;;
        --webarena-judge-model)
            WEBARENA_JUDGE_MODEL="$2"; shift 2;;
        --webarena-judge-base-url)
            WEBARENA_JUDGE_BASE_URL="$2"; shift 2;;
        --webarena-judge-timeout)
            WEBARENA_JUDGE_TIMEOUT="$2"; shift 2;;
        -h|--help)
            cat << EOF
Usage: sbatch $0 [OPTIONS]

This script expects configuration to be passed via CLI arguments.
Use launch_nemotron_webvoyager_parallel.sh as the main entry point.

Arguments (vLLM Server):
  --model-path PATH                    Path to model checkpoint
  --serve-bin PATH                     Path to serve_wrapper.py
  --chat-template PATH                 Path to chat template file
  --tokenizer PATH                     Path to tokenizer model
  --reasoning-parser-plugin PATH       Path to reasoning parser plugin
  --served-model-name NAME             Name exposed via the API
  --port PORT                          vLLM server port
  --max-model-len N                    Maximum model context length
  --tensor-parallel-size N             Tensor parallelism
  --data-parallel-size N               Data parallelism
  --container-image PATH               Container .sqsh image for vLLM
  --container-mounts MOUNTS            Container bind mounts

Arguments (WebVoyager Evaluation):
  --webvoyager-dir PATH                Path to repo root
  --webvoyager-eval-script PATH        Relative path to eval script
  --webvoyager-container-image PATH    Container image for WebVoyager eval
  --webvoyager-container-mounts MOUNTS Container bind mounts
  --webvoyager-result-dir PATH         Result directory
  --webvoyager-benchmark-path PATH     WebVoyager JSONL benchmark path
  --webvoyager-temperature TEMP        Sampling temperature
  --webvoyager-workers N               Parallel eval workers per job
  --webvoyager-max-steps N             Max agent steps per task
  --webvoyager-max-model-len N         Eval context budget length; 0 disables eval-side compaction (default: 0)
  --webvoyager-viewport-width N        Browser viewport width
  --webvoyager-viewport-height N       Browser viewport height
  --webvoyager-timeout N               Deprecated no-op; max_steps bounds task runtime
  --webvoyager-task-ids IDS            Task IDs to run (e.g. "0-50" or "0,5,10")
  --webvoyager-sites SITES             Filter by site/web_name (comma-separated)
  --webvoyager-split-idx IDX           Split index (0-based)
  --webvoyager-split-total N           Total number of splits

Arguments (WebArena Judge):
  --webarena-judge-api-key KEY         Judge API key (prefer WEBARENA_JUDGE_API_KEY env)
  --webarena-judge-model NAME          Judge model
  --webarena-judge-base-url URL        Judge OpenAI-compatible base URL
  --webarena-judge-timeout SECONDS     Judge request timeout
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
[[ -z "$WEBVOYAGER_DIR" ]] && MISSING+=("--webvoyager-dir")
[[ -z "$WEBVOYAGER_EVAL_SCRIPT" ]] && MISSING+=("--webvoyager-eval-script")
[[ -z "$WEBVOYAGER_CONTAINER_IMAGE" ]] && MISSING+=("--webvoyager-container-image")
[[ -z "$WEBVOYAGER_CONTAINER_MOUNTS" ]] && MISSING+=("--webvoyager-container-mounts")
[[ -z "$WEBVOYAGER_BENCHMARK_PATH" ]] && MISSING+=("--webvoyager-benchmark-path")
[[ -z "$WEBARENA_JUDGE_API_KEY" ]] && MISSING+=("WEBARENA_JUDGE_API_KEY")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing required arguments/env: ${MISSING[*]}" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

[[ -z "$WEBVOYAGER_RESULT_DIR" ]] && WEBVOYAGER_RESULT_DIR="$(dirname "$MODEL_PATH")/webvoyager_results"
[[ -z "$WEBVOYAGER_WORKERS" ]] && WEBVOYAGER_WORKERS="16"
[[ -z "$WEBVOYAGER_MAX_STEPS" ]] && WEBVOYAGER_MAX_STEPS="50"
[[ -z "$WEBVOYAGER_MAX_MODEL_LEN" ]] && WEBVOYAGER_MAX_MODEL_LEN="0"
[[ -z "$WEBVOYAGER_VIEWPORT_WIDTH" ]] && WEBVOYAGER_VIEWPORT_WIDTH="1280"
[[ -z "$WEBVOYAGER_VIEWPORT_HEIGHT" ]] && WEBVOYAGER_VIEWPORT_HEIGHT="720"
[[ -z "$WEBVOYAGER_TIMEOUT" ]] && WEBVOYAGER_TIMEOUT="4000"
[[ -z "$WEBVOYAGER_TEMPERATURE" ]] && WEBVOYAGER_TEMPERATURE="0.1"

if [[ ! -d "$WEBVOYAGER_DIR" ]]; then
    echo "ERROR: WebVoyager repo directory not found: $WEBVOYAGER_DIR" >&2
    exit 1
fi

if [[ ! -f "$WEBVOYAGER_DIR/$WEBVOYAGER_EVAL_SCRIPT" ]]; then
    echo "ERROR: WebVoyager eval script not found: $WEBVOYAGER_DIR/$WEBVOYAGER_EVAL_SCRIPT" >&2
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
echo "Nemotron vLLM Server + WebVoyager Evaluation"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
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
echo "WebVoyager Configuration:"
echo "  Repo Dir:            $WEBVOYAGER_DIR"
echo "  Eval Script:         $WEBVOYAGER_EVAL_SCRIPT"
echo "  Benchmark Path:      $WEBVOYAGER_BENCHMARK_PATH"
echo "  Container Image:     $WEBVOYAGER_CONTAINER_IMAGE"
echo "  Result Dir:          $WEBVOYAGER_RESULT_DIR"
echo "  Workers:             $WEBVOYAGER_WORKERS"
echo "  Max Steps:           $WEBVOYAGER_MAX_STEPS"
echo "  Eval Max Model Len:  $WEBVOYAGER_MAX_MODEL_LEN"
echo "  Viewport:            ${WEBVOYAGER_VIEWPORT_WIDTH}x${WEBVOYAGER_VIEWPORT_HEIGHT}"
echo "  Timeout:             disabled (max_steps only)"
echo "  Temperature:         $WEBVOYAGER_TEMPERATURE"
echo "  Judge Model:         $WEBARENA_JUDGE_MODEL"
echo "  Judge Base URL:      $WEBARENA_JUDGE_BASE_URL"
echo "  Judge API Key:       $(webarena_judge_key_fingerprint)"
echo "  Browser Proxy:       $(if [[ -n "$WA_BROWSER_PROXY_SERVER" ]]; then echo enabled; else echo disabled; fi)"
[[ -n "$WEBVOYAGER_SPLIT_IDX" ]] && echo "  Split:               $WEBVOYAGER_SPLIT_IDX / $WEBVOYAGER_SPLIT_TOTAL"
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

VLLM_LOG="logs/vllm_nemotron_webvoyager_${SLURM_JOB_ID}.log"

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
# Run WebVoyager Evaluation
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting WebVoyager evaluation..."
echo ""

WEBVOYAGER_LOG="logs/webvoyager_nemotron_${SLURM_JOB_ID}.log"

EVAL_ARGS="--tool-call"
EVAL_ARGS+=" --model $SERVED_MODEL_NAME"
EVAL_ARGS+=" --result_dir $WEBVOYAGER_RESULT_DIR"
EVAL_ARGS+=" --workers $WEBVOYAGER_WORKERS"
EVAL_ARGS+=" --max_steps $WEBVOYAGER_MAX_STEPS"
EVAL_ARGS+=" --max-model-len $WEBVOYAGER_MAX_MODEL_LEN"
EVAL_ARGS+=" --tokenizer-model $TOKENIZER_MODEL"
EVAL_ARGS+=" --viewport_width $WEBVOYAGER_VIEWPORT_WIDTH"
EVAL_ARGS+=" --viewport_height $WEBVOYAGER_VIEWPORT_HEIGHT"
EVAL_ARGS+=" --temperature $WEBVOYAGER_TEMPERATURE"
EVAL_ARGS+=" --benchmark_path $WEBVOYAGER_BENCHMARK_PATH"
EVAL_ARGS+=" --judge webvoyager"
EVAL_ARGS+=" --thinking"
[[ -n "$WEBVOYAGER_TASK_IDS" ]] && EVAL_ARGS+=" --task_ids $WEBVOYAGER_TASK_IDS"
[[ -n "$WEBVOYAGER_SITES" ]] && EVAL_ARGS+=" --sites $WEBVOYAGER_SITES"
[[ -n "$WEBVOYAGER_SPLIT_IDX" ]] && EVAL_ARGS+=" --split_idx $WEBVOYAGER_SPLIT_IDX"
[[ -n "$WEBVOYAGER_SPLIT_TOTAL" ]] && EVAL_ARGS+=" --split_total $WEBVOYAGER_SPLIT_TOTAL"

export WEBARENA_JUDGE_API_KEY
export WEBARENA_JUDGE_MODEL
export WEBARENA_JUDGE_BASE_URL
export WEBARENA_JUDGE_TIMEOUT
export WA_BROWSER_PROXY_SERVER

srun --jobid=$SLURM_JOB_ID \
     --nodelist="$HEAD_NODE" \
     --container-image="$WEBVOYAGER_CONTAINER_IMAGE" \
     --container-mounts="$WEBVOYAGER_CONTAINER_MOUNTS" \
     --overlap \
     bash -c "
         export VLLM_API_ENDPOINT='http://${HEAD_ADDR}:${PORT}/v1/chat/completions'
         export VLLM_API_KEY='EMPTY'
         cd '$WEBVOYAGER_DIR'
         python $WEBVOYAGER_EVAL_SCRIPT $EVAL_ARGS
     " 2>&1 | tee "$WEBVOYAGER_LOG"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WebVoyager evaluation completed successfully!"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WebVoyager evaluation failed with exit code: $EVAL_EXIT_CODE" >&2
fi

# ----------------------------------------------------------
# Job Summary
# ----------------------------------------------------------
echo ""
echo "==========================================="
echo "Job Summary"
echo "==========================================="
echo "  vLLM log:       $VLLM_LOG"
echo "  WebVoyager log: $WEBVOYAGER_LOG"
echo "  Results dir:    $WEBVOYAGER_RESULT_DIR"
echo "  Job completed at: $(date)"
echo "==========================================="

exit $EVAL_EXIT_CODE
