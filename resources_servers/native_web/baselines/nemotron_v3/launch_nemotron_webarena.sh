#!/bin/bash

#SBATCH -A nemotron_omni_vision
#SBATCH --job-name=nemotron_omni_vision:nemotron-webarena-evals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --partition=batch_block1,interactive
#SBATCH --output=logs/nemotron_webarena_%j.out
#SBATCH --error=logs/nemotron_webarena_%j.err

# ----------------------------------------------------------
# Nemotron vLLM Server + WebArena Evaluation (Single Node)
# This script:
#   1. Launches vLLM on a single node with TP8
#   2. Waits for server readiness
#   3. Runs WebArena-Verified benchmark evaluation
# ----------------------------------------------------------

set -euo pipefail

# ----------------------------------------------------------
# Configuration — all values should be passed via CLI args
# (defaults are managed by launch_nemotron_webarena_parallel.sh)
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

WEBARENA_DIR=""
WEBARENA_CONTAINER_IMAGE=""
WEBARENA_CONTAINER_MOUNTS=""
WEBARENA_RESULT_DIR=""
WEBARENA_BENCHMARK_PATH=""
WEBARENA_TEMPERATURE=""
WEBARENA_WORKERS=""
WEBARENA_MAX_STEPS=""
WEBARENA_TIMEOUT=""
WEBARENA_SPLIT_IDX=""
WEBARENA_SPLIT_TOTAL=""
WEBARENA_TASK_IDS=""
WEBARENA_TASK_TYPE=""
WEBARENA_SITES=""

WA_HOSTNAME="18.116.12.228"
WA_SHOPPING="http://${WA_HOSTNAME}:7770"
WA_SHOPPING_ADMIN="http://${WA_HOSTNAME}:7780/admin"
WA_REDDIT="http://${WA_HOSTNAME}:9999"
WA_GITLAB="http://${WA_HOSTNAME}:8023"
WA_WIKIPEDIA="http://${WA_HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
WA_MAP="http://${WA_HOSTNAME}:3000"

WEBARENA_JUDGE_MODEL="${WEBARENA_JUDGE_MODEL:-us/azure/openai/gpt-4.1}"
WEBARENA_JUDGE_BASE_URL="${WEBARENA_JUDGE_BASE_URL:-https://inference-api.nvidia.com}"
WEBARENA_JUDGE_TIMEOUT="${WEBARENA_JUDGE_TIMEOUT:-120}"

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
        --webarena-split-idx)
            WEBARENA_SPLIT_IDX="$2"; shift 2;;
        --webarena-split-total)
            WEBARENA_SPLIT_TOTAL="$2"; shift 2;;
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
Usage: sbatch $0 [OPTIONS]

This script expects all configuration to be passed via CLI arguments.
Use launch_nemotron_webarena_parallel.sh as the main entry point,
which manages defaults and forwards them here.

Arguments (vLLM Server):
  --model-path PATH                Path to model checkpoint
  --serve-bin PATH                 Path to serve_wrapper.py
  --chat-template PATH             Path to chat template file
  --tokenizer PATH                 Path to tokenizer model
  --reasoning-parser-plugin PATH   Path to reasoning parser plugin
  --served-model-name NAME         Name exposed via the API
  --port PORT                      vLLM server port
  --max-model-len N                Maximum model context length
  --tensor-parallel-size N         Tensor parallelism
  --data-parallel-size N           Data parallelism
  --container-image PATH           Container .sqsh image for vLLM
  --container-mounts MOUNTS        Container bind mounts

Arguments (WebArena Evaluation):
  --webarena-dir PATH              Path to repo root (containing webarena/ dir)
  --webarena-container-image PATH  Container image for WebArena eval
  --webarena-container-mounts MOUNTS Container bind mounts
  --webarena-result-dir PATH       Result directory (default: parent of model-path)
  --webarena-benchmark-path PATH   Benchmark JSONL path
  --webarena-temperature TEMP      Sampling temperature
  --webarena-workers N             Number of parallel eval workers per job
  --webarena-max-steps N           Max agent steps per task
  --webarena-timeout N             Deprecated no-op; task runtime is bounded by --max-steps
  --webarena-split-idx IDX         Split index (0-based) for parallel task distribution
  --webarena-split-total N         Total number of splits
  --webarena-task-ids IDS          Task IDs to run (e.g. "0-50" or "0,5,10")
  --webarena-task-type TYPE        Filter by task type (retrieve/mutate/navigate)
  --webarena-sites SITES           Filter by site (comma-separated)

Arguments (WebArena Site URLs):
  --wa-shopping URL                WebArena shopping site URL
  --wa-shopping-admin URL          WebArena shopping admin URL
  --wa-reddit URL                  WebArena reddit site URL
  --wa-gitlab URL                  WebArena gitlab site URL
  --wa-wikipedia URL               WebArena wikipedia site URL
  --wa-map URL                     WebArena map site URL
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
[[ -z "$WEBARENA_DIR" ]] && MISSING+=("--webarena-dir")
[[ -z "$WEBARENA_CONTAINER_IMAGE" ]] && MISSING+=("--webarena-container-image")
[[ -z "$WEBARENA_CONTAINER_MOUNTS" ]] && MISSING+=("--webarena-container-mounts")
[[ -z "$WA_SHOPPING" ]] && MISSING+=("--wa-shopping")
[[ -z "$WA_SHOPPING_ADMIN" ]] && MISSING+=("--wa-shopping-admin")
[[ -z "$WA_REDDIT" ]] && MISSING+=("--wa-reddit")
[[ -z "$WA_GITLAB" ]] && MISSING+=("--wa-gitlab")
[[ -z "$WA_WIKIPEDIA" ]] && MISSING+=("--wa-wikipedia")
[[ -z "$WA_MAP" ]] && MISSING+=("--wa-map")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing required arguments: ${MISSING[*]}" >&2
    echo "Use -h or --help for usage information"
    exit 1
fi

[[ -z "$WEBARENA_RESULT_DIR" ]] && WEBARENA_RESULT_DIR="$(dirname "$MODEL_PATH")/webarena_results"
[[ -z "$WEBARENA_BENCHMARK_PATH" ]] && WEBARENA_BENCHMARK_PATH="webarea/webarena.jsonl"
[[ -z "$WEBARENA_WORKERS" ]] && WEBARENA_WORKERS="4"
[[ -z "$WEBARENA_MAX_STEPS" ]] && WEBARENA_MAX_STEPS="100"
[[ -z "$WEBARENA_TIMEOUT" ]] && WEBARENA_TIMEOUT="4000"

if [[ ! -d "$WEBARENA_DIR" ]]; then
    echo "ERROR: WebArena directory not found: $WEBARENA_DIR" >&2
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
echo "Nemotron vLLM Server + WebArena Evaluation"
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
echo "WebArena Configuration:"
echo "  WebArena Dir:        $WEBARENA_DIR"
echo "  Container Image:     $WEBARENA_CONTAINER_IMAGE"
echo "  Result Dir:          $WEBARENA_RESULT_DIR"
echo "  Benchmark Path:      $WEBARENA_BENCHMARK_PATH"
echo "  Workers:             $WEBARENA_WORKERS"
echo "  Max Steps:           $WEBARENA_MAX_STEPS"
echo "  Timeout:             disabled (max_steps only)"
[[ -n "$WEBARENA_TEMPERATURE" ]] && echo "  Temperature:         $WEBARENA_TEMPERATURE"
[[ -n "$WEBARENA_SPLIT_IDX" ]] && echo "  Split:               $WEBARENA_SPLIT_IDX / $WEBARENA_SPLIT_TOTAL"
echo "  Judge Model:         $WEBARENA_JUDGE_MODEL"
echo "  Judge Base URL:      $WEBARENA_JUDGE_BASE_URL"
echo "  Judge API Key:       $(webarena_judge_key_fingerprint)"
echo ""
echo "WebArena Sites:"
echo "  Shopping:            $WA_SHOPPING"
echo "  Shopping Admin:      $WA_SHOPPING_ADMIN"
echo "  Reddit:              $WA_REDDIT"
echo "  GitLab:              $WA_GITLAB"
echo "  Wikipedia:           $WA_WIKIPEDIA"
echo "  Map:                 $WA_MAP"
echo "==========================================="
echo ""

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1
export WEBARENA_JUDGE_MODEL
export WEBARENA_JUDGE_BASE_URL
export WEBARENA_JUDGE_TIMEOUT
[[ -n "${WEBARENA_JUDGE_API_KEY:-}" ]] && export WEBARENA_JUDGE_API_KEY

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

VLLM_LOG="logs/vllm_nemotron_webarena_${SLURM_JOB_ID}.log"

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" $CONTAINER_ARGS \
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
# Run WebArena Evaluation
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting WebArena evaluation..."
echo ""

WEBARENA_LOG="logs/webarena_nemotron_${SLURM_JOB_ID}.log"

WEBARENA_ARGS="--tool-call"
WEBARENA_ARGS+=" --model $SERVED_MODEL_NAME"
WEBARENA_ARGS+=" --result_dir $WEBARENA_RESULT_DIR"
WEBARENA_ARGS+=" --benchmark_path $WEBARENA_BENCHMARK_PATH"
WEBARENA_ARGS+=" --workers $WEBARENA_WORKERS"
WEBARENA_ARGS+=" --max_steps $WEBARENA_MAX_STEPS"
WEBARENA_ARGS+=" --thinking"
[[ -n "$WEBARENA_TEMPERATURE" ]] && WEBARENA_ARGS+=" --temperature $WEBARENA_TEMPERATURE"
[[ -n "$WEBARENA_SPLIT_IDX" ]] && WEBARENA_ARGS+=" --split_idx $WEBARENA_SPLIT_IDX"
[[ -n "$WEBARENA_SPLIT_TOTAL" ]] && WEBARENA_ARGS+=" --split_total $WEBARENA_SPLIT_TOTAL"
[[ -n "$WEBARENA_TASK_IDS" ]] && WEBARENA_ARGS+=" --task_ids $WEBARENA_TASK_IDS"
[[ -n "$WEBARENA_TASK_TYPE" ]] && WEBARENA_ARGS+=" --task_type $WEBARENA_TASK_TYPE"
[[ -n "$WEBARENA_SITES" ]] && WEBARENA_ARGS+=" --sites $WEBARENA_SITES"

srun --jobid=$SLURM_JOB_ID \
     --nodelist="$HEAD_NODE" \
     --container-image="$WEBARENA_CONTAINER_IMAGE" \
     --container-mounts="$WEBARENA_CONTAINER_MOUNTS" \
     --overlap \
     bash -c "
         export WA_SHOPPING='$WA_SHOPPING'
         export WA_SHOPPING_ADMIN='$WA_SHOPPING_ADMIN'
         export WA_REDDIT='$WA_REDDIT'
         export WA_GITLAB='$WA_GITLAB'
         export WA_WIKIPEDIA='$WA_WIKIPEDIA'
         export WA_MAP='$WA_MAP'
         export VLLM_API_ENDPOINT='http://${HEAD_ADDR}:${PORT}/v1/chat/completions'
         export VLLM_API_KEY='EMPTY'
         cd '$WEBARENA_DIR'
         python webarena/nvidia/run_eval_parallel.py $WEBARENA_ARGS
     " 2>&1 | tee "$WEBARENA_LOG"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WebArena evaluation completed successfully!"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WebArena evaluation failed with exit code: $EVAL_EXIT_CODE" >&2
fi

# ----------------------------------------------------------
# Job Summary
# ----------------------------------------------------------
echo ""
echo "==========================================="
echo "Job Summary"
echo "==========================================="
echo "  vLLM log:      $VLLM_LOG"
echo "  WebArena log:  $WEBARENA_LOG"
echo "  Job completed at: $(date)"
echo "==========================================="

exit $EVAL_EXIT_CODE
