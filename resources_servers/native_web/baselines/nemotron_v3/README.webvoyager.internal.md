# Running WebVoyager Evaluation with Nemotron

## Quick Start

```bash
export WEBARENA_JUDGE_API_KEY="<nvidia-inference-api-key>"
export WA_BROWSER_PROXY_SERVER="http://<proxy-host>:<proxy-port>"
export CAPSOLVER_API_KEY="<capsolver-api-key>"

./launch_nemotron_webvoyager_parallel.sh --splits 4 --model-path /path/to/your/model
```

This launches parallel SLURM jobs that each spin up a vLLM server, run a split of the WebVoyager benchmark, and finally submit a cleanup job to retry any failures. WebVoyager is latency dominated, so using multiple splits is usually the easiest way to improve wall-clock time.

## Before You Launch

### 1. Configure `launch_nemotron_webvoyager_parallel.sh`

Most defaults should work out of the box. The two settings you will likely need to change are the **container mount paths**, since they need a local empty root directory mounted into the containers:

- **`CONTAINER_MOUNTS`** - bind mounts for the vLLM server container.
- **`WEBVOYAGER_CONTAINER_MOUNTS`** - bind mounts for the WebVoyager evaluation container.

Both follow the format `host_path:container_path,...`. Create an empty directory on your lustre path to serve as the mounted `/root` inside the containers:

```bash
mkdir -p /lustre/fsw/portfolios/nvr/users/<your_username>/root
```

Then update the two variables:

```bash
CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/<your_username>/root:/root,/dev/shm:/dev/shm"
WEBVOYAGER_CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/<your_username>/root:/root"
```

### 2. Set required environment variables

WebVoyager uses live public websites and an LLM/VLM judge, so these environment variables are important:

```bash
export WEBARENA_JUDGE_API_KEY=""
export WA_BROWSER_PROXY_SERVER=""
export CAPSOLVER_API_KEY=""
```

| Variable | Description |
|----------|-------------|
| `WEBARENA_JUDGE_API_KEY` | API key from [NVIDIA Inference](https://inference.nvidia.com/). It is used by the WebVoyager LLM/VLM judge. |
| `WA_BROWSER_PROXY_SERVER` | Browser proxy server address, for example `http://host:port` or `http://user:pass@host:port`. I usually use a random AWS host from `brev.nvidia.com` running Squid; export the necessary ports from the instance and use that address here. |
| `CAPSOLVER_API_KEY` | API key from [CapSolver](https://dashboard.capsolver.com/). Register an account and get an API key. Each run costs roughly 10 cents, so 10 dollars should go a long way. |

The default judge settings are:

```bash
export WEBARENA_JUDGE_MODEL="${WEBARENA_JUDGE_MODEL:-gcp/google/gemini-3-flash-preview}"
export WEBARENA_JUDGE_BASE_URL="${WEBARENA_JUDGE_BASE_URL:-https://inference-api.nvidia.com}"
export WEBARENA_JUDGE_TIMEOUT="${WEBARENA_JUDGE_TIMEOUT:-120}"
```

### 3. Verify access to dependency files

Check that you can access the following files referenced by the script:

```
SERVE_BIN="/lustre/fsw/portfolios/llmservice/users/kchumachenko/nano_v3_vllm/vllm/serve_wrapper.py"
VLLM_CHAT_TMPL="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/mingjiel/workspace/output/nemotron_v3.chat_template.keep_history.jinja"
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/nano-tokenizer"
REASONING_PARSER_PLUGIN="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/NVIDIA-Nemotron-Nano-12B-v2/nano_v3_reasoning_parser.py"
CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/pytorch25.06-moe-avlm-eval-1217-vllm-gpu.sqsh"
WEBVOYAGER_CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/webarena.sqsh"
WEBVOYAGER_BENCHMARK_PATH="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/osworld_internal/webarena/benchmarks/webvoyager.jsonl"
```

If you have permission issues or cannot access them, please reach out to Mingjie Liu (mingjiel@nvidia.com).

### 4. Check SBATCH headers in `launch_nemotron_webvoyager.sh`

Review the SLURM directives at the top of `launch_nemotron_webvoyager.sh` and update if needed:

| Directive | Current Value | Notes |
|-----------|---------------|-------|
| `--account` | `nemotron_omni_vision` | Change to your SLURM account |
| `--job-name` | `nemotron_omni_vision:nemotron-webvoyager-evals` | Optional rename |
| `--partition` | `batch_block1,interactive` | Change to partitions you have access to |

Everything else (nodes, GPUs, time limit, etc.) should not need modification.

## Usage

```bash
./launch_nemotron_webvoyager_parallel.sh --splits <N> --model-path <PATH> [OPTIONS]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `--splits N` | Number of parallel split jobs |
| `--model-path` | Path to the model checkpoint |

**Common optional arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--webvoyager-result-dir` | `<model_parent>/webvoyager_results` | Where results are saved |
| `--webvoyager-benchmark-path` | `webarena/benchmarks/webvoyager.jsonl` | WebVoyager JSONL benchmark path |
| `--webvoyager-workers` | `16` | Parallel browser workers per split job |
| `--webvoyager-max-steps` | `100` | Max agent steps per task |
| `--webvoyager-temperature` | `0.1` | Sampling temperature |
| `--webvoyager-task-ids` | unset | Task IDs to run, e.g. `0-50` or `0,5,10` |
| `--webvoyager-sites` | unset | Filter by site/web_name, comma-separated |
| `--webvoyager-viewport-width` | `1920` | Browser viewport width |
| `--webvoyager-viewport-height` | `1080` | Browser viewport height |

`--model-path` should point to a vLLM-compatible checkpoint, for example:

```
/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/v3_v4_v1369_0218_kimi_distilled_v1_49k/checkpoints/tp_1_hf/iter_0001444/mcore_to_hf
```

Using the example above, results would default to:

```
/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/v3_v4_v1369_0218_kimi_distilled_v1_49k/checkpoints/tp_1_hf/iter_0001444/webvoyager_results
```

Run `./launch_nemotron_webvoyager_parallel.sh --help` for the full list of options.

## Proxy And Captcha Notes

The browser proxy is selected only for known troublesome WebVoyager start URLs. The current proxy-trigger list includes:

- `https://www.allrecipes.com/`
- `https://www.amazon.com/`
- `https://dictionary.cambridge.org/`
- `https://www.google.com/maps/`

Known site behavior:

| Website | Notes |
|---------|-------|
| `https://www.allrecipes.com/` | Blocks the entire site with captcha. Needs proxy plus captcha solvers. |
| `https://www.amazon.com/` | Requires a US IP. If the IP is elsewhere, many problems can fail or become inconsistent, especially tasks involving dollar amounts. |
| `https://dictionary.cambridge.org/` | Requires proxy plus captcha solvers. Sometimes Google vignette ads also block the task. The agent can try workarounds, but they may not always be reliable. |
| `https://www.google.com/maps/` | Aggressively blocks some proxies. Check the proxy before large runs. A US IP works best. |

These captcha and proxy issues can cause up to a 10% performance drop if the proxy is blocked, the IP is not in the expected region, or the solver fails intermittently.

There can also be occasional hiccups from `https://arxiv.org/` and `https://github.com/`. The model can usually navigate around these. Some other sites, such as IMDb, also fail sometimes, but very few benchmark problems encounter them.

To smoke test proxy and captcha handling before a large run:

```bash
python webarena/test_captcha.py --url https://www.allrecipes.com/
python webarena/test_captcha.py --url https://dictionary.cambridge.org/
python webarena/test_captcha.py --url https://www.google.com/maps/
```

## How It Works

1. The parallel script submits `N` SLURM jobs, each running `launch_nemotron_webvoyager.sh` with a different `--webvoyager-split-idx`.
2. Each job starts a vLLM server, waits for it to be ready, then runs its assigned split of WebVoyager tasks.
3. Each WebVoyager worker launches a headed Chrome instance on Xvfb, interacts through pyautogui/browser tools, and writes per-task trajectories and screenshots.
4. Completed tasks are judged with the WebVoyager LLM/VLM judge using `WEBARENA_JUDGE_API_KEY`.
5. After all split jobs finish, a cleanup job runs the full task list without splits to retry any failures.
6. Monitor jobs with `squeue -j <job_ids>`. Logs are written to the `./logs/` directory. These logs can get quite large, so consider cleaning them out periodically (`rm -rf logs/*`).

## Viewing Results

Results are saved under `<result_dir>/`. Each task has a directory such as `task_<id>/` containing:

- `traj.jsonl` - trajectory, one JSON line per step
- `step_*.png` - screenshots
- `result.txt` - numeric score
- `result.json` - detailed task result
- `instruction.txt` - task instruction
- `worker.log` - per-task log
- `network.har` - HAR trace
- `webvoyager_judge_response.json` or `.jsonl` - judge response and parsed verdict

The aggregate summary is written to:

```
<result_dir>/results.json
```

Launch the interactive result viewer:

```bash
python webarena/nvidia/visualize_results.py <result_dir>
```

Then open `http://localhost:8888` in your browser. Use `--port` to change the port if needed:

```bash
python webarena/nvidia/visualize_results.py --port 9999 <result_dir>
```

Note: `visualize_results.py` requires `flask` (`pip install flask`).
