# Running OSWorld Evaluation with Nemotron

## Quick Start

```bash
./launch_nemotron_osworld_parallel.sh --splits 2 --model-path /path/to/your/model
```

This launches parallel SLURM jobs that each spin up a vLLM server, run a split of the OSWorld benchmark, and finally submit a cleanup job to retry any failures. You can increase/decrease the split number based on available resources.
I tested with 2 node it takes ~2.5 hours. Suggest to use splits 2-4, since inference would always be dominated by latency of difficult agentic tasks which could consume up to 100 turns.

## Before You Launch

### 1. Configure `launch_nemotron_osworld_parallel.sh`

Most defaults should work out of the box. The two settings you will likely need to change are the **container mount paths**, since they need a local empty root directory mounted into the containers:

- **`CONTAINER_MOUNTS`** — bind mounts for the vLLM server container.
- **`OSWORLD_CONTAINER_MOUNTS`** — bind mounts for the OSWorld evaluation container.

Both follow the format `host_path:container_path,...`. Create an empty directory on your lustre path to serve as the mounted `/root` inside the container:

```bash
mkdir -p /lustre/fsw/portfolios/nvr/users/<your_username>/root
```

Then update the two variables:

```bash
CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/<your_username>/root:/root,/dev/shm:/dev/shm"
OSWORLD_CONTAINER_MOUNTS="/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/<your_username>/root:/root"
```

### 2. Verify access to dependency files

Check that you can access the following files referenced by the script:

```
SERVE_BIN="/lustre/fsw/portfolios/llmservice/users/kchumachenko/nano_v3_vllm/vllm/serve_wrapper.py"
VLLM_CHAT_TMPL="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/mingjiel/workspace/output/nemotron_v3.chate_template_new.jinja"
TOKENIZER_MODEL="/lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/megatron-lm/nano-tokenizer"
REASONING_PARSER_PLUGIN="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/NVIDIA-Nemotron-Nano-12B-v2/nano_v3_reasoning_parser.py"
OSWORLD_CONTAINER_IMAGE="/lustre/fsw/portfolios/nvr/users/mingjiel/containers/osworld.sqsh"
OSWORLD_SETUP_CACHE_DIR="/lustre/fsw/portfolios/nvr/users/mingjiel/root/osworld_cache/"
NVCF_SINGULARITY_SIF_PATH="/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/nvcf-osworld-eval/osworld-linux.sif"
```

If you have permission issues or cannot access them, please reach out to Mingjie Liu (mingjiel@nvidia.com).

### 3. Check SBATCH headers in `launch_nemotron_osworld.sh`

Review the SLURM directives at the top of `launch_nemotron_osworld.sh` and update if needed:

| Directive     | Current Value                                  | Notes                                    |
|---------------|-------------------------------------------------|------------------------------------------|
| `--account`   | `llmservice_fm_vision`                          | Change to your SLURM account             |
| `--job-name`  | `llmservice_fm_vision:nemotron-osworld-evals`   | Optional rename                          |
| `--partition` | `batch_block1,interactive`                      | Change to partitions you have access to  |

Everything else (nodes, GPUs, time limit, etc.) should not need modification.

## Usage

```bash
./launch_nemotron_osworld_parallel.sh --splits <N> --model-path <PATH> [OPTIONS]
```

**Required arguments:**

| Argument       | Description                   |
|----------------|-------------------------------|
| `--splits N`   | Number of parallel split jobs |
| `--model-path` | Path to the model checkpoint  |

**Common optional arguments:**

| Argument               | Default                  | Description                            |
|------------------------|--------------------------|----------------------------------------|
| `--osworld-result-dir` | Parent of `--model-path` | Where results are saved                |
| `--osworld-num-envs`   | `16`                     | Parallel VM environments per split job |
| `--osworld-temperature`| `0.6`                    | Sampling temperature                   |

`--model-path` should point to a vLLM-compatible checkpoint, for example:

```
/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/v3_v4_v1369_0218_kimi_distilled_v1_49k/checkpoints/tp_1_hf/iter_0001444/mcore_to_hf
```

`--osworld-result-dir` defaults to the parent directory of `--model-path`. Using the example above, results would be saved to:

```
/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/output/v3_v4_v1369_0218_kimi_distilled_v1_49k/checkpoints/tp_1_hf/iter_0001444/pyautogui
```

Run `./launch_nemotron_osworld_parallel.sh --help` for the full list of options.

## How It Works

1. The parallel script submits `N` SLURM jobs, each running `launch_nemotron_osworld.sh` with a different `--osworld-split-idx`.
2. Each job starts a vLLM server, waits for it to be ready, then runs its assigned split of OSWorld tasks.
3. After all split jobs finish, a cleanup job runs the full task list (without splits) to retry any tasks that failed with environments setups.
4. Monitor jobs with `squeue -j <job_ids>`. Logs are written to the `./logs/` directory. These logs can get quite large, so consider cleaning them out periodically (`rm -rf logs/*`).

## Viewing Results

Results are saved under `<result_dir>/pyautogui/screenshot/vllm_local/`. Two scripts are available for inspecting them.

### `visualize_results.py` — Interactive web viewer

Launches a local web UI to browse per-task step-by-step trajectories with annotated screenshots:

```bash
python visualize_results.py <result_dir>/pyautogui/screenshot/vllm_local
```

Then open `http://localhost:8888` in your browser. Use `--port` to change the port:

```bash
python visualize_results.py --port 9999 <result_dir>/pyautogui/screenshot/vllm_local
```

Note: `visualize_results.py` requires `flask` (`pip install flask`).

### `show_result.py` — Print summary statistics

Pass the results directory to get per-domain and overall success rates:

```bash
python show_result.py <result_dir>/pyautogui/screenshot/vllm_local
```

The expected output is as follows:
```
Domain: thunderbird Runned: 15 Success Rate: 66.66666666666666 %
Domain: multi_apps Runned: 93 Success Rate: 20.596871053796583 %
Domain: chrome Runned: 46 Success Rate: 60.86956521739131 %
Domain: vlc Runned: 17 Success Rate: 64.15583863175054 %
Domain: os Runned: 24 Success Rate: 54.166666666666664 %
Domain: libreoffice_calc Runned: 47 Success Rate: 51.06382978723404 %
Domain: vs_code Runned: 23 Success Rate: 60.86956521739131 %
Domain: libreoffice_writer Runned: 23 Success Rate: 43.46844114809873 %
Domain: gimp Runned: 26 Success Rate: 53.84615384615385 %
Domain: libreoffice_impress Runned: 47 Success Rate: 48.729741185149734 %
>>>>>>>>>>>>>
Office Success Rate: 48.633093864173574 %
Daily Success Rate: 62.70063149666358 %
Professional Success Rate: 57.14285714285714 %
Runned: 361 Current Success Rate: 45.97 % 165.96 / 361
```

Note that the complete benchmark should have 361 cases (missing a few is fine). Also temperature is 0.6 with long trajectories so do expect some variance across experiments (1-2%).