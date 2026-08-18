# Nemotron Production-Style Evaluation

Evaluates the Nemotron agent on WebArena-Verified using a production-style browser automation setup: **Xvfb virtual displays, headed Chrome, and pyautogui** for all interactions and screenshots.

## Architecture

Each worker runs in its own process with:
- **Xvfb** virtual display (`:99`, `:100`, `:101`, etc.)
- **Headed Chrome** launched via Playwright on that display
- **pyautogui** for mouse/keyboard actions and full-display screenshots
- **Playwright** for browser lifecycle management, HAR recording, and site logins only

```
Main Process
├── Worker 0 (Xvfb :99)  → Chrome → pyautogui → vLLM API
├── Worker 1 (Xvfb :100) → Chrome → pyautogui → vLLM API
├── Worker 2 (Xvfb :101) → Chrome → pyautogui → vLLM API
└── Worker N (Xvfb :99+N) → Chrome → pyautogui → vLLM API
```

## Prerequisites

```bash
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get update
sudo apt-get install -y xvfb scrot x11-utils python3-tk python3-dev
playwright install

pip install httpx pyautogui python-xlib Pillow
```

## Environment Variables

```bash
# WebArena sites
export WA_SHOPPING="http://<host>:7770"
export WA_SHOPPING_ADMIN="http://<host>:7780/admin"
export WA_REDDIT="http://<host>:9999"
export WA_GITLAB="http://<host>:8023"
export WA_WIKIPEDIA="http://<host>:8888"
export WA_MAP="http://<host>:443"

# vLLM API
export VLLM_API_KEY="your-key-or-EMPTY"
export VLLM_API_ENDPOINT="http://<endpoint>/v1/chat/completions"
```

## Usage

### Single task

```bash
python webarena/nvidia/run_eval.py --task_id 410 --model nemotron
```

### Parallel evaluation

```bash
# 8 workers, all tasks
python webarena/nvidia/run_eval_parallel.py --model nemotron --workers 8

# Specific tasks
python webarena/nvidia/run_eval_parallel.py --model nemotron --task_ids 0,5,10,410

# Range of tasks
python webarena/nvidia/run_eval_parallel.py --model nemotron --task_ids 0-50

# Filter by type
python webarena/nvidia/run_eval_parallel.py --model nemotron --workers 4 --task_type retrieve

# Resume (auto-detects completed tasks)
python webarena/nvidia/run_eval_parallel.py --model nemotron --workers 8
```

### Resource estimates

Each worker uses ~300-500MB (Xvfb + Chrome). Plan accordingly:
- 8 workers: ~2.5-4GB RAM
- 16 workers: ~5-8GB RAM
- 32 workers: ~10-16GB RAM

## Output

Results are saved to `webarena/nvidia/results/<model>/`:
- `task_<id>/traj.jsonl` — trajectory (one JSON line per step)
- `task_<id>/step_*.png` — screenshots from pyautogui
- `task_<id>/result.txt` — score
- `task_<id>/result.json` — detailed result
- `task_<id>/instruction.txt` — augmented instruction
- `task_<id>/worker.log` — per-task log
- `task_<id>/network.har` — HAR trace for evaluation
- `results.json` — aggregate results across all tasks
