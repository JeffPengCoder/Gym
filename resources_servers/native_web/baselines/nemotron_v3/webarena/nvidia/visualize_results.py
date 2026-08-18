#!/usr/bin/env python3
"""
Visualize NVIDIA WebArena/WebVoyager agent evaluation results.

Usage:
    python webarena/nvidia/visualize_results.py webarena/nvidia/results
    python webarena/nvidia/visualize_results.py webarena/nvidia/results_toolcall

Serves a web UI at http://localhost:8888 showing per-task step-by-step
trajectories with annotated screenshots.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, send_file

app = Flask(__name__)
BASE_DIR = None  # set from CLI arg


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_annotations(action_str):
    """Extract drawable annotations from a pyautogui action string.

    Handles both positional (click(960, 540)) and keyword (click(x=0.5, y=0.5))
    argument styles. Coordinates are returned as-is (may be relative 0-1).
    """
    if not action_str:
        return []
    annotations = []
    for line in action_str.strip().split("\n"):
        line = line.strip()

        m = re.match(
            r"pyautogui\.(click|doubleClick|rightClick|tripleClick)"
            r"\((?:x=)?([\d.]+),\s*(?:y=)?([\d.]+)", line)
        if m:
            annotations.append(
                {"type": m.group(1), "x": float(m.group(2)), "y": float(m.group(3))})
            continue

        m = re.match(r"pyautogui\.moveTo\((?:x=)?([\d.]+),\s*(?:y=)?([\d.]+)", line)
        if m:
            annotations.append(
                {"type": "moveTo", "x": float(m.group(1)), "y": float(m.group(2))})
            continue

        m = re.match(r"pyautogui\.dragTo\((?:x=)?([\d.]+),\s*(?:y=)?([\d.]+)", line)
        if m:
            annotations.append(
                {"type": "dragTo", "x": float(m.group(1)), "y": float(m.group(2))})
            continue

        m = re.match(
            r"pyautogui\.scroll\((?:clicks=)?(-?[\d.]+)"
            r"(?:,\s*(?:x=)?([\d.]+),\s*(?:y=)?([\d.]+))?", line)
        if m:
            a = {"type": "scroll", "clicks": int(float(m.group(1)))}
            if m.group(2) and m.group(3):
                a["x"] = float(m.group(2))
                a["y"] = float(m.group(3))
            annotations.append(a)
            continue

        m = re.match(r"pyautogui\.(typewrite|write)\((.+)\)", line)
        if m:
            annotations.append({"type": "typewrite", "value": m.group(2)})
            continue

        m = re.match(r"pyautogui\.(keyDown|keyUp)\(['\"](.+?)['\"]\)", line)
        if m:
            annotations.append({"type": m.group(1), "value": m.group(2)})
            continue

    return annotations


def parse_tool_call_annotations(action_payload):
    """Extract drawable annotations from structured tool-call actions."""
    if not isinstance(action_payload, list):
        return []

    annotations = []
    for call in action_payload:
        if not isinstance(call, dict) or call.get("name") != "computer":
            continue
        args = call.get("arguments") or {}
        actions = args.get("actions") or []
        if not isinstance(actions, list):
            continue

        for action in actions:
            if not isinstance(action, dict):
                continue
            kind = action.get("action")
            coord = action.get("coordinate")

            def add_xy(payload, value):
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    payload["x"] = float(value[0])
                    payload["y"] = float(value[1])
                return payload

            if kind == "left_click":
                annotations.append(add_xy({"type": "click"}, coord))
            elif kind == "middle_click":
                annotations.append(add_xy({"type": "middleClick"}, coord))
            elif kind == "right_click":
                annotations.append(add_xy({"type": "rightClick"}, coord))
            elif kind == "double_click":
                annotations.append(add_xy({"type": "doubleClick"}, coord))
            elif kind == "triple_click":
                annotations.append(add_xy({"type": "tripleClick"}, coord))
            elif kind == "mouse_move":
                annotations.append(add_xy({"type": "moveTo"}, coord))
            elif kind == "scroll":
                params = action.get("scroll_parameters") or {}
                amount = int(params.get("scroll_amount", 1))
                direction = params.get("scroll_direction", "down")
                clicks = -amount if direction in ("down", "left") else amount
                payload = {"type": "scroll", "clicks": clicks, "direction": direction}
                annotations.append(add_xy(payload, coord))
            elif kind == "left_click_drag":
                start = action.get("start_coordinate")
                annotations.append(add_xy({"type": "moveTo"}, start))
                annotations.append(add_xy({"type": "dragTo"}, coord))
            elif kind == "type":
                annotations.append({"type": "typewrite", "value": action.get("text", "")})
            elif kind == "key_press":
                annotations.append({"type": "key_press", "value": "+".join(action.get("keys", []))})

    return annotations


def parse_action_annotations(action):
    """Extract screenshot annotations from either legacy code or tool calls."""
    if isinstance(action, list):
        return parse_tool_call_annotations(action)
    if isinstance(action, str):
        return parse_annotations(action)
    return []


def format_action_for_display(action):
    """Return a readable action payload for the UI."""
    if action is None:
        return ""
    if isinstance(action, str):
        return action
    return json.dumps(action, indent=2, ensure_ascii=False, default=str)


def parse_response_content(content):
    """Parse the model response content into action/code sections."""
    if not content:
        return {"raw": ""}
    sections = {}
    action_m = re.search(r"## Action:\s*\n(.+?)(?=\n## |\Z)", content, re.S)
    if action_m:
        sections["action_description"] = action_m.group(1).strip()
    code_m = re.search(r"```(?:python|code)?\n(.+?)```", content, re.S)
    if code_m:
        sections["code"] = code_m.group(1).strip()
    sections["raw"] = content.strip()
    return sections


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

MAX_TASKS = 1000


def _load_json_or_last_jsonl(path):
    """Load a JSON file or the last record from a JSONL file."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                rows = [json.loads(line) for line in f if line.strip()]
                return rows[-1] if rows else None
            return json.load(f)
    except Exception:
        return None


def _scan_task_dirs():
    """Yield (relative_name, task_dir) for task directories, sorted by name.

    Supports both result_dir/task_0 and result_dir/model_name/task_0 layouts.
    """
    found = []
    for root, dirs, _files in os.walk(BASE_DIR):
        rel_root = os.path.relpath(root, BASE_DIR)
        depth = 0 if rel_root == "." else len(Path(rel_root).parts)
        if depth > 2:
            dirs[:] = []
            continue

        for dirname in dirs:
            if dirname.startswith("task_"):
                task_dir = os.path.join(root, dirname)
                rel_name = os.path.relpath(task_dir, BASE_DIR)
                found.append((rel_name, task_dir))

        dirs[:] = [d for d in dirs if not d.startswith("task_")]

    for rel_name, task_dir in sorted(found):
        yield rel_name, task_dir


def get_tasks():
    """Return list of task dicts (capped at MAX_TASKS)."""
    tasks = []
    for entry, task_dir in _scan_task_dirs():
        if len(tasks) >= MAX_TASKS:
            break

        raw_id = os.path.basename(entry).split("_", 1)[1]
        try:
            task_id = int(raw_id)
        except ValueError:
            task_id = raw_id

        score = None
        result_path = os.path.join(task_dir, "result.txt")
        if os.path.isfile(result_path):
            with open(result_path) as f:
                try:
                    score = float(f.read().strip())
                except ValueError:
                    score = None

        instruction = ""
        inst_path = os.path.join(task_dir, "instruction.txt")
        if os.path.isfile(inst_path):
            with open(inst_path) as f:
                instruction = f.read().strip()

        task_type = ""
        eval_message = ""
        agent_answer = None
        result_json_path = os.path.join(task_dir, "result.json")
        if os.path.isfile(result_json_path):
            try:
                with open(result_json_path) as f:
                    rj = json.load(f)
                    task_type = rj.get("task_type", "")
                    eval_message = rj.get("eval_message", "")
                    agent_answer = rj.get("agent_answer")
            except Exception:
                pass

        judge_result = (
            _load_json_or_last_jsonl(os.path.join(task_dir, "webvoyager_judge_response.json"))
            or _load_json_or_last_jsonl(os.path.join(task_dir, "webvoyager_judge_response.jsonl"))
        )

        tasks.append({
            "task_id": task_id,
            "name": entry,
            "score": score,
            "instruction": instruction,
            "task_type": task_type,
            "eval_message": eval_message,
            "agent_answer": agent_answer,
            "judge": judge_result,
        })

    tasks.sort(key=lambda t: (isinstance(t["task_id"], str), t["task_id"]))
    return tasks


def get_steps(task_name):
    task_dir = os.path.join(BASE_DIR, task_name)
    traj_path = os.path.join(task_dir, "traj.jsonl")
    if not os.path.isfile(traj_path):
        return []

    entries = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    screenshot_map = {}
    for e in entries:
        if "step_num" not in e:
            continue
        screenshot_map[e["step_num"]] = e.get("screenshot_file")

    steps = []
    for e in entries:
        if "step_num" not in e:
            continue
        step_num = e["step_num"]
        if step_num == 0:
            continue

        prev_screenshot = screenshot_map.get(step_num - 1)

        resp = e.get("response")
        nl_action = e.get("natural_language_action", "")

        if isinstance(resp, dict):
            parsed_content = parse_response_content(resp.get("content"))
            reasoning = (resp.get("reasoning_content") or resp.get("reasoning") or "").strip()
        else:
            parsed_content = {"raw": ""}
            reasoning = ""

        info = e.get("info", {})
        if not reasoning and isinstance(info, dict):
            reasoning = info.get("thought", "")
        parsed_action = ""
        tool_results = []
        if isinstance(info, dict):
            parsed_action = info.get("parsed_action", "") or ""
            tool_results = info.get("tool_results", []) or []

        action_payload = e.get("action", "")
        action_display = format_action_for_display(action_payload)
        action_description = parsed_content.get("action_description") or parsed_action or nl_action

        steps.append({
            "step_num": step_num,
            "action": action_payload,
            "action_display": action_display,
            "parsed_action": parsed_action,
            "natural_language_action": nl_action,
            "reasoning": reasoning,
            "parsed_content": parsed_content,
            "action_description": action_description,
            "annotations": parse_action_annotations(action_payload),
            "screenshot": prev_screenshot,
            "done": e.get("done", False),
            "answer": info.get("answer") if isinstance(info, dict) else None,
            "tool_results": tool_results,
        })

    return steps


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/tasks")
def api_tasks():
    return jsonify(get_tasks())


@app.route("/api/steps/<path:task_name>")
def api_steps(task_name):
    return jsonify(get_steps(task_name))


@app.route("/screenshot/<path:task_name>/<filename>")
def serve_screenshot(task_name, filename):
    path = os.path.join(BASE_DIR, task_name, filename)
    if os.path.isfile(path):
        return send_file(path, mimetype="image/png")
    return "Not found", 404


@app.route("/api/worker_log/<path:task_name>")
def api_worker_log(task_name):
    task_dir = os.path.join(BASE_DIR, task_name)
    log_path = os.path.join(task_dir, "worker.log")
    if os.path.isfile(log_path):
        with open(log_path) as f:
            return Response(f.read(), mimetype="text/plain")
    return "No log file found", 404


# ---------------------------------------------------------------------------
# Main HTML page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WebArena Results Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #1a1a2e;
  --sidebar-bg: #16213e;
  --card-bg: #1f2940;
  --accent: #4fc3f7;
  --accent2: #e57373;
  --text: #e0e0e0;
  --text-dim: #90a4ae;
  --success: #66bb6a;
  --fail: #ef5350;
  --border: #2a3a5c;
}
html, body { height: 100%; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }
#app { display: flex; height: 100vh; }

/* Sidebar */
#sidebar {
  width: 340px; min-width: 260px; background: var(--sidebar-bg);
  border-right: 1px solid var(--border); display: flex; flex-direction: column;
  overflow: hidden;
}
#sidebar h1 { padding: 16px 20px; font-size: 18px; color: var(--accent); border-bottom: 1px solid var(--border); flex-shrink: 0; }
#sidebar-content { overflow-y: auto; flex: 1; }
.task-item {
  padding: 8px 20px; cursor: pointer; font-size: 13px;
  color: var(--text-dim);
  display: flex; justify-content: space-between; align-items: center;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.task-item:hover { background: rgba(79,195,247,0.08); color: var(--text); }
.task-item.active { background: rgba(79,195,247,0.15); color: var(--accent); }
.task-item .task-label { display: flex; align-items: center; gap: 8px; overflow: hidden; }
.task-item .task-id { font-family: 'Consolas', monospace; font-weight: 600; flex-shrink: 0; }
.task-item .task-type {
  font-size: 10px; padding: 1px 6px; border-radius: 3px;
  background: rgba(255,255,255,0.08); color: var(--text-dim);
  text-transform: uppercase; flex-shrink: 0;
}
.score-badge {
  font-size: 11px; padding: 1px 8px; border-radius: 10px; font-weight: 700;
  flex-shrink: 0; margin-left: 8px;
}
.score-1 { background: var(--success); color: #1a1a2e; }
.score-0 { background: var(--fail); color: #fff; }
.score-partial { background: #ffa726; color: #1a1a2e; }
.score-none { background: var(--border); color: var(--text-dim); }

.summary-bar {
  padding: 10px 20px; border-bottom: 1px solid var(--border); flex-shrink: 0;
  font-size: 12px; color: var(--text-dim);
}
.summary-bar span { font-weight: 700; }
.summary-bar .s-pass { color: var(--success); }
.summary-bar .s-fail { color: var(--fail); }

/* Main */
#main { flex: 1; overflow-y: auto; padding: 24px; }
#main.empty { display: flex; align-items: center; justify-content: center; }
#main.empty::after { content: 'Select a task from the sidebar'; color: var(--text-dim); font-size: 18px; }

.task-header {
  margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
}
.task-header h2 { font-size: 16px; color: var(--accent); margin-bottom: 4px; }
.task-header .instruction { font-size: 14px; color: var(--text); margin: 8px 0; line-height: 1.5; }
.task-header .meta { font-size: 13px; color: var(--text-dim); }
.judge-box {
  margin-top: 12px; padding: 12px; border: 1px solid var(--border);
  border-radius: 6px; background: rgba(0,0,0,0.22); font-size: 13px;
}
.judge-box h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--accent); margin-bottom: 8px; }
.judge-box .judge-verdict { font-weight: 800; }
.judge-box .judge-success { color: var(--success); }
.judge-box .judge-failure { color: var(--fail); }
.judge-box pre {
  margin-top: 8px; background: rgba(0,0,0,0.25); padding: 8px 10px;
  border-radius: 4px; white-space: pre-wrap; word-break: break-word;
  max-height: 160px; overflow-y: auto; font-family: 'Consolas', 'Fira Code', monospace;
}

/* Step card */
.step-card {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 20px; overflow: hidden;
}
.step-title {
  padding: 10px 16px; font-weight: 600; font-size: 14px;
  background: rgba(255,255,255,0.03); border-bottom: 1px solid var(--border);
  cursor: pointer; display: flex; justify-content: space-between; align-items: center;
  user-select: none;
}
.step-title:hover { background: rgba(255,255,255,0.06); }
.step-title .arrow { transition: transform 0.2s; font-size: 12px; }
.step-title.open .arrow { transform: rotate(90deg); }
.step-body { display: none; }
.step-title.open + .step-body { display: block; }
.step-content {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0;
}
.step-screenshot {
  position: relative; background: #000; display: flex; align-items: flex-start;
  justify-content: center; min-height: 200px; border-right: 1px solid var(--border);
}
.step-screenshot canvas {
  max-width: 100%; height: auto; display: block; cursor: crosshair;
}
.step-info {
  padding: 14px 16px; font-size: 13px; overflow-y: auto; max-height: 600px;
}
.info-section { margin-bottom: 14px; }
.info-section h4 {
  font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
  color: var(--accent); margin-bottom: 6px;
}
.info-section pre {
  background: rgba(0,0,0,0.3); padding: 10px 12px; border-radius: 4px;
  font-size: 12px; white-space: pre-wrap; word-break: break-word;
  font-family: 'Consolas', 'Fira Code', monospace; color: var(--text);
  max-height: 200px; overflow-y: auto; line-height: 1.5;
}
.info-section p {
  line-height: 1.6; color: var(--text); font-size: 13px;
}
.answer-badge {
  display: inline-block; background: rgba(79,195,247,0.15); border: 1px solid var(--accent);
  padding: 4px 10px; border-radius: 4px; font-family: monospace; font-size: 13px;
  color: var(--accent); margin-top: 4px;
}

/* Responsive */
@media (max-width: 1200px) {
  .step-content { grid-template-columns: 1fr; }
  .step-screenshot { border-right: none; border-bottom: 1px solid var(--border); }
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3a4a6c; }

/* Filter bar */
#filter-bar {
  padding: 8px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0;
}
#filter-bar select, #filter-bar input {
  background: var(--card-bg); border: 1px solid var(--border); color: var(--text);
  padding: 4px 8px; border-radius: 4px; font-size: 12px; width: 100%;
  margin-bottom: 4px;
}
#filter-bar label { font-size: 11px; color: var(--text-dim); }

/* Modal */
.modal-overlay {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.6); z-index: 1000;
  display: flex; align-items: center; justify-content: center;
}
.modal-box {
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
  width: 80%; max-width: 900px; max-height: 80vh; display: flex; flex-direction: column;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.modal-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 20px; border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.modal-header h3 { font-size: 15px; color: var(--accent); }
.modal-close {
  background: none; border: none; color: var(--text-dim); font-size: 22px;
  cursor: pointer; padding: 0 4px; line-height: 1;
}
.modal-close:hover { color: var(--text); }
.modal-body {
  padding: 16px 20px; overflow-y: auto; flex: 1;
}
.modal-body pre {
  background: rgba(0,0,0,0.3); padding: 14px; border-radius: 6px;
  font-size: 12px; white-space: pre-wrap; word-break: break-word;
  font-family: 'Consolas', 'Fira Code', monospace; color: var(--text);
  line-height: 1.5;
}

/* Worker log button */
.btn-worker-log {
  background: var(--card-bg); border: 1px solid var(--border); color: var(--accent);
  padding: 5px 14px; border-radius: 5px; cursor: pointer; font-size: 12px;
  font-weight: 600; margin-left: 12px; transition: background 0.15s;
}
.btn-worker-log:hover { background: rgba(79,195,247,0.12); }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>WebArena Results</h1>
    <div id="filter-bar">
      <label>Filter by score</label>
      <select id="score-filter">
        <option value="all">All</option>
        <option value="1">Pass (1.0)</option>
        <option value="0">Fail (0.0)</option>
        <option value="none">Not evaluated</option>
      </select>
      <label>Search task ID</label>
      <input id="task-search" type="text" placeholder="Type to filter...">
    </div>
    <div id="summary-bar" class="summary-bar"></div>
    <div id="sidebar-content"></div>
  </div>
  <div id="main" class="empty"></div>
</div>
<script>
const API = '';
let allTasks = [];
let currentTask = null;

// ---- Sidebar ----
async function loadSidebar() {
  allTasks = await (await fetch(`${API}/api/tasks`)).json();
  renderSummary();
  renderTaskList();
}

function renderSummary() {
  const total = allTasks.length;
  const pass = allTasks.filter(t => t.score === 1.0).length;
  const fail = allTasks.filter(t => t.score !== null && t.score < 1.0).length;
  const pending = allTasks.filter(t => t.score === null).length;
  document.getElementById('summary-bar').innerHTML =
    `<span class="s-pass">${pass}</span> pass / <span class="s-fail">${fail}</span> fail / ${pending} pending &mdash; ${total} total`;
}

function renderTaskList() {
  const container = document.getElementById('sidebar-content');
  container.innerHTML = '';
  const filter = document.getElementById('score-filter').value;
  const search = document.getElementById('task-search').value.toLowerCase();

  for (const t of allTasks) {
    const idStr = String(t.task_id);
    if (search && !idStr.includes(search) && !t.instruction.toLowerCase().includes(search)) continue;
    if (filter === '1' && t.score !== 1.0) continue;
    if (filter === '0' && (t.score === null || t.score >= 1.0)) continue;
    if (filter === 'none' && t.score !== null) continue;

    const item = document.createElement('div');
    item.className = 'task-item';
    item.dataset.name = t.name;

    let badgeClass = 'score-none';
    let scoreText = '?';
    if (t.score !== null) {
      scoreText = t.score.toFixed(1);
      if (t.score === 1.0) badgeClass = 'score-1';
      else if (t.score > 0 && t.score < 1) badgeClass = 'score-partial';
      else badgeClass = 'score-0';
    }

    const typeHtml = t.task_type ? `<span class="task-type">${escHtml(t.task_type)}</span>` : '';

    item.innerHTML = `
      <span class="task-label">
        <span class="task-id">${t.task_id}</span>
        ${typeHtml}
      </span>
      <span class="score-badge ${badgeClass}">${scoreText}</span>`;
    item.addEventListener('click', () => selectTask(t));
    container.appendChild(item);
  }
}

document.getElementById('score-filter').addEventListener('change', renderTaskList);
document.getElementById('task-search').addEventListener('input', renderTaskList);

// ---- Main content ----
async function selectTask(task) {
  currentTask = task;
  document.querySelectorAll('.task-item').forEach(el => el.classList.remove('active'));
  const active = document.querySelector(`.task-item[data-name="${task.name}"]`);
  if (active) active.classList.add('active');

  const main = document.getElementById('main');
  main.classList.remove('empty');
  main.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-dim);">Loading...</div>';

  const steps = await (await fetch(`${API}/api/steps/${task.name}`)).json();
  renderSteps(main, task, steps);
}

function renderSteps(container, task, steps) {
  let scoreText = task.score !== null ? task.score.toFixed(1) : 'N/A';
  let scoreColor = task.score === 1.0 ? 'var(--success)' : task.score === 0.0 ? 'var(--fail)' : 'var(--accent)';
  const judge = task.judge || null;
  let judgeHtml = '';
  if (judge) {
    const parsed = judge.parsed || {};
    const verdict = String(parsed.verdict || '').toUpperCase();
    const verdictClass = verdict === 'SUCCESS' ? 'judge-success' : verdict === 'FAILURE' ? 'judge-failure' : '';
    const judgeScore = judge.score !== undefined && judge.score !== null ? Number(judge.score).toFixed(1) : 'N/A';
    const thought = parsed.thought || '';
    const prediction = judge.prediction !== undefined && judge.prediction !== null ? String(judge.prediction) : '';
    const rawResponse = judge.response || '';
    judgeHtml = `
      <div class="judge-box">
        <h3>WebVoyager Judge</h3>
        <div>Verdict: <span class="judge-verdict ${verdictClass}">${escHtml(verdict || 'UNKNOWN')}</span>
          &nbsp;|&nbsp; Judge score: <strong>${escHtml(judgeScore)}</strong>
          ${task.eval_message ? `&nbsp;|&nbsp; ${escHtml(task.eval_message)}` : ''}
        </div>
        ${prediction ? `<div style="margin-top:6px;">Agent answer: <span class="answer-badge">${escHtml(prediction)}</span></div>` : ''}
        ${thought ? `<pre>${escHtml(thought)}</pre>` : ''}
        ${rawResponse && rawResponse !== thought ? `<details style="margin-top:8px;"><summary>Raw judge response</summary><pre>${escHtml(rawResponse)}</pre></details>` : ''}
      </div>`;
  } else if (task.eval_message) {
    judgeHtml = `<div class="judge-box"><h3>Evaluation</h3><div>${escHtml(task.eval_message)}</div></div>`;
  }

  container.innerHTML = `
    <div class="task-header">
      <h2>Task ${task.task_id}</h2>
      <div class="instruction">${escHtml(task.instruction)}</div>
      <div class="meta">Steps: ${steps.length} &nbsp;|&nbsp; Score: <span style="color:${scoreColor};font-weight:700">${scoreText}</span>
        <button class="btn-worker-log" onclick="showWorkerLog('${task.name}')">View worker.log</button>
      </div>
      ${judgeHtml}
    </div>
  `;

  for (const step of steps) {
    const card = document.createElement('div');
    card.className = 'step-card';

    const isLast = step.done;
    const title = document.createElement('div');
    title.className = 'step-title open';
    let titleText = `Step ${step.step_num}: ${escHtml(step.parsed_action || step.action_description || step.natural_language_action || '').substring(0, 100)}`;
    if (isLast && step.answer) titleText += ` [answer: ${escHtml(String(step.answer)).substring(0, 60)}]`;
    title.innerHTML = `<span>${titleText}</span><span class="arrow">&#9654;</span>`;
    title.addEventListener('click', function() {
      this.classList.toggle('open');
      if (this.classList.contains('open')) {
        const canvas = this.nextElementSibling.querySelector('canvas');
        if (canvas && !canvas.dataset.loaded) loadStepImage(canvas, task.name, step);
      }
    });
    card.appendChild(title);

    const body = document.createElement('div');
    body.className = 'step-body';

    const pc = step.parsed_content || {};
    const actionPayload = step.action_display || '';
    const relativeCode = pc.code || '';
    const actionDesc = step.action_description || pc.action_description || step.parsed_action || step.natural_language_action || '';
    const reasoning = step.reasoning || '';
    const modelMessage = step.natural_language_action || '';
    const toolResults = step.tool_results || [];

    let answerHtml = '';
    if (step.answer) {
      answerHtml = `<div class="info-section"><h4>Answer</h4><div class="answer-badge">${escHtml(String(step.answer))}</div></div>`;
    }

    body.innerHTML = `
      <div class="step-content">
        <div class="step-screenshot"><canvas data-loaded=""></canvas></div>
        <div class="step-info">
          <div class="info-section"><h4>Parsed Action</h4><p>${escHtml(actionDesc)}</p></div>
          ${modelMessage && modelMessage !== actionDesc ? `<div class="info-section"><h4>Model Message</h4><pre>${escHtml(modelMessage)}</pre></div>` : ''}
          <div class="info-section"><h4>Action Payload</h4><pre>${escHtml(actionPayload)}</pre></div>
          ${relativeCode && relativeCode !== actionPayload ? `<div class="info-section"><h4>Response Code</h4><pre>${escHtml(relativeCode)}</pre></div>` : ''}
          ${reasoning ? `<div class="info-section"><h4>Reasoning</h4><pre>${escHtml(reasoning)}</pre></div>` : ''}
          ${toolResults.length ? `<div class="info-section"><h4>Tool Results</h4><pre>${escHtml(toolResults.join('\n'))}</pre></div>` : ''}
          ${answerHtml}
        </div>
      </div>
    `;
    card.appendChild(body);
    container.appendChild(card);

    const canvas = body.querySelector('canvas');
    if (canvas && !canvas.dataset.loaded) loadStepImage(canvas, task.name, step);
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ---- Canvas drawing ----
function loadStepImage(canvas, taskName, step) {
  if (!step.screenshot) {
    canvas.dataset.loaded = '1';
    return;
  }
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    drawAnnotations(ctx, step.annotations, img.width, img.height);
    canvas.dataset.loaded = '1';
  };
  img.onerror = () => {
    canvas.dataset.loaded = '1';
  };
  img.src = `${API}/screenshot/${taskName}/${step.screenshot}`;
}

function resolveCoord(val, size) {
  // Convert relative (0-1) coords to absolute
  if (val <= 1.0) return Math.round(val * size);
  return Math.round(val);
}

function drawAnnotations(ctx, annotations, w, h) {
  if (!annotations || !annotations.length) return;

  let lastMoveTo = null;

  for (const a of annotations) {
    const x = a.x !== undefined ? resolveCoord(a.x, w) : undefined;
    const y = a.y !== undefined ? resolveCoord(a.y, h) : undefined;

    switch (a.type) {
      case 'click':
        drawClickMarker(ctx, x, y, '#ff1744', 'click');
        break;
      case 'doubleClick':
        drawClickMarker(ctx, x, y, '#ff9100', 'dblclick');
        break;
      case 'rightClick':
        drawClickMarker(ctx, x, y, '#d500f9', 'right');
        break;
      case 'middleClick':
        drawClickMarker(ctx, x, y, '#00e5ff', 'middle');
        break;
      case 'tripleClick':
        drawClickMarker(ctx, x, y, '#ffea00', 'triple');
        break;
      case 'moveTo':
        lastMoveTo = {x, y};
        drawDot(ctx, x, y, '#4fc3f7');
        break;
      case 'dragTo':
        if (lastMoveTo) {
          drawArrow(ctx, lastMoveTo.x, lastMoveTo.y, x, y, '#4fc3f7');
          lastMoveTo = null;
        } else {
          drawDot(ctx, x, y, '#4fc3f7');
        }
        break;
      case 'scroll':
        const sx = x !== undefined ? x : Math.round(w / 2);
        const sy = y !== undefined ? y : Math.round(h / 2);
        drawScrollMarker(ctx, sx, sy, a.clicks);
        break;
      case 'typewrite':
        drawTextMarker(ctx, Math.round(w / 2), Math.round(h / 2), `type: ${a.value || ''}`);
        break;
      case 'key_press':
        drawTextMarker(ctx, Math.round(w / 2), Math.round(h / 2), `keys: ${a.value || ''}`);
        break;
    }
  }
}

function drawClickMarker(ctx, x, y, color, label) {
  const r = 18;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - r - 6, y); ctx.lineTo(x + r + 6, y);
  ctx.moveTo(x, y - r - 6); ctx.lineTo(x, y + r + 6);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.font = 'bold 12px sans-serif';
  ctx.fillStyle = '#000';
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeText(label, x + r + 6, y - 6);
  ctx.fillStyle = color;
  ctx.fillText(label, x + r + 6, y - 6);
  ctx.restore();
}

function drawDot(ctx, x, y, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, 2 * Math.PI);
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawArrow(ctx, x1, y1, x2, y2, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 4]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(x1, y1, 8, 0, 2 * Math.PI);
  ctx.fill();
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 16;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - 0.4), y2 - headLen * Math.sin(angle - 0.4));
  ctx.lineTo(x2 - headLen * Math.cos(angle + 0.4), y2 - headLen * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fill();
  ctx.font = 'bold 12px sans-serif';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 3;
  ctx.strokeText('drag start', x1 + 10, y1 - 10);
  ctx.fillText('drag start', x1 + 10, y1 - 10);
  ctx.strokeText('drag end', x2 + 10, y2 - 10);
  ctx.fillText('drag end', x2 + 10, y2 - 10);
  ctx.restore();
}

function drawScrollMarker(ctx, x, y, clicks) {
  ctx.save();
  const color = '#ffeb3b';
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(x - 14, y - 28, 28, 56, 14);
  ctx.stroke();
  const dir = clicks < 0 ? 1 : -1;
  const ay = y + dir * 10;
  ctx.beginPath();
  ctx.moveTo(x, ay + dir * 12);
  ctx.lineTo(x - 8, ay);
  ctx.lineTo(x + 8, ay);
  ctx.closePath();
  ctx.fill();
  ctx.font = 'bold 12px sans-serif';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 3;
  const lbl = `scroll(${clicks})`;
  ctx.strokeText(lbl, x + 20, y + 4);
  ctx.fillText(lbl, x + 20, y + 4);
  ctx.restore();
}

function drawTextMarker(ctx, x, y, text) {
  ctx.save();
  const label = String(text).length > 60 ? String(text).slice(0, 57) + '...' : String(text);
  ctx.font = 'bold 13px sans-serif';
  const metrics = ctx.measureText(label);
  const padX = 10;
  const boxW = metrics.width + padX * 2;
  const boxH = 28;
  ctx.fillStyle = 'rgba(79,195,247,0.9)';
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 2;
  ctx.roundRect(x - boxW / 2, y - boxH / 2, boxW, boxH, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#000';
  ctx.fillText(label, x - boxW / 2 + padX, y + 5);
  ctx.restore();
}

// ---- Worker log modal ----
async function showWorkerLog(taskName) {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });

  overlay.innerHTML = `
    <div class="modal-box">
      <div class="modal-header">
        <h3>worker.log &mdash; ${taskName}</h3>
        <button class="modal-close">&times;</button>
      </div>
      <div class="modal-body"><pre>Loading...</pre></div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('.modal-close').addEventListener('click', () => overlay.remove());

  try {
    const resp = await fetch(`${API}/api/worker_log/${taskName}`);
    const text = await resp.text();
    overlay.querySelector('.modal-body pre').textContent = resp.ok ? text : 'worker.log not found';
  } catch (err) {
    overlay.querySelector('.modal-body pre').textContent = 'Error loading worker.log: ' + err.message;
  }
}

// ---- Init ----
loadSidebar();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize NVIDIA WebArena/WebVoyager agent results")
    parser.add_argument("result_dir", help="Path to results directory (e.g. webarena/nvidia/results_toolcall)")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global BASE_DIR
    BASE_DIR = os.path.abspath(args.result_dir)

    if not os.path.isdir(BASE_DIR):
        print(f"Error: {BASE_DIR} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Serving results from: {BASE_DIR}")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
