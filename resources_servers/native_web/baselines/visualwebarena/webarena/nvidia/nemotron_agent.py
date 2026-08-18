"""
Nemotron agent for WebArena-Verified evaluation (production-style).

Uses Xvfb virtual displays, headed Chrome, and pyautogui for all
interactions and screenshots — matching the production browser
automation setup.

Playwright is used only for:
  - Launching headed Chrome on the Xvfb display
  - Browser context management (HAR recording)
  - Site logins and navigation

All mouse/keyboard actions and screenshots go through pyautogui,
which targets the DISPLAY environment variable of the current process.

Dependencies: httpx, pyautogui, playwright
"""

import ast
import base64
import datetime
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.pyautogui_utils import (
    init_pyautogui,
    take_screenshot,
    execute_action,
    parse_pyautogui_code,
    convert_relative_coords,
)
from common.agent_captcha import maybe_solve_captcha
from common.tab_context import append_traj
from common.visualwebarena_task_images import load_task_input_image_parts

_default_logger = logging.getLogger("nemotron_eval.agent")


# ---------------------------------------------------------------------------
# Prompts (browser-focused, adapted from mm_agents/nvidia/nemotron_agent.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_THINKING = """
You are a GUI agent controlling a web browser. You are given a task instruction, a screenshot of the browser, and your previous interactions. You need to perform a series of actions to complete the task. The browser is already open and logged into the required websites.

For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Wait for the page to load or update.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer to the task. Required for information retrieval tasks - provide the exact data requested."}}, "required": ["status"]}}
""".strip()

SYSTEM_PROMPT_NON_THINKING = """
You are a GUI agent controlling a web browser. You are given a task instruction, a screenshot of the browser, and your previous interactions. You need to perform a series of actions to complete the task. The browser is already open and logged into the required websites.

For each step, provide your response in this format:
## Thought
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Wait for the page to load or update.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer to the task. Required for information retrieval tasks - provide the exact data requested."}}, "required": ["status"]}}
""".strip()

INSTRUCTION_TEMPLATE = (
    "# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, "
    "task instruction and previous steps (if provided).\n"
)

STEP_TEMPLATE = "# Step {step_num}:\n"

TEXT_HISTORY_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n"

ASSISTANT_HISTORY_TEMPLATE_THINKING = "<think>\n{thought}\n</think>\n## Action:\n{action}\n"
ASSISTANT_HISTORY_TEMPLATE_NON_THINKING = "## Thought:\n{thought}\n\n## Action:\n{action}\n"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_terminate_args(code_block: str) -> dict:
    """Extract keyword arguments from a computer.terminate(...) call using ast."""
    result = {"status": None, "answer": None}
    m = re.search(r"computer\.terminate\s*\(", code_block)
    if not m:
        return result

    call_str = code_block[m.start():]
    call_str = call_str.replace("computer.terminate", "_terminate", 1)

    try:
        tree = ast.parse(call_str, mode="eval")
        call_node = tree.body
        if not isinstance(call_node, ast.Call):
            return result
        for kw in call_node.keywords:
            if kw.arg in ("status", "answer"):
                try:
                    result[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    pass
    except SyntaxError:
        paren_start = call_str.index("(")
        depth = 0
        end = None
        for i in range(paren_start, len(call_str)):
            if call_str[i] == "(":
                depth += 1
            elif call_str[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            try:
                tree = ast.parse(call_str[:end].replace("computer.terminate", "_terminate", 1), mode="eval")
                for kw in tree.body.keywords:
                    if kw.arg in ("status", "answer"):
                        try:
                            result[kw.arg] = ast.literal_eval(kw.value)
                        except Exception:
                            pass
            except Exception:
                pass

    return result


def parse_response(response: dict, thinking: bool) -> dict:
    """Parse Nemotron API response into thought, action description, and code."""
    content = response.get("content", "").strip()
    result = {"thought": "", "action": "", "code": "", "original_code": "", "status": "error", "answer": None}

    try:
        if thinking:
            thought = response.get("reasoning_content", "") or response.get("reasoning", "") or ""
            result["thought"] = thought.strip()
            m = re.search(r"^##\s*Action\b", content, flags=re.MULTILINE)
            if m:
                content = content[m.start():]
        else:
            thought_match = re.search(
                r"^##\s*Thought\s*:?\s*[\n\r]+(.*?)(?=^##\s*Action:|^##|\Z)",
                content, re.DOTALL | re.MULTILINE,
            )
            result["thought"] = thought_match.group(1).strip() if thought_match else ""

        action_match = re.search(
            r"^\s*##\s*Action\s*:?\s*[\n\r]+(.*?)(?=^\s*##|\Z)",
            content, re.DOTALL | re.MULTILINE,
        )
        if action_match:
            result["action"] = action_match.group(1).strip()

        code_blocks = re.findall(
            r"```(?:code|python)?\s*(.*?)\s*```",
            content, re.DOTALL | re.IGNORECASE,
        )
        if not code_blocks:
            result["status"] = "error"
            return result

        code_block = code_blocks[-1].strip()
        result["original_code"] = code_block

        if "computer.wait" in code_block.lower():
            result["code"] = "WAIT"
            result["status"] = "wait"
            return result

        if "computer.terminate" in code_block.lower():
            lower_block = code_block.lower()
            terminate_args = _parse_terminate_args(code_block)
            if terminate_args.get("answer") is not None:
                result["answer"] = terminate_args["answer"]

            status_val = terminate_args.get("status", "")
            if not status_val:
                if "failure" in lower_block or "fail" in lower_block:
                    status_val = "failure"
                elif "success" in lower_block:
                    status_val = "success"
                else:
                    status_val = "failure"

            result["status"] = "done" if status_val == "success" else "fail"
            return result

        result["code"] = code_block
        result["status"] = "action"
        return result

    except Exception as e:
        _default_logger.exception(f"Error parsing response: {e}")
        result["status"] = "error"
        return result


# ---------------------------------------------------------------------------
# NemotronAgent (synchronous, pyautogui-based)
# ---------------------------------------------------------------------------

class NemotronAgent:
    """Nemotron agent using pyautogui for browser interaction.

    Usage:
        agent = NemotronAgent(model="nemotron", max_steps=50)
        result = agent.run(instruction, task_dir)
    """

    def __init__(
        self,
        model: str,
        max_steps: int = 50,
        max_image_history: int = 3,
        max_tokens: int = 16384,
        top_p: float = 0.95,
        temperature: float = 1.0,
        thinking: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080,
        wait_seconds: float = 5.0,
        api_timeout: float = 1200.0,
        logger_name: str | None = None,
        captcha_solver=None,
    ):
        self.logger = logging.getLogger(logger_name) if logger_name else _default_logger
        self.model = model
        self.max_steps = max_steps
        self.max_image_history = max_image_history
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.thinking = thinking
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.wait_seconds = wait_seconds
        self.api_timeout = api_timeout
        self.captcha_solver = captcha_solver

        if thinking:
            self.system_prompt = SYSTEM_PROMPT_THINKING
            self.history_template = ASSISTANT_HISTORY_TEMPLATE_THINKING
        else:
            self.system_prompt = SYSTEM_PROMPT_NON_THINKING
            self.history_template = ASSISTANT_HISTORY_TEMPLATE_NON_THINKING

    def _call_api(self, messages: list[dict]) -> dict:
        """Call the vLLM API via httpx (synchronous)."""
        api_key = os.environ.get("VLLM_API_KEY", "")
        if not api_key:
            raise RuntimeError("VLLM_API_KEY environment variable not set")

        endpoint = os.environ.get("VLLM_API_ENDPOINT", "")
        if not endpoint:
            raise RuntimeError("VLLM_API_ENDPOINT environment variable not set")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        for attempt in range(20):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    # "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "temperature": self.temperature if attempt == 0 else max(0.2, self.temperature),
                }
                with httpx.Client(timeout=self.api_timeout, verify=False) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)
                    if resp.status_code != 200:
                        self.logger.error(f"API returned {resp.status_code} (attempt {attempt + 1}): {resp.text}")
                        time.sleep(5)
                        continue

                    data = resp.json()

                finish_reason = data["choices"][0].get("finish_reason")
                if finish_reason == "stop":
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

                    if not content.strip():
                        self.logger.warning(f"API returned null/empty content (attempt {attempt + 1}), retrying...")
                        time.sleep(5)
                        continue

                    return {
                        "content": content,
                        "reasoning_content": reasoning,
                    }
                else:
                    self.logger.warning(f"API did not finish properly (attempt {attempt + 1})")
                    time.sleep(5)

            except Exception as e:
                self.logger.error(f"API call error (attempt {attempt + 1}): {e}")
                time.sleep(5)

        raise RuntimeError("vLLM API max retries exceeded")

    def _build_messages(
        self,
        instruction: str,
        screenshot_b64: str,
        observations: list[bytes],
        cots: list[dict],
        task_input_image_parts: list[dict[str, Any]] | None = None,
    ) -> list[dict]:
        """Build the message list for the API call.

        Uses the same image-window strategy as the OSWorld nemotron agent:
        older steps beyond max_image_history are included as text-only history.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        instruction_prompt = INSTRUCTION_TEMPLATE.format(instruction=instruction)

        num_history_with_images = min(len(cots), self.max_image_history - 1)
        image_window_start = len(cots) - num_history_with_images

        text_history = ""
        if image_window_start > 0:
            parts = []
            for i in range(image_window_start):
                parts.append(
                    STEP_TEMPLATE.format(step_num=i + 1) + TEXT_HISTORY_TEMPLATE.format(
                        thought=cots[i].get("thought", ""),
                        action=cots[i].get("action", ""),
                    )
                )
            text_history = "# Previous History Actions:\n" + "\n".join(parts)

        for i in range(image_window_start, len(cots)):
            user_text = instruction_prompt
            if i == image_window_start and text_history:
                user_text += text_history + "\n"
            user_text += f"You are currently on Step {i + 1}.\n"

            obs_b64 = base64.b64encode(observations[i]).decode()
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{obs_b64}"}},
                    {"type": "text", "text": user_text},
                ],
            })

            messages.append({
                "role": "assistant",
                "content": self.history_template.format(
                    thought=cots[i].get("thought", ""),
                    action=cots[i].get("action", ""),
                ),
            })

        current_suffix = ""
        if num_history_with_images == 0 and text_history:
            current_suffix += text_history + "\n"
        current_suffix += f"You are currently on Step {len(cots) + 1}.\n"
        current_text = instruction_prompt + current_suffix

        if task_input_image_parts:
            current_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": instruction_prompt},
                *task_input_image_parts,
                {"type": "text", "text": current_suffix},
            ]
        else:
            current_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                {"type": "text", "text": current_text},
            ]
        messages.append({"role": "user", "content": current_content})

        return messages

    def run(
        self,
        instruction: str,
        task_dir=None,
        *,
        page=None,
        loop=None,
        task_input_images: list[str | Path] | None = None,
    ) -> dict:
        """Run the agent loop using pyautogui for interaction.

        Trajectory is saved in OSWorld format:
          - traj.jsonl: one JSON line per step
          - step_0_{timestamp}.png, step_{N}_{timestamp}.png

        Returns:
            dict with keys: status, answer, steps, errors
        """
        self._page = page
        self._loop = loop
        observations: list[bytes] = []
        cots: list[dict] = []
        errors: list[str] = []
        task_input_image_parts = load_task_input_image_parts(task_input_images)

        if task_dir:
            task_dir = Path(task_dir)
            task_dir.mkdir(parents=True, exist_ok=True)

        final_status = "fail"
        final_answer = None

        self._maybe_solve_captcha("initial")
        try:
            obs = take_screenshot()
        except Exception as e:
            errors.append(f"Initial screenshot failed: {e}")
            self.logger.error(f"Initial screenshot failed: {e}")
            return {"status": "error", "answer": None, "steps": 0, "errors": errors}

        initial_ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
        initial_screenshot = f"step_0_{initial_ts}.png"
        if task_dir:
            (task_dir / initial_screenshot).write_bytes(obs)
            append_traj(task_dir, {
                "step_num": 0,
                "action": None,
                "natural_language_action": "Initial state",
                "action_timestamp": initial_ts,
                "response": None,
                "reward": 0,
                "done": False,
                "info": {},
                "screenshot_file": initial_screenshot,
            }, page=self._page, loop=self._loop)

        done = False
        step_idx = 0

        while not done and step_idx < self.max_steps:
            self.logger.info(f"===== Step {step_idx + 1}/{self.max_steps} =====")

            screenshot_b64 = base64.b64encode(obs).decode()
            messages = self._build_messages(
                instruction,
                screenshot_b64,
                observations,
                cots,
                task_input_image_parts,
            )

            parsed = None
            for api_attempt in range(3):
                try:
                    response = self._call_api(messages)
                except Exception as e:
                    errors.append(f"API error at step {step_idx + 1} (attempt {api_attempt + 1}): {e}")
                    self.logger.error(f"API error (attempt {api_attempt + 1}): {e}")
                    time.sleep(1)
                    continue

                try:
                    parsed = parse_response(response, self.thinking)
                except Exception as e:
                    errors.append(f"Parse exception at step {step_idx + 1} (attempt {api_attempt + 1}): {e}")
                    self.logger.error(f"Parse exception (attempt {api_attempt + 1}): {e}")
                    time.sleep(1)
                    continue

                if parsed["status"] == "error":
                    errors.append(f"Parse error at step {step_idx + 1} (attempt {api_attempt + 1})")
                    self.logger.warning(f"Parse error (attempt {api_attempt + 1}), retrying API call...")
                    time.sleep(1)
                    continue

                break

            if parsed is None or parsed["status"] == "error":
                self.logger.error(f"All API/parse attempts failed at step {step_idx + 1}, aborting run")
                final_status = "error"
                break

            self.logger.info(f"Status: {parsed['status']}, Action: {parsed['action']}")

            if parsed["status"] == "wait":
                self.logger.info(f"Waiting {self.wait_seconds}s...")
                time.sleep(self.wait_seconds)
                self._maybe_solve_captcha("after wait")
                observations.append(obs)
                cots.append({"thought": parsed["thought"], "action": parsed["action"]})
                try:
                    obs = take_screenshot()
                except Exception as e:
                    errors.append(f"Screenshot failed after wait at step {step_idx + 1}: {e}")
                    final_status = "error"
                    break
                action_ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                screenshot_file = f"step_{step_idx + 1}_{action_ts}.png"
                if task_dir:
                    (task_dir / screenshot_file).write_bytes(obs)
                    append_traj(task_dir, {
                        "step_num": step_idx + 1,
                        "action": "WAIT",
                        "natural_language_action": parsed["action"],
                        "action_timestamp": action_ts,
                        "response": response,
                        "reward": 0,
                        "done": False,
                        "info": {"thought": parsed["thought"]},
                        "screenshot_file": screenshot_file,
                    }, page=self._page, loop=self._loop)
                step_idx += 1
                continue

            if parsed["status"] in ("done", "fail"):
                final_status = parsed["status"]
                final_answer = parsed.get("answer")
                done = True
                self.logger.info(f"Agent terminated: {final_status}, answer: {final_answer}")
                observations.append(obs)
                cots.append({"thought": parsed["thought"], "action": parsed["action"]})
                action_ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                if task_dir:
                    append_traj(task_dir, {
                        "step_num": step_idx + 1,
                        "action": parsed.get("original_code", ""),
                        "natural_language_action": parsed["action"],
                        "action_timestamp": action_ts,
                        "response": response,
                        "reward": 0,
                        "done": True,
                        "info": {
                            "thought": parsed["thought"],
                            "status": final_status,
                            "answer": final_answer,
                        },
                        "screenshot_file": None,
                    }, page=self._page, loop=self._loop)
                break

            action_code = parsed["code"]
            action_dicts = parse_pyautogui_code(action_code)
            if not action_dicts:
                errors.append(f"No actions parsed from code at step {step_idx + 1}")
                self.logger.warning(f"No actions parsed from code: {action_code}")
                step_idx += 1
                continue

            action_dicts = convert_relative_coords(
                action_dicts, self.screen_width, self.screen_height,
            )

            for action in action_dicts:
                try:
                    execute_action(action, self.screen_width, self.screen_height)
                    self.logger.info(f"Executed: {action}")
                except Exception as e:
                    errors.append(f"Action error at step {step_idx + 1}: {e}")
                    self.logger.error(f"Action error: {e}")

            time.sleep(2.0)
            self._maybe_solve_captcha("after action")

            try:
                obs = take_screenshot()
            except Exception as e:
                errors.append(f"Screenshot failed after action at step {step_idx + 1}: {e}")
                self.logger.error(f"Post-action screenshot failed: {e}")
                final_status = "error"
                break

            action_ts = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            screenshot_file = f"step_{step_idx + 1}_{action_ts}.png"
            if task_dir:
                (task_dir / screenshot_file).write_bytes(obs)
                append_traj(task_dir, {
                    "step_num": step_idx + 1,
                    "action": action_code,
                    "natural_language_action": parsed["action"],
                    "action_timestamp": action_ts,
                    "response": response,
                    "reward": 0,
                    "done": False,
                    "info": {"thought": parsed["thought"]},
                    "screenshot_file": screenshot_file,
                }, page=self._page, loop=self._loop)

            observations.append(obs)
            cots.append({"thought": parsed["thought"], "action": parsed["action"]})
            step_idx += 1

        return {
            "status": final_status,
            "answer": final_answer,
            "steps": step_idx,
            "errors": errors,
        }

    def _maybe_solve_captcha(self, phase: str) -> bool:
        return maybe_solve_captcha(
            self.captcha_solver,
            self._page,
            self._loop,
            self.logger,
            phase,
        )
