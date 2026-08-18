"""pyautogui utilities: initialization, screenshots, action parsing & execution."""

import ast
import io
import logging
import os
import re
import time

_default_logger = logging.getLogger(__name__)

os.environ.pop("WAYLAND_DISPLAY", None)

_pyautogui = None


def init_pyautogui():
    """Import and configure pyautogui. Must be called after DISPLAY is set."""
    global _pyautogui
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    _pyautogui = pyautogui
    _default_logger.info(f"pyautogui initialized on DISPLAY={os.environ.get('DISPLAY')}")
    return pyautogui


# ---------------------------------------------------------------------------
# pyautogui code parsing
# ---------------------------------------------------------------------------

_PYAUTOGUI_PARAMS = {
    "click":       ["x", "y", "clicks", "interval", "button", "duration", "pause"],
    "leftClick":   ["x", "y", "duration", "tween", "pause"],
    "rightClick":  ["x", "y", "duration", "tween", "pause"],
    "middleClick": ["x", "y", "duration", "tween", "pause"],
    "doubleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "tripleClick": ["x", "y", "interval", "button", "duration", "pause"],
    "moveTo":      ["x", "y", "duration", "tween", "pause"],
    "dragTo":      ["x", "y", "duration", "button", "mouseDownUp", "pause"],
    "scroll":      ["clicks", "x", "y", "pause"],
    "typewrite":   ["message", "interval", "pause"],
    "write":       ["message", "interval", "pause"],
    "press":       ["keys", "presses", "interval", "pause"],
    "hotkey":      [],  # variadic positional
    "keyDown":     ["key"],
    "keyUp":       ["key"],
}

_KEY_MAP = {
    "ctrl": "ctrl", "alt": "alt", "shift": "shift",
    "enter": "enter", "return": "enter", "tab": "tab",
    "delete": "delete", "backspace": "backspace",
    "escape": "escape", "esc": "escape", "space": "space",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "super": "win", "win": "win", "command": "win",
    "pageup": "pageup", "pagedown": "pagedown",
    "home": "home", "end": "end", "insert": "insert",
    **{f"f{i}": f"f{i}" for i in range(1, 13)},
}


_ACTION_RE = re.compile(
    r"(pyautogui\.(\w+)\([^)]*\))"
    r"|"
    r"(time\.sleep\(([^)]*)\))",
    re.DOTALL,
)


def parse_pyautogui_code(code: str) -> list[dict]:
    """Parse pyautogui calls into action dicts for direct pyautogui execution."""
    actions: list[dict] = []

    for match in _ACTION_RE.finditer(code):
        # --- time.sleep(...) ---
        if match.group(3):
            try:
                seconds = float(match.group(4).strip())
            except (ValueError, TypeError):
                continue
            actions.append({"type": "wait", "seconds": seconds})
            continue

        # --- pyautogui.<func>(...) ---
        full_call = match.group(1)
        func_name = match.group(2)

        # Handle pyautogui.sleep() the same as time.sleep()
        if func_name == "sleep":
            try:
                tree = ast.parse(full_call.replace("pyautogui.", "_pag_.", 1))
                call_node = tree.body[0].value
                positional = [ast.literal_eval(a) for a in call_node.args]
                seconds = float(positional[0]) if positional else 5.0
            except Exception:
                seconds = 5.0
            actions.append({"type": "wait", "seconds": seconds})
            continue

        try:
            tree = ast.parse(full_call.replace("pyautogui.", "_pag_.", 1))
            call_node = tree.body[0].value
            positional = [ast.literal_eval(a) for a in call_node.args]
            keywords = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in call_node.keywords
                if kw.arg is not None
            }
        except Exception:
            continue

        param_names = _PYAUTOGUI_PARAMS.get(func_name, [])
        args: dict = {}
        for i, val in enumerate(positional):
            if func_name == "hotkey":
                args[i] = val
            elif i < len(param_names):
                args[param_names[i]] = val
        args.update(keywords)

        if func_name in ("click", "leftClick"):
            x, y = float(args.get("x", 0)), float(args.get("y", 0))
            a: dict = {"type": "click", "x": x, "y": y}
            if args.get("button"):
                a["button"] = str(args["button"])
            actions.append(a)
        elif func_name == "rightClick":
            actions.append({"type": "click", "x": float(args.get("x", 0)),
                            "y": float(args.get("y", 0)), "button": "right"})
        elif func_name == "middleClick":
            actions.append({"type": "click", "x": float(args.get("x", 0)),
                            "y": float(args.get("y", 0)), "button": "middle"})
        elif func_name in ("doubleClick", "tripleClick"):
            clicks = 3 if func_name == "tripleClick" else 2
            actions.append({"type": "click", "x": float(args.get("x", 0)),
                            "y": float(args.get("y", 0)), "clicks": clicks})
        elif func_name == "moveTo":
            actions.append({"type": "move", "x": float(args.get("x", 0)),
                            "y": float(args.get("y", 0))})
        elif func_name in ("typewrite", "write"):
            text = str(args.get("message", ""))
            interval = float(args.get("interval", 0.02))
            actions.append({"type": "typewrite", "text": text, "interval": interval})
        elif func_name == "hotkey":
            keys = [str(args[i]) for i in sorted(k for k in args if isinstance(k, int))]
            mapped = [_KEY_MAP.get(k.lower(), k) for k in keys if k]
            if mapped:
                actions.append({"type": "hotkey", "keys": mapped})
        elif func_name == "press":
            key = str(args.get("keys", args.get(0, "")))
            actions.append({"type": "press", "key": _KEY_MAP.get(key.lower(), key)})
        elif func_name == "keyDown":
            key = str(args.get("key", args.get(0, "")))
            actions.append({"type": "keyDown", "key": _KEY_MAP.get(key.lower(), key)})
        elif func_name == "keyUp":
            key = str(args.get("key", args.get(0, "")))
            actions.append({"type": "keyUp", "key": _KEY_MAP.get(key.lower(), key)})
        elif func_name == "scroll":
            amount = int(float(args.get("clicks", 3)))
            a = {"type": "scroll", "clicks": amount}
            if "x" in args and "y" in args:
                a["x"] = float(args["x"])
                a["y"] = float(args["y"])
            actions.append(a)
        elif func_name == "dragTo":
            actions.append({"type": "drag", "x": float(args.get("x", 0)),
                            "y": float(args.get("y", 0))})

    return actions


def convert_relative_coords(actions: list[dict], width: int, height: int) -> list[dict]:
    """Convert relative (0-1) coordinates to absolute pixel coords."""
    for action in actions:
        for key in ("x", "dest_x"):
            if key in action:
                v = float(action[key])
                if v <= 1.0:
                    action[key] = int(round(v * width))
                else:
                    action[key] = int(round(v))
        for key in ("y", "dest_y"):
            if key in action:
                v = float(action[key])
                if v <= 1.0:
                    action[key] = int(round(v * height))
                else:
                    action[key] = int(round(v))
    return actions


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def take_screenshot() -> bytes:
    """Capture the full virtual display via Pillow's X11 grab. Returns PNG bytes."""
    from PIL import ImageGrab
    img = ImageGrab.grab()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def execute_action(action: dict, screen_width: int, screen_height: int) -> None:
    """Execute a single parsed action dict via pyautogui."""
    pag = _pyautogui
    if pag is None:
        raise RuntimeError("pyautogui not initialized — call init_pyautogui() first")

    kind = action.get("type", "")

    if kind == "click":
        x, y = action["x"], action["y"]
        button = action.get("button", "left")
        clicks = action.get("clicks", 1)
        pag.click(x, y, button=button, clicks=clicks)
    elif kind == "move":
        pag.moveTo(action["x"], action["y"])
    elif kind == "typewrite":
        text = action.get("text", "")
        interval = action.get("interval", 0.02)
        pag.typewrite(text, interval=interval)
    elif kind == "hotkey":
        keys = action.get("keys", [])
        if keys:
            pag.hotkey(*keys)
    elif kind == "press":
        key = action.get("key", "")
        if key:
            pag.press(key)
    elif kind == "keyDown":
        key = action.get("key", "")
        if key:
            pag.keyDown(key)
    elif kind == "keyUp":
        key = action.get("key", "")
        if key:
            pag.keyUp(key)
    elif kind == "scroll":
        clicks = action.get("clicks", 3)
        x = action.get("x", screen_width // 2)
        y = action.get("y", screen_height // 2)
        pag.scroll(clicks, x=x, y=y)
    elif kind == "drag":
        pag.moveTo(action.get("start_x", screen_width // 2),
                    action.get("start_y", screen_height // 2))
        pag.dragTo(action["x"], action["y"], duration=0.5)
    elif kind == "wait":
        time.sleep(action.get("seconds", 5))

    time.sleep(0.3)
