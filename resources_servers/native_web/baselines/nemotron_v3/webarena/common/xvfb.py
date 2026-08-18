"""Xvfb virtual display management for WebArena evaluation."""

import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_display_available(display_num: int) -> bool:
    """Check if an X11 display number is available.

    If a lock file exists but the owning process is dead, removes the stale
    lock (and socket) so the display can be reused.
    """
    lock_file = f"/tmp/.X{display_num}-lock"
    socket_path = f"/tmp/.X11-unix/X{display_num}"

    if not os.path.exists(lock_file):
        return True

    try:
        pid = int(Path(lock_file).read_text().strip())
        os.kill(pid, 0)
        return False
    except (ProcessLookupError, ValueError):
        logger.info(f"Removing stale lock file for display :{display_num} (pid gone)")
        try:
            os.unlink(lock_file)
        except (FileNotFoundError, OSError):
            pass
        try:
            os.unlink(socket_path)
        except (FileNotFoundError, OSError):
            pass
        return True
    except PermissionError:
        return False


def start_xvfb(width: int = 1920, height: int = 1080, depth: int = 24,
               start_display: int = 99) -> tuple[subprocess.Popen, str]:
    """Start Xvfb on the first available display starting from start_display.

    Returns (process, display_string) e.g. (proc, ":99").
    """
    display_num = start_display
    while not _is_display_available(display_num):
        display_num += 1
        if display_num > start_display + 200:
            raise RuntimeError(f"No available X11 display found in :{start_display}-:{display_num}")

    display = f":{display_num}"
    cmd = [
        "Xvfb", display,
        "-screen", "0", f"{width}x{height}x{depth}",
        "-ac",
        "-nolisten", "tcp",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ["DISPLAY"] = display
    os.environ.pop("WAYLAND_DISPLAY", None)
    logger.info(f"Started Xvfb on {display} ({width}x{height}x{depth}), pid={proc.pid}")

    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(f"Xvfb on {display} exited immediately (code={proc.returncode})")
        ret = subprocess.run(
            ["xdpyinfo", "-display", display],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if ret.returncode == 0:
            logger.info(f"X11 display {display} is ready")
            return proc, display
        time.sleep(0.5)

    proc.kill()
    raise RuntimeError(f"Xvfb on {display} did not become ready within 30s")


def stop_xvfb(proc: subprocess.Popen, display: str | None = None) -> None:
    """Stop an Xvfb process and clean up its lock/socket files."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info(f"Stopped Xvfb pid={proc.pid}")
    if display:
        display_num = display.lstrip(":")
        for path in (f"/tmp/.X{display_num}-lock", f"/tmp/.X11-unix/X{display_num}"):
            try:
                os.unlink(path)
            except (FileNotFoundError, OSError):
                pass
