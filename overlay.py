"""Desktop overlay launcher for the Live Transcriber.

Runs the Streamlit app in the background and shows it inside a frameless,
always-on-top, semi-transparent window (an "invisible interview helper").
Drag it anywhere; press Alt+F4 to close.
"""
import ctypes
import os
import subprocess
import sys
import time
from urllib.request import urlopen

import webview

PORT = 8501
URL = f"http://localhost:{PORT}"
WINDOW_TITLE = "Interview Overlay"
APP_FILE = "interview_helper_local.py"

# See-through level: 1.0 = solid, lower = more transparent. Tune to taste.
OPACITY = 0.82

_st_proc = None


def start_streamlit():
    """Launch the Streamlit server in the background (no browser)."""
    global _st_proc
    here = os.path.dirname(os.path.abspath(__file__))
    _st_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", APP_FILE,
         "--server.headless", "true",
         "--server.port", str(PORT),
         "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false"],
        cwd=here,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until_up(timeout=120):
    """Poll the health endpoint until Streamlit is serving."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urlopen(URL + "/_stcore/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def apply_overlay_style():
    """Make the whole window semi-transparent using the Win32 layered-window
    trick. Runs once the window exists."""
    time.sleep(1.5)
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, WINDOW_TITLE)
        if not hwnd:
            return
        gwl_exstyle = -20
        ws_ex_layered = 0x00080000
        lwa_alpha = 0x2
        style = user32.GetWindowLongW(hwnd, gwl_exstyle)
        user32.SetWindowLongW(hwnd, gwl_exstyle, style | ws_ex_layered)
        user32.SetLayeredWindowAttributes(hwnd, 0, int(255 * OPACITY), lwa_alpha)
    except Exception:
        pass


def on_closed():
    if _st_proc is not None:
        _st_proc.terminate()


def main():
    start_streamlit()
    if not wait_until_up():
        print("The app did not start in time. Try again.")
        return
    window = webview.create_window(
        WINDOW_TITLE, URL,
        width=480, height=780,
        frameless=True,
        on_top=True,
        easy_drag=True,
        background_color="#1e1e2e",
    )
    window.events.closed += on_closed
    webview.start(apply_overlay_style)


if __name__ == "__main__":
    main()
