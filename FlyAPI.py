"""
FlyPy — Trigger->Outputs build with All-in-One Settings GUI, camera previews, rich tooltips,
and comprehensive docstrings (natural-English labels).

On each trigger:
  • Records synchronized clips from two cameras to disk
  • Activates lights (serial if available; simulated otherwise)
  • Presents a looming (growing dot) visual stimulus
  • Logs trial metadata (CSV) with file paths and timestamps

GUI:
  • All settings visible in one window (natural-English labels)
  • Tooltips on every control
  • Per-camera visual index (live preview with overlayed index) and FPS
  • Video Format/Codec dropdown with captions (max res/FPS, size hint)
  • Stimulus delay after recording start  (NEW)
  • Lights delay after recording start    (NEW)
  • Select which screen shows the Stimulus and which screen hosts the GUI
  • Stimulus fullscreen/windowed mode (windowed is draggable)
  • Manual "Trigger Once" button
  • Auto-scales GUI to the maximum usable size of the selected screen (NEW)
  • Help → Check for Updates… menu (now uses an inline updater) (NEW)

This file is heavily annotated. Every class and function includes a docstring with plain English.
"""

__version__ = "1.3.0"  # bump when you cut a release (used by the updater)

import sys
import threading
import time
import os
import csv
from datetime import datetime
import atexit
from collections import deque

import json
import shutil
import tempfile
import zipfile
import urllib.request
from urllib.error import URLError, HTTPError

import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui

# ---------- HiDPI scaling (set BEFORE QApplication is constructed) ----------
try:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
except Exception:
    pass

# Reduce noisy OpenCV backend chatter when available
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# Make stdout/stderr UTF-8 if possible (helps on some Windows consoles)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Optional PsychoPy for high-precision stimulus (falls back to OpenCV)
try:
    from psychopy import visual, core
    PSYCHOPY = True
except Exception:
    visual = None
    core = None
    PSYCHOPY = False


# =========================
# Auto Update Helper (inline)
# =========================
_API_LATEST = "https://api.github.com/repos/{repo}/releases/latest"
_UA = "FlyPy-Updater/1.0 (+https://github.com)"

def _http_get(url: str) -> str:
    """Return the body of a GET request to `url` (as text)."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8")

def _parse_ver(v: str):
    """Parse a semantic-ish version string like 'v1.2.3' into a tuple of ints (1,2,3)."""
    s = v.strip()
    if s[:1] in ("v", "V"):
        s = s[1:]
    parts = []
    for tok in s.split("."):
        try:
            parts.append(int(tok))
        except Exception:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def _is_newer(remote_tag: str, current: str) -> bool:
    """True if remote_tag (e.g., 'v1.4.0') is newer than `current`."""
    return _parse_ver(remote_tag) > _parse_ver(current)

def _find_asset(assets, name: str):
    """Pick an asset (dict) by exact filename from a GitHub release assets list."""
    for a in assets:
        if a.get("name", "").lower() == name.lower():
            return a
    return None

def _download(url: str, dest: str):
    """Download `url` to file `dest`."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

def _app_root() -> str:
    """Folder that contains FlyPy.exe (PyInstaller) or this script (dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(sys.argv[0]))

def _write_update_bat(zip_path: str, app_root: str, exe_name: str = "FlyPy.exe") -> str:
    """Write a small .bat that unpacks the zip over app_root and relaunches the app."""
    bat_path = os.path.join(app_root, "update_on_restart.bat")
    temp_unpack = os.path.join(app_root, "_upd_unpack")
    with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("@echo off\n")
        f.write("setlocal EnableExtensions EnableDelayedExpansion\n")
        f.write("echo Waiting for the app to exit...\n")
        f.write("ping -n 3 127.0.0.1 >nul\n")
        f.write(f"set ZIP=\"{zip_path}\"\n")
        f.write(f"set APPROOT=\"{app_root}\"\n")
        f.write(f"set UNPACK=\"{temp_unpack}\"\n")
        f.write("rmdir /S /Q %UNPACK% 2>nul\n")
        f.write("mkdir %UNPACK%\n")
        f.write("echo Unpacking update...\n")
        f.write("powershell -NoProfile -Command \"Expand-Archive -Path %ZIP% -DestinationPath %UNPACK% -Force\" 2>nul\n")
        f.write("if not exist %UNPACK%\\* (\n")
        f.write("  echo PowerShell unzip failed.\n")
        f.write(")\n")
        f.write("echo Applying update...\n")
        f.write("robocopy %UNPACK% %APPROOT% /E /XO >nul\n")
        f.write("echo Cleaning up...\n")
        f.write("rmdir /S /Q %UNPACK% 2>nul\n")
        f.write("del %ZIP% 2>nul\n")
        f.write("echo Relaunching...\n")
        f.write("start \"\" \"%APPROOT%\\{}\"\n".format(exe_name))
        f.write("exit /b 0\n")
    return bat_path

def check_updates_and_stage(repo: str, current_version: str, asset_name: str):
    """Check GitHub Releases; if newer exists, download asset and prep update script.

    Returns:
        (found, message)
    """
    try:
        meta = json.loads(_http_get(_API_LATEST.format(repo=repo)))
    except (URLError, HTTPError) as e:
        return False, f"Could not contact GitHub: {e}"
    except Exception as e:
        return False, f"Update check failed: {e}"

    remote_tag = str(meta.get("tag_name", "")).strip()
    if not remote_tag:
        return False, "No releases found for this repository."

    if not _is_newer(remote_tag, current_version):
        return False, f"Up to date (current {current_version}, latest {remote_tag})."

    asset = _find_asset(meta.get("assets", []), asset_name)
    if not asset:
        return False, f"New release {remote_tag} found, but asset '{asset_name}' was not attached."
    url = asset.get("browser_download_url")
    if not url:
        return False, "Asset download URL missing."

    tmpdir = tempfile.mkdtemp(prefix="flypy_upd_")
    dest_zip = os.path.join(tmpdir, asset_name)
    try:
        _download(url, dest_zip)
    except Exception as e:
        return False, f"Failed to download update: {e}"

    app_root = _app_root()
    bat = _write_update_bat(dest_zip, app_root)
    return True, f"Update {remote_tag} downloaded. Close the app to finish installing.\n\n" \
                 f"A helper script has been prepared:\n{bat}\n" \
                 f"It will apply the update and relaunch."


# =========================
# Utilities / Config
# =========================

VIDEO_PRESETS = [
    {"id": "mp4_h264", "label": "MP4 (H.264 / avc1) — up to 4K@60 — medium/small filesize", "ext": ".mp4", "fourcc": "avc1", "size_hint": "medium/small"},
    {"id": "mp4_mp4v", "label": "MP4 (MPEG-4 Part 2 / mp4v) — up to 1080p@60 — medium/large filesize", "ext": ".mp4", "fourcc": "mp4v", "size_hint": "medium/large"},
    {"id": "webm_vp9", "label": "WebM (VP9) — up to 4K@60 — small/medium filesize", "ext": ".webm", "fourcc": "VP90", "size_hint": "small/medium"},
    {"id": "avi_xvid", "label": "AVI (XVID / MPEG-4 Part 2) — up to 1080p@60 — large/medium filesize", "ext": ".avi", "fourcc": "XVID", "size_hint": "large/medium"},
    {"id": "mov_h264", "label": "MOV (H.264 / avc1) — up to 4K@60 — medium/small filesize", "ext": ".mov", "fourcc": "avc1", "size_hint": "medium/small"},
    {"id": "mkv_h264", "label": "MKV (H.264 / avc1) — up to 4K@60 — medium/small filesize", "ext": ".mkv", "fourcc": "avc1", "size_hint": "medium/small"},
]
PRESETS_BY_ID = {p["id"]: p for p in VIDEO_PRESETS}

def default_preset_id() -> str:
    """Return the default video preset ID. Chosen for wide compatibility."""
    return "mp4_mp4v"

def ensure_dir(path: str):
    """Create a directory if it does not exist already."""
    os.makedirs(path, exist_ok=True)

def now_stamp() -> str:
    """Return a filesystem-safe timestamp string (YYYY-MM-DD_HH-MM-SS)."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def day_folder(root: str) -> str:
    """Return a path to today's date-stamped folder under `root`, creating it if needed."""
    d = datetime.now().strftime("%Y%m%d")
    p = os.path.join(root, d)
    ensure_dir(p)
    return p

def wait_s(sec: float):
    """Sleep for `sec` seconds, using PsychoPy's timing if available."""
    if PSYCHOPY:
        core.wait(sec)
    else:
        time.sleep(sec)

def get_screen_geometries():
    """Return a list of tuples describing each screen: (x, y, width, height)."""
    screens = QtGui.QGuiApplication.screens()
    geoms = []
    for s in screens:
        g = s.geometry()
        geoms.append((g.x(), g.y(), g.width(), g.height()))
    return geoms


class Config:
    """Mutable configuration object shared across the app (GUI <-> runtime)."""
    def __init__(self):
        # General
        self.simulation_mode = False
        self.sim_trigger_interval = 5.0

        # Recording / output
        self.output_root = "FlyPy_Output"
        self.video_preset_id = default_preset_id()
        self.fourcc = PRESETS_BY_ID[self.video_preset_id]["fourcc"]
        self.record_duration_s = 3.0

        # Stimulus (WHITE background with BLACK dot by default)
        self.stim_duration_s = 1.5
        self.stim_r0_px = 8
        self.stim_r1_px = 400
        self.stim_bg_grey = 1.0  # white background

        # NEW: precise timing from *recording start* to events
        self.stim_delay_s   = 0.0  # delay from recording start to stimulus onset
        self.lights_delay_s = 0.0  # delay from recording start to lights ON

        # Displays
        self.stim_screen_index = 0  # which monitor for the stimulus
        self.stim_fullscreen   = False  # windowed by default (draggable)
        self.gui_screen_index  = 0  # which monitor shows the GUI

        # Cameras
        self.cam0_index = 0
        self.cam1_index = 1
        self.cam0_target_fps = 60
        self.cam1_target_fps = 60


# =========================
# Hardware Bridge (Elegoo UNO R3 via CH340)
# =========================
class HardwareBridge:
    """Adapter around a USB serial device (Elegoo/UNO/CH340) with simulation fallback."""
    def __init__(self, cfg: Config, port: str = None, baud: int = 115200):
        self.cfg = cfg
        self.simulated = cfg.simulation_mode
        self._last_sim = time.time()
        self.ser = None
        self.port = port
               self.baud = baud

        if not self.simulated:
            try:
                import serial, serial.tools.list_ports
                if not self.port:
                    self.port = self._autodetect_port()
                if self.port:
                    try:
                        self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
                        wait_s(1.5)  # allow the MCU to reset after opening serial
                        print(f"[HardwareBridge] Opened {self.port} @ {self.baud} baud")
                    except Exception as e:
                        print(f"[HardwareBridge] Serial open failed on {self.port}: {e} -> simulation.")
                        self.simulated = True
                else:
                    print("[HardwareBridge] No Elegoo/CH340 port found -> simulation.")
                    self.simulated = True
            except ImportError:
                print("[HardwareBridge] pyserial not installed -> simulation.")
                self.simulated = True

    def _autodetect_port(self) -> str:
        """Try to find a likely Elegoo/UNO CH340 port; return a device path or None."""
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            vid = f"{p.vid:04X}" if p.vid is not None else None
            pid = f"{p.pid:04X}" if p.pid is not None else None
            if vid == "1A86" and pid == "7523":  # CH340/CH34x
                return p.device
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if "ch340" in desc or "uno" in desc or "elegoo" in desc:
                return p.device
        return None

    def check_trigger(self) -> bool:
        """Return True exactly when a trigger occurs (simulated or via serial 'T' line)."""
        if self.simulated:
            now = time.time()
            if now - self._last_sim >= self.cfg.sim_trigger_interval:
                self._last_sim = now
                print("[HardwareBridge] (Sim) Trigger.")
                return True
            return False
        try:
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line == "T":
                    return True
        except Exception as e:
            print(f"[HardwareBridge] Read error: {e}")
        return False

    # -------- Outgoing commands (markers & lights) --------
    def _send_line(self, text: str):
        """Send one line of text to the device, or log it if simulating."""
        if self.simulated or not self.ser:
            print(f"[HardwareBridge] (Sim) SEND: {text}")
            return
        try:
            self.ser.write((text.strip() + "\n").encode("utf-8", errors="ignore"))
        except Exception as e:
            print(f"[HardwareBridge] Write error: {e}")

    def mark_start(self): self._send_line("START")
    def mark_stim(self):  self._send_line("STIM")
    def mark_end(self):   self._send_line("END")

    def pulse_ms(self, ms: int = 20): self._send_line(f"PULSE {int(ms)}")
    def light_on(self):   self._send_line("LIGHT ON")
    def light_off(self):  self._send_line("LIGHT OFF")

    # Legacy alias
    def activate_lights(self): self.light_on()

    def close(self):
        """Close the serial port if it was opened."""
        if not self.simulated and self.ser:
            try:
                if getattr(self.ser, "is_open", True):
                    self.ser.close()
                    print("[HardwareBridge] Serial closed.")
            except Exception as e:
                print(f"[HardwareBridge] Close error: {e}")


# =========================
# Camera Recorder w/ Preview
# =========================
class CameraRecorder:
    """OpenCV VideoCapture wrapper with live preview, FPS info, and recording."""
    def __init__(self, index: int, name: str, target_fps: int = 60):
        self.name = name
        self.target_fps = float(target_fps)
        self._preview_times = deque(maxlen=30)
        self._last_preview_frame = None
        self.lock = threading.Lock()
        self.cap = None
        self.synthetic = False
        self.set_index(index)

    def _open(self, index: int):
        backends = [cv2.CAP_ANY]
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for be in backends:
            try:
                cap = cv2.VideoCapture(index, be)
            except TypeError:
                cap = cv2.VideoCapture(index)  # OpenCV < 4.5 fallback
            if cap and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                except Exception:
                    pass
                return cap, False
            try:
                cap.release()
            except Exception:
                pass
        return None, True  # synthetic

    def release(self):
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None

    def set_index(self, index: int):
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap, self.synthetic = self._open(index)
            self.index = index
            if self.synthetic:
                print(f"[Camera {self.name}] index {index} not available -> synthetic preview/recording.")
            else:
                print(f"[Camera {self.name}] bound to index {index}.")

    def set_target_fps(self, fps: float):
        self.target_fps = float(fps)
        with self.lock:
            if self.cap:
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                except Exception:
                    pass

    def reported_fps(self) -> float:
        with self.lock:
            if self.cap:
                v = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                return v if v > 0 else 0.0
        return 0.0

    def measured_preview_fps(self) -> float:
        if len(self._preview_times) < 2:
            return 0.0
        dt = self._preview_times[-1] - self._preview_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._preview_times) - 1) / dt

    def grab_preview(self, w=320, h=240, overlay_index=True):
        with self.lock:
            if self.synthetic:
                frame = np.full((h, w, 3), 255, dtype=np.uint8)
                cv2.putText(frame, f"{self.name} (synthetic)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                cx, cy = (int(time.time()*60) % w, h//2)
                cv2.circle(frame, (cx, cy), 15, (0, 0, 0), 2)
            else:
                ok, bgr = self.cap.read()
                if not ok or bgr is None:
                    frame = np.full((h, w, 3), 255, dtype=np.uint8)
                    cv2.putText(frame, f"{self.name} [drop]", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    frame = cv2.resize(bgr, (w, h))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if overlay_index:
                cv2.putText(frame, f"Index {self.index}", (10, h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            self._last_preview_frame = frame
            self._preview_times.append(time.time())
            return frame

    def _writer(self, path: str, size, fourcc_str: str = "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        return cv2.VideoWriter(path, fourcc, float(self.target_fps), size)

    def _frame_size(self):
        with self.lock:
            if self.cap:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                return (w, h)
        return (640, 480)

    def record_clip(self, path: str, duration_s: float, fourcc_str: str = "mp4v"):
        size = self._frame_size()
        out = self._writer(path, size, fourcc_str)
        if not out or not out.isOpened():
            print(f"[Camera {self.name}] VideoWriter failed for {path} with FOURCC={fourcc_str}. Trying fallback mp4v...")
            try:
                base, _old_ext = os.path.splitext(path)
                fallback_path = base + ".mp4"
                out = self._writer(fallback_path, size, "mp4v")
                if not out or not out.isOpened():
                    print(f"[Camera {self.name}] Fallback mp4v also failed.")
                    return None
                path = fallback_path
                fourcc_str = "mp4v"
            except Exception as e:
                print(f"[Camera {self.name}] Fallback error: {e}")
                return None

        t_end = time.time() + duration_s
        print(f"[Camera {self.name}] Recording -> {path} @ ~{self.target_fps:.1f} fps (FOURCC={fourcc_str})")
        frame_index = 0
        ok_any = False
        while time.time() < t_end:
            with self.lock:
                if self.synthetic:
                    frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
                    cv2.putText(frame, f"{self.name} {now_stamp()}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    cx = int((frame_index * 7) % size[0])
                    cy = int(size[1] / 2)
                    cv2.circle(frame, (cx, cy), 20, (0, 0, 0), 2)
                else:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
                        cv2.putText(frame, f"{self.name} [drop] {now_stamp()}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        ok_any = True
            out.write(frame)
            frame_index += 1
            time.sleep(max(0.0, 1.0 / float(self.target_fps)))
        out.release()
        if not ok_any and not self.synthetic:
            print(f"[Camera {self.name}] Warning: no frames captured from camera index {self.index}.")
        print(f"[Camera {self.name}] Recording complete.")
        return path


# =========================
# Looming Stimulus — persistent window (opens at app start)
# =========================
class LoomingStim:
    """Display a growing dot on a grey/white background (PsychoPy if available, else OpenCV)."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # PsychoPy window cache
        self._pp_win = None
        self._pp_cfg = None  # (screen_idx, fullscr)
        # OpenCV window cache
        self._cv_window_name = "Looming Stimulus"
        self._cv_open = False
        self._cv_size = (800, 600)  # windowed default (draggable)

    def _ensure_psychopy_window(self, screen_idx: int, fullscr: bool, bg_grey: float):
        """Create/update a PsychoPy window matching the requested screen/fullscreen."""
        need_new = False
        if self._pp_win is None:
            need_new = True
        elif self._pp_cfg != (screen_idx, fullscr):
            try:
                self._pp_win.close()
            except Exception:
                pass
            self._pp_win = None
            need_new = True

        if need_new:
            try:
                if fullscr:
                    self._pp_win = visual.Window(color=[bg_grey]*3, units='pix', fullscr=True, screen=screen_idx)
                else:
                    self._pp_win = visual.Window(size=self._cv_size, color=[bg_grey]*3,
                                                 units='pix', fullscr=False, screen=screen_idx, allowGUI=True)
                self._pp_cfg = (screen_idx, fullscr)
            except Exception as e:
                print(f"[Stim] PsychoPy window create error: {e}")
                self._pp_win = None

        if self._pp_win is not None:
            try:
                self._pp_win.color = [bg_grey]*3
            except Exception:
                pass

    def _ensure_opencv_window(self, screen_idx: int, bg_grey: float):
        """Create/move the OpenCV window; keep it open and draggable."""
        try:
            if not self._cv_open:
                cv2.namedWindow(self._cv_window_name, cv2.WINDOW_NORMAL)  # draggable/resizable
                cv2.resizeWindow(self._cv_window_name, self._cv_size[0], self._cv_size[1])
                self._cv_open = True
            geoms = get_screen_geometries()
            if 0 <= screen_idx < len(geoms):
                x, y, w, h = geoms[screen_idx]
                cv2.moveWindow(self._cv_window_name, x + 50, y + 50)
            # draw one background frame so a window is visible immediately
            bg = int(np.clip(bg_grey * 255, 0, 255))
            frame = np.full((self._cv_size[1], self._cv_size[0], 3), bg, dtype=np.uint8)
            cv2.imshow(self._cv_window_name, frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f"[Stim] OpenCV window create/move error: {e}")
            self._cv_open = False

    def open_persistent_window(self, screen_idx: int, fullscreen: bool, bg_grey: float):
        """Ensure a stimulus window exists as soon as the app starts (persistent until quit)."""
        if PSYCHOPY:
            self._ensure_psychopy_window(screen_idx, fullscreen, bg_grey)
            if self._pp_win is not None:
                try:
                    self._pp_win.flip()  # paint background once
                except Exception:
                    pass
            else:
                # fall back to OpenCV flow if PsychoPy failed
                self._ensure_opencv_window(screen_idx, bg_grey)
        else:
            self._ensure_opencv_window(screen_idx, bg_grey)

    def run(self, duration_s: float, r0: int, r1: int, bg_grey: float,
            screen_idx: int, fullscreen: bool):
        """Show the looming dot for `duration_s` seconds."""
        print("[Stim] Looming start.")
        if PSYCHOPY:
            try:
                self._ensure_psychopy_window(screen_idx, fullscreen, bg_grey)
                if self._pp_win is not None:
                    dot = visual.Circle(self._pp_win, radius=r0, fillColor='black', lineColor='black')
                    t0 = time.time()
                    while True:
                        t = time.time() - t0
                        if t >= duration_s:
                            break
                        r = r0 + (r1 - r0) * (t / duration_s)
                        dot.radius = r
                        dot.draw()
                        self._pp_win.flip()
                    print("[Stim] Looming done (PsychoPy).")
                    return
            except Exception as e:
                print(f"[Stim] PsychoPy error: {e} -> OpenCV fallback.")

        # OpenCV fallback (window already exists; keep it open between trials)
        try:
            self._ensure_opencv_window(screen_idx, bg_grey)
            size = self._cv_size
            bg = int(np.clip(bg_grey * 255, 0, 255))
            t0 = time.time()
            while True:
                t = time.time() - t0
                if t >= duration_s:
                    break
                r = int(r0 + (r1 - r0) * (t / duration_s))
                frame = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
                cv2.circle(frame, (size[0]//2, size[1]//2), r, (0, 0, 0), -1)
                cv2.imshow(self._cv_window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to end the stimulus early
                    break
            print("[Stim] Looming done (OpenCV).")
        except Exception as e:
            print(f"[Stim] OpenCV display unavailable ({e}). Logging-only fallback.")
            wait_s(duration_s)
            print("[Stim] Looming done (no display).")

    def close(self):
        """Tear down windows at application shutdown."""
        try:
            if self._pp_win is not None:
                self._pp_win.close()
        except Exception:
            pass
        self._pp_win = None
        self._pp_cfg = None
        if self._cv_open:
            try:
                cv2.destroyWindow(self._cv_window_name)
            except Exception:
                pass
        self._cv_open = False


# =========================
# Trial Orchestrator
# =========================
class TrialRunner:
    """Coordinates cameras, lights, stimulus, and logging for each trigger."""
    def __init__(self, cfg: Config, hardware: HardwareBridge, cam0: CameraRecorder, cam1: CameraRecorder, logger_path: str):
        self.cfg = cfg
        self.hardware = hardware
        self.cam0 = cam0
        self.cam1 = cam1
        self.stim = LoomingStim(cfg)
        self.trial_idx = 0

        new_file = not os.path.exists(logger_path)
        ensure_dir(os.path.dirname(logger_path))
        self.log_file = open(logger_path, "a", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_file)
        if new_file:
            self.log_writer.writerow([
                "trial", "timestamp",
                "cam0_path", "cam1_path",
                "record_duration_s",
                "lights_delay_s", "stim_delay_s", "stim_duration_s",
                "stim_screen_index", "stim_fullscreen",
                "cam0_target_fps", "cam1_target_fps",
                "video_preset_id", "fourcc"
            ])

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass
        try:
            self.stim.close()
        except Exception:
            pass

    def _preset(self):
        return PRESETS_BY_ID.get(self.cfg.video_preset_id, PRESETS_BY_ID[default_preset_id()])

    def run_trial(self):
        """Perform one full trial: start recording → (lights delay) lights on → (stim delay) stimulus."""
        self.trial_idx += 1
        ts = now_stamp()
        out_dir = day_folder(self.cfg.output_root)

        preset = self._preset()
        ext = preset["ext"]
        fourcc = preset["fourcc"]

        cam0_path_req = os.path.join(out_dir, f"{ts}_trial{self.trial_idx:04d}_cam0{ext}")
        cam1_path_req = os.path.join(out_dir, f"{ts}_trial{self.trial_idx:04d}_cam1{ext}")

        print(f"[Trial {self.trial_idx}] START {ts}")

        # Start camera recordings first (the origin point for both delays)
        results = {"cam0": None, "cam1": None}
        def rec0(): results["cam0"] = self.cam0.record_clip(cam0_path_req, self.cfg.record_duration_s, fourcc)
        def rec1(): results["cam1"] = self.cam1.record_clip(cam1_path_req, self.cfg.record_duration_s, fourcc)

        t0 = threading.Thread(target=rec0, daemon=True)
        t1 = threading.Thread(target=rec1, daemon=True)
        t0.start(); t1.start()

        # Mark "START" at recording start
        self.hardware.mark_start()

        # Lights: delay from the start of recording to lights ON
        ldelay = max(0.0, float(self.cfg.lights_delay_s))
        if ldelay > 0:
            print(f"[Trial {self.trial_idx}] Waiting {ldelay:.3f}s to turn lights ON...")
            wait_s(ldelay)
            self.hardware.light_on()
        else:
            self.hardware.light_on()

        # Stimulus: delay from start of recording to stimulus onset
        sdelay = max(0.0, float(self.cfg.stim_delay_s))
        now_to_stim = max(0.0, sdelay - ldelay) if sdelay > ldelay else sdelay  # simple stagger if both set
        if now_to_stim > 0:
            print(f"[Trial {self.trial_idx}] Waiting {now_to_stim:.3f}s before stimulus onset...")
            wait_s(now_to_stim)

        # Mark STIM at actual onset and run looming during recording
        self.hardware.mark_stim()
        self.stim.run(self.cfg.stim_duration_s,
                      self.cfg.stim_r0_px, self.cfg.stim_r1_px,
                      self.cfg.stim_bg_grey,
                      screen_idx=self.cfg.stim_screen_index,
                      fullscreen=self.cfg.stim_fullscreen)

        # Finish: wait for cameras, mark end, lights off
        t0.join(); t1.join()
        self.hardware.mark_end()
        self.hardware.light_off()

        cam0_path_actual = results["cam0"] or cam0_path_req
        cam1_path_actual = results["cam1"] or cam1_path_req

        self.log_writer.writerow([
            self.trial_idx, ts,
            cam0_path_actual, cam1_path_actual,
            self.cfg.record_duration_s,
            self.cfg.lights_delay_s, self.cfg.stim_delay_s, self.cfg.stim_duration_s,
            self.cfg.stim_screen_index, int(self.cfg.stim_fullscreen),
            self.cam0.target_fps, self.cam1.target_fps,
            self.cfg.video_preset_id, fourcc
        ])
        self.log_file.flush()
        print(f"[Trial {self.trial_idx}] END  (files: {cam0_path_actual} , {cam1_path_actual})")


# =========================
# All-in-One GUI
# =========================
class SettingsGUI(QtWidgets.QWidget):
    """Single-window GUI for settings, previews, and controls."""
    start_experiment = QtCore.pyqtSignal()
    stop_experiment  = QtCore.pyqtSignal()
    apply_settings   = QtCore.pyqtSignal()
    manual_trigger   = QtCore.pyqtSignal()
    check_updates    = QtCore.pyqtSignal()

    def __init__(self, cfg: Config, cam0: CameraRecorder, cam1: CameraRecorder):
        super().__init__()
        self.cfg  = cfg
        self.cam0 = cam0
        self.cam1 = cam1

        self.setWindowTitle(f"FlyPy — Trigger->Outputs (GUI)  v{__version__}")

        root = QtWidgets.QVBoxLayout(self)

        # Menu bar (Help → Check for Updates…)
        menubar = QtWidgets.QMenuBar()
        helpmenu = menubar.addMenu("Help")
        act_update = QtWidgets.QAction("Check for Updates…", self)
        helpmenu.addAction(act_update)
        act_update.triggered.connect(self.check_updates.emit)
        root.addWidget(menubar)

        # Top controls row
        controls = QtWidgets.QHBoxLayout()
        self.bt_start   = QtWidgets.QPushButton("Start")
        self.bt_stop    = QtWidgets.QPushButton("Stop")
        self.bt_trigger = QtWidgets.QPushButton("Trigger Once (Manual)")
        self.bt_apply   = QtWidgets.QPushButton("Apply Settings")
        self.bt_start.setToolTip("Watch for triggers. On each trigger: record, lights, stimulus, log.")
        self.bt_stop.setToolTip("Stop watching for triggers.")
        self.bt_trigger.setToolTip("Run one trial immediately (no hardware needed).")
        self.bt_apply.setToolTip("Apply changes from the panels below.")
        controls.addWidget(self.bt_start)
        controls.addWidget(self.bt_stop)
        controls.addWidget(self.bt_trigger)
        controls.addStretch(1)
        controls.addWidget(self.bt_apply)
        root.addLayout(controls)

        # Panels grid
        panels = QtWidgets.QGridLayout()
        panels.setColumnStretch(0, 1)
        panels.setColumnStretch(1, 1)
        root.addLayout(panels)

        # ------- General -------
        gen = QtWidgets.QGroupBox("General settings")
        gen.setToolTip("Top-level behavior and where files are saved.")
        gl = QtWidgets.QFormLayout(gen)
        gl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)

        self.lbl_sim = QtWidgets.QLabel("Simulation mode: OFF (hardware triggers active)")
        self.lbl_sim.setWordWrap(True)
        self.lbl_sim.setToolTip("If ON, triggers are generated on a timer.")
        gl.addRow(self.lbl_sim)

        self.sb_sim_interval = QtWidgets.QDoubleSpinBox()
        self.sb_sim_interval.setRange(0.5, 3600.0); self.sb_sim_interval.setDecimals(2)
        self.sb_sim_interval.setValue(self.cfg.sim_trigger_interval)
        self.sb_sim_interval.setToolTip("Time between simulated triggers (seconds).")
        gl.addRow(QtWidgets.QLabel("Interval between simulated triggers (seconds):"), self.sb_sim_interval)

        self.le_root = QtWidgets.QLineEdit(self.cfg.output_root)
        self.le_root.setToolTip("Folder where date-stamped trial folders/videos will be saved.")
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        rhl = QtWidgets.QHBoxLayout(); rhl.addWidget(self.le_root); rhl.addWidget(self.btn_browse)
        gl.addRow(QtWidgets.QLabel("Output folder for all trials:"), rhl)

        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.setToolTip("Select container/codec for saved videos.")
        self._preset_id_by_index = {}
        current_index = 0
        for i, p in enumerate(VIDEO_PRESETS):
            self.cb_format.addItem(p["label"]); self.cb_format.setItemData(i, p["id"]); self._preset_id_by_index[i] = p["id"]
            if p["id"] == self.cfg.video_preset_id: current_index = i
        self.cb_format.setCurrentIndex(current_index)
        gl.addRow(QtWidgets.QLabel("Video file format / codec:"), self.cb_format)

        self.sb_rec_dur = QtWidgets.QDoubleSpinBox()
        self.sb_rec_dur.setRange(0.2, 600.0); self.sb_rec_dur.setDecimals(2); self.sb_rec_dur.setValue(self.cfg.record_duration_s)
        self.sb_rec_dur.setToolTip("How long to record for each trigger (seconds).")
        gl.addRow(QtWidgets.QLabel("Recording duration per trigger (seconds):"), self.sb_rec_dur)

        panels.addWidget(gen, 0, 0)

        # ------- Stimulus & Timing -------
        stim = QtWidgets.QGroupBox("Stimulus & Timing")
        stim.setToolTip("Looming dot parameters and timing relative to recording start.")
        sl = QtWidgets.QFormLayout(stim)
        sl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)

        self.sb_stim_dur = QtWidgets.QDoubleSpinBox()
        self.sb_stim_dur.setRange(0.1, 30.0); self.sb_stim_dur.setDecimals(2); self.sb_stim_dur.setValue(self.cfg.stim_duration_s)
        self.sb_stim_dur.setToolTip("How long the looming dot is shown (seconds).")
        sl.addRow(QtWidgets.QLabel("Stimulus display duration (seconds):"), self.sb_stim_dur)

        self.sb_r0 = QtWidgets.QSpinBox(); self.sb_r0.setRange(1, 2000); self.sb_r0.setValue(self.cfg.stim_r0_px)
        self.sb_r0.setToolTip("Starting dot radius (px).")
        sl.addRow(QtWidgets.QLabel("Starting dot radius (pixels):"), self.sb_r0)

        self.sb_r1 = QtWidgets.QSpinBox(); self.sb_r1.setRange(1, 4000); self.sb_r1.setValue(self.cfg.stim_r1_px)
        self.sb_r1.setToolTip("Final dot radius (px).")
        sl.addRow(QtWidgets.QLabel("Final dot radius (pixels):"), self.sb_r1)

        self.sb_bg = QtWidgets.QDoubleSpinBox(); self.sb_bg.setRange(0.0, 1.0); self.sb_bg.setSingleStep(0.05); self.sb_bg.setValue(self.cfg.stim_bg_grey)
        self.sb_bg.setToolTip("Background shade (0=black, 1=white).")
        sl.addRow(QtWidgets.QLabel("Stimulus background shade (0=black, 1=white):"), self.sb_bg)

        # NEW: delays (from recording start) — both independently configurable
        self.sb_light_delay = QtWidgets.QDoubleSpinBox()
        self.sb_light_delay.setRange(0.0, 10.0); self.sb_light_delay.setDecimals(3); self.sb_light_delay.setValue(self.cfg.lights_delay_s)
        self.sb_light_delay.setToolTip("Delay (seconds) from recording start to LIGHTS ON.")
        sl.addRow(QtWidgets.QLabel("Delay from recording start → lights ON (seconds):"), self.sb_light_delay)

        self.sb_stim_delay = QtWidgets.QDoubleSpinBox()
        self.sb_stim_delay.setRange(0.0, 10.0); self.sb_stim_delay.setDecimals(3); self.sb_stim_delay.setValue(self.cfg.stim_delay_s)
        self.sb_stim_delay.setToolTip("Delay (seconds) from recording start to STIMULUS onset.")
        sl.addRow(QtWidgets.QLabel("Delay from recording start → stimulus ON (seconds):"), self.sb_stim_delay)

        panels.addWidget(stim, 0, 1)

        # ------- Display / Screens -------
        disp = QtWidgets.QGroupBox("Display & Windows")
        disp.setToolTip("Choose which monitor shows the Stimulus and the GUI. Fullscreen or windowed.")
        dl = QtWidgets.QFormLayout(disp)
        dl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)

        screens = QtGui.QGuiApplication.screens()
        def screen_label(i, s):
            g = s.geometry()
            return f"Screen {i} — {g.width()}×{g.height()} @ ({g.x()},{g.y()})"

        self.cb_stim_screen = QtWidgets.QComboBox()
        self.cb_gui_screen = QtWidgets.QComboBox()
        for i, s in enumerate(screens):
            lbl = screen_label(i, s); self.cb_stim_screen.addItem(lbl); self.cb_gui_screen.addItem(lbl)
        self.cb_stim_screen.setCurrentIndex(self.cfg.stim_screen_index if self.cfg.stim_screen_index < len(screens) else 0)
        self.cb_gui_screen.setCurrentIndex(self.cfg.gui_screen_index if self.cfg.gui_screen_index < len(screens) else 0)
        self.cb_stim_screen.setToolTip("Which monitor should the looming stimulus appear on.")
        self.cb_gui_screen.setToolTip("Which monitor should host this GUI window.")

        self.cb_stim_fullscr = QtWidgets.QCheckBox("Stimulus fullscreen on selected screen")
        self.cb_stim_fullscr.setChecked(bool(self.cfg.stim_fullscreen))
        self.cb_stim_fullscr.setToolTip("Unchecked = windowed (draggable). Checked = fullscreen.")

        dl.addRow(QtWidgets.QLabel("Stimulus display screen:"), self.cb_stim_screen)
        dl.addRow(QtWidgets.QLabel("GUI display screen:"), self.cb_gui_screen)
        dl.addRow(self.cb_stim_fullscr)

        panels.addWidget(disp, 1, 0, 1, 2)

        # ------- Camera panels -------
        self.cam_groups = []
        for idx, cam, target_default in [(0, self.cam0, self.cfg.cam0_target_fps),
                                         (1, self.cam1, self.cfg.cam1_target_fps)]:
            gb = QtWidgets.QGroupBox(f"Camera {idx} — preview & frame rate")
            fl = QtWidgets.QGridLayout(gb)

            preview = QtWidgets.QLabel()
            preview.setFixedSize(640, 480)
            preview.setFrameShape(QtWidgets.QFrame.Box)
            preview.setAlignment(QtCore.Qt.AlignCenter)
            preview.setToolTip("Live preview (pauses while recording).")
            fl.addWidget(preview, 0, 0, 5, 1)

            spin_index = QtWidgets.QSpinBox(); spin_index.setRange(0, 15)
            spin_index.setValue(getattr(cam, "index", 0))
            spin_index.setToolTip("OpenCV device index.")
            fl.addWidget(QtWidgets.QLabel("Which camera to use (OpenCV device index):"), 0, 1)
            fl.addWidget(spin_index, 0, 2)

            spin_fps = QtWidgets.QSpinBox(); spin_fps.setRange(1, 10000); spin_fps.setValue(int(target_default))
            spin_fps.setAccelerated(True)
            spin_fps.setToolTip("Target recording FPS (actual may vary).")
            fl.addWidget(QtWidgets.QLabel("Target recording frame rate (fps):"), 1, 1)
            fl.addWidget(spin_fps, 1, 2)

            lbl_rep = QtWidgets.QLabel("Driver-reported frame rate: —")
            lbl_mea = QtWidgets.QLabel("Measured preview frame rate (GUI): —")
            lbl_tar = QtWidgets.QLabel(f"Recording target frame rate (intended): {int(target_default)}")
            lbl_rep.setWordWrap(True); lbl_mea.setWordWrap(True); lbl_tar.setWordWrap(True)
            fl.addWidget(lbl_rep, 2, 1, 1, 2)
            fl.addWidget(lbl_mea, 3, 1, 1, 2)
            fl.addWidget(lbl_tar, 4, 1, 1, 2)

            fl.setColumnStretch(0, 3); fl.setColumnStretch(1, 1); fl.setColumnStretch(2, 1)

            self.cam_groups.append({
                "group": gb, "preview": preview,
                "spin_index": spin_index, "spin_fps": spin_fps,
                "lbl_rep": lbl_rep, "lbl_mea": lbl_mea, "lbl_tar": lbl_tar,
                "cam": cam
            })
            panels.addWidget(gb, 2, idx)

        # Status
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        # Signals
        self.bt_start.clicked.connect(self.start_experiment.emit)
        self.bt_stop.clicked.connect(self.stop_experiment.emit)
        self.bt_trigger.clicked.connect(self.manual_trigger.emit)
        self.bt_apply.clicked.connect(self.apply_settings.emit)
        self.btn_browse.clicked.connect(self._pick_folder)

        # Initial text
        self._refresh_general_labels()

    def _pick_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Root", self.le_root.text() or ".")
        if path: self.le_root.setText(path)

    def _refresh_general_labels(self):
        self.lbl_sim.setText(
            f"Simulation mode: {'ON (timer-based triggers)' if self.cfg.simulation_mode else 'OFF (hardware triggers active)'}"
        )

    def update_cam_fps_labels(self):
        for g in self.cam_groups:
            cam: CameraRecorder = g["cam"]
            rep = cam.reported_fps()
            mea = cam.measured_preview_fps()
            g["lbl_rep"].setText("Driver-reported frame rate: " + (f"{rep:.1f}" if rep > 0 else "(unknown)"))
            g["lbl_mea"].setText(f"Measured preview frame rate (GUI): {mea:.1f}")
            g["lbl_tar"].setText(f"Recording target frame rate (intended): {int(cam.target_fps)}")

    def set_preview_image(self, cam_idx: int, img_rgb: np.ndarray):
        g = self.cam_groups[cam_idx]
        h, w, _ = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, w*3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(g["preview"].size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        g["preview"].setPixmap(pix)


# =========================
# Main Application
# =========================
class MainApp(QtWidgets.QApplication):
    """Top-level Qt app wiring Config, Hardware, Cameras, GUI, Stimulus, and the trigger loop."""
    def __init__(self, argv):
        super().__init__(argv)
        # Config + Simulation prompt
        self.cfg = Config()
        reply = QtWidgets.QMessageBox.question(None, "Simulation Mode",
                                               "Run in SIMULATION MODE?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        self.cfg.simulation_mode = (reply == QtWidgets.QMessageBox.Yes)

        # Hardware + cameras
        self.hardware = HardwareBridge(self.cfg)
        self.cam0 = CameraRecorder(self.cfg.cam0_index, "cam0", self.cfg.cam0_target_fps)
        self.cam1 = CameraRecorder(self.cfg.cam1_index, "cam1", self.cfg.cam1_target_fps)

        # Output + logger
        ensure_dir(self.cfg.output_root)
        self.trial_runner = TrialRunner(self.cfg, self.hardware, self.cam0, self.cam1,
                                        logger_path=os.path.join(self.cfg.output_root, "trials_log.csv"))

        # GUI
        self.gui = SettingsGUI(self.cfg, self.cam0, self.cam1)
        self.gui.start_experiment.connect(self.start_loop)
        self.gui.stop_experiment.connect(self.stop_loop)
        self.gui.apply_settings.connect(self.apply_from_gui)
        self.gui.manual_trigger.connect(self.trigger_once)
        self.gui.check_updates.connect(self.on_check_updates)
        self.gui.show()

        # Place & scale GUI to max usable size of the selected screen
        self.position_and_maximize_gui(self.cfg.gui_screen_index)

        # Make sure the stimulus window exists right away (persistent until quit)
        self.trial_runner.stim.open_persistent_window(self.cfg.stim_screen_index,
                                                      self.cfg.stim_fullscreen,
                                                      self.cfg.stim_bg_grey)

        # Preview timer
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(500)  # ms
        self.preview_timer.timeout.connect(self.update_previews)
        self.preview_timer.start()

        # State flags
        self.running = False
        self.in_trial = False
        self.thread = None

        # Cleanup hooks
        self.aboutToQuit.connect(self.cleanup)
        atexit.register(self.cleanup)

        self.gui._refresh_general_labels()

    # ---------- helpers ----------
    def position_and_maximize_gui(self, screen_idx: int):
        """Center and maximize the GUI window on the chosen screen."""
        screens = QtGui.QGuiApplication.screens()
        if not screens:
            self.gui.showMaximized()
            return
        if screen_idx < 0 or screen_idx >= len(screens):
            screen_idx = 0
        g = screens[screen_idx].availableGeometry()
        w = self.gui.frameGeometry()
        w.moveCenter(g.center())
        self.gui.move(w.topLeft())
        # Maximize to the available geometry (auto-scale to max size)
        self.gui.showMaximized()

    # ---------- settings ----------
    def apply_from_gui(self):
        """Push current GUI values into Config/Cameras and update windows."""
        # General
        self.cfg.sim_trigger_interval = float(self.gui.sb_sim_interval.value())
        self.cfg.output_root = self.gui.le_root.text().strip() or self.cfg.output_root

        idx = self.gui.cb_format.currentIndex()
        preset_id = self.gui.cb_format.itemData(idx) or self.gui._preset_id_by_index.get(idx, default_preset_id())
        self.cfg.video_preset_id = preset_id
        self.cfg.fourcc = PRESETS_BY_ID[self.cfg.video_preset_id]["fourcc"]
        self.cfg.record_duration_s = float(self.gui.sb_rec_dur.value())

        # Stimulus & timing
        self.cfg.stim_duration_s = float(self.gui.sb_stim_dur.value())
        self.cfg.stim_r0_px = int(self.gui.sb_r0.value())
        self.cfg.stim_r1_px = int(self.gui.sb_r1.value())
        self.cfg.stim_bg_grey = float(self.gui.sb_bg.value())
        self.cfg.lights_delay_s = float(self.gui.sb_light_delay.value())  # NEW
        self.cfg.stim_delay_s   = float(self.gui.sb_stim_delay.value())   # NEW
        self.cfg.stim_screen_index = int(self.gui.cb_stim_screen.currentIndex())
        self.cfg.stim_fullscreen   = bool(self.gui.cb_stim_fullscr.isChecked())

        # GUI screen placement
        self.cfg.gui_screen_index = int(self.gui.cb_gui_screen.currentIndex())
        self.position_and_maximize_gui(self.cfg.gui_screen_index)

        # Cameras
        cam0_new_idx = int(self.gui.cam_groups[0]["spin_index"].value())
        cam0_new_tfps = int(self.gui.cam_groups[0]["spin_fps"].value())
        if cam0_new_idx != getattr(self.cam0, "index", -1):
            self.cam0.set_index(cam0_new_idx)
        self.cam0.set_target_fps(cam0_new_tfps)

        cam1_new_idx = int(self.gui.cam_groups[1]["spin_index"].value())
        cam1_new_tfps = int(self.gui.cam_groups[1]["spin_fps"].value())
        if cam1_new_idx != getattr(self.cam1, "index", -1):
            self.cam1.set_index(cam1_new_idx)
        self.cam1.set_target_fps(cam1_new_tfps)

        # Mirror back for logging completeness
        self.cfg.cam0_index = getattr(self.cam0, "index", 0)
        self.cfg.cam1_index = getattr(self.cam1, "index", 1)
        self.cfg.cam0_target_fps = int(self.cam0.target_fps)
        self.cfg.cam1_target_fps = int(self.cam1.target_fps)

        # Reconfigure stimulus persistent window immediately
        self.trial_runner.stim.open_persistent_window(self.cfg.stim_screen_index,
                                                      self.cfg.stim_fullscreen,
                                                      self.cfg.stim_bg_grey)

        ensure_dir(self.cfg.output_root)
        self.gui.lbl_status.setText("Status: Settings applied.")
        print("[MainApp] Settings applied.")

    # ---------- preview ----------
    def update_previews(self):
        if self.in_trial:
            self.gui.lbl_status.setText("Status: Trial running (preview paused).")
            self.gui.update_cam_fps_labels()
            return
        p0 = self.gui.cam_groups[0]["preview"]; p1 = self.gui.cam_groups[1]["preview"]
        img0 = self.cam0.grab_preview(w=p0.width(), h=p0.height())
        img1 = self.cam1.grab_preview(w=p1.width(), h=p1.height())
        self.gui.set_preview_image(0, img0); self.gui.set_preview_image(1, img1)
        self.gui.update_cam_fps_labels()
        self.gui.lbl_status.setText("Status: Waiting / Idle.")

    # ---------- loops & triggers ----------
    def loop(self):
        self.gui.lbl_status.setText("Status: Watching for triggers...")
        print("[MainApp] Trigger loop started.")
        while self.running:
            try:
                if not self.in_trial and self.hardware.check_trigger():
                    self.in_trial = True
                    self.gui.lbl_status.setText("Status: Trial running...")
                    self.trial_runner.run_trial()
                    self.in_trial = False
                    self.gui.lbl_status.setText("Status: Trial finished.")
                time.sleep(0.002)
            except Exception as e:
                print(f"[MainApp] Loop error: {e}")
                self.gui.lbl_status.setText(f"Status: Error - {e}")
                time.sleep(0.05)
        print("[MainApp] Trigger loop stopped.")

    def trigger_once(self):
        if self.in_trial: return
        def _run():
            try:
                self.in_trial = True
                self.gui.lbl_status.setText("Status: Trial running (manual trigger)...")
                self.trial_runner.run_trial()
                self.gui.lbl_status.setText("Status: Trial finished (manual).")
            finally:
                self.in_trial = False
        threading.Thread(target=_run, daemon=True).start()

    def start_loop(self):
        if not self.running:
            self.apply_from_gui()  # ensure latest settings
            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop_loop(self):
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
            self.thread = None
        self.gui.lbl_status.setText("Status: Stopped.")

    # ---------- updater ----------
    def on_check_updates(self):
        """Menu handler: Help → Check for Updates… (uses the inline updater)."""
        # Configure your repo and asset name here:
        REPO = "your-github-user/your-repo"    # e.g., "neuro-lab/FlyPy"
        ASSET_NAME = "FlyPy-Full.zip"          # the asset attached to Releases

        def worker():
            try:
                found, msg = check_updates_and_stage(repo=REPO,
                                                     current_version=__version__,
                                                     asset_name=ASSET_NAME)
            except Exception as e:
                found, msg = False, f"Update check failed: {e}"
            QtCore.QMetaObject.invokeMethod(self, "_show_update_msg",
                                            QtCore.Qt.QueuedConnection,
                                            QtCore.Q_ARG(str, msg))
        threading.Thread(target=worker, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _show_update_msg(self, msg: str):
        QtWidgets.QMessageBox.information(None, "Updates", msg)

    # ---------- cleanup ----------
    def cleanup(self):
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
            self.thread = None
        try:
            self.trial_runner.close()
        except Exception:
            pass
        try:
            self.hardware.close()
        except Exception:
            pass
        for cam in (self.cam0, self.cam1):
            try:
                cam.release()
            except Exception:
                pass
        print("[MainApp] Cleanup complete.")


if __name__ == "__main__":
    app = MainApp(sys.argv)
    sys.exit(app.exec_())
