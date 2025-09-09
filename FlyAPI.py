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
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
except Exception:
    pass

# Make stdout/stderr UTF-8 if possible (helps on some Windows consoles)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Optional PsychoPy (lazy-loaded to speed startup). We import only on first use.
import importlib
PSYCHOPY = None  # None=unknown (not tried), True=loaded OK, False=unavailable
visual = None
core = None

def _ensure_psychopy_loaded() -> bool:
    """Import PsychoPy submodules on first use. Returns True if available."""
    global PSYCHOPY, visual, core
    if PSYCHOPY is True:
        return True
    if PSYCHOPY is False:
        return False
    try:
        # Defer heavy import until actually needed
        psychopy = importlib.import_module("psychopy")  # noqa: F401
        visual = importlib.import_module("psychopy.visual")
        core   = importlib.import_module("psychopy.core")
        PSYCHOPY = True
        return True
    except Exception:
        visual = None
        core = None
        PSYCHOPY = False
        return False


# =========================
# Utilities
# =========================

def default_preset_id() -> str:
    """Return the default video preset ID. Chosen for wide compatibility."""
    return "mp4_mp4v"

def _parse_cli(argv):
    """Very small CLI parser for power users."""
    args = {"simulate": False, "prewarm_stim": False}
    try:
        import argparse
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--simulate", action="store_true")
        ap.add_argument("--prewarm-stim", action="store_true")
        ns, _rest = ap.parse_known_args(argv[1:] if isinstance(argv, (list, tuple)) else [])
        args["simulate"] = bool(getattr(ns, "simulate", False))
        args["prewarm_stim"] = bool(getattr(ns, "prewarm_stim", False))
    except Exception:
        pass
    return args

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
    if _ensure_psychopy_loaded():
        try:
            core.wait(sec)
            return
        except Exception:
            pass
    time.sleep(sec)

def get_screen_geometries():
    """Return a list of tuples describing each screen: (x, y, width, height)."""
    screens = QtGui.QGuiApplication.screens()
    geoms = []
    for s in screens:
        g = s.geometry()
        geoms.append((g.x(), g.y(), g.width(), g.height()))
    return geoms


# =========================
# Video presets (container/codec choices)
# =========================

VIDEO_PRESETS = [
    {
        "id": "mp4_mp4v",
        "label": "MP4 / mp4v — very compatible (H.263-like); ~large files; 60–200 fps ok",
        "fourcc": "mp4v",
    },
    {
        "id": "avi_xvid",
        "label": "AVI / XVID — broad compatibility; moderate size",
        "fourcc": "XVID",
    },
    {
        "id": "avi_mjpg",
        "label": "AVI / MJPG — huge files; light CPU; high fps ok",
        "fourcc": "MJPG",
    },
]
PRESETS_BY_ID = {p["id"]: p for p in VIDEO_PRESETS}


# =========================
# Config
# =========================

class Config:
    """Mutable configuration object shared across the app (GUI <-> runtime)."""
    def __init__(self):
        # General
        self.simulation_mode = False
        self.sim_trigger_interval = 5.0
        self.prewarm_stim = False  # do not pre-open the stimulus window by default

        # Recording / output
        self.output_root = "FlyPy_Output"
        self.video_preset_id = default_preset_id()
        self.fourcc = PRESETS_BY_ID[self.video_preset_id]["fourcc"]
        self.record_duration_s = 3.0

        # Stimulus (WHITE background with BLACK dot by default)
        self.stim_duration_s = 1.5
        self.stim_r0_px = 8
        self.stim_r1_px = 240
        self.stim_bg_grey = 1.0  # 1.0 = white
        self.lights_delay_s = 0.0  # NEW
        self.stim_delay_s   = 0.0  # NEW

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
        self._opened = False  # have we attempted to open yet?

    def _open_if_needed(self):
        """Open the serial port the first time we actually need it."""
        if self.simulated or self._opened:
            return
        self._opened = True
        try:
            import serial, serial.tools.list_ports
            if not self.port:
                self.port = self._autodetect_port()
            if self.port:
                try:
                    self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
                    # Move MCU settle delay out of startup and into first real use
                    wait_s(1.2)
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
        self._open_if_needed()
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
        self._open_if_needed()
        if self.simulated or not self.ser:
            print(f"[HardwareBridge] (Sim) SEND: {text}")
            return
        try:
            self.ser.write((text.strip() + "\n").encode("utf-8", errors="ignore"))
        except Exception as e:
            print(f"[HardwareBridge] Write error: {e}")

    def mark_start(self):
        self._send_line("MARK START")

    def mark_end(self):
        self._send_line("MARK END")

    def lights_on(self):
        self._send_line("LIGHT ON")

    def lights_off(self):
        self._send_line("LIGHT OFF")

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
        self.index = index  # defer opening until first use

    def _ensure_open(self):
        """Open the VideoCapture lazily on first use."""
        if self.cap is None:
            cap, synth = self._open(self.index)
            self.cap, self.synthetic = cap, synth
            if synth:
                print(f"[Camera {self.name}] index {self.index} not available -> synthetic preview/recording.")
            else:
                print(f"[Camera {self.name}] opened (lazy) on index {self.index}.")

    def _open(self, index: int):
        backends = [cv2.CAP_ANY]
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap and cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                    except Exception:
                        pass
                    return cap, False
                if cap:
                    cap.release()
            except Exception:
                pass
        # synthetic fallback
        return None, True

    def set_index(self, index: int):
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.synthetic = False
            self.index = index
            print(f"[Camera {self.name}] bound to index {index} (will open lazily).")

    def set_target_fps(self, fps: int):
        with self.lock:
            self.target_fps = float(fps)
            if self.cap:
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                except Exception:
                    pass

    def grab_preview(self, w=320, h=240, overlay_index=True):
        self._ensure_open()
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
                cv2.putText(frame, f"Index: {getattr(self, 'index', '?')}", (10, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

            t = time.time()
            self._preview_times.append(t)
            if len(self._preview_times) >= 2:
                dt = self._preview_times[-1] - self._preview_times[0]
                fps = (len(self._preview_times) - 1) / dt if dt > 0 else 0.0
            else:
                fps = 0.0
            cv2.putText(frame, f"~{fps:.1f} FPS", (w-120, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

            self._last_preview_frame = frame
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
        self._ensure_open()
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
                    # draw a moving synthetic marker
                    cx = (frame_index * 7) % size[0]
                    cv2.circle(frame, (cx, size[1]//2), 25, (0, 0, 0), 2)
                    ok_any = True
                else:
                    ok, bgr = self.cap.read()
                    if not ok or bgr is None:
                        frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
                        cv2.putText(frame, "[drop]", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    else:
                        ok_any = True
                        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            out.write(frame)
            frame_index += 1
            time.sleep(max(0.0, 1.0 / float(self.target_fps)))
        out.release()
        if not ok_any and not self.synthetic:
            print(f"[Camera {self.name}] Warning: no frames captured from camera index {self.index}.")
        print(f"[Camera {self.name}] Recording complete.")
        return path

    def last_preview(self):
        return self._last_preview_frame

    def release(self):
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None


# =========================
# Looming Stimulus — (optional pre-warm; lazy PsychoPy)
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
        """(Optional) Pre-warm a stimulus window. Disabled by default for fast startup."""
        if _ensure_psychopy_loaded():
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
        if _ensure_psychopy_loaded():
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

        # --- OpenCV fallback ---
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
# Trial Runner (records, lights, stimulus, CSV log)
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
                "timestamp", "trial_idx",
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

    def _trial_folder(self):
        root = day_folder(self.cfg.output_root)
        tstamp = now_stamp()
        p = os.path.join(root, f"trial_{tstamp}")
        ensure_dir(p)
        return p

    def _record_both(self, folder: str):
        """Record both cameras concurrently for the configured duration."""
        dur = float(self.cfg.record_duration_s)
        fourcc_str = self.cfg.fourcc

        cam0_path = os.path.join(folder, f"cam0.{self._ext_for_fourcc(fourcc_str)}")
        cam1_path = os.path.join(folder, f"cam1.{self._ext_for_fourcc(fourcc_str)}")

        result = {"cam0": None, "cam1": None}

        def rec_cam(cam: CameraRecorder, path: str, key: str):
            result[key] = cam.record_clip(path, dur, fourcc_str=fourcc_str)

        t0 = threading.Thread(target=rec_cam, args=(self.cam0, cam0_path, "cam0"))
        t1 = threading.Thread(target=rec_cam, args=(self.cam1, cam1_path, "cam1"))
        t0.start(); t1.start()
        t0.join(); t1.join()

        return result["cam0"], result["cam1"]

    def _ext_for_fourcc(self, fourcc_str: str) -> str:
        # map FOURCC to filename extension
        if fourcc_str.lower() in ("mp4v", "avc1", "h264"):
            return "mp4"
        if fourcc_str.lower() in ("mjpg", "xvid", "divx"):
            return "avi"
        return "mp4"

    def run_one(self):
        """Execute one full trial: record, lights, stimulus, and log."""
        folder = self._trial_folder()
        self.hardware.mark_start()

        # Begin recording immediately
        print("[Trial] Recording both cameras...")
        cam0_path, cam1_path = None, None
        rec_thread = threading.Thread(target=lambda: self._store_cam_paths(folder))
        # Actually record synchronously here to make delays meaningful
        cam0_path, cam1_path = self._record_both(folder)

        # Lights and stimulus timing (relative to recording start)
        if self.cfg.lights_delay_s > 0:
            print(f"[Trial] Waiting {self.cfg.lights_delay_s:.3f}s before LIGHTS ON...")
            wait_s(self.cfg.lights_delay_s)
        self.hardware.lights_on()

        if self.cfg.stim_delay_s > 0:
            print(f"[Trial] Waiting {self.cfg.stim_delay_s:.3f}s before STIMULUS...")
            wait_s(self.cfg.stim_delay_s)
        self.stim.run(
            duration_s=self.cfg.stim_duration_s,
            r0=int(self.cfg.stim_r0_px),
            r1=int(self.cfg.stim_r1_px),
            bg_grey=float(self.cfg.stim_bg_grey),
            screen_idx=int(self.cfg.stim_screen_index),
            fullscreen=bool(self.cfg.stim_fullscreen),
        )

        self.hardware.lights_off()
        self.hardware.mark_end()

        # Log CSV row
        self.trial_idx += 1
        self.log_writer.writerow([
            now_stamp(), self.trial_idx,
            cam0_path or "", cam1_path or "",
            float(self.cfg.record_duration_s),
            float(self.cfg.lights_delay_s), float(self.cfg.stim_delay_s), float(self.cfg.stim_duration_s),
            int(self.cfg.stim_screen_index), bool(self.cfg.stim_fullscreen),
            int(self.cam0.target_fps), int(self.cam1.target_fps),
            self.cfg.video_preset_id, self.cfg.fourcc
        ])
        self.log_file.flush()
        print("[Trial] Logged.")

    def _store_cam_paths(self, folder):
        pass  # legacy compatibility (record now handled inline)


# =========================
# GUI
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
        self.bt_trigger = QtWidgets.QPushButton("Trigger Once")
        self.bt_apply   = QtWidgets.QPushButton("Apply Settings")
        controls.addWidget(self.bt_start); controls.addWidget(self.bt_stop)
        controls.addWidget(self.bt_trigger); controls.addWidget(self.bt_apply)
        root.addLayout(controls)

        # Signals for buttons
        self.bt_start.clicked.connect(self.start_experiment.emit)
        self.bt_stop.clicked.connect(self.stop_experiment.emit)
        self.bt_trigger.clicked.connect(self.manual_trigger.emit)
        self.bt_apply.clicked.connect(self.apply_settings.emit)

        # Status label
        self.lbl_status = QtWidgets.QLabel("Status: Idle.")
        root.addWidget(self.lbl_status)

        # Panels grid
        panels = QtWidgets.QGridLayout()
        root.addLayout(panels)

        # ------- General -------
        gen = QtWidgets.QGroupBox("General")
        gen.setToolTip("Global options and output folders.")
        gl = QtWidgets.QFormLayout(gen)
        gl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)

        self.sb_sim_interval = QtWidgets.QDoubleSpinBox()
        self.sb_sim_interval.setRange(0.1, 3600.0); self.sb_sim_interval.setDecimals(2)
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
        gl.addRow(QtWidgets.QLabel("Video format / codec:"), self.cb_format)

        self.sb_rec_dur = QtWidgets.QDoubleSpinBox()
        self.sb_rec_dur.setRange(0.1, 600.0); self.sb_rec_dur.setDecimals(2); self.sb_rec_dur.setValue(self.cfg.record_duration_s)
        self.sb_rec_dur.setToolTip("Recording duration per trigger (seconds).")
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

        self.cb_prewarm = QtWidgets.QCheckBox("Pre-warm stimulus window at launch (slower startup, faster first trial)")
        self.cb_prewarm.setChecked(bool(getattr(self.cfg, "prewarm_stim", False)))
        self.cb_prewarm.setToolTip("If checked, the stimulus window opens during startup. Leave unchecked for fastest launch.")
        dl.addRow(self.cb_prewarm)

        panels.addWidget(disp, 1, 0, 1, 2)

        # ------- Camera panels -------
        self.cam_groups = []
        for idx, cam, target_default in [(0, self.cam0, self.cfg.cam0_target_fps),
                                         (1, self.cam1, self.cfg.cam1_target_fps)]:
            gb = QtWidgets.QGroupBox(f"Camera {idx}")
            fl = QtWidgets.QGridLayout(gb)

            preview = QtWidgets.QLabel()
            preview.setFixedSize(360, 240)
            preview.setStyleSheet("background-color: #ddd; border: 1px solid #aaa;")
            preview.setAlignment(QtCore.Qt.AlignCenter)
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

            lbl_rep = QtWidgets.QLabel("Driver-reported fps: ~0.0")
            fl.addWidget(lbl_rep, 2, 1, 1, 2)

            self.cam_groups.append({
                "group": gb, "preview": preview, "spin_index": spin_index,
                "spin_fps": spin_fps, "lbl_rep": lbl_rep
            })
            panels.addWidget(gb, 2 + idx, 0, 1, 2)

        # Footer
        self.lbl_general = QtWidgets.QLabel("")
        root.addWidget(self.lbl_general)

    def set_preview_image(self, cam_idx: int, img_rgb: np.ndarray):
        """Update preview image for camera `cam_idx`."""
        if img_rgb is None:
            return
        h, w, _ = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.cam_groups[cam_idx]["preview"].setPixmap(pix)

    def update_cam_fps_labels(self):
        for g in self.cam_groups:
            # We only have approximate preview fps; display what we computed in the overlay text
            g["lbl_rep"].setText(f"Driver-reported fps: (see overlay)")

    def _refresh_general_labels(self):
        """Update the summary label with high-level state."""
        self.lbl_general.setText(
            f"Simulation mode: {'ON (timer-based triggers)' if self.cfg.simulation_mode else 'OFF (hardware triggers active)'}"
        )


# =========================
# Main Application
# =========================

class MainApp(QtWidgets.QApplication):
    """Top-level Qt app wiring Config, Hardware, Cameras, GUI, Stimulus, and the trigger loop."""
    def __init__(self, argv):
        super().__init__(argv)
        # Config (no blocking modal prompt). Default simulation OFF; allow CLI flags.
        self.cfg = Config()
        cli = _parse_cli(argv)
        if cli.get("simulate"):
            self.cfg.simulation_mode = True

        # Hardware + cameras
        self.hardware = HardwareBridge(self.cfg)
        self.cam0 = CameraRecorder(self.cfg.cam0_index, "cam0", self.cfg.cam0_target_fps)
        self.cam1 = CameraRecorder(self.cfg.cam1_index, "cam1", self.cfg.cam1_target_fps)

        # Logger + trial runner
        out_root = self.cfg.output_root
        ensure_dir(out_root)
        log_path = os.path.join(out_root, "trials_log.csv")
        self.trial_runner = TrialRunner(self.cfg, self.hardware, self.cam0, self.cam1, log_path)

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

        # Do NOT pre-open the stimulus window by default (faster startup). Optionally prewarm later.
        # Preview timer starts after the window first paints.
        QtCore.QTimer.singleShot(200, self._start_previews)

        # Optional CLI prewarm
        if getattr(self.cfg, "prewarm_stim", False) or cli.get("prewarm_stim"):
            QtCore.QTimer.singleShot(300, lambda: self.trial_runner.stim.open_persistent_window(
                self.cfg.stim_screen_index, self.cfg.stim_fullscreen, self.cfg.stim_bg_grey))

        # State flags

    def _start_previews(self):
        """Start the periodic preview refresh."""
        try:
            self.preview_timer = QtCore.QTimer(self)
            self.preview_timer.setInterval(500)
            self.preview_timer.timeout.connect(self.update_previews)
            self.preview_timer.start()
        except Exception as e:
            print(f"[MainApp] Preview timer error: {e}")

        # State flags
        self.running = False
        self.in_trial = False
        self.thread = None

        # Cleanup hooks
        self.aboutToQuit.connect(self.cleanup)
        atexit.register(self.cleanup)

        self.gui._refresh_general_labels()

    # ---------- placement ----------
    def position_and_maximize_gui(self, screen_index: int):
        """Move the GUI to the selected screen and maximize to usable area."""
        try:
            screens = QtGui.QGuiApplication.screens()
            if 0 <= screen_index < len(screens):
                geo = screens[screen_index].availableGeometry()
                self.gui.setGeometry(geo)
                self.gui.showMaximized()
            else:
                self.gui.showMaximized()
        except Exception:
            self.gui.showMaximized()

    # ---------- apply settings ----------
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

        # Reconfigure stimulus persistent window immediately (only if prewarm is enabled)
        self.cfg.prewarm_stim = bool(self.gui.cb_prewarm.isChecked())
        if self.cfg.prewarm_stim:
            self.trial_runner.stim.open_persistent_window(self.cfg.stim_screen_index,
                                                          self.cfg.stim_fullscreen,
                                                          self.cfg.stim_bg_grey)
        else:
            # Close any prewarmed window if user disabled it
            try:
                self.trial_runner.stim.close()
            except Exception:
                pass

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
        try:
            while self.running:
                if self.hardware.check_trigger():
                    self.in_trial = True
                    self.gui.lbl_status.setText("Status: Triggered — running trial...")
                    try:
                        self.trial_runner.run_one()
                    except Exception as e:
                        print(f"[MainApp] Trial error: {e}")
                    self.in_trial = False
                    self.gui.lbl_status.setText("Status: Waiting / Idle.")
                QtWidgets.QApplication.processEvents()
                time.sleep(0.01)
        finally:
            print("[MainApp] Trigger loop exiting.")

    def start_loop(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.gui.lbl_status.setText("Status: Trigger loop running.")
        print("[MainApp] Start.")

    def stop_loop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None
        self.gui.lbl_status.setText("Status: Stopped.")
        print("[MainApp] Stop.")

    def trigger_once(self):
        """Manual single trial (ignores trigger loop state)."""
        if self.in_trial:
            return
        self.in_trial = True
        try:
            self.trial_runner.run_one()
        except Exception as e:
            print(f"[MainApp] Manual trial error: {e}")
        self.in_trial = False

    # ---------- updater ----------
    def on_check_updates(self):
        """Download a release artifact and stage for update (see auto_update.py in repository)."""
        try:
            from auto_update import check_and_stage_update
            ok = check_and_stage_update(__version__)
            QtWidgets.QMessageBox.information(self.gui, "Update",
                                              "Update staged. Exit & relaunch to finish." if ok
                                              else "Already up to date.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self.gui, "Update error", str(e))

    # ---------- cleanup ----------
    def cleanup(self):
        print("[MainApp] Cleaning up…")
        try:
            self.hardware.close()
        except Exception:
            pass
        for cam in (self.cam0, self.cam1):
            try:
                cam.release()
            except Exception:
                pass
        try:
            self.trial_runner.stim.close()
        except Exception:
            pass
        print("[MainApp] Cleanup complete.")


if __name__ == "__main__":
    app = MainApp(sys.argv)
    sys.exit(app.exec_())
