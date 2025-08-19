"""
FlyPy — Trigger->Outputs build with All-in-One Settings GUI, camera previews, rich tooltips,
and comprehensive docstrings (natural-English labels).

On each trigger:
  • Activates lights (serial if available; simulated otherwise)
  • Records synchronized clips from two cameras to disk
  • Presents a looming (growing dot) visual stimulus
  • Logs trial metadata (CSV) with file paths and timestamps

GUI:
  • All settings visible in one window (natural-English labels)
  • Tooltips on every control
  • Per-camera visual index (live preview with overlayed index) and FPS
  • Video Format/Codec dropdown with captions (max res/FPS, size hint)
  • Stimulus delay after recording start
  • Select which screen shows the Stimulus and which screen hosts the GUI
  • Stimulus fullscreen/windowed mode
  • Manual "Trigger Once" button

This file is heavily annotated. Every class and function includes a docstring written in natural English,
and most non-obvious lines include inline comments to make the control/data flow easy to follow.
"""

import sys
import threading
import time
import os
import csv
from datetime import datetime
import atexit
from collections import deque

import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui

# ---------- HiDPI scaling (set BEFORE QApplication is constructed) ----------
try:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
except Exception:
    # If Qt can't set these attributes (rare), we just proceed without HiDPI scaling.
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
# Utilities / Config
# =========================

# Video format presets: container + FOURCC + caption
# Note: Actual achievable resolution/FPS depends on your hardware and build of OpenCV/FFmpeg.
VIDEO_PRESETS = [
    {
        "id": "mp4_h264",
        "label": "MP4 (H.264 / avc1) — up to 4K@60 — medium/small filesize",
        "ext": ".mp4",
        "fourcc": "avc1",
        "size_hint": "medium/small",
    },
    {
        "id": "mp4_mp4v",
        "label": "MP4 (MPEG-4 Part 2 / mp4v) — up to 1080p@60 — medium/large filesize",
        "ext": ".mp4",
        "fourcc": "mp4v",
        "size_hint": "medium/large",
    },
    {
        "id": "webm_vp9",
        "label": "WebM (VP9) — up to 4K@60 — small/medium filesize",
        "ext": ".webm",
        "fourcc": "VP90",
        "size_hint": "small/medium",
    },
    {
        "id": "avi_xvid",
        "label": "AVI (XVID / MPEG-4 Part 2) — up to 1080p@60 — large/medium filesize",
        "ext": ".avi",
        "fourcc": "XVID",
        "size_hint": "large/medium",
    },
    {
        "id": "mov_h264",
        "label": "MOV (H.264 / avc1) — up to 4K@60 — medium/small filesize",
        "ext": ".mov",
        "fourcc": "avc1",
        "size_hint": "medium/small",
    },
    {
        "id": "mkv_h264",
        "label": "MKV (H.264 / avc1) — up to 4K@60 — medium/small filesize",
        "ext": ".mkv",
        "fourcc": "avc1",
        "size_hint": "medium/small",
    },
]

PRESETS_BY_ID = {p["id"]: p for p in VIDEO_PRESETS}

def default_preset_id() -> str:
    """Return the default video preset ID.

    We choose 'mp4_mp4v' because it is broadly compatible across platforms
    without requiring extra codecs.
    """
    return "mp4_mp4v"

def ensure_dir(path: str):
    """Create a directory if it does not exist already.

    This is used for the general Output Root and the date-stamped trial folders.
    It's safe to call repeatedly due to `exist_ok=True`.
    """
    os.makedirs(path, exist_ok=True)

def now_stamp() -> str:
    """Return a filesystem-safe timestamp string (YYYY-MM-DD_HH-MM-SS).

    This timestamp is used in video filenames and CSV logs to keep trials organized.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def day_folder(root: str) -> str:
    """Return a path to today's date-stamped folder under `root`, creating it if needed.

    We store all outputs per-day to keep the output directory clean and to simplify bookkeeping.
    Example: <root>/20250819/
    """
    d = datetime.now().strftime("%Y%m%d")
    p = os.path.join(root, d)
    ensure_dir(p)
    return p

def wait_s(sec: float):
    """Sleep for `sec` seconds, using PsychoPy's timing if available.

    This ensures stimulus timing behaves the same regardless of whether PsychoPy is installed.
    """
    if PSYCHOPY:
        core.wait(sec)
    else:
        time.sleep(sec)

def get_screen_geometries():
    """Return a list of tuples describing each screen: (x, y, width, height).

    We use this to position the OpenCV stimulus window onto a specific monitor.
    """
    screens = QtGui.QGuiApplication.screens()
    geoms = []
    for s in screens:
        g = s.geometry()
        geoms.append((g.x(), g.y(), g.width(), g.height()))
    return geoms


class Config:
    """Mutable configuration object shared across the app (GUI <-> runtime).

    Each field maps directly to a GUI control. This makes it simple to apply
    and persist user choices during an experiment run.
    """
    def __init__(self):
        """Initialize default configuration values used by GUI and core logic.

        General:
          - simulation_mode: If True, synthesize triggers on a timer.
          - sim_trigger_interval: Seconds between simulated triggers.

        Recording & Output:
          - output_root: Top-level folder where trial folders are created.
          - video_preset_id/fourcc: What format/codec to use when writing videos.
          - record_duration_s: How long each camera records per trigger.

        Stimulus:
          - stim_duration_s: How long the looming dot runs.
          - stim_r0_px / stim_r1_px: Start/end radius of the dot (in pixels).
          - stim_bg_grey: Background shade (0=black, 1=white).
          - stim_delay_s: Delay between recording start and stimulus onset.
          - stim_screen_index: Which monitor to show the stimulus on.
          - stim_fullscreen: Whether the stimulus should cover the whole screen.

        GUI:
          - gui_screen_index: Which monitor the GUI window should appear on.

        Cameras:
          - camN_index: OpenCV device indices for each camera.
          - camN_target_fps: Intended frames-per-second for recording.
        """
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
        self.stim_delay_s = 0.0  # delay between recording start and stimulus
        self.stim_screen_index = 0  # which monitor for the stimulus
        self.stim_fullscreen = False  # stimulus fullscreen/windowed

        # GUI placement
        self.gui_screen_index = 0  # which monitor shows the GUI

        # Cameras
        self.cam0_index = 0
        self.cam1_index = 1
        self.cam0_target_fps = 60
        self.cam1_target_fps = 60


# =========================
# Hardware Bridge (Elegoo UNO R3 via CH340)
# =========================
class HardwareBridge:
    """Small adapter around a USB serial device that provides trigger input and marker output.

    When simulation is ON, we simply synthesize triggers every `sim_trigger_interval` seconds.
    When simulation is OFF, we try to auto-detect an Elegoo/UNO on a CH340 serial port and
    listen for lines containing 'T', which signals a trigger.

    We also provide small helper methods to send text commands to the microcontroller
    to mark START/STIM/END boundaries or control lights.
    """
    def __init__(self, cfg: Config, port: str = None, baud: int = 115200):
        """Create a hardware bridge.

        Args:
            cfg: Global configuration (we read simulation flag and interval).
            port: Optional explicit serial port (e.g., 'COM3' or '/dev/ttyUSB0').
            baud: Baud rate for the serial connection (defaults to 115200).
        """
        self.cfg = cfg
        self.simulated = cfg.simulation_mode
        self._last_sim = time.time()
        self.ser = None
        self.port = port
        self.baud = baud

        # Try to open a real serial port if we are not simulating
        if not self.simulated:
            try:
                import serial, serial.tools.list_ports
                if not self.port:
                    self.port = self._autodetect_port()
                if self.port:
                    try:
                        self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
                        wait_s(1.5)  # allow the microcontroller to reset after opening serial
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
        # First pass: look for specific USB VID/PID known for CH340
        for p in serial.tools.list_ports.comports():
            vid = f"{p.vid:04X}" if p.vid is not None else None
            pid = f"{p.pid:04X}" if p.pid is not None else None
            if vid == "1A86" and pid == "7523":  # CH340/CH34x
                return p.device
        # Second pass: fuzzy match by human-readable description
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if "ch340" in desc or "uno" in desc or "elegoo" in desc:
                return p.device
        return None

    def check_trigger(self) -> bool:
        """Return True exactly when a trigger occurs.

        Simulation mode:
          - Return True when the simulated interval elapses.

        Hardware mode:
          - Non-blocking read from serial; if a full line equals 'T', return True.
        """
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
        """Send one line of text to the device, or log it if simulating.

        This method centralizes error handling so that our marker/light helpers are tiny.
        """
        if self.simulated or not self.ser:
            print(f"[HardwareBridge] (Sim) SEND: {text}")
            return
        try:
            self.ser.write((text.strip() + "\n").encode("utf-8", errors="ignore"))
        except Exception as e:
            print(f"[HardwareBridge] Write error: {e}")

    def mark_start(self):
        """Tell the device we are starting a trial (useful for syncing and lights)."""
        self._send_line("START")

    def mark_stim(self):
        """Tell the device the stimulus just started (timestamp alignment marker)."""
        self._send_line("STIM")

    def mark_end(self):
        """Tell the device the trial finished (end marker)."""
        self._send_line("END")

    def pulse_ms(self, ms: int = 20):
        """Ask the device to produce a TTL pulse for `ms` milliseconds."""
        self._send_line(f"PULSE {int(ms)}")

    def light_on(self):
        """Turn on lights via the device (or log in simulation)."""
        self._send_line("LIGHT ON")

    def light_off(self):
        """Turn off lights via the device (or log in simulation)."""
        self._send_line("LIGHT OFF")

    # Backward-compat with legacy naming
    def activate_lights(self):
        """Legacy alias for `light_on()` to avoid breaking old call sites."""
        self.light_on()

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
    """Thin wrapper around OpenCV VideoCapture that adds:

    - Live preview frames for the GUI
    - Easy switching of device index
    - A simple file writer with target FPS
    - Basic FPS diagnostics (driver-reported + preview-measured)
    """
    def __init__(self, index: int, name: str, target_fps: int = 60):
        """Bind to a camera index and set an intended recording FPS.

        Args:
            index: OpenCV device index (0, 1, ...).
            name: Friendly label for logs (e.g., "cam0").
            target_fps: Intended FPS for recordings (actual FPS may vary by hardware).
        """
        self.name = name
        self.target_fps = float(target_fps)
        self._preview_times = deque(maxlen=30)  # rolling timestamps for preview FPS
        self._last_preview_frame = None
        self.lock = threading.Lock()            # guards self.cap during capture/record
        self.cap = None
        self.synthetic = False                  # if True, we render synthetic frames
        self.set_index(index)

    def _open(self, index: int):
        """Try to open the physical device at `index` using a few backends.

        Returns:
            (cap, False) if a real device opened successfully
            (None, True) if we will use a synthetic camera
        """
        backends = [cv2.CAP_ANY]
        if os.name == "nt":
            # On Windows, DirectShow and Media Foundation are both worth trying explicitly.
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
        """Release the camera device if it is currently open."""
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None

    def set_index(self, index: int):
        """Switch to a new OpenCV device index, falling back to a synthetic camera if needed."""
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
        """Set the intended recording FPS and hint that value to the capture driver."""
        self.target_fps = float(fps)
        with self.lock:
            if self.cap:
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                except Exception:
                    pass

    def reported_fps(self) -> float:
        """Return the driver-reported FPS from CAP_PROP_FPS (may be 0 or unreliable)."""
        with self.lock:
            if self.cap:
                v = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                return v if v > 0 else 0.0
        return 0.0

    def measured_preview_fps(self) -> float:
        """Compute GUI preview FPS from recent frame timestamps."""
        if len(self._preview_times) < 2:
            return 0.0
        dt = self._preview_times[-1] - self._preview_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._preview_times) - 1) / dt

    def grab_preview(self, w=320, h=240, overlay_index=True):
        """Return an RGB frame sized for the GUI preview area.

        This method never blocks for long: if the camera read fails,
        we return a simple "drop" frame so the GUI remains responsive.
        """
        with self.lock:
            if self.synthetic:
                # Simple animation to prove the UI is refreshing
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
        """Create and return a cv2.VideoWriter using the given FOURCC and current target FPS."""
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        return cv2.VideoWriter(path, fourcc, float(self.target_fps), size)

    def _frame_size(self):
        """Return the current capture frame size (w, h), falling back to 640x480."""
        with self.lock:
            if self.cap:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                return (w, h)
        return (640, 480)

    def record_clip(self, path: str, duration_s: float, fourcc_str: str = "mp4v"):
        """Record a video to `path` for `duration_s` seconds at `target_fps`.

        We attempt to open the writer using the selected FOURCC. If that fails,
        we fall back to 'mp4v' so a recording is still produced.

        Returns:
            The path actually written (may be changed to .mp4 if we fell back), or None on failure.
        """
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
                    # Render a simple white frame with a moving dot and timestamp
                    frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
                    cv2.putText(frame, f"{self.name} {now_stamp()}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    cx = int((frame_index * 7) % size[0])
                    cy = int(size[1] / 2)
                    cv2.circle(frame, (cx, cy), 20, (0, 0, 0), 2)
                else:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        # If a frame drops, write a placeholder so the writer stays open
                        frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
                        cv2.putText(frame, f"{self.name} [drop] {now_stamp()}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        ok_any = True
            out.write(frame)
            frame_index += 1
            # Throttle loop to target FPS in a cross-platform friendly way
            time.sleep(max(0.0, 1.0 / float(self.target_fps)))
        out.release()
        if not ok_any and not self.synthetic:
            print(f"[Camera {self.name}] Warning: no frames captured from camera index {self.index}.")
        print(f"[Camera {self.name}] Recording complete.")
        return path


# =========================
# Looming Stimulus (white background, black dot) — persistent window
# =========================
class LoomingStim:
    """Display a growing black dot on a white (or chosen grey) background.

    Implementation details:
    - If PsychoPy is available, we use it for accurate timing and full-screen support.
    - If PsychoPy isn't available, we create a persistent OpenCV named window
      and paint frames manually. The window stays open across trials so you
      can drag it to another monitor one time and keep it there.
    """
    def __init__(self, cfg: Config):
        """Create a stimulus manager bound to the shared configuration."""
        self.cfg = cfg
        # PsychoPy window cache (so we don't re-create it every trial)
        self._pp_win = None
        self._pp_cfg = None  # (screen_idx, fullscr)
        # OpenCV window cache
        self._cv_window_name = "Looming Stimulus"
        self._cv_open = False
        self._cv_size = (800, 600)  # default size for windowed mode

    # ---------- internal helpers ----------
    def _ensure_psychopy_window(self, screen_idx: int, fullscr: bool, bg_grey: float):
        """Create or update the PsychoPy window so it matches screen/fullscreen settings."""
        need_new = False
        if self._pp_win is None:
            need_new = True
        elif self._pp_cfg != (screen_idx, fullscr):
            # Screen index or full-screen state changed → rebuild window
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
                    # Windowed & resizable → easy to drag anywhere
                    self._pp_win = visual.Window(size=self._cv_size, color=[bg_grey]*3,
                                                 units='pix', fullscr=False, screen=screen_idx, allowGUI=True)
                self._pp_cfg = (screen_idx, fullscr)
            except Exception as e:
                print(f"[Stim] PsychoPy window create error: {e}")
                self._pp_win = None

        # Keep background color up-to-date (if the user changed it)
        if self._pp_win is not None:
            try:
                self._pp_win.color = [bg_grey]*3
            except Exception:
                pass

    def _ensure_opencv_window(self, screen_idx: int, bg_grey: float):
        """Create the OpenCV window if needed and move it to the chosen monitor."""
        try:
            if not self._cv_open:
                cv2.namedWindow(self._cv_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._cv_window_name, self._cv_size[0], self._cv_size[1])
                self._cv_open = True
            # Position the window near the top-left of the selected screen
            geoms = get_screen_geometries()
            if 0 <= screen_idx < len(geoms):
                x, y, w, h = geoms[screen_idx]
                cv2.moveWindow(self._cv_window_name, x + 50, y + 50)
        except Exception as e:
            print(f"[Stim] OpenCV window create/move error: {e}")
            self._cv_open = False

    # ---------- public API ----------
    def run(self, duration_s: float, r0: int, r1: int, bg_grey: float,
            screen_idx: int, fullscreen: bool):
        """Show the looming stimulus for `duration_s` seconds.

        Args:
            duration_s: Length of the stimulus in seconds.
            r0: Starting radius in pixels.
            r1: Ending radius in pixels.
            bg_grey: Background shade (0=black, 1=white).
            screen_idx: Which monitor to present on.
            fullscreen: If True, cover the entire selected screen; otherwise open a window.
        """
        print("[Stim] Looming start.")

        if PSYCHOPY:
            try:
                # Use a persistent, correctly-configured PsychoPy window
                self._ensure_psychopy_window(screen_idx, fullscreen, bg_grey)
                if self._pp_win is not None:
                    dot = visual.Circle(self._pp_win, radius=r0, fillColor='black', lineColor='black')
                    t0 = time.time()
                    while True:
                        t = time.time() - t0
                        if t >= duration_s:
                            break
                        # Linearly scale radius over time (r0 -> r1)
                        r = r0 + (r1 - r0) * (t / duration_s)
                        dot.radius = r
                        dot.draw()
                        self._pp_win.flip()
                    print("[Stim] Looming done (PsychoPy).")
                    return
            except Exception as e:
                print(f"[Stim] PsychoPy error: {e} -> OpenCV fallback.")

        # OpenCV fallback: persistent namedWindow positioned to selected screen
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
                # ESC allows an early exit without crashing the window
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            # We intentionally do NOT destroy the window, so it persists between trials.
            print("[Stim] Looming done (OpenCV).")
        except Exception as e:
            print(f"[Stim] OpenCV display unavailable ({e}). Logging-only fallback.")
            wait_s(duration_s)
            print("[Stim] Looming done (no display).")

    def close(self):
        """Tear down the persistent windows (PsychoPy and OpenCV) when quitting."""
        # PsychoPy window cleanup
        try:
            if self._pp_win is not None:
                self._pp_win.close()
        except Exception:
            pass
        self._pp_win = None
        self._pp_cfg = None

        # OpenCV window cleanup
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
    """Glue object that coordinates cameras, lights, stimulus, and logging for each trigger.

    Usage:
      - `run_trial()` is called when a trigger occurs.
      - It records both cameras in parallel, waits the configured delay,
        runs the stimulus, and writes a row to the CSV log.
    """
    def __init__(self, cfg: Config, hardware: HardwareBridge, cam0: CameraRecorder, cam1: CameraRecorder, logger_path: str):
        """Create a TrialRunner and open (or create) the CSV log file."""
        self.cfg = cfg
        self.hardware = hardware
        self.cam0 = cam0
        self.cam1 = cam1
        self.stim = LoomingStim(cfg)
        self.trial_idx = 0

        # CSV logger set-up (append mode; create header if file is new)
        new_file = not os.path.exists(logger_path)
        ensure_dir(os.path.dirname(logger_path))
        self.log_file = open(logger_path, "a", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_file)
        if new_file:
            self.log_writer.writerow([
                "trial", "timestamp",
                "cam0_path", "cam1_path",
                "record_duration_s", "stim_duration_s", "stim_delay_s",
                "stim_screen_index", "stim_fullscreen",
                "cam0_target_fps", "cam1_target_fps",
                "video_preset_id", "fourcc"
            ])

    def close(self):
        """Close the CSV log and stimulus windows at application shutdown."""
        try:
            self.log_file.close()
        except Exception:
            pass
        try:
            self.stim.close()
        except Exception:
            pass

    def _preset(self):
        """Return the active video preset dict (container/FOURCC/labels)."""
        return PRESETS_BY_ID.get(self.cfg.video_preset_id, PRESETS_BY_ID[default_preset_id()])

    def run_trial(self):
        """Perform one full trial from start to end.

        Steps:
          1) Increment trial index and compute output paths.
          2) Turn on lights and send START marker.
          3) Launch camera recordings in two parallel threads.
          4) Wait the configured delay (`stim_delay_s`), then send STIM marker.
          5) Run the looming stimulus on the selected monitor.
          6) Join camera threads, send END marker, and turn lights off.
          7) Write a CSV row including file paths and key settings.

        All significant moments also print to the console for quick diagnostics.
        """
        self.trial_idx += 1
        ts = now_stamp()
        out_dir = day_folder(self.cfg.output_root)

        preset = self._preset()
        ext = preset["ext"]
        fourcc = preset["fourcc"]

        # Requested output paths. Note: writer may fall back to .mp4 if FOURCC is unsupported.
        cam0_path_req = os.path.join(out_dir, f"{ts}_trial{self.trial_idx:04d}_cam0{ext}")
        cam1_path_req = os.path.join(out_dir, f"{ts}_trial{self.trial_idx:04d}_cam1{ext}")

        print(f"[Trial {self.trial_idx}] START {ts}")
        self.hardware.activate_lights()
        self.hardware.mark_start()

        # Record both cameras in parallel, capturing the actual written paths from each
        results = {"cam0": None, "cam1": None}
        def rec0():
            results["cam0"] = self.cam0.record_clip(cam0_path_req, self.cfg.record_duration_s, fourcc)
        def rec1():
            results["cam1"] = self.cam1.record_clip(cam1_path_req, self.cfg.record_duration_s, fourcc)

        t0 = threading.Thread(target=rec0, daemon=True)
        t1 = threading.Thread(target=rec1, daemon=True)
        t0.start(); t1.start()

        # Configurable delay between recording start and stimulus onset
        delay = max(0.0, float(self.cfg.stim_delay_s))
        if delay > 0:
            print(f"[Trial {self.trial_idx}] Waiting {delay:.3f}s before stimulus...")
            wait_s(delay)

        # Mark stimulus onset right before we actually draw the first frame
        self.hardware.mark_stim()

        # Show looming stimulus while cameras are still recording
        self.stim.run(self.cfg.stim_duration_s,
                      self.cfg.stim_r0_px, self.cfg.stim_r1_px,
                      self.cfg.stim_bg_grey,
                      screen_idx=self.cfg.stim_screen_index,
                      fullscreen=self.cfg.stim_fullscreen)

        # Close out the trial: wait for cameras, mark end, turn lights off
        t0.join(); t1.join()
        self.hardware.mark_end()
        self.hardware.light_off()

        # Final written paths (may differ from requested if we fell back)
        cam0_path_actual = results["cam0"] or cam0_path_req
        cam1_path_actual = results["cam1"] or cam1_path_req

        # Append a log row with key parameters for reproducibility
        self.log_writer.writerow([
            self.trial_idx, ts,
            cam0_path_actual, cam1_path_actual,
            self.cfg.record_duration_s, self.cfg.stim_duration_s, self.cfg.stim_delay_s,
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
    """Single-window GUI exposing all settings, camera previews, windows/screens and loop controls.

    Signals:
      - start_experiment(): user clicked "Start"
      - stop_experiment():  user clicked "Stop"
      - apply_settings():   user clicked "Apply Settings"
      - manual_trigger():   user clicked "Trigger Once (Manual)"
    """
    start_experiment = QtCore.pyqtSignal()
    stop_experiment  = QtCore.pyqtSignal()
    apply_settings   = QtCore.pyqtSignal()
    manual_trigger   = QtCore.pyqtSignal()

    def __init__(self, cfg: Config, cam0: CameraRecorder, cam1: CameraRecorder):
        """Build the entire GUI and connect signal handlers.

        The layout is designed for a 1920×1080 screen. The top area contains global and
        stimulus settings; the bottom half contains two camera panels with live previews.
        """
        super().__init__()
        self.cfg  = cfg
        self.cam0 = cam0
        self.cam1 = cam1

        self.setWindowTitle("FlyPy — Trigger->Outputs (All-in-One GUI)")
        # Fixed-size 1920x1080 canvas as requested
        self.setFixedSize(1920, 1080)

        root = QtWidgets.QVBoxLayout(self)

        # --- Controls row: Start / Stop / Trigger Once / Apply ---
        controls = QtWidgets.QHBoxLayout()
        self.bt_start   = QtWidgets.QPushButton("Start")
        self.bt_stop    = QtWidgets.QPushButton("Stop")
        self.bt_trigger = QtWidgets.QPushButton("Trigger Once (Manual)")  # Manual single trial for quick testing
        self.bt_apply   = QtWidgets.QPushButton("Apply Settings")
        self.bt_start.setToolTip("Begin watching for triggers. On each trigger: lights on, record both cameras, show looming stimulus, log trial.")
        self.bt_stop.setToolTip("Stop watching for triggers. Safe to close the app after this.")
        self.bt_trigger.setToolTip("Run a single trial immediately (for testing without hardware trigger).")
        self.bt_apply.setToolTip("Apply changes from the panels below without restarting the app.")
        controls.addWidget(self.bt_start)
        controls.addWidget(self.bt_stop)
        controls.addWidget(self.bt_trigger)
        controls.addStretch(1)
        controls.addWidget(self.bt_apply)
        root.addLayout(controls)

        # --- Panels container ---
        panels = QtWidgets.QGridLayout()
        panels.setColumnStretch(0, 1)
        panels.setColumnStretch(1, 1)
        root.addLayout(panels)

        # General panel
        gen = QtWidgets.QGroupBox("General settings")
        gen.setToolTip("Top-level behavior and where files are saved.")
        gl = QtWidgets.QFormLayout(gen)
        gl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)

        self.lbl_sim = QtWidgets.QLabel("Simulation mode: OFF (hardware triggers active)")
        self.lbl_sim.setWordWrap(True)
        self.lbl_sim.setToolTip("If ON, triggers are generated on a timer instead of coming from hardware.")
        gl.addRow(self.lbl_sim)

        self.sb_sim_interval = QtWidgets.QDoubleSpinBox()
        self.sb_sim_interval.setRange(0.5, 3600.0); self.sb_sim_interval.setDecimals(2)
        self.sb_sim_interval.setValue(self.cfg.sim_trigger_interval)
        self.sb_sim_interval.setToolTip("Time between simulated triggers (in seconds). Use ≥2 s on laptops to keep CPU low.")
        _lbl_simint = QtWidgets.QLabel("Interval between simulated triggers (seconds):"); _lbl_simint.setWordWrap(True)
        gl.addRow(_lbl_simint, self.sb_sim_interval)

        self.le_root = QtWidgets.QLineEdit(self.cfg.output_root)
        self.le_root.setToolTip("Folder where all date-stamped trial folders and videos will be saved.")
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.setToolTip("Choose a different output folder.")
        rhl = QtWidgets.QHBoxLayout()
        rhl.addWidget(self.le_root); rhl.addWidget(self.btn_browse)
        _lbl_root = QtWidgets.QLabel("Output folder for all trials:"); _lbl_root.setWordWrap(True)
        gl.addRow(_lbl_root, rhl)

        # Video format/codec dropdown with captions
        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.setToolTip("Select the container and codec for saved videos. Captions show typical max resolution/FPS and approximate file size.")
        self._preset_id_by_index = {}
        current_index = 0
        for i, p in enumerate(VIDEO_PRESETS):
            self.cb_format.addItem(p["label"])
            self.cb_format.setItemData(i, p["id"])
            self._preset_id_by_index[i] = p["id"]
            if p["id"] == self.cfg.video_preset_id:
                current_index = i
        self.cb_format.setCurrentIndex(current_index)
        _lbl_fmt = QtWidgets.QLabel("Video file format / codec:"); _lbl_fmt.setWordWrap(True)
        gl.addRow(_lbl_fmt, self.cb_format)

        self.sb_rec_dur = QtWidgets.QDoubleSpinBox()
        self.sb_rec_dur.setRange(0.2, 600.0); self.sb_rec_dur.setDecimals(2); self.sb_rec_dur.setValue(self.cfg.record_duration_s)
        self.sb_rec_dur.setToolTip("How long to record for each trigger (seconds). Longer clips = larger files.")
        _lbl_recdur = QtWidgets.QLabel("Recording duration per trigger (seconds):"); _lbl_recdur.setWordWrap(True)
        gl.addRow(_lbl_recdur, self.sb_rec_dur)

        panels.addWidget(gen, 0, 0)

        # Stimulus panel
        stim = QtWidgets.QGroupBox("Looming stimulus (growing black dot on white background)")
        stim.setToolTip("Duration and size change of the looming dot shown after each trigger.")
        sl = QtWidgets.QFormLayout(stim)
        sl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)

        self.sb_stim_dur = QtWidgets.QDoubleSpinBox()
        self.sb_stim_dur.setRange(0.1, 30.0); self.sb_stim_dur.setDecimals(2); self.sb_stim_dur.setValue(self.cfg.stim_duration_s)
        self.sb_stim_dur.setToolTip("How long the looming dot is shown (seconds). Cameras still record for the full duration above.")
        _lbl_sd = QtWidgets.QLabel("Stimulus display duration (seconds):"); _lbl_sd.setWordWrap(True)
        sl.addRow(_lbl_sd, self.sb_stim_dur)

        self.sb_r0 = QtWidgets.QSpinBox(); self.sb_r0.setRange(1, 2000); self.sb_r0.setValue(self.cfg.stim_r0_px)
        self.sb_r0.setToolTip("Starting dot radius in pixels. Smaller values (8–20 px) start more subtly.")
        _lbl_r0 = QtWidgets.QLabel("Starting dot radius (pixels):"); _lbl_r0.setWordWrap(True)
        sl.addRow(_lbl_r0, self.sb_r0)

        self.sb_r1 = QtWidgets.QSpinBox(); self.sb_r1.setRange(1, 4000); self.sb_r1.setValue(self.cfg.stim_r1_px)
        self.sb_r1.setToolTip("Final dot radius in pixels at the end of the stimulus.")
        _lbl_r1 = QtWidgets.QLabel("Final dot radius (pixels):"); _lbl_r1.setWordWrap(True)
        sl.addRow(_lbl_r1, self.sb_r1)

        self.sb_bg = QtWidgets.QDoubleSpinBox(); self.sb_bg.setRange(0.0, 1.0); self.sb_bg.setSingleStep(0.05); self.sb_bg.setValue(self.cfg.stim_bg_grey)
        self.sb_bg.setToolTip("Background shade for PsychoPy display (0 = black, 1 = white). OpenCV fallback uses this shade as well.")
        _lbl_bg = QtWidgets.QLabel("Stimulus background shade (0=black, 1=white):"); _lbl_bg.setWordWrap(True)
        sl.addRow(_lbl_bg, self.sb_bg)

        # Stimulus delay
        self.sb_stim_delay = QtWidgets.QDoubleSpinBox()
        self.sb_stim_delay.setRange(0.0, 10.0); self.sb_stim_delay.setDecimals(3)
        self.sb_stim_delay.setValue(self.cfg.stim_delay_s)
        self.sb_stim_delay.setToolTip("Delay (seconds) after recording starts before the stimulus begins.")
        _lbl_delay = QtWidgets.QLabel("Delay before stimulus after recording start (seconds):"); _lbl_delay.setWordWrap(True)
        sl.addRow(_lbl_delay, self.sb_stim_delay)

        panels.addWidget(stim, 0, 1)

        # Display / Screens panel
        disp = QtWidgets.QGroupBox("Display & Windows")
        disp.setToolTip("Choose which monitor shows the Stimulus window and the GUI. Stimulus can be fullscreen or windowed.")
        dl = QtWidgets.QFormLayout(disp)
        dl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)

        # Build screen choices
        screens = QtGui.QGuiApplication.screens()
        def screen_label(i, s):
            g = s.geometry()
            return f"Screen {i} — {g.width()}×{g.height()} @ ({g.x()},{g.y()})"

        self.cb_stim_screen = QtWidgets.QComboBox()
        self.cb_gui_screen = QtWidgets.QComboBox()
        for i, s in enumerate(screens):
            lbl = screen_label(i, s)
            self.cb_stim_screen.addItem(lbl)
            self.cb_gui_screen.addItem(lbl)
        self.cb_stim_screen.setCurrentIndex(self.cfg.stim_screen_index if self.cfg.stim_screen_index < len(screens) else 0)
        self.cb_gui_screen.setCurrentIndex(self.cfg.gui_screen_index if self.cfg.gui_screen_index < len(screens) else 0)
        self.cb_stim_screen.setToolTip("Which monitor should the looming stimulus appear on.")
        self.cb_gui_screen.setToolTip("Which monitor should host this GUI window.")

        self.cb_stim_fullscr = QtWidgets.QCheckBox("Stimulus fullscreen on selected screen")
        self.cb_stim_fullscr.setChecked(bool(self.cfg.stim_fullscreen))
        self.cb_stim_fullscr.setToolTip("If checked, the stimulus uses fullscreen on the chosen monitor; otherwise it opens a resizable window you can drag anywhere.")

        dl.addRow(QtWidgets.QLabel("Stimulus display screen:"), self.cb_stim_screen)
        dl.addRow(QtWidgets.QLabel("GUI display screen:"), self.cb_gui_screen)
        dl.addRow(self.cb_stim_fullscr)

        panels.addWidget(disp, 1, 0, 1, 2)

        # Camera panels
        self.cam_groups = []
        for idx, cam, target_default in [(0, self.cam0, self.cfg.cam0_target_fps),
                                         (1, self.cam1, self.cfg.cam1_target_fps)]:
            gb = QtWidgets.QGroupBox(f"Camera {idx} — preview & frame rate")
            gb.setToolTip("Live preview shows which camera you're using. Frame-rate panel shows driver-reported, preview-measured, and target recording FPS.")
            fl = QtWidgets.QGridLayout(gb)

            # Left: preview (larger for 1920x1080 canvas)
            preview = QtWidgets.QLabel()
            preview.setFixedSize(640, 480)
            preview.setFrameShape(QtWidgets.QFrame.Box)
            preview.setAlignment(QtCore.Qt.AlignCenter)
            preview.setToolTip("Live preview (pauses while recording). The overlay at the bottom shows the OpenCV device index in use.")
            fl.addWidget(preview, 0, 0, 5, 1)

            # Right: settings & info (wrapped labels)

            # Camera index selection
            spin_index = QtWidgets.QSpinBox()
            spin_index.setRange(0, 15)
            spin_index.setValue(getattr(cam, "index", 0))
            spin_index.setToolTip("Which camera to use (OpenCV device index). Change if the preview shows the wrong device; click Apply to rebind.")
            _lbl_idx = QtWidgets.QLabel("Which camera to use (OpenCV device index):"); _lbl_idx.setWordWrap(True)
            fl.addWidget(_lbl_idx, 0, 1)
            fl.addWidget(spin_index, 0, 2)

            # FPS selection — typable up to 10,000
            spin_fps = QtWidgets.QSpinBox()
            spin_fps.setRange(1, 10000)
            spin_fps.setValue(int(target_default))
            spin_fps.setAccelerated(True)
            spin_fps.setKeyboardTracking(True)
            spin_fps.setToolTip("Target recording frame rate (FPS). Type directly or use arrows. Range: 1–10,000. Actual FPS may be limited by camera/driver.")
            _lbl_tf = QtWidgets.QLabel("Target recording frame rate (fps):"); _lbl_tf.setWordWrap(True)
            fl.addWidget(_lbl_tf, 1, 1)
            fl.addWidget(spin_fps, 1, 2)

            lbl_rep = QtWidgets.QLabel("Driver-reported frame rate (may be 0 on some webcams): —")
            lbl_rep.setWordWrap(True)
            lbl_rep.setToolTip("Driver-reported FPS (CAP_PROP_FPS). Some webcams return 0 or an inaccurate value here.")
            fl.addWidget(lbl_rep, 2, 1, 1, 2)

            lbl_mea = QtWidgets.QLabel("Measured preview frame rate (GUI): —")
            lbl_mea.setWordWrap(True)
            lbl_mea.setToolTip("Measured frame rate of the GUI preview (not the recorded file).")
            fl.addWidget(lbl_mea, 3, 1, 1, 2)

            lbl_tar = QtWidgets.QLabel(f"Recording target frame rate (intended): {int(target_default)}")
            lbl_tar.setWordWrap(True)
            lbl_tar.setToolTip("Intended recording FPS used by the video writer.")
            fl.addWidget(lbl_tar, 4, 1, 1, 2)

            # Stretch so preview gets more space
            fl.setColumnStretch(0, 3)  # preview
            fl.setColumnStretch(1, 1)  # labels
            fl.setColumnStretch(2, 1)  # editors

            self.cam_groups.append({
                "group": gb,
                "preview": preview,
                "spin_index": spin_index,
                "spin_fps": spin_fps,
                "lbl_rep": lbl_rep,
                "lbl_mea": lbl_mea,
                "lbl_tar": lbl_tar,
                "cam": cam
            })

            panels.addWidget(gb, 2, idx)

        # Status
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setToolTip("Overall state: Idle / Watching for triggers / Trial running / Errors.")
        root.addWidget(self.lbl_status)

        # Signals
        self.bt_start.clicked.connect(self.start_experiment.emit)
        self.bt_stop.clicked.connect(self.stop_experiment.emit)
        self.bt_trigger.clicked.connect(self.manual_trigger.emit)
        self.bt_apply.clicked.connect(self.apply_settings.emit)
        self.btn_browse.clicked.connect(self._pick_folder)

        # Initial text
        self._refresh_general_labels()

        # Preview timer flag (controlled by Main)
        self.preview_paused = False

    def _pick_folder(self):
        """Open a directory picker and store the user's selection in the Output Root field."""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Root", self.le_root.text() or ".")
        if path:
            self.le_root.setText(path)

    def _refresh_general_labels(self):
        """Reflect the current simulation mode in the big label at the top of General settings."""
        self.lbl_sim.setText(
            f"Simulation mode: {'ON (timer-based triggers)' if self.cfg.simulation_mode else 'OFF (hardware triggers active)'}"
        )

    def update_cam_fps_labels(self):
        """Update the three FPS labels (driver-reported, GUI-measured, target) for each camera."""
        for g in self.cam_groups:
            cam: CameraRecorder = g["cam"]
            rep = cam.reported_fps()
            mea = cam.measured_preview_fps()
            g["lbl_rep"].setText(
                f"Driver-reported frame rate (may be 0 on some webcams): {rep:.1f}"
                if rep > 0 else
                "Driver-reported frame rate (may be 0 on some webcams): (unknown)"
            )
            g["lbl_mea"].setText(f"Measured preview frame rate (GUI): {mea:.1f}")
            g["lbl_tar"].setText(f"Recording target frame rate (intended): {int(cam.target_fps)}")

    def set_preview_image(self, cam_idx: int, img_rgb: np.ndarray):
        """Render a numpy RGB image into the preview QLabel for camera `cam_idx`."""
        g = self.cam_groups[cam_idx]
        h, w, _ = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, w*3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            g["preview"].size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        g["preview"].setPixmap(pix)


# =========================
# Main Application
# =========================
class MainApp(QtWidgets.QApplication):
    """Top-level Qt application that wires together Config, Hardware, Cameras, GUI, and the trigger loop.

    The MainApp owns the long-lived objects and controls the background loop that watches for triggers.
    """
    def __init__(self, argv):
        """Construct subsystems, prompt for Simulation Mode, and display the GUI."""
        super().__init__(argv)
        # Config + Simulation prompt
        self.cfg = Config()
        reply = QtWidgets.QMessageBox.question(
            None, "Simulation Mode",
            "Run in SIMULATION MODE?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
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
        self.gui.show()

        # Place GUI on selected screen initially (centered on that monitor)
        self.position_gui(self.cfg.gui_screen_index)

        # Preview timer (updates when idle)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(500)  # ms
        self.preview_timer.timeout.connect(self.update_previews)
        self.preview_timer.start()

        # State flags
        self.running = False   # whether the trigger-watching loop is running
        self.in_trial = False  # whether a trial is currently executing
        self.thread = None     # background loop thread

        # Cleanup hooks to ensure cameras, serial, and files close properly
        self.aboutToQuit.connect(self.cleanup)
        atexit.register(self.cleanup)

        # Initialize labels after sim prompt
        self.gui._refresh_general_labels()

    def position_gui(self, screen_idx: int):
        """Move and center the GUI window on the chosen screen."""
        screens = QtGui.QGuiApplication.screens()
        if not screens:
            return
        if screen_idx < 0 or screen_idx >= len(screens):
            screen_idx = 0
        g = screens[screen_idx].availableGeometry()
        w = self.gui.frameGeometry()
        w.moveCenter(g.center())
        self.gui.move(w.topLeft())

    def apply_from_gui(self):
        """Read current widget values and write them into Config/Cameras.

        This is the single place where UI → runtime updates are applied.
        It keeps the rest of the code clean and predictable.
        """
        # General
        self.cfg.sim_trigger_interval = float(self.gui.sb_sim_interval.value())
        self.cfg.output_root = self.gui.le_root.text().strip() or self.cfg.output_root

        # Format/codec preset
        idx = self.gui.cb_format.currentIndex()
        preset_id = self.gui.cb_format.itemData(idx)
        if not preset_id:
            preset_id = self.gui._preset_id_by_index.get(idx, default_preset_id())
        self.cfg.video_preset_id = preset_id
        self.cfg.fourcc = PRESETS_BY_ID[self.cfg.video_preset_id]["fourcc"]

        self.cfg.record_duration_s = float(self.gui.sb_rec_dur.value())

        # Stimulus
        self.cfg.stim_duration_s = float(self.gui.sb_stim_dur.value())
        self.cfg.stim_r0_px = int(self.gui.sb_r0.value())
        self.cfg.stim_r1_px = int(self.gui.sb_r1.value())
        self.cfg.stim_bg_grey = float(self.gui.sb_bg.value())
        self.cfg.stim_delay_s = float(self.gui.sb_stim_delay.value())
        self.cfg.stim_screen_index = int(self.gui.cb_stim_screen.currentIndex())
        self.cfg.stim_fullscreen = bool(self.gui.cb_stim_fullscr.isChecked())

        # GUI screen placement
        self.cfg.gui_screen_index = int(self.gui.cb_gui_screen.currentIndex())
        self.position_gui(self.cfg.gui_screen_index)

        # Cameras (rebind as needed and update target FPS)
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

        # Mirror back into cfg for logging completeness
        self.cfg.cam0_index = getattr(self.cam0, "index", 0)
        self.cfg.cam1_index = getattr(self.cam1, "index", 1)
        self.cfg.cam0_target_fps = int(self.cam0.target_fps)
        self.cfg.cam1_target_fps = int(self.cam1.target_fps)

        ensure_dir(self.cfg.output_root)
        self.gui.lbl_status.setText("Status: Settings applied.")
        print("[MainApp] Settings applied.")

    def update_previews(self):
        """Refresh the camera previews and FPS labels when not in a trial."""
        if self.in_trial:
            # While recording, we pause preview updates to avoid starving the writers.
            self.gui.lbl_status.setText("Status: Trial running (preview paused).")
            self.gui.update_cam_fps_labels()
            return

        # Render frames at the preview labels' actual sizes for best quality/perf
        p0 = self.gui.cam_groups[0]["preview"]
        p1 = self.gui.cam_groups[1]["preview"]
        img0 = self.cam0.grab_preview(w=p0.width(), h=p0.height())
        img1 = self.cam1.grab_preview(w=p1.width(), h=p1.height())
        self.gui.set_preview_image(0, img0)
        self.gui.set_preview_image(1, img1)

        self.gui.update_cam_fps_labels()
        self.gui.lbl_status.setText("Status: Waiting / Idle.")

    def loop(self):
        """Background thread that watches for triggers and runs trials.

        The loop is simple by design: poll for triggers, run a trial when one occurs,
        and update the GUI status messages so the user knows what's happening.
        """
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
                time.sleep(0.002)  # tiny sleep keeps CPU usage sane
            except Exception as e:
                print(f"[MainApp] Loop error: {e}")
                self.gui.lbl_status.setText(f"Status: Error - {e}")
                time.sleep(0.05)
        print("[MainApp] Trigger loop stopped.")

    def trigger_once(self):
        """Run a single trial immediately (manual button). Does not affect the background loop."""
        if self.in_trial:
            return  # ignore if already mid-trial
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
        """Start the background trigger-watching loop (idempotent)."""
        if not self.running:
            self.apply_from_gui()  # ensure we use the latest settings
            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop_loop(self):
        """Stop the background loop and update the status label."""
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
            self.thread = None
        self.gui.lbl_status.setText("Status: Stopped.")

    def cleanup(self):
        """Gracefully shut down threads, close files/serial/cameras, and print a final message."""
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
    # Create and run the Qt app. This call blocks until the window is closed.
    app = MainApp(sys.argv)
    sys.exit(app.exec_())
