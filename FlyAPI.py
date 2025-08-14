"""
FlyPy — Trigger→Outputs build with All-in-One Settings GUI, camera previews, rich tooltips,
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
def ensure_dir(path: str):
    """Create directory `path` if it doesn't exist.

    Usage:
        ensure_dir("FlyPy_Output/20250813")

    GUI impact:
        None directly. Called when applying settings (to ensure the Output Root exists)
        and when writing trial files inside date-stamped subfolders.
    """
    os.makedirs(path, exist_ok=True)


def now_stamp():
    """Return a filesystem-safe timestamp string (YYYY-MM-DD_HH-MM-SS).

    Usage:
        ts = now_stamp()

    GUI impact:
        None. Used for file naming shown in logs and indirectly visible via output folder.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def day_folder(root: str):
    """Return/create today's date-stamped subfolder within `root`.

    Args:
        root: Output Root path (typically from GUI "Output folder for all trials")

    Returns:
        Absolute/relative path to `<root>/<YYYYMMDD>`.

    GUI impact:
        Reads the "Output folder for all trials" setting (via cfg) to organize trial videos.
    """
    d = datetime.now().strftime("%Y%m%d")
    p = os.path.join(root, d)
    ensure_dir(p)
    return p


def wait_s(sec: float):
    """Wait for `sec` seconds using PsychoPy's clock if available, else time.sleep.

    Args:
        sec: Seconds to wait.

    GUI impact:
        None. Timing helper used for serial reset and headless stimulus fallback duration.
    """
    if PSYCHOPY:
        core.wait(sec)
    else:
        time.sleep(sec)


class Config:
    """Holds all runtime settings; the GUI reads/writes these.

    Visible GUI names → Config fields:

      General settings
        • “Interval between simulated triggers (seconds)” → sim_trigger_interval
        • “Output folder for all trials”                  → output_root
        • “Video codec (FOURCC code)”                    → fourcc
        • “Recording duration per trigger (seconds)”     → record_duration_s
        • Simulation mode (Yes/No dialog at startup)     → simulation_mode

      Looming stimulus (growing dot)
        • “Stimulus display duration (seconds)”          → stim_duration_s
        • “Starting dot radius (pixels)”                 → stim_r0_px
        • “Final dot radius (pixels)”                    → stim_r1_px
        • “Stimulus background shade (0=black, 1=white)” → stim_bg_grey

      Camera N — preview & frame rate
        • “Which camera to use (OpenCV device index)”    → camN_index
        • “Target recording frame rate (fps)”            → camN_target_fps
        • FPS labels are informational; they don't change settings.
    """
    def __init__(self):
        """Initialize default configuration values used by GUI and core logic."""
        # General
        self.simulation_mode = False
        self.sim_trigger_interval = 5.0

        # Recording / output
        self.output_root = "FlyPy_Output"
        self.fourcc = "mp4v"
        self.record_duration_s = 3.0

        # Stimulus
        self.stim_duration_s = 1.5
        self.stim_r0_px = 8
        self.stim_r1_px = 400
        self.stim_bg_grey = 0.1

        # Cameras
        self.cam0_index = 0
        self.cam1_index = 1
        self.cam0_target_fps = 60
        self.cam1_target_fps = 60


# =========================
# Hardware Bridge
# =========================
class HardwareBridge:
    """Bridge for trigger and light commands over serial, with simulation fallback.

    GUI settings used:
      • cfg.simulation_mode  — toggles simulated triggers
      • cfg.sim_trigger_interval — interval for simulated triggers
    """
    def __init__(self, cfg: Config, port="COM3", baud=9600):
        """Attempt to open serial port; otherwise run in simulation mode.

        Args:
            cfg: Shared Config with simulation settings.
            port: Serial port for hardware trigger/light controller.
            baud: Baud rate.

        GUI impact:
            Reads cfg.simulation_mode at creation time.
            No direct writes to GUI; status printed to console.
        """
        self.cfg = cfg
        self.simulated = cfg.simulation_mode
        self._last_sim = time.time()
        self.ser = None
        if not self.simulated:
            try:
                import serial
                try:
                    self.ser = serial.Serial(port, baud, timeout=0.01)
                    wait_s(2)  # allow MCU reset
                except Exception as e:
                    print(f"[HardwareBridge] Serial open failed: {e} → simulation.")
                    self.simulated = True
            except ImportError:
                print("[HardwareBridge] pyserial missing → simulation.")
                self.simulated = True

    def check_trigger(self) -> bool:
        """Return True when a trigger is detected (simulated or serial).

        Behavior:
            • Simulated: True every cfg.sim_trigger_interval seconds.
            • Serial: reads lines; "T" indicates a trigger.

        GUI impact:
            Reads cfg.sim_trigger_interval (General settings).
            Drives the main loop state (Status label will reflect trial start).
        """
        if self.simulated:
            now = time.time()
            if now - self._last_sim >= self.cfg.sim_trigger_interval:
                self._last_sim = now
                print("[HardwareBridge] (Sim) Trigger.")
                return True
            return False
        # Real serial
        try:
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                return (line == "T")
        except Exception as e:
            print(f"[HardwareBridge] Read error: {e}")
        return False

    def activate_lights(self):
        """Activate lights (serial 'L' byte) or log in simulation.

        GUI impact:
            None directly; effect is observable as console log and (if hardware attached) lights turning on.
        """
        if not self.simulated and self.ser:
            try:
                self.ser.write(b"L")
                print("[HardwareBridge] Lights command sent.")
            except Exception as e:
                print(f"[HardwareBridge] Write error: {e}")
        else:
            print("[HardwareBridge] (Sim) Lights ON.")

    def close(self):
        """Close serial port if open.

        GUI impact:
            None; invoked by application cleanup.
        """
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
    """OpenCV VideoCapture wrapper with live preview, FPS info, and recording.

    GUI settings used/written:
      • “Which camera to use (OpenCV device index)” (SpinBox) → set_index()
      • “Target recording frame rate (fps)” (SpinBox)         → set_target_fps()
      • Preview panel fetches frames via grab_preview()
      • Labels show reported_fps(), measured_preview_fps(), target_fps
    """
    def __init__(self, index: int, name: str, target_fps: int = 60):
        """Bind a camera by index and set a target recording FPS.

        Args:
            index: OpenCV camera index.
            name: Friendly name ("cam0"/"cam1") for logging/overlays.
            target_fps: Intended FPS for recording.

        GUI impact:
            Initial values populate the Camera panel. Rebinding/retargeting occurs via Apply.
        """
        self.name = name
        self.target_fps = float(target_fps)
        self._preview_times = deque(maxlen=30)
        self._last_preview_frame = None
        self.lock = threading.Lock()
        self.cap = None
        self.synthetic = False
        self.set_index(index)

    def _open(self, index: int):
        """(Private) Try to open a VideoCapture at `index`; return (cap, synthetic_flag)."""
        cap = cv2.VideoCapture(index)
        if cap and cap.isOpened():
            # try hinting FPS (many drivers ignore)
            cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
            return cap, False
        return None, True  # synthetic

    def release(self):
        """Release the underlying VideoCapture (if any).

        GUI impact:
            None. Called during cleanup and when rebinding to a new camera index.
        """
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None

    def set_index(self, index: int):
        """Bind to a new OpenCV camera `index`. Falls back to synthetic if unavailable.

        Args:
            index: Desired OpenCV index.

        GUI impact:
            Called when user changes “Which camera to use (OpenCV device index)” and presses “Apply Settings”.
            Affects the live preview and subsequent recordings for this camera.
        """
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap, self.synthetic = self._open(index)
            self.index = index
            if self.synthetic:
                print(f"[Camera {self.name}] index {index} not available → synthetic preview/recording.")
            else:
                print(f"[Camera {self.name}] bound to index {index}.")

    def set_target_fps(self, fps: float):
        """Update the target FPS for recording and hint capture FPS to the driver.

        Args:
            fps: Target frames per second.

        GUI impact:
            Called when user changes “Target recording frame rate (fps)” and presses “Apply Settings”.
            Updates the “Recording target frame rate (intended)” label and video writer behavior.
        """
        self.target_fps = float(fps)
        with self.lock:
            if self.cap:
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                except Exception:
                    pass

    def reported_fps(self) -> float:
        """Return driver-reported FPS (CAP_PROP_FPS), may be 0 or inaccurate.

        GUI impact:
            Populates the “Driver-reported frame rate (may be 0 on some webcams)” label in the Camera panel.
        """
        with self.lock:
            if self.cap:
                v = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                return v if v > 0 else 0.0
        return 0.0

    def measured_preview_fps(self) -> float:
        """Compute measured preview FPS from recent preview timestamps.

        GUI impact:
            Populates the “Measured preview frame rate (GUI)” label in the Camera panel.
        """
        if len(self._preview_times) < 2:
            return 0.0
        dt = self._preview_times[-1] - self._preview_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._preview_times) - 1) / dt

    def grab_preview(self, w=320, h=240, overlay_index=True):
        """Return an RGB preview frame for GUI display (HxWx3 uint8).

        Args:
            w, h: Desired preview size (GUI preview label dimensions).
            overlay_index: If True, overlays “Index N” text on the frame.

        GUI impact:
            Called on a timer to update each camera's preview pane and FPS labels.
        """
        with self.lock:
            if self.synthetic:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, f"{self.name} (synthetic)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cx, cy = (int(time.time()*60) % w, h//2)
                cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
            else:
                ok, bgr = self.cap.read()
                if not ok or bgr is None:
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{self.name} [drop]", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    frame = cv2.resize(bgr, (w, h))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if overlay_index:
                cv2.putText(frame, f"Index {self.index}", (10, h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            self._last_preview_frame = frame
            self._preview_times.append(time.time())
            return frame

    def _writer(self, path: str, size):
        """(Private) Build a cv2.VideoWriter for (path, size) using current target FPS."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, float(self.target_fps), size)

    def _frame_size(self):
        """Return current capture size (width, height); defaults to 640x480 when unknown."""
        with self.lock:
            if self.cap:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                return (w, h)
        return (640, 480)

    def record_clip(self, path: str, duration_s: float, fourcc_str: str = "mp4v"):
        """Record a video clip to `path` for `duration_s` seconds at `target_fps`.

        Args:
            path: Output file path (.mp4 or .avi; container should match FOURCC).
            duration_s: Duration to record.
            fourcc_str: FOURCC string selected in GUI ("mp4v", "avc1", "XVID").

        GUI impact:
            Uses Camera panel's Target FPS and General settings' FOURCC; files appear under Output folder/date.
        """
        size = self._frame_size()
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(path, fourcc, float(self.target_fps), size)
        if not out or not out.isOpened():
            print(f"[Camera {self.name}] VideoWriter failed for {path}")
            return

        t_end = time.time() + duration_s
        print(f"[Camera {self.name}] Recording → {path} @ ~{self.target_fps:.1f} fps")
        frame_index = 0
        while time.time() < t_end:
            with self.lock:
                if self.synthetic:
                    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                    cv2.putText(frame, f"{self.name} {now_stamp()}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cx = int((frame_index * 7) % size[0])
                    cy = int(size[1] / 2)
                    cv2.circle(frame, (cx, cy), 20, (255, 255, 255), 2)
                else:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                        cv2.putText(frame, f"{self.name} [drop] {now_stamp()}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame)
            frame_index += 1
            time.sleep(max(0.0, 1.0 / float(self.target_fps)))
        out.release()
        print(f"[Camera {self.name}] Recording complete.")


# =========================
# Looming Stimulus
# =========================
class LoomingStim:
    """Renders a growing-dot ("looming") stimulus via PsychoPy or OpenCV fallback.

    GUI settings used:
      • Stimulus display duration (seconds), Starting/Final radius (pixels), Background shade (0–1)
    """
    def run(self, duration_s: float, r0: int, r1: int, bg_grey: float):
        """Present the looming stimulus for `duration_s` with radius ranging r0→r1.

        Args:
            duration_s: Duration of the looming stimulus.
            r0: Starting radius in pixels.
            r1: Ending radius in pixels.
            bg_grey: Psychopy background grey level [0–1] (ignored by OpenCV fallback).

        GUI impact:
            Reads current Stimulus panel values at trial time.
        """
        print("[Stim] Looming start.")
        if PSYCHOPY:
            try:
                win = visual.Window(size=(800, 600), color=[bg_grey]*3, units='pix', fullscr=False)
                dot = visual.Circle(win, radius=r0, fillColor='white', lineColor='white')
                t0 = time.time()
                while True:
                    t = time.time() - t0
                    if t >= duration_s:
                        break
                    r = r0 + (r1 - r0) * (t / duration_s)
                    dot.radius = r
                    dot.draw()
                    win.flip()
                win.close()
                print("[Stim] Looming done (PsychoPy).")
                return
            except Exception as e:
                print(f"[Stim] PsychoPy error: {e} → OpenCV fallback.")

        # OpenCV fallback
        try:
            size = (800, 600)
            cv2.namedWindow("Looming Stimulus", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Looming Stimulus", size[0], size[1])
            t0 = time.time()
            while True:
                t = time.time() - t0
                if t >= duration_s:
                    break
                r = int(r0 + (r1 - r0) * (t / duration_s))
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                cv2.circle(frame, (size[0]//2, size[1]//2), r, (255, 255, 255), -1)
                cv2.imshow("Looming Stimulus", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cv2.destroyWindow("Looming Stimulus")
            print("[Stim] Looming done (OpenCV).")
        except Exception as e:
            print(f"[Stim] OpenCV display unavailable ({e}). Logging-only fallback.")
            wait_s(duration_s)
            print("[Stim] Looming done (no display).")


# =========================
# Trial Orchestrator
# =========================
class TrialRunner:
    """Coordinates lights, cameras, stimulus, and logging for each trigger event.

    GUI settings used:
      • Output folder for all trials, Video codec (FOURCC), Recording duration per trigger (seconds)
      • Stimulus panel fields
      • Camera target FPS (written to CSV)
    """
    def __init__(self, cfg: Config, hardware: HardwareBridge, cam0: CameraRecorder, cam1: CameraRecorder, logger_path: str):
        """Initialize the trial runner and open/create the CSV logger.

        Args:
            cfg: Shared Config (output, durations, stimulus params).
            hardware: HardwareBridge for lights and trigger source.
            cam0, cam1: CameraRecorder instances to record per trial.
            logger_path: Path to CSV log file under Output folder.

        GUI impact:
            Writes to CSV files inside Output folder (visible to user). No direct widget updates.
        """
        self.cfg = cfg
        self.hardware = hardware
        self.cam0 = cam0
        self.cam1 = cam1
        self.stim = LoomingStim()
        self.trial_idx = 0
        # CSV logger
        new_file = not os.path.exists(logger_path)
        ensure_dir(os.path.dirname(logger_path))
        self.log_file = open(logger_path, "a", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_file)
        if new_file:
            self.log_writer.writerow([
                "trial", "timestamp", "cam0_path", "cam1_path",
                "record_duration_s", "stim_duration_s", "cam0_target_fps", "cam1_target_fps"
            ])

    def close(self):
        """Close the CSV logger file handle.

        GUI impact:
            None (file is flushed/closed on exit).
        """
        try:
            self.log_file.close()
        except Exception:
            pass

    def run_trial(self):
        """Execute one complete trial: lights → cameras (parallel) → looming → log.

        Reads from GUI/Config:
            • cfg.output_root, cfg.fourcc, cfg.record_duration_s
            • cfg.stim_duration_s, cfg.stim_r0_px, cfg.stim_r1_px, cfg.stim_bg_grey
            • cam0.target_fps, cam1.target_fps (written to CSV)

        Side-effects:
            • Writes two video files into Output folder/date
            • Appends a row to the CSV log
            • Console status prints
        """
        self.trial_idx += 1
        ts = now_stamp()
        out_dir = day_folder(self.cfg.output_root)
        # extension determines container/codec mapping via FOURCC provided
        ext = ".mp4" if self.cfg.fourcc.lower() in ("mp4v", "avc1", "h264") else ".avi"
        cam0_path = os.path.join(out_dir, f"{ts}_trial{self.trial_idx:04d}_cam0{ext}")
        cam1_path = os.path.join(out_dir, f"{ts}_trial{self.trial_idx:04d}_cam1{ext}")

        print(f"[Trial {self.trial_idx}] START {ts}")
        self.hardware.activate_lights()

        # Record both cameras in parallel
        t0 = threading.Thread(target=self.cam0.record_clip, args=(cam0_path, self.cfg.record_duration_s, self.cfg.fourcc), daemon=True)
        t1 = threading.Thread(target=self.cam1.record_clip, args=(cam1_path, self.cfg.record_duration_s, self.cfg.fourcc), daemon=True)
        t0.start(); t1.start()

        # Run looming while cameras record
        self.stim.run(self.cfg.stim_duration_s, self.cfg.stim_r0_px, self.cfg.stim_r1_px, self.cfg.stim_bg_grey)

        t0.join(); t1.join()

        self.log_writer.writerow([
            self.trial_idx, ts, cam0_path, cam1_path,
            self.cfg.record_duration_s, self.cfg.stim_duration_s,
            self.cam0.target_fps, self.cam1.target_fps
        ])
        self.log_file.flush()
        print(f"[Trial {self.trial_idx}] END  (files: {cam0_path} , {cam1_path})")


# =========================
# All-in-One GUI
# =========================
class SettingsGUI(QtWidgets.QWidget):
    """Single-window GUI exposing all settings, camera previews, and loop controls.

    Signals:
      • start_experiment() — user clicked "Start"
      • stop_experiment()  — user clicked "Stop"
      • apply_settings()   — user clicked "Apply Settings"

    Panels/controls map to Config and CameraRecorder instances passed by MainApp.
    """
    start_experiment = QtCore.pyqtSignal()
    stop_experiment  = QtCore.pyqtSignal()
    apply_settings   = QtCore.pyqtSignal()

    def __init__(self, cfg: Config, cam0: CameraRecorder, cam1: CameraRecorder):
        """Build all GUI widgets and bind local references.

        Args:
            cfg: Shared Config (read to initialize widgets).
            cam0, cam1: CameraRecorder objects for previews and live info.

        GUI impact:
            Initializes all displayed settings to match current Config/camera state.
        """
        super().__init__()
        self.cfg  = cfg
        self.cam0 = cam0
        self.cam1 = cam1

        self.setWindowTitle("FlyPy — Trigger→Outputs (All-in-One GUI)")
        self.setMinimumWidth(980)

        root = QtWidgets.QVBoxLayout(self)

        # --- Controls row: Start / Stop ---
        controls = QtWidgets.QHBoxLayout()
        self.bt_start = QtWidgets.QPushButton("Start")
        self.bt_stop  = QtWidgets.QPushButton("Stop")
        self.bt_apply = QtWidgets.QPushButton("Apply Settings")
        self.bt_start.setToolTip("Begin watching for triggers. On each trigger: lights on, record both cameras, show looming stimulus, log trial.")
        self.bt_stop.setToolTip("Stop watching for triggers. Safe to close the app after this.")
        self.bt_apply.setToolTip("Apply changes from the panels below without restarting the app.")
        controls.addWidget(self.bt_start)
        controls.addWidget(self.bt_stop)
        controls.addStretch(1)
        controls.addWidget(self.bt_apply)
        root.addLayout(controls)

        # --- Panels container ---
        panels = QtWidgets.QGridLayout()
        root.addLayout(panels)

        # General panel
        gen = QtWidgets.QGroupBox("General settings")
        gen.setToolTip("Top-level behavior and where files are saved.")
        gl = QtWidgets.QFormLayout(gen)

        self.lbl_sim = QtWidgets.QLabel("Simulation mode: OFF (hardware triggers active)")
        self.lbl_sim.setToolTip("If ON, triggers are generated on a timer instead of coming from hardware.")
        gl.addRow(self.lbl_sim)

        self.sb_sim_interval = QtWidgets.QDoubleSpinBox()
        self.sb_sim_interval.setRange(0.5, 3600.0); self.sb_sim_interval.setDecimals(2)
        self.sb_sim_interval.setValue(self.cfg.sim_trigger_interval)
        self.sb_sim_interval.setToolTip("Time between simulated triggers (in seconds). Use ≥2 s on laptops to keep CPU low.")
        gl.addRow("Interval between simulated triggers (seconds):", self.sb_sim_interval)

        self.le_root = QtWidgets.QLineEdit(self.cfg.output_root)
        self.le_root.setToolTip("Folder where all date-stamped trial folders and videos will be saved.")
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.setToolTip("Choose a different output folder.")
        rhl = QtWidgets.QHBoxLayout()
        rhl.addWidget(self.le_root); rhl.addWidget(self.btn_browse)
        gl.addRow("Output folder for all trials:", rhl)

        self.cb_fourcc = QtWidgets.QComboBox()
        self.cb_fourcc.addItems(["mp4v", "avc1", "XVID"])
        self.cb_fourcc.setCurrentText(self.cfg.fourcc)
        self.cb_fourcc.setToolTip("Video codec (FOURCC code). 'mp4v' is broadly compatible; 'avc1' compresses better if available.")
        gl.addRow("Video codec (FOURCC code):", self.cb_fourcc)

        self.sb_rec_dur = QtWidgets.QDoubleSpinBox()
        self.sb_rec_dur.setRange(0.2, 600.0); self.sb_rec_dur.setDecimals(2); self.sb_rec_dur.setValue(self.cfg.record_duration_s)
        self.sb_rec_dur.setToolTip("How long to record for each trigger (seconds). Longer clips = larger files.")
        gl.addRow("Recording duration per trigger (seconds):", self.sb_rec_dur)

        panels.addWidget(gen, 0, 0)

        # Stimulus panel
        stim = QtWidgets.QGroupBox("Looming stimulus (growing dot)")
        stim.setToolTip("Duration and size change of the looming dot shown after each trigger.")
        sl = QtWidgets.QFormLayout(stim)
        self.sb_stim_dur = QtWidgets.QDoubleSpinBox()
        self.sb_stim_dur.setRange(0.1, 30.0); self.sb_stim_dur.setDecimals(2); self.sb_stim_dur.setValue(self.cfg.stim_duration_s)
        self.sb_stim_dur.setToolTip("How long the looming dot is shown (seconds). Cameras still record for the full duration above.")
        sl.addRow("Stimulus display duration (seconds):", self.sb_stim_dur)
        self.sb_r0 = QtWidgets.QSpinBox(); self.sb_r0.setRange(1, 2000); self.sb_r0.setValue(self.cfg.stim_r0_px)
        self.sb_r0.setToolTip("Starting dot radius in pixels. Smaller values (8–20 px) start more subtly.")
        sl.addRow("Starting dot radius (pixels):", self.sb_r0)
        self.sb_r1 = QtWidgets.QSpinBox(); self.sb_r1.setRange(1, 4000); self.sb_r1.setValue(self.cfg.stim_r1_px)
        self.sb_r1.setToolTip("Final dot radius in pixels at the end of the stimulus.")
        sl.addRow("Final dot radius (pixels):", self.sb_r1)
        self.sb_bg = QtWidgets.QDoubleSpinBox(); self.sb_bg.setRange(0.0, 1.0); self.sb_bg.setSingleStep(0.05); self.sb_bg.setValue(self.cfg.stim_bg_grey)
        self.sb_bg.setToolTip("Background shade for PsychoPy display (0 = black, 1 = white). Ignored by OpenCV fallback.")
        sl.addRow("Stimulus background shade (0=black, 1=white):", self.sb_bg)
        panels.addWidget(stim, 0, 1)

        # Camera panels
        self.cam_groups = []
        for idx, cam, target_default in [(0, self.cam0, self.cfg.cam0_target_fps),
                                         (1, self.cam1, self.cfg.cam1_target_fps)]:
            gb = QtWidgets.QGroupBox(f"Camera {idx} — preview & frame rate")
            gb.setToolTip("Live preview shows which camera you're using. Frame-rate panel shows driver-reported, preview-measured, and target recording FPS.")
            fl = QtWidgets.QGridLayout(gb)

            # Left: preview
            preview = QtWidgets.QLabel()
            preview.setFixedSize(360, 270)
            preview.setFrameShape(QtWidgets.QFrame.Box)
            preview.setAlignment(QtCore.Qt.AlignCenter)
            preview.setToolTip("Live preview (pauses while recording). The overlay at the bottom shows the OpenCV device index in use.")
            fl.addWidget(preview, 0, 0, 5, 1)

            # Right: settings & info
            spin_index = QtWidgets.QSpinBox(); spin_index.setRange(0, 15)
            spin_index.setValue(cam.index)
            spin_index.setToolTip("Which camera to use (OpenCV device index). Change if the preview shows the wrong device; click Apply to rebind.")
            fl.addWidget(QtWidgets.QLabel("Which camera to use (OpenCV device index):"), 0, 1)
            fl.addWidget(spin_index, 0, 2)

            spin_fps = QtWidgets.QSpinBox(); spin_fps.setRange(1, 240); spin_fps.setValue(int(target_default))
            spin_fps.setToolTip("Target recording frame rate (FPS). Actual FPS may be limited by the camera/driver.")
            fl.addWidget(QtWidgets.QLabel("Target recording frame rate (fps):"), 1, 1)
            fl.addWidget(spin_fps, 1, 2)

            lbl_rep = QtWidgets.QLabel("Driver-reported frame rate (may be 0 on some webcams): —")
            lbl_rep.setToolTip("Driver-reported FPS (CAP_PROP_FPS). Some webcams return 0 or an inaccurate value here.")
            fl.addWidget(lbl_rep, 2, 1, 1, 2)

            lbl_mea = QtWidgets.QLabel("Measured preview frame rate (GUI): —")
            lbl_mea.setToolTip("Measured frame rate of the GUI preview (not the recorded file).")
            fl.addWidget(lbl_mea, 3, 1, 1, 2)

            lbl_tar = QtWidgets.QLabel(f"Recording target frame rate (intended): {int(target_default)}")
            lbl_tar.setToolTip("Intended recording FPS used by the video writer.")
            fl.addWidget(lbl_tar, 4, 1, 1, 2)

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

            panels.addWidget(gb, 1, idx)

        # Status
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setToolTip("Overall state: Idle / Watching for triggers / Trial running / Errors.")
        root.addWidget(self.lbl_status)

        # Signals
        self.bt_start.clicked.connect(self.start_experiment.emit)
        self.bt_stop.clicked.connect(self.stop_experiment.emit)
        self.bt_apply.clicked.connect(self.apply_settings.emit)
        self.btn_browse.clicked.connect(self._pick_folder)

        # Initial text
        self._refresh_general_labels()

        # Preview timer flag (controlled by Main)
        self.preview_paused = False

    def _pick_folder(self):
        """Open a directory picker and write the selected path to Output Root line edit.

        GUI impact:
            Updates the “Output folder for all trials” field; changes take effect after “Apply Settings”.
        """
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Root", self.le_root.text() or ".")
        if path:
            self.le_root.setText(path)

    def _refresh_general_labels(self):
        """Refresh labels derived from config (e.g., Simulation mode text).

        GUI impact:
            Updates “Simulation mode: ON/OFF …” label based on cfg.simulation_mode.
        """
        self.lbl_sim.setText(
            f"Simulation mode: {'ON (timer-based triggers)' if self.cfg.simulation_mode else 'OFF (hardware triggers active)'}"
        )

    def update_cam_fps_labels(self):
        """Update Driver-reported/Preview/Target FPS labels for both cameras.

        GUI impact:
            Writes to per-camera labels in the Camera panels.
        """
        for g in self.cam_groups:
            cam: CameraRecorder = g["cam"]
            rep = cam.reported_fps()
            mea = cam.measured_preview_fps()
            g["lbl_rep"].setText(f"Driver-reported frame rate (may be 0 on some webcams): {rep:.1f}" if rep > 0 else "Driver-reported frame rate (may be 0 on some webcams): (unknown)")
            g["lbl_mea"].setText(f"Measured preview frame rate (GUI): {mea:.1f}")
            g["lbl_tar"].setText(f"Recording target frame rate (intended): {int(cam.target_fps)}")

    def set_preview_image(self, cam_idx: int, img_rgb: np.ndarray):
        """Render a numpy RGB image into the preview QLabel for camera `cam_idx`.

        Args:
            cam_idx: 0 or 1 (Camera panels' index).
            img_rgb: HxWx3 uint8 RGB image produced by CameraRecorder.grab_preview().

        GUI impact:
            Visually updates the “preview & frame rate” box for the selected camera.
        """
        g = self.cam_groups[cam_idx]
        h, w, _ = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, w*3, QtGui.QImage.Format_RGB888)
        g["preview"].setPixmap(QtGui.QPixmap.fromImage(qimg))


# =========================
# Main Application
# =========================
class MainApp(QtWidgets.QApplication):
    """Qt Application wiring together Config, Hardware, Cameras, GUI, and the trigger loop.

    GUI settings used/written:
      • Reads all panels via apply_from_gui()
      • Updates Status label and camera preview/fps labels periodically
    """
    def __init__(self, argv):
        """Construct all subsystems, prompt for Simulation Mode, and show the GUI.

        Args:
            argv: sys.argv from Python entry point.

        GUI impact:
            Shows Yes/No dialog to set cfg.simulation_mode, initializes panels, starts preview timer.
        """
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
        self.gui.show()

        # Preview timer (updates when idle)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(500)  # ms
        self.preview_timer.timeout.connect(self.update_previews)
        self.preview_timer.start()

        # Status
        self.running = False
        self.in_trial = False
        self.thread = None

        # Cleanup hooks
        self.aboutToQuit.connect(self.cleanup)
        atexit.register(self.cleanup)

        # Initialize labels after sim prompt
        self.gui._refresh_general_labels()

    def apply_from_gui(self):
        """Read current widget values and write them into Config/Cameras.

        Reads/writes:
            • General settings → cfg.sim_trigger_interval, cfg.output_root, cfg.fourcc, cfg.record_duration_s
            • Looming stimulus → cfg.stim_* fields
            • Camera panels → cam.set_index(), cam.set_target_fps()
            • Updates cfg.cam* fields from CameraRecorder state
            • Ensures Output folder exists

        GUI impact:
            Sets Status to “Settings applied.” and prints console confirmation.
        """
        # General
        self.cfg.sim_trigger_interval = float(self.gui.sb_sim_interval.value())
        self.cfg.output_root = self.gui.le_root.text().strip() or self.cfg.output_root
        self.cfg.fourcc = self.gui.cb_fourcc.currentText()
        self.cfg.record_duration_s = float(self.gui.sb_rec_dur.value())

        # Stimulus
        self.cfg.stim_duration_s = float(self.gui.sb_stim_dur.value())
        self.cfg.stim_r0_px = int(self.gui.sb_r0.value())
        self.cfg.stim_r1_px = int(self.gui.sb_r1.value())
        self.cfg.stim_bg_grey = float(self.gui.sb_bg.value())

        # Cameras
        cam0_new_idx = int(self.gui.cam_groups[0]["spin_index"].value())
        cam0_new_tfps = int(self.gui.cam_groups[0]["spin_fps"].value())
        if cam0_new_idx != self.cam0.index:
            self.cam0.set_index(cam0_new_idx)
        self.cam0.set_target_fps(cam0_new_tfps)

        cam1_new_idx = int(self.gui.cam_groups[1]["spin_index"].value())
        cam1_new_tfps = int(self.gui.cam_groups[1]["spin_fps"].value())
        if cam1_new_idx != self.cam1.index:
            self.cam1.set_index(cam1_new_idx)
        self.cam1.set_target_fps(cam1_new_tfps)

        self.cfg.cam0_index = self.cam0.index
        self.cfg.cam1_index = self.cam1.index
        self.cfg.cam0_target_fps = int(self.cam0.target_fps)
        self.cfg.cam1_target_fps = int(self.cam1.target_fps)

        ensure_dir(self.cfg.output_root)
        self.gui.lbl_status.setText("Status: Settings applied.")
        print("[MainApp] Settings applied.")

    def update_previews(self):
        """Refresh camera previews and FPS labels when not recording a trial.

        GUI impact:
            • Updates preview panes for both cameras
            • Updates Driver-reported/Preview/Target FPS labels
            • Writes the Status label (“Waiting / Idle” or “Trial running” when paused)
        """
        if self.in_trial:
            self.gui.lbl_status.setText("Status: Trial running (preview paused).")
            self.gui.update_cam_fps_labels()
            return
        img0 = self.cam0.grab_preview()
        img1 = self.cam1.grab_preview()
        self.gui.set_preview_image(0, img0)
        self.gui.set_preview_image(1, img1)
        self.gui.update_cam_fps_labels()
        self.gui.lbl_status.setText("Status: Waiting / Idle.")

    def loop(self):
        """Background trigger loop. On trigger, runs a trial end-to-end.

        Reads:
            • hardware.check_trigger() (Simulation interval or real serial)
        Calls:
            • trial_runner.run_trial() which uses cfg and camera settings

        GUI impact:
            Writes status messages to Status label and prints to console.
        """
        self.gui.lbl_status.setText("Status: Watching for triggers…")
        print("[MainApp] Trigger loop started.")
        while self.running:
            try:
                if not self.in_trial and self.hardware.check_trigger():
                    self.in_trial = True
                    self.gui.lbl_status.setText("Status: Trial running…")
                    self.trial_runner.run_trial()
                    self.in_trial = False
                    self.gui.lbl_status.setText("Status: Trial finished.")
                time.sleep(0.002)
            except Exception as e:
                print(f"[MainApp] Loop error: {e}")
                self.gui.lbl_status.setText(f"Status: Error — {e}")
                time.sleep(0.05)
        print("[MainApp] Trigger loop stopped.")

    def start_loop(self):
        """Start the trigger loop thread (idempotent).

        GUI impact:
            Applies current settings before starting; Status reflects “Watching for triggers…”.
        """
        if not self.running:
            self.apply_from_gui()  # ensure latest settings at start
            self.running = True
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def stop_loop(self):
        """Stop the trigger loop thread and update Status.

        GUI impact:
            Sets Status to “Stopped.” after the worker thread has been joined.
        """
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
            self.thread = None
        self.gui.lbl_status.setText("Status: Stopped.")

    def cleanup(self):
        """Gracefully shut down background thread, log file, hardware, and cameras.

        GUI impact:
            None directly; prints “Cleanup complete.” and ensures resources are released.
        """
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
