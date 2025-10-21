# ========================= FlyAPI.py (v1.6.0) =========================
"""
FlyPy — Unified Trigger → Cameras + Lights + Looming Stimulus
Scaled GUI + Scrollable content + PySpin detection + Max-FPS probe.
"""

__version__ = "1.6.0"

import os, sys, time, csv, atexit, threading, importlib
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

# ---- OpenCV (primary preview path) ----
try:
    import cv2
    HAVE_OPENCV = True
except Exception:
    HAVE_OPENCV = False
    cv2 = None  # type: ignore

# ---- Qt GUI ----

class PopoutPreview(QtWidgets.QDialog):
    """Lightweight high-FPS preview window that pulls frames from a callable."""
    def __init__(self, title, frame_source, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.label = QtWidgets.QLabel()
        self.label.setMinimumSize(320, 240)
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.label)
        self._src = frame_source
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(0)  # run as fast as the event loop allows

    def _tick(self):
        try:
            img = self._src()
            if img is None:
                return
            if img.ndim == 2:
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_Grayscale8)
            else:
                rgb = img[:, :, ::-1].copy()
                qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
                self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        except Exception as e:
            # don't spam; silent drop
            pass
from PyQt5 import QtWidgets, QtCore, QtGui
try:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
except Exception:
    pass

# ---- PsychoPy (optional, for looming stimulus) ----
PSY_LOADED = None
visual = None
core = None

def _ensure_psychopy_loaded() -> bool:
    global PSY_LOADED, visual, core
    if PSY_LOADED is True: return True
    if PSY_LOADED is False: return False
    try:
        importlib.import_module("psychopy")  # noqa
        visual = importlib.import_module("psychopy.visual")
        core   = importlib.import_module("psychopy.core")
        PSY_LOADED = True
        return True
    except Exception:
        visual = None; core = None; PSY_LOADED = False
        return False

# ---- PySpin (Spinnaker) optional backend ----
HAVE_PYSPIN = False
PySpin = None
_SPIN_SYSTEM = None
_SPIN_SYS_REFCOUNT = 0

def _maybe_add_spinnaker_paths():
    candidates = []
    for env_var in ("SPINNAKER_PATH", "SPINNAKER"):
        p = os.environ.get(env_var)
        if p and os.path.isdir(p):
            candidates.append(p)
    candidates.extend([
        r"C:\Program Files\Teledyne FLIR\Spinnaker",
        r"C:\Program Files\FLIR Systems\Spinnaker",
    ])
    add = []
    for base in candidates:
        for sub in ("bin64", "lib64"):
            p = os.path.join(base, sub)
            if os.path.isdir(p):
                add.append(p)
    if add:
        cur = os.environ.get("PATH", "")
        for p in add:
            if p not in cur:
                cur = p + os.pathsep + cur
        os.environ["PATH"] = cur

def _try_import_pyspin():
    global HAVE_PYSPIN, PySpin
    try:
        import PySpin as _PySpin  # type: ignore
        PySpin = _PySpin
        HAVE_PYSPIN = True
    except Exception:
        HAVE_PYSPIN = False
        PySpin = None

_maybe_add_spinnaker_paths()
_try_import_pyspin()

def _spin_system_acquire():
    """Reference-counted PySpin.System singleton."""
    global _SPIN_SYSTEM, _SPIN_SYS_REFCOUNT, HAVE_PYSPIN, PySpin
    if not HAVE_PYSPIN: return None
    if _SPIN_SYSTEM is None:
        _SPIN_SYSTEM = PySpin.System.GetInstance()
    _SPIN_SYS_REFCOUNT += 1
    return _SPIN_SYSTEM

def _spin_system_release():
    global _SPIN_SYSTEM, _SPIN_SYS_REFCOUNT, HAVE_PYSPIN, PySpin
    if not HAVE_PYSPIN or _SPIN_SYSTEM is None: return
    _SPIN_SYS_REFCOUNT = max(0, _SPIN_SYS_REFCOUNT - 1)
    if _SPIN_SYS_REFCOUNT == 0:
        try:
            _SPIN_SYSTEM.ReleaseInstance()
        except Exception:
            pass
        _SPIN_SYSTEM = None

# ------------- utils -------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def now_stamp() -> str: return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def day_folder(root: str) -> str:
    d = datetime.now().strftime("%Y%m%d")
    path = os.path.join(root, d); ensure_dir(path); return path
def wait_s(sec: float):
    if _ensure_psychopy_loaded():
        try: core.wait(sec)  # type: ignore
        except Exception: time.sleep(sec)
        return
    time.sleep(sec)

VIDEO_PRESETS = [
    {"id": "mp4_mp4v", "label": "MP4 / mp4v — very compatible; moderate CPU", "fourcc": "mp4v"},
    {"id": "avi_xvid", "label": "AVI / XVID — broad compatibility; larger files", "fourcc": "XVID"},
    {"id": "avi_mjpg", "label": "AVI / MJPG — very large files; light CPU", "fourcc": "MJPG"},
]
PRESETS_BY_ID = {p["id"]: p for p in VIDEO_PRESETS}
def default_preset_id() -> str: return "mp4_mp4v"

class Config:
    def __init__(self):
        self.simulation_mode = False
        self.sim_trigger_interval = 5.0
        self.output_root = "FlyPy_Output"
        self.prewarm_stim = False

        self.video_preset_id = default_preset_id()
        self.fourcc = PRESETS_BY_ID[self.video_preset_id]["fourcc"]
        self.record_duration_s = 3.0

        self.stim_duration_s = 1.5
        self.stim_r0_px = 8
        self.stim_r1_px = 240
        self.stim_bg_grey = 1.0
        self.lights_delay_s = 0.0
        self.stim_delay_s = 0.0

        self.stim_screen_index = 0
        self.stim_fullscreen = False
        self.gui_screen_index = 0

        self.cam0_backend = "OpenCV"
        self.cam1_backend = "OpenCV"
        self.cam0_id = "0"      # OpenCV index OR PySpin serial
        self.cam1_id = "1"
        self.cam0_target_fps = 60
        self.cam1_target_fps = 60

# --------- Hardware Bridge (serial) ---------
class HardwareBridge:
    def __init__(self, cfg: Config, port: str = None, baud: int = 115200):
        self.cfg = cfg
        self.simulated = cfg.simulation_mode
        self.port = port; self.baud = baud
        self._opened = False; self._last_sim = time.time()
        self.ser = None

    def _open_if_needed(self):
        if self.simulated or self._opened: return
        self._opened = True
        try:
            import serial, serial.tools.list_ports  # type: ignore
            if not self.port: self.port = self._autodetect_port()
            if self.port:
                try:
                    self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
                    wait_s(1.2); print(f"[HW] Serial open: {self.port} @ {self.baud}")
                except Exception as e:
                    print(f"[HW] Open failed: {e} → simulation"); self.simulated = True
            else:
                print("[HW] No CH340/UNO port found → simulation"); self.simulated = True
        except Exception:
            print("[HW] pyserial not available → simulation"); self.simulated = True

    def _autodetect_port(self) -> Optional[str]:
        try:
            import serial.tools.list_ports  # type: ignore
            for p in serial.tools.list_ports.comports():
                vid = f"{p.vid:04X}" if p.vid is not None else None
                pid = f"{p.pid:04X}" if p.pid is not None else None
                if vid == "1A86" and pid == "7523": return p.device
            for p in serial.tools.list_ports.comports():
                d = (p.description or "").lower()
                if "ch340" in d or "uno" in d or "elegoo" in d: return p.device
        except Exception: pass
        return None

    def check_trigger(self) -> bool:
        if self.simulated:
            now = time.time()
            if now - self._last_sim >= self.cfg.sim_trigger_interval:
                self._last_sim = now; print("[HW] (Sim) Trigger"); return True
            return False
        self._open_if_needed()
        try:
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line == "T": return True
        except Exception as e:
            print(f"[HW] Read error: {e}")
        return False

    def _send(self, text: str):
        self._open_if_needed()
        if self.simulated or not self.ser: print(f"[HW] (Sim) SEND: {text}"); return
        try: self.ser.write((text.strip()+"\n").encode("utf-8", errors="ignore"))
        except Exception as e: print(f"[HW] Write error: {e}")

    def mark_start(self): self._send("MARK START")
    def mark_end(self):   self._send("MARK END")
    def lights_on(self):  self._send("LIGHT ON")
    def lights_off(self): self._send("LIGHT OFF")

    def close(self):
        if not self.simulated and self.ser:
            try: self.ser.close()
            except Exception: pass
        self.ser = None; self._opened = False

# --------- Cameras ---------
class BaseCamera:
    def open(self): raise NotImplementedError
    def get_frame(self): raise NotImplementedError
    def release(self): raise NotImplementedError
    def frame_size(self) -> Tuple[int,int]: raise NotImplementedError
    def max_fps(self) -> Optional[float]: return None

class OpenCVCamera(BaseCamera):
    def __init__(self, index: int, target_fps: float):
        if not HAVE_OPENCV: raise RuntimeError("OpenCV is not installed")
        self.index = index; self.target_fps = float(target_fps); self.cap = None

    def open(self):
        backends = [cv2.CAP_ANY]
        if os.name == "nt": backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for be in backends:
            try:
                cap = cv2.VideoCapture(self.index, be)
                if cap and cap.isOpened():
                    try: cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
                    except Exception: pass
                    self.cap = cap; return
                if cap: cap.release()
            except Exception: pass
        self.cap = None

    def get_frame(self):
        if not self.cap: return None
        ok, bgr = self.cap.read()
        if not ok or bgr is None: return None
        return bgr

    def frame_size(self):
        if self.cap:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            return (w, h)
        return (640, 480)

    def release(self):
        if self.cap:
            try: self.cap.release()
            except Exception: pass
            self.cap = None

class PySpinCamera(BaseCamera):
    def __init__(self, serial: str, target_fps: float):
        if not HAVE_PYSPIN: raise RuntimeError("PySpin not available")
        self.serial = str(serial or "").strip()
        self.target_fps = float(target_fps)
        self.cam = None
        self._nm = None
        self._size = (640, 480)
        self._max_fps = None
        self._opened = False

    def _set_enum(self, nodemap, name, entry):
        try:
            node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                val = node.GetEntryByName(entry)
                if PySpin.IsAvailable(val) and PySpin.IsReadable(val):
                    node.SetIntValue(val.GetValue())
        except Exception:
            pass

    def _set_bool(self, nodemap, name, value: bool):
        try:
            node = PySpin.CBooleanPtr(nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                node.SetValue(bool(value))
        except Exception:
            pass

    def _set_float(self, nodemap, name, value: float, clamp_to_max=True):
        try:
            node = PySpin.CFloatPtr(nodemap.GetNode(name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                lo = node.GetMin(); hi = node.GetMax()
                v = max(lo, min(hi, float(value))) if clamp_to_max else float(value)
                node.SetValue(v)
        except Exception:
            pass

    def open(self):
        sys_obj = _spin_system_acquire()
        cams = sys_obj.GetCameras() if sys_obj else None
        try:
            if not cams or cams.GetSize() == 0: raise RuntimeError("No PySpin cameras found")
            chosen = None
            if self.serial:
                for cam in cams:
                    try:
                        sn = cam.TLDevice.DeviceSerialNumber.ToString()
                    except Exception:
                        sn = ""
                    if sn == self.serial:
                        chosen = cam; break
            if chosen is None:
                chosen = cams.GetByIndex(0)
            self.cam = chosen
            self.cam.Init()
            self._nm = self.cam.GetNodeMap()

            # PixelFormat: prefer Mono8 (fast, grayscale); otherwise GUI converts
            self._set_enum(self._nm, "PixelFormat", "Mono8")

            # Frame rate: enable and set; capture max for UI
            try:
                fr_node = PySpin.CFloatPtr(self._nm.GetNode("AcquisitionFrameRate"))
                if PySpin.IsAvailable(fr_node) and PySpin.IsWritable(fr_node):
                    self._max_fps = float(fr_node.GetMax())
                    self._set_bool(self._nm, "AcquisitionFrameRateEnable", True)
                    self._set_float(self._nm, "AcquisitionFrameRate", self.target_fps)
            except Exception:
                pass

            # Continuous
            self._set_enum(self._nm, "AcquisitionMode", "Continuous")

            # Stream buffer handling = NewestOnly
            try:
                s_map = self.cam.GetTLStreamNodeMap()
                self._set_enum(s_map, "StreamBufferHandlingMode", "NewestOnly")
            except Exception:
                pass

            # Size for writer
            try:
                w = int(PySpin.CIntegerPtr(self._nm.GetNode("Width")).GetValue())
                h = int(PySpin.CIntegerPtr(self._nm.GetNode("Height")).GetValue())
                self._size = (w, h)
            except Exception:
                self._size = (640, 480)

            self.cam.BeginAcquisition()
            self._opened = True
        except Exception as e:
            if self.cam:
                try:
                    self.cam.DeInit()
                except Exception:
                    pass
                self.cam = None
            raise e
        finally:
            if cams is not None:
                cams.Clear()

    def get_frame(self):
        if not self._opened or not self.cam: return None
        try:
            img = self.cam.GetNextImage(1000)
            try:
                if img.IsIncomplete():
                    return None
                arr = img.GetNDArray()
                if arr.ndim == 2:
                    if HAVE_OPENCV:
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                    else:
                        arr = np.stack([arr,arr,arr], axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                return arr
            finally:
                img.Release()
        except Exception:
            return None

    def frame_size(self):
        return self._size

    def max_fps(self) -> Optional[float]:
        return self._max_fps

    def release(self):
        try:
            if self.cam:
                try:
                    if self._opened:
                        self.cam.EndAcquisition()
                except Exception:
                    pass
                try:
                    self.cam.DeInit()
                except Exception:
                    pass
        finally:
            self.cam = None; self._nm = None; self._opened = False
            _spin_system_release()

class CameraNode:
    def __init__(self, name: str, backend: str, ident: str, target_fps: int):
        self.name = name; self.backend = backend; self.ident = ident
        self.target_fps = float(target_fps)
        self.dev: Optional[BaseCamera] = None
        self.synthetic = False
        self.preview_times = deque(maxlen=30)
        self._max_fps_cache: Optional[float] = None

    def _open_if_needed(self):
        if self.dev is not None: return
        try:
            if self.backend == "PySpin":
                if not HAVE_PYSPIN: raise RuntimeError("PySpin backend not available")
                dev = PySpinCamera(self.ident, self.target_fps)  # accepts blank → first camera
                dev.open()
                self.dev = dev
                self._max_fps_cache = dev.max_fps()
                print(f"[{self.name}] PySpin open: serial={'(auto)' if not self.ident else self.ident}, max_fps={self._max_fps_cache}")
            else:
                if not HAVE_OPENCV: raise RuntimeError("OpenCV not installed")
                idx = int(self.ident or "0")
                dev = OpenCVCamera(idx, self.target_fps); dev.open()
                if dev.cap is None:  # type: ignore
                    self.synthetic = True; self.dev = None
                    print(f"[{self.name}] OpenCV index {idx} not available → synthetic")
                else:
                    self.dev = dev; print(f"[{self.name}] OpenCV open: index {idx}")
        except Exception as e:
            print(f"[{self.name}] Open error: {e} → synthetic")
            self.dev = None; self.synthetic = True

    def set_backend_ident(self, backend: str, ident: str):
        self.release()
        self.backend = backend; self.ident = ident; self.synthetic = False
        self._max_fps_cache = None
        print(f"[{self.name}] set backend={backend} ident={ident or '(auto)'} (lazy open)")

    def set_target_fps(self, fps: int):
        self.target_fps = float(fps)

    def grab_preview(self, w: int, h: int):
        self._open_if_needed()
        if self.synthetic:
            frame = np.full((h, w, 3), 240, dtype=np.uint8)
            if HAVE_OPENCV:
                cv2.putText(frame, f"{self.name} (synthetic)", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            self.preview_times.append(time.time())
            return frame
        if self.dev is None:
            frame = np.full((h, w, 3), 220, dtype=np.uint8)
            if HAVE_OPENCV:
                cv2.putText(frame, f"{self.name} (opening…)", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2, cv2.LINE_AA)
            self.preview_times.append(time.time())
            return frame
        img = self.dev.get_frame()
        if img is None:
            frame = np.full((h, w, 3), 255, dtype=np.uint8)
            if HAVE_OPENCV:
                cv2.putText(frame, f"{self.name} [drop]", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        else:
            if HAVE_OPENCV:
                frame = cv2.resize(img, (w, h))
            else:
                frame = img
        self.preview_times.append(time.time())
        return frame

    def driver_fps(self) -> float:
        if len(self.preview_times) < 2: return 0.0
        dt = self.preview_times[-1] - self.preview_times[0]
        n = len(self.preview_times) - 1
        return (n / dt) if dt > 0 else 0.0

    
def record_clip(self, path, duration_s, fourcc, async_writer=False):
    """Record a fixed-duration clip to disk, robust to color/gray channel mismatches.
    Lazy-create writer on first frame to infer channels; convert frames as needed.
    """
    import time, numpy as _np, cv2 as _cv2
    self.start_grabber()
    size = (int(self.width or 640), int(self.height or 480))
    fps  = float(self.fps or 60.0)

    writer = None
    n = 0
    t0 = time.time()
    stop_t = t0 + float(duration_s)

    try:
        while time.time() < stop_t:
            img = self.grab_preview()
            if img is None:
                img = _np.zeros((size[1], size[0], 3), dtype=_np.uint8)

            if img.ndim == 2:
                ch = 1
            else:
                ch = 1 if img.shape[2] == 1 else 3

            if writer is None:
                writer = _cv2.VideoWriter(path, fourcc, fps, size, (ch == 3))
                if not writer.isOpened():
                    raise RuntimeError(f"VideoWriter failed to open: {path}")

            # Convert to expected channels
            # We infer expectation from the writer's isColor arg (ch at creation time).
            expect_color = (ch == 3)
            if expect_color and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)):
                img = _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
            elif (not expect_color) and (img.ndim == 3 and img.shape[2] == 3):
                img = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)

            if (img.shape[1], img.shape[0]) != size:
                img = _cv2.resize(img, size, interpolation=_cv2.INTER_NEAREST)

            writer.write(img)
            n += 1

        elapsed = max(1e-9, time.time() - t0)
        return {"path": path, "frames": n, "avg_fps": n / elapsed}
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

        if not ok_any and not self.synthetic:
            print(f"[{self.name}] Warning: no frames captured")
        return path

    def max_fps_cached(self) -> Optional[float]:
        return self._max_fps_cache

    def release(self):
        try:
            if self.dev: self.dev.release()
        except Exception: pass
        self.dev = None; self.synthetic = False

# ---- Looming stimulus ----
class LoomingStim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._pp_win = None; self._pp_cfg = None
        self._cv_window_name = "Looming Stimulus"
        self._cv_open = False; self._cv_size = (800, 600)

    def _pp_window(self, screen_idx: int, fullscreen: bool, bg_grey: float):
        need_new = (self._pp_win is None) or (self._pp_cfg != (screen_idx, fullscreen))
        if need_new and self._pp_win is not None:
            try: self._pp_win.close()
            except Exception: pass
            self._pp_win = None
        if need_new:
            try:
                if fullscreen:
                    self._pp_win = visual.Window(color=[bg_grey]*3, units='pix', fullscr=True, screen=screen_idx)
                else:
                    self._pp_win = visual.Window(size=self._cv_size, color=[bg_grey]*3, units='pix', fullscr=False, screen=screen_idx, allowGUI=True)
                self._pp_cfg = (screen_idx, fullscreen)
            except Exception as e:
                print(f"[Stim] PsychoPy window error: {e}")
                self._pp_win = None
        if self._pp_win is not None:
            try: self._pp_win.color = [bg_grey]*3
            except Exception: pass

    def _cv_window(self, screen_idx: int, bg_grey: float):
        try:
            if not self._cv_open:
                cv2.namedWindow(self._cv_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._cv_window_name, self._cv_size[0], self._cv_size[1])
                self._cv_open = True
            geoms = QtGui.QGuiApplication.screens()
            if 0 <= screen_idx < len(geoms):
                g = geoms[screen_idx].geometry()
                cv2.moveWindow(self._cv_window_name, g.x()+50, g.y()+50)
            bg = int(max(0, min(255, int(bg_grey*255))))
            frame = np.full((self._cv_size[1], self._cv_size[0], 3), bg, dtype=np.uint8)
            cv2.imshow(self._cv_window_name, frame); cv2.waitKey(1)
        except Exception as e:
            print(f"[Stim] OpenCV window error: {e}"); self._cv_open = False

    def open_persistent(self, screen_idx: int, fullscreen: bool, bg_grey: float):
        if _ensure_psychopy_loaded():
            self._pp_window(screen_idx, fullscreen, bg_grey)
            if self._pp_win is not None:
                try: self._pp_win.flip()
                except Exception: pass
        else:
            self._cv_window(screen_idx, bg_grey)

    def run(self, duration_s: float, r0: int, r1: int, bg_grey: float, screen_idx: int, fullscreen: bool):
        print("[Stim] Looming start")
        if _ensure_psychopy_loaded():
            try:
                self._pp_window(screen_idx, fullscreen, bg_grey)
                if self._pp_win is not None:
                    dot = visual.Circle(self._pp_win, radius=r0, fillColor='black', lineColor='black')
                    t0 = time.time()
                    while True:
                        t = time.time() - t0
                        if t >= duration_s: break
                        r = r0 + (r1 - r0) * (t / duration_s)
                        dot.radius = r; dot.draw(); self._pp_win.flip()
                    print("[Stim] Done (PsychoPy)"); return
            except Exception as e:
                print(f"[Stim] PsychoPy error: {e} → OpenCV fallback")
        try:
            self._cv_window(screen_idx, bg_grey); size = self._cv_size
            bg = int(max(0, min(255, int(bg_grey*255))))
            t0 = time.time()
            while True:
                t = time.time() - t0
                if t >= duration_s: break
                r = int(r0 + (r1 - r0) * (t / duration_s))
                frame = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
                if HAVE_OPENCV:
                    cv2.circle(frame, (size[0]//2, size[1]//2), r, (0,0,0), -1)
                    cv2.imshow(self._cv_window_name, frame)
                    if cv2.waitKey(1) & 0xFF == 27: break
            print("[Stim] Done (OpenCV)")
        except Exception as e:
            print(f"[Stim] Fallback display unavailable: {e}"); wait_s(duration_s); print("[Stim] Done (timing only)")

    def close(self):
        try:
            if self._pp_win is not None: self._pp_win.close()
        except Exception: pass
        self._pp_win = None; self._pp_cfg = None
        if self._cv_open and HAVE_OPENCV:
            try: cv2.destroyWindow(self._cv_window_name)
            except Exception: pass
            self._cv_open = False

# --------- Trial Runner ---------
class TrialRunner:
    def __init__(self, cfg: Config, hw: HardwareBridge, cam0, cam1, log_path: str):
        self.cfg = cfg; self.hw = hw; self.cam0 = cam0; self.cam1 = cam1
        self.stim = LoomingStim(cfg)
        ensure_dir(os.path.dirname(log_path) or ".")
        new = not os.path.exists(log_path)
        self.log = open(log_path, "a", newline="", encoding="utf-8")
        self.csvw = csv.writer(self.log)
        if new:
            self.csvw.writerow(["timestamp","trial_idx","cam0_path","cam1_path","record_duration_s",
                                "lights_delay_s","stim_delay_s","stim_duration_s",
                                "stim_screen_index","stim_fullscreen",
                                "cam0_backend","cam0_ident","cam0_target_fps",
                                "cam1_backend","cam1_ident","cam1_target_fps",
                                "video_preset_id","fourcc","cam0_max_fps","cam1_max_fps"])
        self.trial_idx = 0

    def close(self):
        try: self.log.close()
        except Exception: pass
        try: self.stim.close()
        except Exception: pass

    def _ext_for_fourcc(self, fourcc: str) -> str:
        s = fourcc.lower()
        if s in ("mp4v","avc1","h264"): return "mp4"
        if s in ("mjpg","xvid","divx"): return "avi"
        return "mp4"

    def _trial_folder(self) -> str:
        base = day_folder(self.cfg.output_root)
        p = os.path.join(base, f"trial_{now_stamp()}"); ensure_dir(p); return p

    def _record_both(self, folder: str):
        fourcc = self.cfg.fourcc; ext = self._ext_for_fourcc(fourcc)
        out0 = os.path.join(folder, f"cam0.{ext}")
        out1 = os.path.join(folder, f"cam1.{ext}")
        res = {"c0": None, "c1": None}
        def rec(cam, pth, key): res[key] = cam.record_clip(pth, float(self.cfg.record_duration_s), fourcc)
        t0 = threading.Thread(target=rec, args=(self.cam0,out0,"c0"))
        t1 = threading.Thread(target=rec, args=(self.cam1,out1,"c1"))
        t0.start(); t1.start(); t0.join(); t1.join()
        return res["c0"], res["c1"]

    def run_one(self):
        folder = self._trial_folder()
        self.hw.mark_start(); print("[Trial] Recording…")
        c0, c1 = self._record_both(folder)

        if self.cfg.lights_delay_s > 0:
            print(f"[Trial] Wait {self.cfg.lights_delay_s:.3f}s → LIGHTS ON"); wait_s(self.cfg.lights_delay_s)
        self.hw.lights_on()

        if self.cfg.stim_delay_s > 0:
            print(f"[Trial] Wait {self.cfg.stim_delay_s:.3f}s → STIM"); wait_s(self.cfg.stim_delay_s)
        self.stim.run(self.cfg.stim_duration_s, self.cfg.stim_r0_px, self.cfg.stim_r1_px,
                      self.cfg.stim_bg_grey, self.cfg.stim_screen_index, self.cfg.stim_fullscreen)

        self.hw.lights_off(); self.hw.mark_end()
        self.trial_idx += 1
        self.csvw.writerow([now_stamp(), self.trial_idx, c0 or "", c1 or "", float(self.cfg.record_duration_s),
                            float(self.cfg.lights_delay_s), float(self.cfg.stim_delay_s), float(self.cfg.stim_duration_s),
                            int(self.cfg.stim_screen_index), bool(self.cfg.stim_fullscreen),
                            self.cam0.backend, self.cam0.ident, int(self.cam0.target_fps),
                            self.cam1.backend, self.cam1.ident, int(self.cam1.target_fps),
                            self.cfg.video_preset_id, self.cfg.fourcc,
                            self.cam0.max_fps_cached() or "", self.cam1.max_fps_cached() or ""])
        self.log.flush(); print("[Trial] Logged")

# --------- Settings GUI ---------
class SettingsGUI(QtWidgets.QWidget):
    start_experiment = QtCore.pyqtSignal()
    stop_experiment  = QtCore.pyqtSignal()
    apply_settings   = QtCore.pyqtSignal()
    manual_trigger   = QtCore.pyqtSignal()

    def __init__(self, cfg: Config, cam0, cam1):
        super().__init__()
        self.cfg = cfg; self.cam0 = cam0; self.cam1 = cam1
        self.setWindowTitle(f"FlyPy — v{__version__}")

        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self); scroll.setWidgetResizable(True)
        pane = QtWidgets.QWidget(); root = QtWidgets.QVBoxLayout(pane)

        row = QtWidgets.QHBoxLayout()
        self.bt_start = QtWidgets.QPushButton("Start")
        self.bt_stop  = QtWidgets.QPushButton("Stop")
        self.bt_trig  = QtWidgets.QPushButton("Trigger Once")
        self.bt_apply = QtWidgets.QPushButton("Apply Settings")
        self.bt_detect= QtWidgets.QPushButton("Detect Cameras")
        for b in (self.bt_start, self.bt_stop, self.bt_trig, self.bt_apply, self.bt_detect):
            row.addWidget(b)
        root.addLayout(row)
        self.bt_start.clicked.connect(self.start_experiment.emit)
        self.bt_stop.clicked.connect(self.stop_experiment.emit)
        self.bt_trig.clicked.connect(self.manual_trigger.emit)
        self.bt_apply.clicked.connect(self.apply_settings.emit)
        self.bt_detect.clicked.connect(self._detect_cams)

        self.lbl_status = QtWidgets.QLabel("Status: Idle."); root.addWidget(self.lbl_status)

        grid = QtWidgets.QGridLayout(); root.addLayout(grid)

        gen = QtWidgets.QGroupBox("General"); gl = QtWidgets.QFormLayout(gen)
        self.cb_sim = QtWidgets.QCheckBox("Simulation Mode (timer triggers)"); self.cb_sim.setChecked(self.cfg.simulation_mode); gl.addRow(self.cb_sim)
        self.sb_sim = QtWidgets.QDoubleSpinBox(); self.sb_sim.setRange(0.1, 3600.0); self.sb_sim.setDecimals(2); self.sb_sim.setValue(self.cfg.sim_trigger_interval); gl.addRow("Interval between simulated triggers (s):", self.sb_sim)
        self.le_root = QtWidgets.QLineEdit(self.cfg.output_root); btn_browse = QtWidgets.QPushButton("Browse…")
        rowr = QtWidgets.QHBoxLayout(); rowr.addWidget(self.le_root); rowr.addWidget(btn_browse); gl.addRow("Output folder:", rowr)
        self.cb_fmt = QtWidgets.QComboBox(); self._id_by_idx = {}
        current = 0
        for i, p in enumerate(VIDEO_PRESETS):
            self.cb_fmt.addItem(p["label"]); self.cb_fmt.setItemData(i, p["id"]); self._id_by_idx[i] = p["id"]
            if p["id"] == self.cfg.video_preset_id: current = i
        self.cb_fmt.setCurrentIndex(current); gl.addRow("Video format / codec:", self.cb_fmt)
        self.sb_rec = QtWidgets.QDoubleSpinBox(); self.sb_rec.setRange(0.1, 600.0); self.sb_rec.setDecimals(2); self.sb_rec.setValue(self.cfg.record_duration_s); gl.addRow("Recording duration (s):", self.sb_rec)
        grid.addWidget(gen, 0, 0)

        stim = QtWidgets.QGroupBox("Stimulus & Timing"); sl = QtWidgets.QFormLayout(stim)
        self.sb_stim_dur = QtWidgets.QDoubleSpinBox(); self.sb_stim_dur.setRange(0.1, 30.0); self.sb_stim_dur.setDecimals(2); self.sb_stim_dur.setValue(self.cfg.stim_duration_s)
        self.sb_r0 = QtWidgets.QSpinBox(); self.sb_r0.setRange(1, 2000); self.sb_r0.setValue(self.cfg.stim_r0_px)
        self.sb_r1 = QtWidgets.QSpinBox(); self.sb_r1.setRange(1, 4000); self.sb_r1.setValue(self.cfg.stim_r1_px)
        self.sb_bg = QtWidgets.QDoubleSpinBox(); self.sb_bg.setRange(0.0, 1.0); self.sb_bg.setSingleStep(0.05); self.sb_bg.setValue(self.cfg.stim_bg_grey)
        self.sb_light_delay = QtWidgets.QDoubleSpinBox(); self.sb_light_delay.setRange(0.0, 10.0); self.sb_light_delay.setDecimals(3); self.sb_light_delay.setValue(self.cfg.lights_delay_s)
        self.sb_stim_delay  = QtWidgets.QDoubleSpinBox(); self.sb_stim_delay.setRange(0.0, 10.0); self.sb_stim_delay.setDecimals(3); self.sb_stim_delay.setValue(self.cfg.stim_delay_s)
        for w,label in [(self.sb_stim_dur,"Stimulus duration (s):"),(self.sb_r0,"Start radius (px):"),(self.sb_r1,"Final radius (px):"),
                        (self.sb_bg,"Background shade (0–1):"),(self.sb_light_delay,"Delay: record → lights ON (s):"),(self.sb_stim_delay,"Delay: record → stimulus ON (s):")]:
            sl.addRow(label, w)
        grid.addWidget(stim, 0, 1)

        disp = QtWidgets.QGroupBox("Display & Windows"); dl = QtWidgets.QFormLayout(disp)
        self.cb_stim_screen = QtWidgets.QComboBox(); self.cb_gui_screen  = QtWidgets.QComboBox()
        for i, s in enumerate(QtGui.QGuiApplication.screens()):
            g = s.geometry(); label = f"Screen {i} — {g.width()}×{g.height()} @({g.x()},{g.y()})"
            self.cb_stim_screen.addItem(label); self.cb_gui_screen.addItem(label)
        self.cb_stim_screen.setCurrentIndex(min(self.cfg.stim_screen_index, self.cb_stim_screen.count()-1))
        self.cb_gui_screen.setCurrentIndex(min(self.cfg.gui_screen_index, self.cb_gui_screen.count()-1))
        self.cb_full = QtWidgets.QCheckBox("Stimulus fullscreen"); self.cb_full.setChecked(self.cfg.stim_fullscreen)
        self.cb_prewarm = QtWidgets.QCheckBox("Pre-warm stimulus window at launch"); self.cb_prewarm.setChecked(self.cfg.prewarm_stim)
        for w,label in [(self.cb_stim_screen,"Stimulus display screen:"),(self.cb_gui_screen,"GUI display screen:")]:
            dl.addRow(label, w)
        dl.addRow(self.cb_full); dl.addRow(self.cb_prewarm)
        grid.addWidget(disp, 1, 0, 1, 2)

        self.cam_boxes = []
        for idx, node in enumerate((self.cam0, self.cam1)):
            gb = QtWidgets.QGroupBox(f"Camera {idx}"); glb = QtWidgets.QGridLayout(gb)
            preview = QtWidgets.QLabel(); preview.setFixedSize(360, 240); preview.setStyleSheet("background:#ddd;border:1px solid #aaa;"); preview.setAlignment(QtCore.Qt.AlignCenter)
            glb.addWidget(preview, 0, 0, 6, 1)
            cb_backend = QtWidgets.QComboBox(); cb_backend.addItem("OpenCV"); cb_backend.addItem("PySpin")
            cb_backend.setCurrentIndex(0 if node.backend=="OpenCV" else 1)
            glb.addWidget(QtWidgets.QLabel("Backend:"), 0, 1); glb.addWidget(cb_backend, 0, 2)
            le_ident = QtWidgets.QLineEdit(node.ident); le_ident.setPlaceholderText("OpenCV index (e.g., 0) OR PySpin serial (blank = auto-first)")
            glb.addWidget(QtWidgets.QLabel("Device index / serial:"), 1, 1); glb.addWidget(le_ident, 1, 2)
            # --- Search & select cameras (live dropdown) ---
            row = 1
            btn_search = QtWidgets.QPushButton("Search cameras")
            cb_found = QtWidgets.QComboBox(); cb_found.setMinimumWidth(260)
            def _search_and_fill():
                # Enumerate based on current backend selection
                backend = cb_backend.currentText()
                found = []
                try:
                    if backend == "PySpin":
                        for d in enum_pyspin():
                            # Display serial + model
                            s = d.get("serial") or d.get("id") or ""
                            name = d.get("model", "")
                            found.append((f"{s} {name}".strip(), s))
                    else:
                        for d in enum_opencv():
                            # Show index + name
                            idx2 = d.get("index")
                            name  = d.get("name", "")
                            found.append((f"index:{idx2} {name}".strip(), f"index:{idx2}"))
                except Exception as e:
                    print("[Search] enumerate error:", e)

                cb_found.blockSignals(True); cb_found.clear()
                for label, val in found:
                    cb_found.addItem(label, val)
                cb_found.blockSignals(False)
                if found:
                    # Preselect first and copy to ident
                    le_ident.setText(str(found[0][1]))
            btn_apply_found = QtWidgets.QPushButton("Use selected")
            def _apply_found():
                v = cb_found.currentData()
                if v is not None:
                    le_ident.setText(str(v))

            glb.addWidget(btn_search, row+1, 1)
            glb.addWidget(cb_found,  row+1, 2)
            glb.addWidget(btn_apply_found, row+2, 2)
            btn_search.clicked.connect(_search_and_fill)
            btn_apply_found.clicked.connect(_apply_found)

            sb_fps = QtWidgets.QSpinBox(); sb_fps.setRange(1, 20000); sb_fps.setValue(int(node.target_fps))
            glb.addWidget(QtWidgets.QLabel("Target FPS:"), 2, 1); glb.addWidget(sb_fps, 2, 2)
            lbl_rep = QtWidgets.QLabel("Driver-reported FPS: ~0.0  |  Max: ?")
            glb.addWidget(lbl_rep, 3, 1, 1, 2)
            btn_apply_cam = QtWidgets.QPushButton("Apply to this camera")
            glb.addWidget(btn_apply_cam, 4, 1, 1, 2)
            i = idx
            btn_apply_cam.clicked.connect(lambda _, ii=i, cb=cb_backend, le=le_ident, sb=sb_fps: self._apply_one(ii, cb, le, sb))
            self.cam_boxes.append({"gb": gb, "preview": preview, "cb_backend": cb_backend, "le_ident": le_ident, "sb_fps": sb_fps, "lbl_rep": lbl_rep})
            grid.addWidget(gb, 2+idx, 0, 1, 2)

        btn_browse.clicked.connect(self._browse)
        scroll.setWidget(pane); outer.addWidget(scroll)
        self._update_footer = QtWidgets.QLabel(""); outer.addWidget(self._update_footer)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.le_root.text() or os.getcwd())
        if d: self.le_root.setText(d)

    def _apply_one(self, idx: int, cb_backend, le_ident, sb_fps):
        backend = "OpenCV" if cb_backend.currentIndex() == 0 else "PySpin"
        ident = le_ident.text().strip()  # can be blank → PySpin auto-first
        fps = int(sb_fps.value())
        if idx == 0:
            self.cam0.set_backend_ident(backend, ident); self.cam0.set_target_fps(fps)
        else:
            self.cam1.set_backend_ident(backend, ident); self.cam1.set_target_fps(fps)

    def _detect_cams(self):
        msgs: List[str] = []
        # PySpin
        if HAVE_PYSPIN:
            try:
                sys_obj = _spin_system_acquire()
                cams = sys_obj.GetCameras() if sys_obj else None
                n = cams.GetSize() if cams else 0
                msgs.append(f"PySpin cameras: {n}")
                for i in range(n):
                    c = cams.GetByIndex(i)
                    try:
                        sn = c.TLDevice.DeviceSerialNumber.ToString()
                        model = c.TLDevice.DeviceModelName.ToString()
                        tl = c.TLDevice.TLType.ToString()
                        msgs.append(f"  idx={i} sn={sn} model={model} TL={tl}")
                    except Exception:
                        msgs.append(f"  idx={i} (details unavailable)")
                if cams: cams.Clear()
            except Exception as e:
                msgs.append(f"PySpin enumeration error: {e}")
            finally:
                _spin_system_release()
        else:
            msgs.append("PySpin not available (install SDK wheel in the venv).")

        # OpenCV quick probe (0..3)
        if HAVE_OPENCV:
            avail = []
            for i in range(4):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY)
                    ok = cap.isOpened()
                    if ok: avail.append(i)
                    if cap: cap.release()
                except Exception:
                    pass
            msgs.append(f"OpenCV indices available: {avail if avail else 'none'}")

        QtWidgets.QMessageBox.information(self, "Camera detection", "\n".join(msgs))

    def set_preview_image(self, cam_idx: int, img_rgb: np.ndarray):
        if img_rgb is None: return
        h, w, _ = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.cam_boxes[cam_idx]["preview"].setPixmap(pix)

    def update_cam_label(self, idx: int, fps: float, maxfps: Optional[float]):
        max_txt = f"{maxfps:.0f}" if maxfps else "?"
        self.cam_boxes[idx]["lbl_rep"].setText(f"Driver-reported FPS: ~{fps:.1f}  |  Max: {max_txt}")

# --------- Main App ---------

class _CameraToolbar(QtWidgets.QWidget):
    def __init__(self, app, cam0, cam1, parent=None):
        super().__init__(parent)
        self.app = app
        self.cam0 = cam0
        self.cam1 = cam1
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.btn_refresh = QtWidgets.QPushButton("Refresh Cameras")
        self.btn_prev0 = QtWidgets.QPushButton("Preview A")
        self.btn_prev1 = QtWidgets.QPushButton("Preview B")
        self.btn_probe = QtWidgets.QPushButton("Probe Max FPS")
        lay.addWidget(self.btn_refresh); lay.addWidget(self.btn_prev0); lay.addWidget(self.btn_prev1); lay.addWidget(self.btn_probe)
        self.btn_refresh.clicked.connect(self.app.refresh_cameras)
        self.btn_prev0.clicked.connect(lambda: self._open_prev(self.cam0, "Camera A"))
        self.btn_prev1.clicked.connect(lambda: self._open_prev(self.cam1, "Camera B"))
        self.btn_probe.clicked.connect(self.app.probe_max_fps)

    def _open_prev(self, cam, title):
        dlg = PopoutPreview(title, frame_source=cam.grab_preview)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dlg.resize(640, 480)
        dlg.show()
        # Keep a reference
        if not hasattr(self.app, "_previews"):
            self.app._previews = []
        self.app._previews.append(dlg)
class MainApp(QtWidgets.QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.cfg = Config()
        try:
            import argparse
            ap = argparse.ArgumentParser(add_help=False)
            ap.add_argument("--simulate", action="store_true")
            ap.add_argument("--prewarm-stim", action="store_true", dest="prewarm_stim")
            ns, _ = ap.parse_known_args(argv[1:])
            if ns.simulate: self.cfg.simulation_mode = True
            if getattr(ns, 'prewarm_stim', False): self.cfg.prewarm_stim = True
        except Exception: pass

        self.hw = HardwareBridge(self.cfg)
        self.cam0 = CameraNode("cam0", self.cfg.cam0_backend, self.cfg.cam0_id, self.cfg.cam0_target_fps)
        self.cam1 = CameraNode("cam1", self.cfg.cam1_backend, self.cfg.cam1_id, self.cfg.cam1_target_fps)

        ensure_dir(self.cfg.output_root)
        log_path = os.path.join(self.cfg.output_root, "trials_log.csv")
        self.runner = TrialRunner(self.cfg, self.hw, self.cam0, self.cam1, log_path)

        self.gui = SettingsGUI(self.cfg, self.cam0, self.cam1)
        self.gui.start_experiment.connect(self.start_loop)
        self.gui.stop_experiment.connect(self.stop_loop)
        self.gui.apply_settings.connect(self.apply_from_gui)
        self.gui.manual_trigger.connect(self.trigger_once)

        self.show_scaled_gui(self.cfg.gui_screen_index)

        self.running = False; self.in_trial = False; self.thread = None
        self.preview_timer = QtCore.QTimer(self); self.preview_timer.setInterval(500)
        self.preview_timer.timeout.connect(self.update_previews); self.preview_timer.start()

        self.aboutToQuit.connect(self.cleanup); atexit.register(self.cleanup)

        if self.cfg.prewarm_stim:
            QtCore.QTimer.singleShot(300, lambda: self.runner.stim.open_persistent(
                self.cfg.stim_screen_index, self.cfg.stim_fullscreen, self.cfg.stim_bg_grey))

        self._print_startup_summary()

    def _print_startup_summary(self):
        print("=== FlyPy Startup ===")
        print(f"Version: {__version__}")
        print(f"OpenCV: {'OK' if HAVE_OPENCV else 'MISSING'}")
        print(f"PsychoPy: {'will use' if _ensure_psychopy_loaded() else 'not available (OpenCV fallback)'}")
        print(f"PySpin: {'OK' if HAVE_PYSPIN else 'MISSING'}")
        print("======================")

    def show_scaled_gui(self, screen_index: int):
        screens = QtGui.QGuiApplication.screens()
        geo = (screens[screen_index].availableGeometry()
               if 0 <= screen_index < len(screens)
               else QtGui.QGuiApplication.primaryScreen().availableGeometry())
        target_w = max(980, int(geo.width() * 0.9))
        target_h = max(700, int(geo.height() * 0.9))
        target_w = min(target_w, geo.width()); target_h = min(target_h, geo.height())
        self.gui.resize(target_w, target_h)
        x = geo.x() + (geo.width() - target_w) // 2; y = geo.y() + (geo.height() - target_h) // 2
        self.gui.move(x, y); self.gui.show()

    def position_gui(self, screen_index: int):
        screens = QtGui.QGuiApplication.screens()
        if 0 <= screen_index < len(screens):
            geo = screens[screen_index].availableGeometry()
            w = min(self.gui.width(), geo.width()); h = min(self.gui.height(), geo.height())
            x = geo.x() + (geo.width() - w) // 2; y = geo.y() + (geo.height() - h) // 2
            self.gui.resize(w, h); self.gui.move(x, y); self.gui.show()
        else:
            self.gui.show()

    def apply_from_gui(self):
        prev_sim = self.cfg.simulation_mode
        self.cfg.simulation_mode = bool(self.gui.cb_sim.isChecked())
        self.cfg.sim_trigger_interval = float(self.gui.sb_sim.value())
        self.cfg.output_root = self.gui.le_root.text().strip() or self.cfg.output_root

        idx = self.gui.cb_fmt.currentIndex()
        preset_id = self.gui.cb_fmt.itemData(idx) or self.gui._id_by_idx.get(idx, default_preset_id())
        self.cfg.video_preset_id = preset_id; self.cfg.fourcc = PRESETS_BY_ID[preset_id]["fourcc"]
        self.cfg.record_duration_s = float(self.gui.sb_rec.value())

        self.cfg.stim_duration_s = float(self.gui.sb_stim_dur.value())
        self.cfg.stim_r0_px = int(self.gui.sb_r0.value())
        self.cfg.stim_r1_px = int(self.gui.sb_r1.value())
        self.cfg.stim_bg_grey = float(self.gui.sb_bg.value())
        self.cfg.lights_delay_s = float(self.gui.sb_light_delay.value())
        self.cfg.stim_delay_s = float(self.gui.sb_stim_delay.value())
        self.cfg.stim_screen_index = int(self.gui.cb_stim_screen.currentIndex())
        self.cfg.stim_fullscreen = bool(self.gui.cb_full.isChecked())
        self.cfg.prewarm_stim = bool(self.gui.cb_prewarm.isChecked())

        self.cfg.gui_screen_index = int(self.gui.cb_gui_screen.currentIndex())
        self.position_gui(self.cfg.gui_screen_index)

        for i, node in enumerate((self.cam0, self.cam1)):
            box = self.gui.cam_boxes[i]
            backend = "OpenCV" if box["cb_backend"].currentIndex() == 0 else "PySpin"
            ident = box["le_ident"].text().strip()  # may be blank for PySpin auto
            fps = int(box["sb_fps"].value())
            node.set_backend_ident(backend, ident); node.set_target_fps(fps)
            if i == 0:
                self.cfg.cam0_backend, self.cfg.cam0_id, self.cfg.cam0_target_fps = backend, ident, fps
            else:
                self.cfg.cam1_backend, self.cfg.cam1_id, self.cfg.cam1_target_fps = backend, ident, fps

        if prev_sim != self.cfg.simulation_mode:
            if self.cfg.simulation_mode:
                self.hw.close(); self.hw.simulated = True
            else:
                self.hw.simulated = False; self.hw._opened = False; self.hw.ser = None

        ensure_dir(self.cfg.output_root)
        self.gui.lbl_status.setText("Status: Settings applied."); print("[Main] Settings applied")

        if self.cfg.prewarm_stim:
            self.runner.stim.open_persistent(self.cfg.stim_screen_index, self.cfg.stim_fullscreen, self.cfg.stim_bg_grey)
        else:
            try: self.runner.stim.close()
            except Exception: pass

    def update_previews(self):
        if self.in_trial:
            self.gui.lbl_status.setText("Status: Trial running (preview paused)")
            self.gui.update_cam_label(0, self.cam0.driver_fps(), self.cam0.max_fps_cached())
            self.gui.update_cam_label(1, self.cam1.driver_fps(), self.cam1.max_fps_cached())
            return
        p0 = self.gui.cam_boxes[0]["preview"]; p1 = self.gui.cam_boxes[1]["preview"]
        w0, h0 = p0.width(), p0.height(); w1, h1 = p1.width(), p1.height()
        img0 = self.cam0.grab_preview(w0, h0); img1 = self.cam1.grab_preview(w1, h1)
        self.gui.set_preview_image(0, img0); self.gui.set_preview_image(1, img1)
        self.gui.update_cam_label(0, self.cam0.driver_fps(), self.cam0.max_fps_cached())
        self.gui.update_cam_label(1, self.cam1.driver_fps(), self.cam1.max_fps_cached())
        self.gui.lbl_status.setText("Status: Waiting / Idle.")

    def loop(self):
        self.gui.lbl_status.setText("Status: Watching for triggers…"); print("[Main] Trigger loop started")
        try:
            while self.running:
                if self.hw.check_trigger():
                    self.in_trial = True; self.gui.lbl_status.setText("Status: Triggered — running trial…")
                    try: self.runner.run_one()
                    except Exception as e: print(f"[Main] Trial error: {e}")
                    self.in_trial = False; self.gui.lbl_status.setText("Status: Waiting / Idle.")
                QtWidgets.QApplication.processEvents(); time.sleep(0.01)
        finally:
            print("[Main] Trigger loop exiting")

    def start_loop(self):
        if self.running: return
        self.running = True; self.thread = threading.Thread(target=self.loop, daemon=True); self.thread.start()
        self.gui.lbl_status.setText("Status: Trigger loop running."); print("[Main] Start")

    def stop_loop(self):
        if not self.running: return
        self.running = False
        if self.thread: self.thread.join(); self.thread = None
        self.gui.lbl_status.setText("Status: Stopped."); print("[Main] Stop")

    def trigger_once(self):
        if self.in_trial: return
        self.in_trial = True
        try: self.runner.run_one()
        except Exception as e: print(f"[Main] Manual trial error: {e}")
        self.in_trial = False

    def cleanup(self):
        print("[Main] Cleanup…")
        try: self.hw.close()
        except Exception: pass
        for node in (self.cam0, self.cam1):
            try: node.release()
            except Exception: pass
        try: self.runner.stim.close()
        except Exception: pass
        print("[Main] Cleanup done")

def main():
    app = MainApp(sys.argv); sys.exit(app.exec_())

if __name__ == "__main__":
    main()
# ======================= end FlyAPI.py (v1.6.0) =======================
