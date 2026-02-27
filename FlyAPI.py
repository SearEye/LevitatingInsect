# FlyAPI.py
# FlyPy — Unified Trigger → Cameras + Lights + Looming Stimulus
# v1.44.1
#
# FIX (v1.44.1):
# - Fix stimulus “instant max size” + stimulus/GUI freeze when Trigger Once or beam-break trigger fires.
#   Root cause: main Qt thread was being blocked (Trigger Once ran trial inline) + heavy camera writer loops
#   could starve the stimulus update loop (GIL contention), making the first visible stimulus frame happen late.
#   Fixes:
#     1) All trials (manual + hardware-triggered) now run in a background worker thread, so the Qt GUI thread
#        stays responsive and windows can repaint.
#     2) Camera writer/grabber loops now include tiny cooperative yields to reduce GIL starvation and keep the
#        stimulus animation smooth even under load.
#
# CHANGE (v1.44.0):
# - Stimulus window recording is now RENDERED AFTER the camera clips finish writing.
#   During the trial we still PRESENT the stimulus in real-time (so behavior is unchanged),
#   but we DO NOT capture PsychoPy/OpenCV window frames while cameras are recording.
#   After camera outputs complete, we deterministically render the stimulus video from the
#   same timeline parameters (record_duration, lights_delay, stim_delay, stim_duration),
#   producing a time-accurate stimulus .AVI/.MP4 without AVI corruption from concurrent load.
#
# Keeps all prior v1.43.2 fixes and image scaling behavior.

import os, sys, time, csv, json, atexit, threading, queue, logging, shutil, importlib
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

__version__ = "1.44.1"

# -------------------- Logging --------------------
def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _now() -> str: return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def _day() -> str: return datetime.now().strftime("%Y%m%d")
LOG_DIR_DEFAULT = r"C:\\Users\\Murpheylab\\Desktop\\LevitatingInsect-main\\logs"

def _init_log():
    log_dir = LOG_DIR_DEFAULT if os.name == "nt" else os.path.join(os.getcwd(), "logs")
    _ensure_dir(log_dir)
    tmp = os.path.join(log_dir, f"FlyPy_run_{_now()}.log.tmp")
    lg = logging.getLogger("FlyPy"); lg.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(tmp, encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    if not lg.handlers: lg.addHandler(fh); lg.addHandler(ch)
    lg.info(f"=== FlyPy Log Start v{__version__} ===")
    return lg, tmp

LOGGER, _LOG_TMP = _init_log()

def _finalize_log():
    try:
        for h in list(LOGGER.handlers):
            try: h.flush(); h.close()
            except Exception: pass
            LOGGER.removeHandler(h)
    except Exception: pass
    try:
        final = _LOG_TMP.replace(".tmp", f"__ENDED_{_now()}.log")
        if os.path.exists(_LOG_TMP):
            base, ext = os.path.splitext(final); i = 1
            while os.path.exists(final): final = f"{base}_{i}{ext}"; i += 1
            shutil.move(_LOG_TMP, final)
    except Exception: pass

def _excepthook(t, v, tb):
    import traceback
    try: LOGGER.critical("UNCAUGHT:\n%s", "".join(traceback.format_exception(t, v, tb)))
    except Exception: pass
    sys.__excepthook__(t, v, tb)

sys.excepthook = _excepthook
atexit.register(_finalize_log)

try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------- Optional libs --------------------
try:
    import cv2; HAVE_OPENCV = True
except Exception as e:
    LOGGER.warning("OpenCV not available: %s", e); HAVE_OPENCV = False; cv2 = None  # type: ignore

PSY_OK = None; visual = None; core = None
def _psy_ok() -> bool:
    global PSY_OK, visual, core
    if PSY_OK is not None: return PSY_OK
    try:
        importlib.import_module("psychopy")
        visual = importlib.import_module("psychopy.visual")
        core   = importlib.import_module("psychopy.core")
        PSY_OK = True; LOGGER.info("PsychoPy OK"); return True
    except Exception:
        PSY_OK = False; return False

def _wait(s: float):
    if _psy_ok():
        try: core.wait(s); return
        except Exception: pass
    time.sleep(s)

try:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
except Exception:
    pass

# -------------------- Presets --------------------
VIDEO_PRESETS = [
    {"id": "avi_mjpg", "label": "AVI / MJPG — fast, large", "fourcc": "MJPG"},
    {"id": "avi_xvid", "label": "AVI / XVID — compatible", "fourcc": "XVID"},
    {"id": "mp4_mp4v", "label": "MP4 / mp4v", "fourcc": "mp4v"},
]
PRESETS_BY_ID = {p["id"]: p for p in VIDEO_PRESETS}

DEFAULT_STIM_PRESETS = [
    {"id": "std_050_r8_300_bg1",  "label": "Standard (0.50s r8→300 bg=1)", "stim_kind": "circle", "dur": 0.50, "r0": 8, "r1": 300, "bg": 1.0, "protected": True},
    {"id": "fast_025_r8_300_bg1", "label": "Fast (0.25s r8→300 bg=1)",     "stim_kind": "circle", "dur": 0.25, "r0": 8, "r1": 300, "bg": 1.0, "protected": True},
    {"id": "slow_100_r8_300_bg1", "label": "Slow (1.00s r8→300 bg=1)",     "stim_kind": "circle", "dur": 1.00, "r0": 8, "r1": 300, "bg": 1.0, "protected": True},
]

def _app_data_dir() -> str:
    try:
        if os.name == "nt":
            base = os.environ.get("APPDATA") or os.path.expanduser("~")
            d = os.path.join(base, "FlyPy")
        else:
            d = os.path.join(os.path.expanduser("~"), ".flypy")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return os.getcwd()

STIM_PRESET_FILE = os.path.join(_app_data_dir(), "stim_presets.json")

def load_user_stim_presets() -> List[Dict]:
    try:
        if not os.path.exists(STIM_PRESET_FILE): return []
        with open(STIM_PRESET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict): data = data.get("user_presets", [])
        if not isinstance(data, list): return []
        out = []
        for p in data:
            if not isinstance(p, dict): continue
            q = dict(p); q.pop("protected", None)
            if "label" in q and "stim_kind" in q: out.append(q)
        return out
    except Exception as e:
        LOGGER.warning("[Presets] Load user stim presets failed: %s", e)
        return []

def save_user_stim_presets(presets: List[Dict]) -> None:
    try:
        safe = []
        for p in presets:
            if not isinstance(p, dict): continue
            q = dict(p); q.pop("protected", None)
            safe.append(q)
        with open(STIM_PRESET_FILE, "w", encoding="utf-8") as f:
            json.dump({"user_presets": safe}, f, indent=2)
    except Exception as e:
        LOGGER.warning("[Presets] Save user stim presets failed: %s", e)

# -------------------- Config --------------------
class Config:
    def __init__(self):
        self.simulation_mode = False
        self.sim_trigger_interval = 5.0
        self.output_root = "FlyPy_Output"
        self.prewarm_stim = False

        self.video_preset_id = "avi_mjpg"
        self.fourcc = "MJPG"

        self.record_duration_s = 3.0
        self.record_start_timeout_s = 1.5

        self.stim_duration_s = 1.5
        self.stim_r0_px = 8
        self.stim_r1_px = 240
        self.stim_bg_grey = 1.0
        self.lights_delay_s = 0.0
        self.stim_delay_s = 0.0

        self.stim_screen_index = 0
        self.stim_fullscreen = False
        self.gui_screen_index = 0

        self.stim_kind = "circle"  # "circle" | "image" | "png"(alias)
        self.stim_png_path = ""
        self.stim_png_keep_aspect = True
        self.stim_keep_window_open = True

        self.cam0_backend = "PySpin"
        self.cam1_backend = "PySpin"
        self.cam0_id = ""
        self.cam1_id = ""
        self.cam0_target_fps = 522
        self.cam1_target_fps = 522
        self.cam0_width = 0
        self.cam1_width = 0
        self.cam0_height = 0
        self.cam1_height = 0
        self.cam0_exposure_us = 1500
        self.cam1_exposure_us = 1500
        self.cam0_hw_trigger = True
        self.cam1_hw_trigger = True

        self.cam_async_writer = True

        self.min_trigger_interval_s = 0.30
        self.token_trigger = "T"

        self.hw_no_frame_fallback = True
        self.video_write_fps_cap = 240.0

# -------------------- Hardware Bridge --------------------
class HardwareBridge:
    def __init__(self, cfg: Config, port: str = None, baud: int = 115200):
        self.cfg = cfg; self.port = port; self.baud = baud
        self.ser = None; self._opened = False; self._pyserial_ok = True
        self._last_sim = time.time(); self._last_t = 0.0
        self.simulated = bool(cfg.simulation_mode)
        try: import serial  # noqa
        except Exception as e:
            self._pyserial_ok = False; LOGGER.info("[HW] pyserial missing: %s", e)

    def _auto(self) -> Optional[str]:
        if not self._pyserial_ok: return None
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                d = (p.description or "").lower()
                if any(x in d for x in ("ch340", "uno", "elegoo", "arduino")):
                    return p.device
        except Exception:
            pass
        return None

    def _open(self):
        if self.simulated or not self._pyserial_ok or (self._opened and self.ser): return
        try:
            import serial
            if not self.port: self.port = self._auto()
            if not self.port: return
            self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
            try: self.ser.reset_input_buffer()
            except Exception: pass
            self._opened = True; LOGGER.info("[HW] Serial open: %s", self.port)
        except Exception as e:
            self.ser = None; self._opened = False
            LOGGER.warning("[HW] Serial open fail: %s", e)

    def check_trigger(self) -> bool:
        if self.simulated:
            now = time.time()
            if now - self._last_sim >= self.cfg.sim_trigger_interval:
                self._last_sim = now; return True
            return False
        if not self._pyserial_ok: return False
        self._open()
        if not self.ser: return False
        fired = False
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", "ignore").strip()
                if not line: continue
                if line == self.cfg.token_trigger: fired = True
        except Exception:
            return False
        if not fired: return False
        now = time.time()
        if now - self._last_t < self.cfg.min_trigger_interval_s: return False
        self._last_t = now; return True

    def _send(self, s: str):
        if self.simulated: return
        try:
            self._open()
            if self.ser: self.ser.write((s.strip() + "\n").encode("utf-8", "ignore"))
        except Exception:
            pass

    def mark_start(self): self._send("MARK START")
    def mark_end(self):   self._send("MARK END")
    def lights_on(self):  self._send("LIGHT ON")
    def lights_off(self): self._send("LIGHT OFF")

    def close(self):
        try:
            if self.ser: self.ser.close()
        except Exception:
            pass
        self.ser = None; self._opened = False

# -------------------- Cameras --------------------
class BaseCamera:
    def open(self): ...
    def get_frame(self): ...
    def release(self): ...
    def frame_size(self) -> Tuple[int, int]: ...
    def start_acquisition(self): ...
    def stop_acquisition(self): ...

class OpenCVCamera(BaseCamera):
    def __init__(self, index: int, fps: float):
        if not HAVE_OPENCV: raise RuntimeError("OpenCV not installed")
        self.index = index; self.fps = float(fps); self.cap = None

    def open(self):
        be = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for b in be:
            try:
                c = cv2.VideoCapture(self.index, b)
                if c and c.isOpened():
                    try:
                        c.set(cv2.CAP_PROP_FPS, self.fps)
                        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception:
                        pass
                    self.cap = c; LOGGER.info("[OpenCV] index %d opened", self.index); return
                if c: c.release()
            except Exception:
                pass
        self.cap = None

    def start_acquisition(self): pass

    def get_frame(self):
        if not self.cap: return None
        ok, f = self.cap.read()
        if not ok or f is None: return None
        if f.ndim == 2: f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        return f

    def frame_size(self):
        if self.cap:
            return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480))
        return (640, 480)

    def stop_acquisition(self): pass

    def release(self):
        try:
            if self.cap: self.cap.release()
        except Exception:
            pass
        self.cap = None

HAVE_PYSPIN = False; PySpin = None; _SPIN_SYS = None
def _spin_get():
    global PySpin, HAVE_PYSPIN, _SPIN_SYS
    if not HAVE_PYSPIN:
        try:
            import PySpin as _ps; PySpin = _ps; HAVE_PYSPIN = True
        except Exception as e:
            LOGGER.info("[PySpin] missing: %s", e); return None
    if _SPIN_SYS is None:
        _SPIN_SYS = PySpin.System.GetInstance()
    return _SPIN_SYS

def _spin_rel():
    global _SPIN_SYS
    if _SPIN_SYS is not None:
        try: _SPIN_SYS.ReleaseInstance()
        except Exception: pass
    _SPIN_SYS = None

def _spin_enum() -> List[Dict[str, str]]:
    sysi = _spin_get(); out = []
    if sysi is None: return out
    cams = sysi.GetCameras()
    try:
        n = cams.GetSize()
        for i in range(n):
            try:
                cam = cams.GetByIndex(i); d = cam.GetTLDeviceNodeMap()
                try: sn = PySpin.CStringPtr(d.GetNode("DeviceSerialNumber")).GetValue()
                except Exception:
                    try: sn = PySpin.CStringPtr(d.GetNode("DeviceID")).GetValue()
                    except Exception: sn = f"idx={i}"
                try: mdl = PySpin.CStringPtr(d.GetNode("DeviceModelName")).GetValue()
                except Exception: mdl = "Model"
                out.append({"serial": str(sn), "model": str(mdl), "display": f"PySpin {sn} — {mdl}"})
            except Exception:
                pass
    finally:
        try: cams.Clear()
        except Exception: pass
    return out

def _e(nm, val):
    try:
        node = PySpin.CEnumerationPtr(nm)
        if not PySpin.IsWritable(node): return False
        ent = node.GetEntryByName(val)
        if not PySpin.IsReadable(ent): return False
        node.SetIntValue(ent.GetValue()); return True
    except Exception:
        return False

def _f(nm, val):
    try:
        node = PySpin.CFloatPtr(nm)
        if not PySpin.IsWritable(node): return False
        lo, hi = node.GetMin(), node.GetMax()
        node.SetValue(max(lo, min(hi, float(val)))); return True
    except Exception:
        return False

def _b(nm, val: bool):
    try:
        node = PySpin.CBooleanPtr(nm)
        if not PySpin.IsWritable(node): return False
        node.SetValue(bool(val)); return True
    except Exception:
        return False

class SpinnakerCamera(BaseCamera):
    def __init__(self, serial: str, fps: float, width: int = 0, height: int = 0, exp_us: int = 1500, hw_trigger: bool = False):
        self.serial = serial.strip(); self.fps = float(fps)
        self.req_w = int(width); self.req_h = int(height); self.exp = int(exp_us)
        self.hw_trigger = bool(hw_trigger)
        self.cam = None; self.node = None; self.stream = None; self._acq = False
        self._size = (640, 480); self._mono = True

    def open(self):
        sysi = _spin_get()
        if sysi is None: raise RuntimeError("PySpin not available")
        lst = sysi.GetCameras()
        try:
            idx = 0
            if self.serial:
                for i in range(lst.GetSize()):
                    try:
                        cam = lst.GetByIndex(i); d = cam.GetTLDeviceNodeMap()
                        try: sn = PySpin.CStringPtr(d.GetNode("DeviceSerialNumber")).GetValue()
                        except Exception:
                            try: sn = PySpin.CStringPtr(d.GetNode("DeviceID")).GetValue()
                            except Exception: sn = None
                        if str(sn) == self.serial:
                            idx = i; break
                    except Exception:
                        pass

            self.cam = lst.GetByIndex(idx); self.cam.Init()
            self.node = self.cam.GetNodeMap(); self.stream = self.cam.GetTLStreamNodeMap()
            _e(self.stream.GetNode("StreamBufferHandlingMode"), "NewestOnly")
            try:
                mode = PySpin.CEnumerationPtr(self.stream.GetNode("StreamBufferCountMode"))
                if PySpin.IsWritable(mode):
                    mode.SetIntValue(mode.GetEntryByName("Manual").GetValue())
                    cnt = PySpin.CIntegerPtr(self.stream.GetNode("StreamBufferCountManual"))
                    if PySpin.IsWritable(cnt): cnt.SetValue(max(int(cnt.GetMin()), min(int(cnt.GetMax()), 192)))
            except Exception:
                pass

            if not _e(self.node.GetNode("PixelFormat"), "Mono8"):
                _e(self.node.GetNode("PixelFormat"), "BayerRG8"); self._mono = False

            try:
                w = PySpin.CIntegerPtr(self.node.GetNode("Width"))
                h = PySpin.CIntegerPtr(self.node.GetNode("Height"))
                ox = PySpin.CIntegerPtr(self.node.GetNode("OffsetX"))
                oy = PySpin.CIntegerPtr(self.node.GetNode("OffsetY"))
                maxw, maxh = int(w.GetMax()), int(h.GetMax())
                rw = maxw if self.req_w <= 0 or self.req_w > maxw else int(self.req_w // 2 * 2)
                rh = maxh if self.req_h <= 0 or self.req_h > maxh else int(self.req_h // 2 * 2)
                cx = max(0, (maxw - rw) // 4 * 2); cy = max(0, (maxh - rh) // 4 * 2)
                if PySpin.IsWritable(ox): ox.SetValue(cx)
                if PySpin.IsWritable(oy): oy.SetValue(cy)
                w.SetValue(rw); h.SetValue(rh)
                self._size = (int(rw), int(rh))
            except Exception:
                pass

            _e(self.node.GetNode("ExposureAuto"), "Off")
            if self.exp > 0:
                period_us = 1e6 / max(1.0, self.fps)
                _f(self.node.GetNode("ExposureTime"), min(self.exp, int(period_us * 0.85)))
            _e(self.node.GetNode("GainAuto"), "Off")
            _e(self.node.GetNode("AcquisitionMode"), "Continuous")
        finally:
            try: lst.Clear()
            except Exception: pass

    def _trig_on(self, on: bool):
        _e(self.node.GetNode("TriggerMode"), "On" if on else "Off")

    def _trig_src(self, src: str):
        _e(self.node.GetNode("TriggerSelector"), "FrameStart")
        _e(self.node.GetNode("TriggerSource"), src)
        _e(self.node.GetNode("TriggerActivation"), "RisingEdge")
        _e(self.node.GetNode("TriggerOverlap"), "ReadOut")

    def configure_hw(self):
        self._trig_on(False)
        _b(self.node.GetNode("AcquisitionFrameRateEnable"), False)
        self._trig_src("Line0")
        self._trig_on(True)

    def configure_free(self):
        self._trig_on(False)
        _b(self.node.GetNode("AcquisitionFrameRateEnable"), True)
        _f(self.node.GetNode("AcquisitionFrameRate"), float(self.fps))

    def _ensure(self):
        if self.cam and not self._acq:
            try:
                self.cam.BeginAcquisition()
                self._acq = True
            except Exception:
                pass

    def get_frame(self):
        if not self.cam: return None
        self._ensure()
        try:
            img = self.cam.GetNextImage(100)
            if img.IsIncomplete(): img.Release(); return None
            arr = img.GetNDArray()
            w, h = img.GetWidth(), img.GetHeight()
            img.Release()
            if arr.ndim == 2:
                if HAVE_OPENCV: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else: arr = np.repeat(arr[..., None], 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            self._size = (int(w), int(h))
            return arr
        except Exception:
            return None

    def soft_trigger_frame(self):
        if not self.cam: return None
        self._ensure()
        try:
            self._trig_on(True); self._trig_src("Software")
            cmd = PySpin.CCommandPtr(self.node.GetNode("TriggerSoftware"))
            if PySpin.IsWritable(cmd): cmd.Execute()
        except Exception:
            pass
        return self.get_frame()

    def start_acquisition(self): self._ensure()
    def frame_size(self): return self._size

    def stop_acquisition(self):
        if self.cam and self._acq:
            try: self.cam.EndAcquisition()
            except Exception: pass
            self._acq = False

    def release(self):
        try:
            self.stop_acquisition()
            if self.cam: self.cam.DeInit()
        except Exception:
            pass
        self.cam = None

# -------------------- Camera Node --------------------
class CameraNode:
    def __init__(self, name: str, backend: str, ident: str, fps: int, adv=None):
        self.name = name; self.backend = backend; self.ident = ident; self.fps = float(fps)
        self.dev: Optional[BaseCamera] = None
        self.synthetic = False
        self.tbuf = deque(maxlen=120)
        self.adv = adv or {}

    def _open(self):
        if self.dev is not None or self.synthetic: return
        try:
            if self.backend == "PySpin":
                d = SpinnakerCamera(
                    self.ident, self.fps,
                    width=int(self.adv.get("width", 0) or 0),
                    height=int(self.adv.get("height", 0) or 0),
                    exp_us=int(self.adv.get("exposure_us", 1500) or 1500),
                    hw_trigger=bool(self.adv.get("hw_trigger", False)),
                )
                d.open(); self.dev = d
                LOGGER.info("[%s] PySpin open %s @%.1ffps", self.name, self.ident or "(first)", self.fps)
            else:
                if not HAVE_OPENCV: raise RuntimeError("OpenCV not installed")
                idx = int(self.ident or "0"); d = OpenCVCamera(idx, self.fps); d.open()
                if getattr(d, "cap", None) is None:
                    self.synthetic = True; self.dev = None
                    LOGGER.info("[%s] OpenCV %d not avail → synthetic", self.name, idx)
                else:
                    self.dev = d; LOGGER.info("[%s] OpenCV open %d", self.name, idx)
        except Exception as e:
            LOGGER.warning("[%s] open fail: %s → synthetic", self.name, e)
            self.dev = None; self.synthetic = True

    def set_backend_ident(self, backend: str, ident: str, adv=None):
        self.release()
        self.backend = backend; self.ident = ident; self.synthetic = False
        if adv is not None: self.adv = adv

    def set_target_fps(self, fps: int):
        self.fps = float(fps)

    def grab_preview(self, w: int, h: int):
        self._open()
        if self.synthetic or self.dev is None:
            frm = np.full((max(h, 1), max(w, 1), 3), 255, np.uint8)
            if HAVE_OPENCV: cv2.putText(frm, f"{self.name} (synthetic)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            self.tbuf.append(time.time()); return frm
        if isinstance(self.dev, SpinnakerCamera) and bool(self.adv.get("hw_trigger", False)):
            img = self.dev.soft_trigger_frame()
        else:
            img = self.dev.get_frame()
        if img is None:
            frm = np.full((max(h, 1), max(w, 1), 3), 255, np.uint8)
            if HAVE_OPENCV: cv2.putText(frm, f"{self.name} [drop]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            frm = cv2.resize(img, (w, h)) if HAVE_OPENCV and (img.shape[1] != w or img.shape[0] != h) else img
        self.tbuf.append(time.time()); return frm

    def driver_fps(self) -> float:
        if len(self.tbuf) < 2: return 0.0
        dt = self.tbuf[-1] - self.tbuf[0]; n = len(self.tbuf) - 1
        return (n / dt) if dt > 0 else 0.0

    # -------------------- REAL-TIME RECORDING FIX --------------------
    def record_clip(
        self,
        out_path: str,
        duration_s: float,
        fourcc_str: str,
        async_writer: bool = True,
        start_evt: Optional[threading.Event] = None,
        force_soft: bool = False,
    ) -> str:
        self._open()

        if not HAVE_OPENCV:
            _ensure_dir(os.path.dirname(out_path) or ".")
            with open(out_path + ".txt", "w", encoding="utf-8") as f:
                f.write(f"{self.name} placeholder (OpenCV missing)\n")
            if start_evt: start_evt.set()
            return out_path + ".txt"

        if self.synthetic or self.dev is None:
            w, h = (640, 480)
        else:
            w, h = self.dev.frame_size()

        try: os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        except Exception: pass

        ext = os.path.splitext(out_path)[1].lower()
        if not ext:
            out_path += ".avi" if fourcc_str.upper() in ("MJPG", "XVID", "DIVX") else ".mp4"

        if isinstance(self.dev, SpinnakerCamera):
            try:
                if force_soft:
                    self.dev.configure_free()
                else:
                    if bool(self.adv.get("hw_trigger", False)):
                        self.dev.configure_hw()
                    else:
                        self.dev.configure_free()
                self.dev.start_acquisition()
            except Exception:
                pass
        else:
            try:
                if self.dev is not None: self.dev.start_acquisition()
            except Exception:
                pass

        lock = threading.Lock()
        latest = {"frame": None, "ts": None, "got_any": False, "stop": False}

        def grabber():
            hw = bool(self.adv.get("hw_trigger", False))
            grace_s = 0.35
            t_start = time.perf_counter()
            did_fallback = False

            while True:
                if latest["stop"]:
                    break

                frame = None
                if self.synthetic or self.dev is None:
                    frame = np.full((h, w, 3), 255, np.uint8)
                    if not latest["got_any"]:
                        if HAVE_OPENCV: cv2.putText(frame, f"{self.name} (synthetic)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    if isinstance(self.dev, SpinnakerCamera):
                        frame = self.dev.get_frame()
                        if (frame is None) and hw and (not force_soft) and (not did_fallback):
                            if (time.perf_counter() - t_start) >= grace_s:
                                try:
                                    self.dev.configure_free()
                                    did_fallback = True
                                    LOGGER.warning("[%s] HW trigger produced no frames; falling back to FREE-RUN for this clip.", self.name)
                                except Exception:
                                    pass
                        if frame is None and hw and (not force_soft):
                            try:
                                frame = self.dev.soft_trigger_frame()
                            except Exception:
                                frame = None
                    else:
                        frame = self.dev.get_frame()

                if frame is None:
                    time.sleep(0.001)
                    continue

                if frame.shape[1] != w or frame.shape[0] != h:
                    try:
                        frame = cv2.resize(frame, (w, h))
                    except Exception:
                        frame = np.full((h, w, 3), 255, np.uint8)

                with lock:
                    latest["frame"] = frame
                    latest["ts"] = time.perf_counter()
                    latest["got_any"] = True

                # v1.44.1: cooperative yield to reduce starving other timing loops (stimulus)
                time.sleep(0)

        tgrab = threading.Thread(target=grabber, daemon=True)
        tgrab.start()

        start_deadline = time.perf_counter() + 1.5
        while (time.perf_counter() < start_deadline):
            with lock:
                if latest["got_any"]:
                    break
            time.sleep(0.005)

        if start_evt: start_evt.set()

        out_fps = float(max(1.0, min(float(self.fps), 240.0)))

        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h), True)
            if not writer or not writer.isOpened():
                base = os.path.splitext(out_path)[0]
                out_path = base + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h), True)
        except Exception as e:
            LOGGER.error("[%s] VideoWriter open failed: %s", self.name, e)
            latest["stop"] = True
            try: tgrab.join(timeout=0.5)
            except Exception: pass
            with open(out_path + ".txt", "w", encoding="utf-8") as f:
                f.write(f"{self.name} writer failed\n")
            return out_path + ".txt"

        t0 = time.perf_counter()
        t_end = t0 + float(max(0.01, duration_s))
        period = 1.0 / out_fps
        next_tick = t0

        frames_written = 0
        last = np.full((h, w, 3), 255, np.uint8)

        while True:
            now = time.perf_counter()
            if now >= t_end:
                break
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue

            with lock:
                frm = latest["frame"]
            if frm is None:
                frm = last
            else:
                last = frm

            try:
                writer.write(frm)
                frames_written += 1
            except Exception:
                pass

            # v1.44.1: cooperative yield every few frames (keeps stimulus smooth under load)
            if (frames_written & 0x3) == 0:
                time.sleep(0)

            next_tick += period
            if next_tick < now - (2 * period):
                next_tick = now + period

        latest["stop"] = True
        try: tgrab.join(timeout=1.0)
        except Exception: pass

        try: writer.release()
        except Exception: pass

        if isinstance(self.dev, SpinnakerCamera):
            try: self.dev.stop_acquisition()
            except Exception: pass
        else:
            try:
                if self.dev is not None: self.dev.stop_acquisition()
            except Exception:
                pass

        if frames_written < 2:
            LOGGER.warning("[%s] Wrote only %d frames; appending 2 frames for playability.", self.name, frames_written)
            try:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), out_fps, (w, h), True)
                if writer and writer.isOpened():
                    for _ in range(2):
                        writer.write(last)
                writer.release()
            except Exception:
                pass

        LOGGER.info("[%s] Saved time-accurate clip: %s (%.2fs @ %.1f fps, frames=%d)",
                    self.name, out_path, float(duration_s), out_fps, frames_written)
        return out_path

    def probe_max_fps(self, seconds: float = 3.0) -> Tuple[float, int, int]:
        self._open(); frames = 0; drops = 0; t0 = time.time()
        try:
            if self.dev is not None: self.dev.start_acquisition()
        except Exception: pass
        while (time.time() - t0) < seconds:
            img = None if (self.synthetic or self.dev is None) else self.dev.get_frame()
            if img is None: drops += 1
            else: frames += 1
        try:
            if self.dev is not None: self.dev.stop_acquisition()
        except Exception: pass
        el = max(1e-6, time.time() - t0)
        return (frames / el, frames, drops)

    def release(self):
        try:
            if self.dev: self.dev.release()
        except Exception: pass
        self.dev = None; self.synthetic = False

# -------------------- Stimulus --------------------
class LoomingStim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._pp_win = None; self._pp_cfg = None
        self._pp_img = None; self._pp_img_path = ""; self._pp_img_native = (0, 0)
        self._cv_name = "Looming Stim"; self._cv_open = False
        self._cv_size = (800, 600); self._cv_img = None; self._cv_img_path = ""

    @staticmethod
    def _ease_cubic(k: float) -> float:
        k = 0.0 if k < 0 else (1.0 if k > 1 else k)
        return k * k * k

    def _pp_window(self, screen: int, fullscreen: bool, bg: float):
        need = False
        if self._pp_win is None: need = True
        elif self._pp_cfg != (screen, fullscreen):
            try: self._pp_win.close()
            except Exception: pass
            self._pp_win = None; need = True
        if need:
            try:
                if fullscreen:
                    self._pp_win = visual.Window(color=[bg]*3, units="pix", fullscr=True, screen=screen, allowGUI=False)
                else:
                    self._pp_win = visual.Window(size=self._cv_size, color=[bg]*3, units="pix", fullscr=False, screen=screen, allowGUI=True)
                self._pp_cfg = (screen, fullscreen)
                self._pp_img = None
                self._pp_img_path = ""
                self._pp_img_native = (0, 0)
            except Exception as e:
                LOGGER.warning("[Stim] PsychoPy window: %s", e); self._pp_win = None
        if self._pp_win is not None:
            try: self._pp_win.color = [bg]*3
            except Exception: pass

    def _cv_window(self, screen: int, bg: float, fullscreen: bool):
        if not HAVE_OPENCV: return
        try:
            if not self._cv_open:
                cv2.namedWindow(self._cv_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._cv_name, *self._cv_size)
                self._cv_open = True
            scrs = QtGui.QGuiApplication.screens()
            if 0 <= screen < len(scrs):
                g = scrs[screen].geometry(); cv2.moveWindow(self._cv_name, g.x(), g.y())
            try:
                cv2.setWindowProperty(self._cv_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            except Exception:
                pass
            bg255 = int(max(0, min(255, int(bg * 255))))
            frm = np.full((self._cv_size[1], self._cv_size[0], 3), bg255, np.uint8)
            cv2.imshow(self._cv_name, frm); cv2.waitKey(1)
        except Exception as e:
            LOGGER.warning("[Stim] OpenCV window: %s", e); self._cv_open = False

    def open_persistent(self, screen: int, fullscreen: bool, bg: float):
        if _psy_ok():
            self._pp_window(screen, fullscreen, bg)
            if self._pp_win is not None:
                try: self._pp_win.flip()
                except Exception: pass
        else:
            self._cv_window(screen, bg, fullscreen)

    def close(self):
        try:
            if self._pp_win is not None: self._pp_win.close()
        except Exception:
            pass
        self._pp_win = None; self._pp_cfg = None
        self._pp_img = None; self._pp_img_path = ""; self._pp_img_native = (0, 0)
        if self._cv_open and HAVE_OPENCV:
            try: cv2.destroyWindow(self._cv_name)
            except Exception: pass
            self._cv_open = False
        self._cv_img = None; self._cv_img_path = ""

    def _pp_get_image(self, path: str):
        if not path: return None
        p = os.path.abspath(path)
        if p == self._pp_img_path and self._pp_img is not None:
            return self._pp_img
        if not os.path.exists(p):
            LOGGER.warning("[Stim] Image path does not exist: %s", p)
            self._pp_img = None; self._pp_img_path = ""; self._pp_img_native = (0, 0)
            return None
        try:
            from PIL import Image
            im = Image.open(p).convert("RGBA")
            arr = np.asarray(im)  # HxWx4 uint8
            h, w = arr.shape[:2]
            self._pp_img = visual.ImageStim(self._pp_win, image=arr, units="pix", size=(w, h), interpolate=True)
            self._pp_img_path = p
            self._pp_img_native = (w, h)
        except Exception as e:
            LOGGER.warning("[Stim] Image load (PsychoPy/PIL): %s", e)
            self._pp_img = None; self._pp_img_path = ""; self._pp_img_native = (0, 0)
        return self._pp_img

    def _cv_get_image(self, path: str):
        if not HAVE_OPENCV or not path: return None
        if path == self._cv_img_path and self._cv_img is not None: return self._cv_img
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: raise RuntimeError("cv2.imread None")
            self._cv_img = img; self._cv_img_path = path
        except Exception as e:
            LOGGER.warning("[Stim] Image load (OpenCV): %s", e); self._cv_img = None; self._cv_img_path = ""
        return self._cv_img

    def _cv_overlay(self, frame_bgr: np.ndarray, t_sec: float, stim_on: bool):
        if not HAVE_OPENCV or frame_bgr is None: return frame_bgr
        h, w = frame_bgr.shape[:2]
        margin = 12
        time_txt = f"{t_sec:0.3f}s"
        stim_txt = "STIMULUS ON" if stim_on else "STIMULUS OFF"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_time = 0.8; scale_stim = 0.8; thick = 2
        (tw, th), _ = cv2.getTextSize(time_txt, font, scale_time, thick)
        x_time = max(0, w - margin - tw); y_time = max(th + margin, h - margin)
        (sw, sh), _ = cv2.getTextSize(stim_txt, font, scale_stim, thick)
        x_stim = max(0, w - margin - sw); y_stim = max(sh + margin, y_time - th - 10)

        def draw_outlined(text, x, y, scale, color):
            cv2.putText(frame_bgr, text, (x, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

        draw_outlined(stim_txt, x_stim, y_stim, scale_stim, (0, 0, 255))
        draw_outlined(time_txt, x_time, y_time, scale_time, (255, 255, 255))
        return frame_bgr

    def _size_from_radius(self, r_px: int, nat_w: int, nat_h: int, keep_aspect: bool) -> Tuple[int, int]:
        side = max(1, int(2 * max(1, r_px)))
        if not keep_aspect or nat_w <= 0 or nat_h <= 0:
            return (side, side)
        scale = float(side) / float(max(nat_w, nat_h))
        sw = max(1, int(round(nat_w * scale)))
        sh = max(1, int(round(nat_h * scale)))
        return (sw, sh)

    # -------- offline deterministic render (no window capture) --------
    def render_timeline_video(
        self,
        total_s: float,
        stim_onset_s: float,
        stim_dur_s: float,
        r0: int, r1: int, bg: float,
        record_path: str,
        record_fourcc: str = "MJPG",
        record_fps: float = 60.0,
        record_size: Tuple[int, int] = (640, 480),
    ) -> Optional[str]:
        if not HAVE_OPENCV or not record_path:
            return None

        total_s = float(max(0.01, total_s))
        stim_onset_s = float(max(0.0, stim_onset_s))
        stim_dur_s = float(max(0.0, stim_dur_s))
        stim_end_s = stim_onset_s + stim_dur_s

        kind = (self.cfg.stim_kind or "circle").strip().lower()
        if kind == "png": kind = "image"
        is_image_kind = (kind == "image" or kind.startswith("image") or ("image" in kind) or kind in ("jpg", "jpeg", "bmp", "webp", "png"))
        path = (self.cfg.stim_png_path or "").strip()
        use_img = bool(is_image_kind and path)

        out_w, out_h = int(record_size[0]), int(record_size[1])
        rec_fps = float(max(1.0, record_fps))

        try: os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        except Exception: pass

        writer = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*str(record_fourcc))
            writer = cv2.VideoWriter(record_path, fourcc, rec_fps, (out_w, out_h), True)
            if not writer or not writer.isOpened():
                base = os.path.splitext(record_path)[0]
                record_path = base + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(record_path, fourcc, rec_fps, (out_w, out_h), True)
        except Exception as e:
            LOGGER.warning("[StimRender] VideoWriter open failed: %s", e)
            writer = None

        if writer is None:
            return None

        bg255 = int(max(0, min(255, int(bg * 255))))
        img = None
        nat_w = nat_h = 0
        if use_img:
            img = self._cv_get_image(path)
            if img is None:
                use_img = False
            else:
                nat_h, nat_w = img.shape[:2]

        n_frames = int(max(2, round(total_s * rec_fps)))
        for fi in range(n_frames):
            t = float(fi) / rec_fps
            if t > total_s:
                t = total_s
            stim_on = (t >= stim_onset_s) and (t < stim_end_s)

            frm = np.full((out_h, out_w, 3), bg255, np.uint8)

            if stim_on:
                k_lin = (t - stim_onset_s) / max(1e-6, stim_dur_s)
                k = self._ease_cubic(k_lin)
                r = int(round(r0 + (r1 - r0) * k))
                r = max(1, r)

                if use_img and img is not None:
                    sw, sh = self._size_from_radius(r, nat_w, nat_h, bool(self.cfg.stim_png_keep_aspect))
                    try:
                        scaled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                        y0 = out_h // 2 - sh // 2
                        x0 = out_w // 2 - sw // 2
                        if scaled.ndim == 3 and scaled.shape[2] == 4:
                            b, g, rch, a = cv2.split(scaled)
                            rgb = cv2.merge((b, g, rch))
                            alpha = a.astype(np.float32) / 255.0
                            y1 = max(0, y0); x1 = max(0, x0)
                            y2 = min(out_h, y0 + sh); x2 = min(out_w, x0 + sw)
                            H = max(0, y2 - y1); W = max(0, x2 - x1)
                            if H > 0 and W > 0:
                                roi = frm[y1:y2, x1:x2]
                                rgb2 = rgb[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W]
                                a2 = alpha[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W][..., None]
                                frm[y1:y2, x1:x2] = (a2 * rgb2 + (1 - a2) * roi).astype(np.uint8)
                        else:
                            y1 = max(0, y0); x1 = max(0, x0)
                            y2 = min(out_h, y0 + sh); x2 = min(out_w, x0 + sw)
                            H = max(0, y2 - y1); W = max(0, x2 - x1)
                            if H > 0 and W > 0:
                                frm[y1:y2, x1:x2] = scaled[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W]
                    except Exception:
                        pass
                else:
                    cv2.circle(frm, (out_w // 2, out_h // 2), r, (0, 0, 0), -1)

            frm = self._cv_overlay(frm, float(t), bool(stim_on))
            try:
                writer.write(frm)
            except Exception:
                pass

        try: writer.release()
        except Exception: pass

        LOGGER.info("[StimRender] Rendered stimulus AFTER cameras: %s (%.2fs @ %.1f fps, frames=%d)",
                    record_path, total_s, rec_fps, n_frames)
        return record_path

    # -------- live presentation (during trial) --------
    def present_timeline(
        self,
        trial_t0_perf: float,
        total_s: float,
        stim_onset_s: float,
        stim_dur_s: float,
        r0: int, r1: int, bg: float,
        screen: int, fullscreen: bool,
        record_path: Optional[str] = None,
        record_fourcc: str = "MJPG",
        record_fps: float = 60.0,
        record_size: Tuple[int, int] = (640, 480)
    ):
        total_s = float(max(0.01, total_s))
        stim_onset_s = float(max(0.0, stim_onset_s))
        stim_dur_s = float(max(0.0, stim_dur_s))
        stim_end_s = stim_onset_s + stim_dur_s

        kind = (self.cfg.stim_kind or "circle").strip().lower()
        if kind == "png": kind = "image"
        is_image_kind = (kind == "image" or kind.startswith("image") or ("image" in kind) or kind in ("jpg", "jpeg", "bmp", "webp", "png"))
        path = (self.cfg.stim_png_path or "").strip()
        use_img = bool(is_image_kind and path)

        writer = None
        out_w, out_h = int(record_size[0]), int(record_size[1])
        rec_fps = float(max(1.0, record_fps))
        if record_path and HAVE_OPENCV:
            try: os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
            except Exception: pass
            try:
                fourcc = cv2.VideoWriter_fourcc(*str(record_fourcc))
                writer = cv2.VideoWriter(record_path, fourcc, rec_fps, (out_w, out_h), True)
                if not writer or not writer.isOpened():
                    base = os.path.splitext(record_path)[0]
                    record_path = base + ".avi"
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(record_path, fourcc, rec_fps, (out_w, out_h), True)
            except Exception as e:
                LOGGER.warning("[StimRec] VideoWriter open failed: %s", e)
                writer = None

        if self.cfg.stim_keep_window_open:
            self.open_persistent(screen, fullscreen, bg)
        else:
            self.close()

        stim_period = 1.0 / rec_fps
        next_tick = trial_t0_perf

        if _psy_ok():
            try:
                self._pp_window(screen, fullscreen, bg)
                if self._pp_win is None:
                    raise RuntimeError("PsychoPy window unavailable")

                try: win_w, win_h = self._pp_win.size
                except Exception: win_w, win_h = (800, 600)
                margin = 16
                x_right = (win_w / 2) - margin
                y_bottom = (-win_h / 2) + margin

                time_txt = visual.TextStim(self._pp_win, text="0.000s", color="white", height=24, units="pix", pos=(x_right, y_bottom), alignText="right")
                stim_txt = visual.TextStim(self._pp_win, text="STIMULUS OFF", color="red", height=24, units="pix", pos=(x_right, y_bottom + 34), alignText="right")

                stim_img = None
                nat_w = nat_h = 0
                if use_img:
                    stim_img = self._pp_get_image(path)
                    if stim_img is None:
                        LOGGER.warning("[Stim] Image selected but failed to load; falling back to circle.")
                        use_img = False
                    else:
                        nat_w, nat_h = self._pp_img_native

                dot = None
                if not use_img:
                    dot = visual.Circle(self._pp_win, radius=max(1, r0), fillColor="black", lineColor="black")

                end_perf = trial_t0_perf + total_s
                while True:
                    now = time.perf_counter()
                    if now >= end_perf: break

                    if now < next_tick:
                        time.sleep(min(0.002, next_tick - now))
                        continue
                    next_tick += stim_period
                    if next_tick < now - (2 * stim_period):
                        next_tick = now + stim_period

                    t = now - trial_t0_perf

                    hold_after = bool(self.cfg.stim_keep_window_open)
                    stim_on = (t >= stim_onset_s) and (t < stim_end_s or (hold_after and t >= stim_end_s))

                    time_txt.text = f"{t:0.3f}s"
                    stim_txt.text = "STIMULUS ON" if stim_on else "STIMULUS OFF"

                    if stim_on:
                        if hold_after and t >= stim_end_s:
                            k = 1.0
                        else:
                            k_lin = (t - stim_onset_s) / max(1e-6, stim_dur_s)
                            k = self._ease_cubic(k_lin)
                        r = int(round(r0 + (r1 - r0) * k))
                        r = max(1, r)

                        if use_img and stim_img is not None:
                            if nat_w <= 0 or nat_h <= 0:
                                try:
                                    iw, ih = stim_img.size
                                    nat_w, nat_h = int(iw), int(ih)
                                except Exception:
                                    nat_w, nat_h = (1, 1)

                            sw, sh = self._size_from_radius(r, nat_w, nat_h, bool(self.cfg.stim_png_keep_aspect))
                            stim_img.size = (sw, sh)
                            stim_img.pos = (0, 0)
                            stim_img.draw()
                        elif dot is not None:
                            dot.radius = r
                            dot.draw()

                    stim_txt.draw()
                    time_txt.draw()
                    self._pp_win.flip()

                    if writer is not None and HAVE_OPENCV:
                        try:
                            frame_rgb = None
                            try: frame_rgb = self._pp_win._getFrame(buffer="front")
                            except Exception: frame_rgb = self._pp_win._getFrame(buffer="back")
                            if frame_rgb is not None:
                                frame_bgr = frame_rgb[..., ::-1].copy()
                                if frame_bgr.dtype != np.uint8:
                                    frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
                                if (frame_bgr.shape[1], frame_bgr.shape[0]) != (out_w, out_h):
                                    frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                                frame_bgr = self._cv_overlay(frame_bgr, float(t), bool(stim_on))
                                writer.write(frame_bgr)
                        except Exception:
                            pass

                if bool(self.cfg.stim_keep_window_open):
                    try:
                        t = max(0.0, stim_end_s)
                        time_txt.text = f"{t:0.3f}s"
                        stim_txt.text = "STIMULUS ON"
                        r = max(1, int(r1))
                        if use_img and stim_img is not None:
                            if nat_w <= 0 or nat_h <= 0:
                                try:
                                    iw, ih = stim_img.size
                                    nat_w, nat_h = int(iw), int(ih)
                                except Exception:
                                    nat_w, nat_h = (1, 1)
                            sw, sh = self._size_from_radius(r, nat_w, nat_h, bool(self.cfg.stim_png_keep_aspect))
                            stim_img.size = (sw, sh)
                            stim_img.pos = (0, 0)
                            stim_img.draw()
                        elif dot is not None:
                            dot.radius = r
                            dot.draw()
                        stim_txt.draw()
                        time_txt.draw()
                        self._pp_win.flip()
                    except Exception:
                        pass

            except Exception as e:
                LOGGER.warning("[Stim] PsychoPy err → OpenCV: %s", e)
            finally:
                if writer is not None:
                    try: writer.release()
                    except Exception: pass
                if (not self.cfg.stim_keep_window_open):
                    self.close()
                return

        if not HAVE_OPENCV:
            _wait(total_s); return

        self._cv_window(screen, bg, fullscreen)
        size = self._cv_size
        bg255 = int(max(0, min(255, int(bg * 255))))
        img = None
        nat_w = nat_h = 0
        if use_img:
            img = self._cv_get_image(path)
            if img is None:
                use_img = False
            else:
                nat_h, nat_w = img.shape[:2]

        start_perf = trial_t0_perf
        end_perf = start_perf + total_s
        next_tick = start_perf
        while True:
            now = time.perf_counter()
            if now >= end_perf: break

            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += stim_period
            if next_tick < now - (2 * stim_period):
                next_tick = now + stim_period

            t = now - start_perf

            hold_after = bool(self.cfg.stim_keep_window_open)
            stim_on = (t >= stim_onset_s) and (t < stim_end_s or (hold_after and t >= stim_end_s))

            frm = np.full((size[1], size[0], 3), bg255, np.uint8)

            if stim_on:
                if hold_after and t >= stim_end_s:
                    k = 1.0
                else:
                    k_lin = (t - stim_onset_s) / max(1e-6, stim_dur_s)
                    k = self._ease_cubic(k_lin)
                r = int(round(r0 + (r1 - r0) * k))
                r = max(1, r)

                if use_img and img is not None:
                    sw, sh = self._size_from_radius(r, nat_w, nat_h, bool(self.cfg.stim_png_keep_aspect))
                    try:
                        scaled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                        y0 = size[1] // 2 - sh // 2
                        x0 = size[0] // 2 - sw // 2
                        if scaled.ndim == 3 and scaled.shape[2] == 4:
                            b, g, rch, a = cv2.split(scaled)
                            rgb = cv2.merge((b, g, rch))
                            alpha = a.astype(np.float32) / 255.0
                            y1 = max(0, y0); x1 = max(0, x0)
                            y2 = min(size[1], y0 + sh); x2 = min(size[0], x0 + sw)
                            H = max(0, y2 - y1); W = max(0, x2 - x1)
                            if H > 0 and W > 0:
                                roi = frm[y1:y2, x1:x2]
                                rgb2 = rgb[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W]
                                a2 = alpha[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W][..., None]
                                frm[y1:y2, x1:x2] = (a2 * rgb2 + (1 - a2) * roi).astype(np.uint8)
                        else:
                            y1 = max(0, y0); x1 = max(0, x0)
                            y2 = min(size[1], y0 + sh); x2 = min(size[0], x0 + sw)
                            H = max(0, y2 - y1); W = max(0, x2 - x1)
                            if H > 0 and W > 0:
                                frm[y1:y2, x1:x2] = scaled[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W]
                    except Exception:
                        pass
                else:
                    cv2.circle(frm, (size[0] // 2, size[1] // 2), r, (0, 0, 0), -1)

            frm = self._cv_overlay(frm, float(t), bool(stim_on))
            cv2.imshow(self._cv_name, frm)
            if cv2.waitKey(1) & 0xFF == 27: break

            if writer is not None:
                try:
                    rec = frm
                    if (rec.shape[1], rec.shape[0]) != (out_w, out_h):
                        rec = cv2.resize(rec, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    writer.write(rec)
                except Exception:
                    pass

        if bool(self.cfg.stim_keep_window_open) and HAVE_OPENCV and self._cv_open:
            try:
                frm = np.full((size[1], size[0], 3), bg255, np.uint8)
                r = max(1, int(r1))
                if use_img and img is not None:
                    sw, sh = self._size_from_radius(r, nat_w, nat_h, bool(self.cfg.stim_png_keep_aspect))
                    scaled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                    y0 = size[1] // 2 - sh // 2
                    x0 = size[0] // 2 - sw // 2
                    if scaled.ndim == 3 and scaled.shape[2] == 4:
                        b, g, rch, a = cv2.split(scaled)
                        rgb = cv2.merge((b, g, rch))
                        alpha = a.astype(np.float32) / 255.0
                        y1 = max(0, y0); x1 = max(0, x0)
                        y2 = min(size[1], y0 + sh); x2 = min(size[0], x0 + sw)
                        H = max(0, y2 - y1); W = max(0, x2 - x1)
                        if H > 0 and W > 0:
                            roi = frm[y1:y2, x1:x2]
                            rgb2 = rgb[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W]
                            a2 = alpha[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W][..., None]
                            frm[y1:y2, x1:x2] = (a2 * rgb2 + (1 - a2) * roi).astype(np.uint8)
                    else:
                        y1 = max(0, y0); x1 = max(0, x0)
                        y2 = min(size[1], y0 + sh); x2 = min(size[0], x0 + sw)
                        H = max(0, y2 - y1); W = max(0, x2 - x1)
                        if H > 0 and W > 0:
                            frm[y1:y2, x1:x2] = scaled[(y1 - y0):(y1 - y0) + H, (x1 - x0):(x1 - x0) + W]
                else:
                    cv2.circle(frm, (size[0] // 2, size[1] // 2), r, (0, 0, 0), -1)
                frm = self._cv_overlay(frm, float(max(0.0, stim_end_s)), True)
                cv2.imshow(self._cv_name, frm)
                cv2.waitKey(1)
            except Exception:
                pass

        if writer is not None:
            try: writer.release()
            except Exception: pass
        if not self.cfg.stim_keep_window_open:
            self.close()

    def run(self, dur_s: float, r0: int, r1: int, bg: float, screen: int, fullscreen: bool):
        t0 = time.perf_counter()
        self.present_timeline(
            trial_t0_perf=t0,
            total_s=float(dur_s),
            stim_onset_s=0.0,
            stim_dur_s=float(dur_s),
            r0=int(r0), r1=int(r1), bg=float(bg),
            screen=int(screen), fullscreen=bool(fullscreen),
            record_path=None
        )

# -------------------- Trial Runner --------------------
def _day_folder(root: str) -> str:
    p = os.path.join(root, _day()); os.makedirs(p, exist_ok=True); return p

class TrialRunner:
    def __init__(self, cfg: Config, hw: HardwareBridge, cam0: "CameraNode", cam1: "CameraNode", log_path: str):
        self.cfg = cfg; self.hw = hw; self.cam0 = cam0; self.cam1 = cam1
        self.stim = LoomingStim(cfg)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        new = not os.path.exists(log_path)
        self.log = open(log_path, "a", newline="", encoding="utf-8")
        self.csvw = csv.writer(self.log)
        if new:
            self.csvw.writerow([
                "timestamp","trial_idx","cam0_path","cam1_path","stim_path","record_duration_s",
                "lights_delay_s","stim_delay_s","stim_duration_s","stim_screen_index","stim_fullscreen",
                "stim_kind","stim_img_path","keep_aspect","keep_window",
                "cam0_backend","cam0_ident","cam0_target_fps","cam0_w","cam0_h","cam0_exp_us","cam0_hwtrig",
                "cam1_backend","cam1_ident","cam1_target_fps","cam1_w","cam1_h","cam1_exp_us","cam1_hwtrig",
                "video_preset_id","fourcc"
            ])
        self.trial_idx = 0

    def close(self):
        try: self.log.close()
        except Exception: pass
        try: self.stim.close()
        except Exception: pass

    def _ext(self, fcc: str) -> str:
        return "mp4" if fcc.lower() in ("mp4v", "avc1", "h264") else "avi"

    def _folder(self) -> str:
        p = os.path.join(_day_folder(self.cfg.output_root), f"trial_{_now()}"); os.makedirs(p, exist_ok=True); return p

    def _start_recorders(self, folder: str, force_soft: bool):
        fcc = self.cfg.fourcc; ext = self._ext(fcc)
        out0 = os.path.join(folder, f"cam0.{ext}")
        out1 = os.path.join(folder, f"cam1.{ext}")
        res = {"c0": None, "c1": None}
        s0 = threading.Event(); s1 = threading.Event()

        t0 = threading.Thread(
            target=lambda: res.__setitem__("c0",
                self.cam0.record_clip(out0, float(self.cfg.record_duration_s), fcc,
                                      async_writer=self.cfg.cam_async_writer, start_evt=s0, force_soft=force_soft)),
            daemon=True
        )
        t1 = threading.Thread(
            target=lambda: res.__setitem__("c1",
                self.cam1.record_clip(out1, float(self.cfg.record_duration_s), fcc,
                                      async_writer=self.cfg.cam_async_writer, start_evt=s1, force_soft=force_soft)),
            daemon=True
        )
        t0.start(); t1.start()
        return res, (t0, t1), (s0, s1)

    def run_one(self, force_soft: bool = False):
        folder = self._folder()
        if self.cfg.stim_keep_window_open:
            self.stim.open_persistent(self.cfg.stim_screen_index, self.cfg.stim_fullscreen, self.cfg.stim_bg_grey)

        self.hw.mark_start()
        trial_t0_perf = time.perf_counter()

        res, threads, starts = self._start_recorders(folder, force_soft)

        fcc = self.cfg.fourcc
        ext = self._ext(fcc)
        stim_path = os.path.join(folder, f"stimulus.{ext}")
        res["stim"] = stim_path

        stim_onset = float(self.cfg.lights_delay_s) + float(self.cfg.stim_delay_s)

        stim_thread = threading.Thread(
            target=lambda: self.stim.present_timeline(
                trial_t0_perf=trial_t0_perf,
                total_s=float(self.cfg.record_duration_s),
                stim_onset_s=stim_onset,
                stim_dur_s=float(self.cfg.stim_duration_s),
                r0=int(self.cfg.stim_r0_px),
                r1=int(self.cfg.stim_r1_px),
                bg=float(self.cfg.stim_bg_grey),
                screen=int(self.cfg.stim_screen_index),
                fullscreen=bool(self.cfg.stim_fullscreen),
                record_path=None,   # <-- critical: no capture during camera write
            ),
            daemon=True
        )
        stim_thread.start()

        t_dead = time.time() + float(self.cfg.record_start_timeout_s)
        while time.time() < t_dead and not (starts[0].is_set() and starts[1].is_set()):
            time.sleep(0.01)

        if self.cfg.lights_delay_s > 0: _wait(self.cfg.lights_delay_s)
        self.hw.lights_on()
        if self.cfg.stim_delay_s > 0: _wait(self.cfg.stim_delay_s)

        _wait(float(self.cfg.stim_duration_s))

        self.hw.lights_off()
        self.hw.mark_end()

        for t in threads: t.join()
        try: stim_thread.join(timeout=5.0)
        except Exception: pass

        try:
            rendered = self.stim.render_timeline_video(
                total_s=float(self.cfg.record_duration_s),
                stim_onset_s=stim_onset,
                stim_dur_s=float(self.cfg.stim_duration_s),
                r0=int(self.cfg.stim_r0_px),
                r1=int(self.cfg.stim_r1_px),
                bg=float(self.cfg.stim_bg_grey),
                record_path=stim_path,
                record_fourcc=str(fcc),
                record_fps=60.0,
                record_size=(640, 480),
            )
            if rendered:
                res["stim"] = rendered
        except Exception as e:
            LOGGER.warning("[StimRender] failed: %s", e)

        self.trial_idx += 1
        self.csvw.writerow([
            _now(), self.trial_idx,
            res["c0"] or "", res["c1"] or "", res.get("stim") or "", float(self.cfg.record_duration_s),
            float(self.cfg.lights_delay_s), float(self.cfg.stim_delay_s), float(self.cfg.stim_duration_s),
            int(self.cfg.stim_screen_index), bool(self.cfg.stim_fullscreen),
            str(self.cfg.stim_kind), str(self.cfg.stim_png_path), bool(self.cfg.stim_png_keep_aspect), bool(self.cfg.stim_keep_window_open),
            self.cam0.backend, self.cam0.ident, int(self.cam0.fps), self.cam0.adv.get("width", 0), self.cam0.adv.get("height", 0), self.cam0.adv.get("exposure_us", 0), self.cam0.adv.get("hw_trigger", False),
            self.cam1.backend, self.cam1.ident, int(self.cam1.fps), self.cam1.adv.get("width", 0), self.cam1.adv.get("height", 0), self.cam1.adv.get("exposure_us", 0), self.cam1.adv.get("hw_trigger", False),
            self.cfg.video_preset_id, self.cfg.fourcc
        ])
        self.log.flush()

# -------------------- Device enumeration --------------------
def enumerate_opencv(max_index: int = 6) -> List[str]:
    if not HAVE_OPENCV: return []
    out = []
    for i in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if cap and cap.isOpened(): out.append(f"OpenCV index {i}")
        except Exception:
            pass
        finally:
            try:
                if cap: cap.release()
            except Exception:
                pass
    return out

# -------------------- GUI --------------------
class SettingsGUI(QtWidgets.QWidget):
    start_experiment = QtCore.pyqtSignal()
    stop_experiment = QtCore.pyqtSignal()
    apply_settings = QtCore.pyqtSignal()
    manual_trigger = QtCore.pyqtSignal()
    probe_requested = QtCore.pyqtSignal()
    refresh_devices_requested = QtCore.pyqtSignal()
    reset_stimulus_requested = QtCore.pyqtSignal()

    def __init__(self, cfg: Config, cam0: CameraNode, cam1: CameraNode):
        super().__init__()
        self.cfg = cfg; self.cam0 = cam0; self.cam1 = cam1
        self.setWindowTitle(f"FlyPy — v{__version__}")
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self); scroll.setWidgetResizable(True)
        pane = QtWidgets.QWidget(); root = QtWidgets.QVBoxLayout(pane)

        row0 = QtWidgets.QHBoxLayout()
        row0.addWidget(QtWidgets.QLabel("Quick Preset:"))
        self.cb_preset = QtWidgets.QComboBox()
        self.cb_preset.addItem("Blackfly 522 fps (PySpin Mono8, ROI 640×512)")
        self.cb_preset.addItem("Blackfly 300 fps (PySpin Mono8, ROI 720×540)")
        self.cb_preset.addItem("OpenCV baseline (laptop cam)")
        self.bt_apply_preset = QtWidgets.QPushButton("Apply Preset")
        self.bt_probe = QtWidgets.QPushButton("Probe Max FPS")
        self.bt_refresh = QtWidgets.QPushButton("Refresh Cameras")
        for w in (self.cb_preset, self.bt_apply_preset, self.bt_probe, self.bt_refresh): row0.addWidget(w)
        root.addLayout(row0)
        self.bt_apply_preset.clicked.connect(self._apply_preset)
        self.bt_probe.clicked.connect(lambda: self.probe_requested.emit())
        self.bt_refresh.clicked.connect(lambda: self.refresh_devices_requested.emit())

        row = QtWidgets.QHBoxLayout()
        self.bt_start = QtWidgets.QPushButton("Start")
        self.bt_stop = QtWidgets.QPushButton("Stop")
        self.bt_trig = QtWidgets.QPushButton("Trigger Once")
        self.bt_apply = QtWidgets.QPushButton("Apply Settings")
        for w in (self.bt_start, self.bt_stop, self.bt_trig, self.bt_apply): row.addWidget(w)
        root.addLayout(row)
        self.bt_start.clicked.connect(self.start_experiment.emit)
        self.bt_stop.clicked.connect(self.stop_experiment.emit)
        self.bt_trig.clicked.connect(self.manual_trigger.emit)
        self.bt_apply.clicked.connect(self.apply_settings.emit)

        self.lbl_status = QtWidgets.QLabel("Status: Idle.")
        root.addWidget(self.lbl_status)

        grid = QtWidgets.QGridLayout()
        root.addLayout(grid)

        gen = QtWidgets.QGroupBox("General"); gl = QtWidgets.QFormLayout(gen)
        self.cb_sim = QtWidgets.QCheckBox("Test/Simulation Mode"); self.cb_sim.setChecked(self.cfg.simulation_mode); gl.addRow(self.cb_sim)
        self.sb_sim = QtWidgets.QDoubleSpinBox(); self.sb_sim.setRange(0.1, 3600.0); self.sb_sim.setDecimals(2); self.sb_sim.setValue(self.cfg.sim_trigger_interval); gl.addRow("Simulated trigger interval (s):", self.sb_sim)
        self.le_root = QtWidgets.QLineEdit(self.cfg.output_root); btn_browse = QtWidgets.QPushButton("Browse…")
        rowr = QtWidgets.QHBoxLayout(); rowr.addWidget(self.le_root); rowr.addWidget(btn_browse); gl.addRow("Output folder:", rowr)

        self.cb_fmt = QtWidgets.QComboBox(); self._id_by_idx = {}; cur = 0
        for i, p in enumerate(VIDEO_PRESETS):
            self.cb_fmt.addItem(p["label"]); self.cb_fmt.setItemData(i, p["id"]); self._id_by_idx[i] = p["id"]
            if p["id"] == self.cfg.video_preset_id: cur = i
        self.cb_fmt.setCurrentIndex(cur); gl.addRow("Video format / codec:", self.cb_fmt)

        self.sb_rec = QtWidgets.QDoubleSpinBox(); self.sb_rec.setRange(0.1, 600.0); self.sb_rec.setDecimals(2); self.sb_rec.setValue(self.cfg.record_duration_s); gl.addRow("Recording duration (s):", self.sb_rec)
        grid.addWidget(gen, 0, 0, 1, 2)

        btn_browse.clicked.connect(lambda: self._browse())

        stim = QtWidgets.QGroupBox("Stimulus & Timing"); sl = QtWidgets.QFormLayout(stim)
        self.sb_stim_dur = QtWidgets.QDoubleSpinBox(); self.sb_stim_dur.setRange(0.05, 60.0); self.sb_stim_dur.setDecimals(3); self.sb_stim_dur.setValue(self.cfg.stim_duration_s)
        self.sb_r0 = QtWidgets.QSpinBox(); self.sb_r0.setRange(1, 4000); self.sb_r0.setValue(self.cfg.stim_r0_px)
        self.sb_r1 = QtWidgets.QSpinBox(); self.sb_r1.setRange(1, 8000); self.sb_r1.setValue(self.cfg.stim_r1_px)
        self.sb_bg = QtWidgets.QDoubleSpinBox(); self.sb_bg.setRange(0.0, 1.0); self.sb_bg.setSingleStep(0.05); self.sb_bg.setValue(self.cfg.stim_bg_grey)
        self.sb_light_delay = QtWidgets.QDoubleSpinBox(); self.sb_light_delay.setRange(0.0, 10.0); self.sb_light_delay.setDecimals(3); self.sb_light_delay.setValue(self.cfg.lights_delay_s)
        self.sb_stim_delay = QtWidgets.QDoubleSpinBox(); self.sb_stim_delay.setRange(0.0, 10.0); self.sb_stim_delay.setDecimals(3); self.sb_stim_delay.setValue(self.cfg.stim_delay_s)

        self.cb_stim_kind = QtWidgets.QComboBox()
        self.cb_stim_kind.addItem("Circle", "circle")
        self.cb_stim_kind.addItem("Image (PNG/JPG)", "image")
        self.cb_stim_kind.setCurrentIndex(0 if (self.cfg.stim_kind or "circle") != "image" else 1)

        self.le_stim_png = QtWidgets.QLineEdit(self.cfg.stim_png_path or "")
        self.le_stim_png.setPlaceholderText("Path to image (png/jpg/jpeg/bmp/webp)")
        self.bt_stim_png = QtWidgets.QPushButton("Browse…")
        row_png = QtWidgets.QHBoxLayout(); row_png.addWidget(self.le_stim_png); row_png.addWidget(self.bt_stim_png)
        self.cb_png_aspect = QtWidgets.QCheckBox("Image: keep aspect"); self.cb_png_aspect.setChecked(bool(self.cfg.stim_png_keep_aspect))
        self.cb_keep_open = QtWidgets.QCheckBox("Keep stimulus window open"); self.cb_keep_open.setChecked(bool(self.cfg.stim_keep_window_open))

        self.cb_stim_preset = QtWidgets.QComboBox()
        self.bt_preset_save = QtWidgets.QPushButton("Save Preset…")
        self.bt_preset_delete = QtWidgets.QPushButton("Delete Preset")
        row_p = QtWidgets.QHBoxLayout(); row_p.addWidget(self.cb_stim_preset, 1); row_p.addWidget(self.bt_preset_save); row_p.addWidget(self.bt_preset_delete)
        w_p = QtWidgets.QWidget(); w_p.setLayout(row_p)

        self._user_stim_presets = load_user_stim_presets()

        def _refresh_presets(select_label: str = None):
            cur_lbl = select_label or (self.cb_stim_preset.currentText() if self.cb_stim_preset.count() else None)
            self.cb_stim_preset.blockSignals(True)
            self.cb_stim_preset.clear()
            combined = []
            combined.extend(DEFAULT_STIM_PRESETS)
            for p in self._user_stim_presets:
                q = dict(p); q["protected"] = False
                combined.append(q)
            self._stim_presets = combined
            for p in combined:
                self.cb_stim_preset.addItem(p.get("label", "(unnamed)"), p)
            if cur_lbl:
                for i in range(self.cb_stim_preset.count()):
                    if self.cb_stim_preset.itemText(i) == cur_lbl:
                        self.cb_stim_preset.setCurrentIndex(i); break
            self.cb_stim_preset.blockSignals(False)

        def _apply_preset():
            p = self.cb_stim_preset.currentData()
            if not isinstance(p, dict): return
            kind = str(p.get("stim_kind", "circle"))
            self.cb_stim_kind.setCurrentIndex(1 if kind == "image" else 0)
            if "dur" in p: self.sb_stim_dur.setValue(float(p["dur"]))
            if "r0" in p: self.sb_r0.setValue(int(p["r0"]))
            if "r1" in p: self.sb_r1.setValue(int(p["r1"]))
            if "bg" in p: self.sb_bg.setValue(float(p["bg"]))
            if "png_path" in p: self.le_stim_png.setText(str(p.get("png_path") or ""))
            if "keep_aspect" in p: self.cb_png_aspect.setChecked(bool(p.get("keep_aspect")))
            if "keep_open" in p: self.cb_keep_open.setChecked(bool(p.get("keep_open")))
            on = (self.cb_stim_kind.currentData() == "image")
            self.le_stim_png.setEnabled(on); self.bt_stim_png.setEnabled(on); self.cb_png_aspect.setEnabled(on)

        def _current_preset_dict(label: str) -> Dict:
            kind = str(self.cb_stim_kind.currentData() or "circle")
            return {
                "id": f"user_{int(time.time())}",
                "label": label,
                "stim_kind": ("image" if kind != "circle" else "circle"),
                "dur": float(self.sb_stim_dur.value()),
                "r0": int(self.sb_r0.value()),
                "r1": int(self.sb_r1.value()),
                "bg": float(self.sb_bg.value()),
                "png_path": str(self.le_stim_png.text() or "").strip(),
                "keep_aspect": bool(self.cb_png_aspect.isChecked()),
                "keep_open": bool(self.cb_keep_open.isChecked()),
            }

        def _save_preset():
            name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset name:")
            if not ok: return
            name = str(name or "").strip()
            if not name: return
            self._user_stim_presets.append(_current_preset_dict(name))
            save_user_stim_presets(self._user_stim_presets)
            _refresh_presets(select_label=name)

        def _delete_preset():
            p = self.cb_stim_preset.currentData()
            if not isinstance(p, dict): return
            if bool(p.get("protected")):
                QtWidgets.QMessageBox.information(self, "Protected Preset", "This preset is part of the thesis standard procedure and cannot be deleted.")
                return
            label = str(p.get("label", "")).strip()
            for i, up in enumerate(list(self._user_stim_presets)):
                if str(up.get("label", "")).strip() == label:
                    self._user_stim_presets.pop(i); break
            save_user_stim_presets(self._user_stim_presets)
            _refresh_presets()

        self.bt_preset_save.clicked.connect(_save_preset)
        self.bt_preset_delete.clicked.connect(_delete_preset)
        _refresh_presets()
        self.cb_stim_preset.currentIndexChanged.connect(_apply_preset)

        sl.addRow("Stimulus Preset:", w_p)
        sl.addRow("Stimulus total time (s):", self.sb_stim_dur)
        sl.addRow("Start radius (px):", self.sb_r0)
        sl.addRow("End radius (px):", self.sb_r1)
        sl.addRow("Background shade (0=black,1=white):", self.sb_bg)
        sl.addRow("Delay: record → lights ON (s):", self.sb_light_delay)
        sl.addRow("Delay: record → stimulus ON (s):", self.sb_stim_delay)
        sl.addRow("Stimulus Type:", self.cb_stim_kind)
        w_png = QtWidgets.QWidget(); w_png.setLayout(row_png); sl.addRow("Image path:", w_png)
        sl.addRow(self.cb_png_aspect); sl.addRow(self.cb_keep_open)
        grid.addWidget(stim, 1, 0, 1, 2)

        def _browse_png():
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", self.le_stim_png.text() or os.getcwd(),
                                                        "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
            if p: self.le_stim_png.setText(p)

        self.bt_stim_png.clicked.connect(_browse_png)

        def _toggle_img():
            on = (self.cb_stim_kind.currentData() == "image")
            self.le_stim_png.setEnabled(on); self.bt_stim_png.setEnabled(on); self.cb_png_aspect.setEnabled(on)
        self.cb_stim_kind.currentIndexChanged.connect(_toggle_img); _toggle_img()

        disp = QtWidgets.QGroupBox("Display & Windows"); dl = QtWidgets.QFormLayout(disp)
        self.cb_stim_screen = QtWidgets.QComboBox()
        self.cb_gui_screen = QtWidgets.QComboBox()
        self.bt_refresh_displays = QtWidgets.QPushButton("Refresh Displays")
        self.bt_reset_stim = QtWidgets.QPushButton("Open/Reset Stimulus Window")
        row_disp = QtWidgets.QHBoxLayout()
        row_disp.addWidget(self.cb_stim_screen); row_disp.addWidget(self.bt_refresh_displays); row_disp.addWidget(self.bt_reset_stim)
        self.cb_full = QtWidgets.QCheckBox("Borderless fullscreen"); self.cb_full.setChecked(self.cfg.stim_fullscreen)
        self.cb_prewarm = QtWidgets.QCheckBox("Pre-warm stimulus window"); self.cb_prewarm.setChecked(self.cfg.prewarm_stim)
        dl.addRow("Stimulus display screen:", row_disp)
        dl.addRow("GUI display screen:", self.cb_gui_screen)
        dl.addRow(self.cb_full); dl.addRow(self.cb_prewarm)
        grid.addWidget(disp, 2, 0, 1, 2)

        def _populate_screens():
            self.cb_stim_screen.clear(); self.cb_gui_screen.clear()
            scrs = QtGui.QGuiApplication.screens(); prim = QtGui.QGuiApplication.primaryScreen()
            for i, s in enumerate(scrs):
                g = s.geometry()
                try: name = s.name()
                except Exception: name = f"Screen {i}"
                label = f"{i}: {name}{' (Primary)' if (s is prim) else ''} — {g.width()}×{g.height()} @({g.x()},{g.y()})"
                self.cb_stim_screen.addItem(label); self.cb_gui_screen.addItem(label)
            self.cb_stim_screen.setCurrentIndex(min(self.cfg.stim_screen_index, max(0, self.cb_stim_screen.count() - 1)))
            self.cb_gui_screen.setCurrentIndex(min(self.cfg.gui_screen_index, max(0, self.cb_gui_screen.count() - 1)))

        _populate_screens()
        self.bt_refresh_displays.clicked.connect(_populate_screens)
        self.bt_reset_stim.clicked.connect(self.reset_stimulus_requested.emit)

        self.cam_boxes = []
        for idx, node in enumerate((cam0, cam1)):
            gb = QtWidgets.QGroupBox(f"Camera {idx}"); glb = QtWidgets.QGridLayout(gb)
            preview = QtWidgets.QLabel("Preview OFF"); preview.setFixedSize(360, 240)
            preview.setStyleSheet("background:#fff;border:1px solid #aaa;"); preview.setAlignment(QtCore.Qt.AlignCenter)
            cb_show = QtWidgets.QCheckBox("Show Preview"); cb_show.setChecked(False)
            glb.addWidget(preview, 0, 0, 7, 1); glb.addWidget(cb_show, 7, 0, 1, 1)

            cb_backend = QtWidgets.QComboBox(); cb_backend.addItem("OpenCV"); cb_backend.addItem("PySpin")
            cb_backend.setCurrentIndex(0 if node.backend == "OpenCV" else 1)
            glb.addWidget(QtWidgets.QLabel("Backend:"), 0, 1); glb.addWidget(cb_backend, 0, 2)

            cb_device = QtWidgets.QComboBox(); cb_device.setEditable(False); cb_device.setMinimumWidth(280)
            glb.addWidget(QtWidgets.QLabel("Device:"), 1, 1); glb.addWidget(cb_device, 1, 2)

            le_ident = QtWidgets.QLineEdit(node.ident); le_ident.setPlaceholderText("OpenCV index or PySpin serial")
            glb.addWidget(QtWidgets.QLabel("Manual index/serial:"), 2, 1); glb.addWidget(le_ident, 2, 2)

            sb_fps = QtWidgets.QSpinBox(); sb_fps.setRange(1, 10000); sb_fps.setValue(int(node.fps))
            glb.addWidget(QtWidgets.QLabel("Target FPS:"), 3, 1); glb.addWidget(sb_fps, 3, 2)

            adv_frame = QtWidgets.QFrame(); adv_layout = QtWidgets.QFormLayout(adv_frame)
            sb_w = QtWidgets.QSpinBox(); sb_w.setRange(0, 20000); sb_w.setSingleStep(2); sb_w.setValue(int(node.adv.get("width", 0) or 0))
            sb_h = QtWidgets.QSpinBox(); sb_h.setRange(0, 20000); sb_h.setSingleStep(2); sb_h.setValue(int(node.adv.get("height", 0) or 0))
            sb_exp = QtWidgets.QSpinBox(); sb_exp.setRange(20, 1000000); sb_exp.setSingleStep(50); sb_exp.setValue(int(node.adv.get("exposure_us", 1500) or 1500))
            cb_hwtrig = QtWidgets.QCheckBox("Hardware trigger (Line0)"); cb_hwtrig.setChecked(bool(node.adv.get("hw_trigger", True)))
            adv_layout.addRow("ROI Width (0=max):", sb_w); adv_layout.addRow("ROI Height (0=max):", sb_h)
            adv_layout.addRow("Exposure (µs):", sb_exp); adv_layout.addRow(cb_hwtrig)
            adv_frame.setVisible(False); btn_adv = QtWidgets.QPushButton("Advanced…")
            btn_adv.clicked.connect(lambda _=False, f=adv_frame, b=btn_adv: (f.setVisible(not f.isVisible()), b.setText("Hide Advanced" if f.isVisible() else "Advanced…")))
            glb.addWidget(btn_adv, 4, 1, 1, 2); glb.addWidget(adv_frame, 5, 1, 2, 2)

            lbl_rep = QtWidgets.QLabel("Driver-reported FPS: ~0.0"); glb.addWidget(lbl_rep, 8, 1, 1, 2)

            self.cam_boxes.append({
                "gb": gb, "preview": preview, "cb_show": cb_show, "cb_backend": cb_backend, "cb_device": cb_device,
                "le_ident": le_ident, "sb_fps": sb_fps, "sb_w": sb_w, "sb_h": sb_h, "sb_exp": sb_exp,
                "cb_hw": cb_hwtrig, "lbl_rep": lbl_rep
            })
            grid.addWidget(gb, 3 + idx, 0, 1, 2)

        scroll.setWidget(pane); outer.addWidget(scroll)
        self._update_footer = QtWidgets.QLabel("Note: Cameras are time-accurate (real seconds). Stimulus video is post-rendered to avoid corruption under load.")
        outer.addWidget(self._update_footer)

    def _apply_preset(self):
        idx = self.cb_preset.currentIndex()
        for i in range(self.cb_fmt.count()):
            if (self.cb_fmt.itemData(i) or "").lower() == "avi_mjpg":
                self.cb_fmt.setCurrentIndex(i); break
        self.sb_bg.setValue(1.0); self.cb_keep_open.setChecked(True); self.cb_prewarm.setChecked(True)

        if idx in (0, 1):
            fps = 522 if idx == 0 else 300
            w, h = (640, 512) if idx == 0 else (720, 540)
            exp = 1500 if idx == 0 else 2500
            for box in self.cam_boxes:
                box["cb_backend"].setCurrentIndex(1)
                box["sb_fps"].setValue(fps)
                box["sb_w"].setValue(w); box["sb_h"].setValue(h)
                box["sb_exp"].setValue(exp)
                box["cb_hw"].setChecked(True)
        else:
            for n, box in enumerate(self.cam_boxes):
                box["cb_backend"].setCurrentIndex(0)
                box["le_ident"].setText(str(n))
                box["sb_fps"].setValue(60)
                box["sb_w"].setValue(640); box["sb_h"].setValue(480)
                box["sb_exp"].setValue(5000)
                box["cb_hw"].setChecked(False)

        self.sb_rec.setValue(2.0)
        self.apply_settings.emit()

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.le_root.text() or os.getcwd())
        if d: self.le_root.setText(d)

    def set_preview_image(self, cam_idx: int, img_bgr: Optional[np.ndarray]):
        if img_bgr is None:
            self.cam_boxes[cam_idx]["preview"].setText("Preview OFF")
            self.cam_boxes[cam_idx]["preview"].setPixmap(QtGui.QPixmap())
            return
        h, w, _ = img_bgr.shape
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if HAVE_OPENCV else img_bgr[..., ::-1].copy()
        qimg = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.cam_boxes[cam_idx]["preview"].setPixmap(pix)

    def update_cam_fps_labels(self, f0: float, f1: float):
        self.cam_boxes[0]["lbl_rep"].setText(f"Driver-reported FPS: ~{f0:.1f}")
        self.cam_boxes[1]["lbl_rep"].setText(f"Driver-reported FPS: ~{f1:.1f}")

# -------------------- Main App --------------------
class MainApp(QtWidgets.QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.cfg = Config()
        try:
            import argparse
            ap = argparse.ArgumentParser(add_help=False)
            ap.add_argument("--simulate", action="store_true")
            ap.add_argument("--prewarm-stim", dest="prewarm_stim", action="store_true")
            ns, _ = ap.parse_known_args(argv[1:])
            if ns.simulate: self.cfg.simulation_mode = True
            if getattr(ns, "prewarm_stim", False): self.cfg.prewarm_stim = True
        except Exception:
            pass

        self.hw = HardwareBridge(self.cfg)
        self.cam0 = CameraNode("cam0", self.cfg.cam0_backend, self.cfg.cam0_id, self.cfg.cam0_target_fps,
                               adv={"width": self.cfg.cam0_width, "height": self.cfg.cam0_height, "exposure_us": self.cfg.cam0_exposure_us, "hw_trigger": self.cfg.cam0_hw_trigger})
        self.cam1 = CameraNode("cam1", self.cfg.cam1_backend, self.cfg.cam1_id, self.cfg.cam1_target_fps,
                               adv={"width": self.cfg.cam1_width, "height": self.cfg.cam1_height, "exposure_us": self.cfg.cam1_exposure_us, "hw_trigger": self.cfg.cam1_hw_trigger})

        os.makedirs(self.cfg.output_root, exist_ok=True)
        log_path = os.path.join(self.cfg.output_root, "trials_log.csv")
        self.runner = TrialRunner(self.cfg, self.hw, self.cam0, self.cam1, log_path)

        self.gui = SettingsGUI(self.cfg, self.cam0, self.cam1)
        self.gui.start_experiment.connect(self.start_loop)
        self.gui.stop_experiment.connect(self.stop_loop)
        self.gui.apply_settings.connect(self.apply_settings_from_gui)
        self.gui.manual_trigger.connect(self.trigger_once)
        self.gui.probe_requested.connect(self.start_probe)
        self.gui.refresh_devices_requested.connect(self.refresh_devices)
        self.gui.reset_stimulus_requested.connect(self.reset_stimulus_window)

        self.show_scaled_gui(self.cfg.gui_screen_index)

        self.running = False
        self.in_trial = False
        self.thread = None

        # v1.44.1: dedicated trial worker handle to avoid multiple overlapping trials
        self._trial_thread = None
        self._trial_lock = threading.Lock()

        self.preview_timer = QtCore.QTimer(self); self.preview_timer.setInterval(300)
        self.preview_timer.timeout.connect(self.update_previews); self.preview_timer.start()
        self.aboutToQuit.connect(self.cleanup)

        if self.cfg.prewarm_stim or self.cfg.stim_keep_window_open:
            QtCore.QTimer.singleShot(300, lambda: self.runner.stim.open_persistent(self.cfg.stim_screen_index, self.cfg.stim_fullscreen, self.cfg.stim_bg_grey))

        LOGGER.info("Startup v%s | OpenCV:%s PySpin:%s", __version__, "OK" if HAVE_OPENCV else "NO", "OK" if _spin_get() else "NO")
        QtCore.QTimer.singleShot(200, self.refresh_devices)

    def _set_status(self, txt: str):
        # safe from any thread
        try:
            QtCore.QMetaObject.invokeMethod(
                self.gui.lbl_status, "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, txt)
            )
        except Exception:
            try:
                self.gui.lbl_status.setText(txt)
            except Exception:
                pass

    def _run_trial_async(self, force_soft: bool):
        # v1.44.1: run trial in background so Qt event loop keeps repainting (fixes GUI/stim freeze)
        with self._trial_lock:
            if self.in_trial:
                return
            self.in_trial = True

        def worker():
            try:
                self._set_status("Status: Triggered — running trial…")
                self.runner.run_one(force_soft=force_soft)
            except Exception as e:
                LOGGER.error("[Main] Trial error: %s", e)
            finally:
                self.in_trial = False
                self._set_status("Status: Waiting / Idle.")

        self._trial_thread = threading.Thread(target=worker, daemon=True)
        self._trial_thread.start()

    def show_scaled_gui(self, screen_idx: int):
        scrs = QtGui.QGuiApplication.screens()
        geo = scrs[screen_idx].availableGeometry() if 0 <= screen_idx < len(scrs) else QtGui.QGuiApplication.primaryScreen().availableGeometry()
        w = max(980, int(geo.width() * 0.9)); h = max(720, int(geo.height() * 0.9))
        w = min(w, geo.width()); h = min(h, geo.height())
        self.gui.resize(w, h)
        self.gui.move(geo.x() + (geo.width() - w) // 2, geo.y() + (geo.height() - h) // 2)
        self.gui.show()

    def refresh_devices(self):
        try:
            ocv = enumerate_opencv(6) if HAVE_OPENCV else []
            spin = _spin_enum()
            devs = [{"backend": "PySpin", "ident": d["serial"], "label": d["display"]} for d in spin] + \
                   [{"backend": "OpenCV", "ident": s.split()[-1], "label": s} for s in ocv]

            for box in self.gui.cam_boxes:
                cb = box["cb_device"]; cb.blockSignals(True)
                cb.clear(); cb.addItem("— Select device —"); cb.model().item(0).setEnabled(False)
                for d in devs: cb.addItem(d["label"], d)
                cb.blockSignals(False)

                def on_change(i, box=box, cb=cb):
                    data = cb.itemData(i)
                    if isinstance(data, dict) and "backend" in data:
                        box["cb_backend"].setCurrentIndex(0 if data["backend"] == "OpenCV" else 1)
                        box["le_ident"].setText(str(data["ident"]))

                try: cb.currentIndexChanged.disconnect()
                except Exception: pass
                cb.currentIndexChanged.connect(on_change)

            id0 = self.gui.cam_boxes[0]["le_ident"].text().strip()
            id1 = self.gui.cam_boxes[1]["le_ident"].text().strip()
            be0 = "OpenCV" if self.gui.cam_boxes[0]["cb_backend"].currentIndex() == 0 else "PySpin"
            be1 = "OpenCV" if self.gui.cam_boxes[1]["cb_backend"].currentIndex() == 0 else "PySpin"

            def _set(box, b, i):
                box["cb_backend"].setCurrentIndex(0 if b == "OpenCV" else 1)
                box["le_ident"].setText(str(i))
                for k in range(box["cb_device"].count()):
                    data = box["cb_device"].itemData(k)
                    if isinstance(data, dict) and data.get("backend") == b and str(data.get("ident")) == str(i):
                        box["cb_device"].setCurrentIndex(k); break

            if len(devs) >= 2 and ((not id0 and not id1) or (be0 == be1 and id0 == id1)):
                _set(self.gui.cam_boxes[0], devs[0]["backend"], devs[0]["ident"])
                _set(self.gui.cam_boxes[1], devs[1]["backend"], devs[1]["ident"])
                self.apply_settings_from_gui()

        except Exception as e:
            LOGGER.error("[Devices] %s", e)

    def apply_settings_from_gui(self):
        try:
            self.cfg.simulation_mode = bool(self.gui.cb_sim.isChecked())
            self.cfg.sim_trigger_interval = float(self.gui.sb_sim.value())
            self.cfg.output_root = self.gui.le_root.text().strip() or self.cfg.output_root

            idx = self.gui.cb_fmt.currentIndex()
            self.cfg.video_preset_id = self.gui.cb_fmt.itemData(idx) or "avi_mjpg"
            self.cfg.fourcc = PRESETS_BY_ID[self.cfg.video_preset_id]["fourcc"]

            self.cfg.record_duration_s = float(self.gui.sb_rec.value())
            self.cfg.stim_duration_s = float(self.gui.sb_stim_dur.value())
            self.cfg.stim_r0_px = int(self.gui.sb_r0.value())
            self.cfg.stim_r1_px = int(self.gui.sb_r1.value())
            self.cfg.stim_bg_grey = float(self.gui.sb_bg.value())
            self.cfg.lights_delay_s = float(self.gui.sb_light_delay.value())
            self.cfg.stim_delay_s = float(self.gui.sb_stim_delay.value())

            self.cfg.stim_kind = str(self.gui.cb_stim_kind.currentData() or "circle")
            self.cfg.stim_png_path = str(self.gui.le_stim_png.text() or "").strip()
            self.cfg.stim_png_keep_aspect = bool(self.gui.cb_png_aspect.isChecked())
            self.cfg.stim_keep_window_open = bool(self.gui.cb_keep_open.isChecked())

            self.cfg.stim_screen_index = int(self.gui.cb_stim_screen.currentIndex())
            self.cfg.gui_screen_index = int(self.gui.cb_gui_screen.currentIndex())
            self.cfg.stim_fullscreen = bool(self.gui.cb_full.isChecked())
            self.cfg.prewarm_stim = bool(self.gui.cb_prewarm.isChecked())

            for i, node in enumerate((self.cam0, self.cam1)):
                box = self.gui.cam_boxes[i]
                backend = "OpenCV" if box["cb_backend"].currentIndex() == 0 else "PySpin"
                ident = box["le_ident"].text().strip() or ("0" if i == 0 else "1")
                fps = int(box["sb_fps"].value())
                user_hw = bool(box["cb_hw"].isChecked())
                forced_hw = (backend == "PySpin" and not self.cfg.simulation_mode)
                adv = {
                    "width": int(box["sb_w"].value() or 0),
                    "height": int(box["sb_h"].value() or 0),
                    "exposure_us": int(box["sb_exp"].value() or 0),
                    "hw_trigger": bool(forced_hw or user_hw),
                }
                node.set_backend_ident(backend, ident, adv=adv)
                node.set_target_fps(fps)

                if i == 0:
                    self.cfg.cam0_backend, self.cfg.cam0_id, self.cfg.cam0_target_fps = backend, ident, fps
                    self.cfg.cam0_width, self.cfg.cam0_height = adv["width"], adv["height"]
                    self.cfg.cam0_exposure_us, self.cfg.cam0_hw_trigger = adv["exposure_us"], adv["hw_trigger"]
                else:
                    self.cfg.cam1_backend, self.cfg.cam1_id, self.cfg.cam1_target_fps = backend, ident, fps
                    self.cfg.cam1_width, self.cfg.cam1_height = adv["width"], adv["height"]
                    self.cfg.cam1_exposure_us, self.cfg.cam1_hw_trigger = adv["exposure_us"], adv["hw_trigger"]

            self.hw.simulated = bool(self.cfg.simulation_mode)
            os.makedirs(self.cfg.output_root, exist_ok=True)
            self.gui.lbl_status.setText("Status: Settings applied.")

            if self.cfg.prewarm_stim or self.cfg.stim_keep_window_open:
                self.runner.stim.open_persistent(self.cfg.stim_screen_index, self.cfg.stim_fullscreen, self.cfg.stim_bg_grey)
            else:
                try: self.runner.stim.close()
                except Exception: pass

        except Exception as e:
            LOGGER.error("[Main] apply_settings: %s", e)

    def reset_stimulus_window(self):
        try: self.apply_settings_from_gui()
        except Exception: pass
        try: self.runner.stim.close()
        except Exception: pass
        self.runner.stim.open_persistent(self.cfg.stim_screen_index, self.cfg.stim_fullscreen, self.cfg.stim_bg_grey)
        self.gui.lbl_status.setText("Status: Stimulus window reset.")

    def update_previews(self):
        try:
            if self.in_trial:
                self.gui.lbl_status.setText("Status: Trial running (preview paused)")
                self.gui.update_cam_fps_labels(self.cam0.driver_fps(), self.cam1.driver_fps())
                return
            for i, node in enumerate((self.cam0, self.cam1)):
                if not self.gui.cam_boxes[i]["cb_show"].isChecked():
                    self.gui.set_preview_image(i, None); continue
                p = self.gui.cam_boxes[i]["preview"]; w, h = p.width(), p.height()
                img = node.grab_preview(w, h)
                self.gui.set_preview_image(i, img)
            self.gui.update_cam_fps_labels(self.cam0.driver_fps(), self.cam1.driver_fps())
            self.gui.lbl_status.setText(f"Status: Waiting / Idle.{ ' [SIM ON]' if self.cfg.simulation_mode else ''}")
        except Exception as e:
            LOGGER.error("[GUI] previews: %s", e)

    def loop(self):
        LOGGER.info("[Main] Trigger loop started")
        self._set_status("Status: Watching for triggers…")
        try:
            while self.running:
                if (not self.in_trial) and self.hw.check_trigger():
                    self._run_trial_async(force_soft=False)
                    # prevent stacking triggers; wait until trial ends (non-blocking to GUI)
                    while self.running and self.in_trial:
                        time.sleep(0.01)
                time.sleep(0.005)
        finally:
            LOGGER.info("[Main] Trigger loop exit")

    def start_loop(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self._set_status("Status: Trigger loop running.")
        LOGGER.info("[Main] Start")

    def stop_loop(self):
        if not self.running: return
        self.running = False
        if self.thread: self.thread.join(timeout=2.0); self.thread = None
        self._set_status("Status: Stopped.")
        LOGGER.info("[Main] Stop")

    def trigger_once(self):
        # v1.44.1: do NOT run trial inline on the Qt GUI thread
        if self.in_trial: return
        self._run_trial_async(force_soft=True)

    def start_probe(self):
        try: self.apply_settings_from_gui()
        except Exception as e: LOGGER.error("[Probe] apply failed: %s", e)
        self.preview_timer.stop()
        self.gui.lbl_status.setText("Status: Probing max FPS…")

        def worker():
            try:
                r0 = self.cam0.probe_max_fps(3.0)
                r1 = self.cam1.probe_max_fps(3.0)
                txt = (f"Probe window: 3.0 s\n\n"
                       f"Camera 0 → FPS: {r0[0]:.1f}  (frames={r0[1]}, drops={r0[2]})\n"
                       f"Camera 1 → FPS: {r1[0]:.1f}  (frames={r1[1]}, drops={r1[2]})\n"
                       f"Tip: set Target FPS ≈ 90% of measured for stability.")
            except Exception as e:
                txt = f"Probe failed: {e}"
            QtCore.QMetaObject.invokeMethod(self, "_finish_probe", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, txt))

        threading.Thread(target=worker, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _finish_probe(self, msg: str):
        try:
            self.preview_timer.start()
            self.gui.lbl_status.setText("Status: Probe finished.")
            QtWidgets.QMessageBox.information(self.gui, "Probe Max FPS", msg)
        except Exception as e:
            LOGGER.error("[Probe] finish: %s", e)

    def cleanup(self):
        LOGGER.info("[Main] Cleanup…")
        try: self.preview_timer.stop()
        except Exception: pass
        try: self.hw.close()
        except Exception: pass
        for n in (self.cam0, self.cam1):
            try: n.release()
            except Exception: pass
        try: self.runner.stim.close()
        except Exception: pass
        try: _spin_rel()
        except Exception: pass
        LOGGER.info("[Main] Cleanup done")

def main():
    app = MainApp(sys.argv)
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
