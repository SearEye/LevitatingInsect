#!/usr/bin/env python3
# FlyAPI.py — Full FlyPy GUI (cameras + recording + stimulus) — robust edition
# - Two-camera selection via dropdowns + Refresh button.
# - Previews OFF by default; user can toggle each preview.
# - Continuous FPS display even when preview is off.
# - Recording uses full native resolution (no preview-scaling); MJPG/XVID fallback.
# - Looming stimulus (growing dot), optional falling motion, monitor picker, fullscreen toggle.
# - Hardened PySpin enumeration & cleanup to avoid "interface still referenced" crash.
# - Advanced settings (exposure/gain) with best-effort apply for OpenCV and PySpin backends.

import os, sys, time, threading, pathlib, gc
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ----------------------- Qt & OpenCV -----------------------
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2

CV_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF]  # try DirectShow first on Windows

# ----------------------- Optional PySpin -------------------
try:
    import PySpin
    HAS_PYSPIN = True
except Exception:
    HAS_PYSPIN = False

APP_VERSION = "1.32.0"
OUT_DIR = pathlib.Path("FlyPy_Output"); OUT_DIR.mkdir(exist_ok=True)

def banner():
    print("=== FlyPy Startup ===")
    print(f"Version: {APP_VERSION}")
    print("OpenCV: OK")
    print("PySpin: OK" if HAS_PYSPIN else "PySpin: not available")
    print("======================")

# ----------------------- Utilities -------------------------
def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def qimage_from_bgr(bgr: np.ndarray) -> Optional[QtGui.QImage]:
    """Safe QImage creator (deep-copy) from BGR/GRAY numpy array."""
    if bgr is None:
        return None
    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    bytes_per_line = rgb.strides[0]
    # Deep copy so Qt owns the memory; avoids 'memoryview' TypeError
    return QtGui.QImage(rgb.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

# ----------------------- OpenCV backend --------------------
@dataclass
class OpenCVCam:
    index: int
    friendly: str
    backend: int = field(default=cv2.CAP_DSHOW)
    cap: Optional[cv2.VideoCapture] = field(default=None, init=False)
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    size: Tuple[int,int] = field(default=(0,0), init=False)

    @staticmethod
    def enumerate(max_indices: int = 6) -> List["OpenCVCam"]:
        cams = []
        for idx in range(max_indices):
            opened = False
            last_be = None
            cap = None
            for be in CV_BACKENDS:
                last_be = be
                cap = cv2.VideoCapture(idx, be)
                if cap and cap.isOpened():
                    opened = True
                    break
                if cap: cap.release()
            if not opened:
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cams.append(OpenCVCam(index=idx, friendly=f"OpenCV idx {idx} ({w}x{h})", backend=last_be))
            cap.release()
        return cams

    def open(self) -> bool:
        with self.lock:
            if self.cap and self.cap.isOpened():
                return True
            for be in (self.backend, *(b for b in CV_BACKENDS if b != self.backend)):
                cap = cv2.VideoCapture(self.index, be)
                if cap and cap.isOpened():
                    self.cap = cap
                    self.backend = be
                    break
                if cap: cap.release()
            if not (self.cap and self.cap.isOpened()):
                print(f"[cam{self.index}] OpenCV failed to open")
                return False
            # Use BGR output (avoid YUY2 conversion overhead)
            if self.backend == cv2.CAP_DSHOW:
                try: self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                except Exception: pass
            self._cache_size()
            print(f"[cam{self.index}] OpenCV open: index {self.index} ({self.size[0]}x{self.size[1]})")
            return True

    def _cache_size(self):
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))

    def request_full_resolution(self):
        if not (self.cap and self.cap.isOpened()):
            return
        # Probe a set of large -> small
        candidates = [
            (4096,2160),(3840,2160),(2560,1440),(1920,1200),(1920,1080),
            (1600,1200),(1280,1024),(1280,720),(1024,768),(800,600),(640,480)
        ]
        best = self.size
        for (w,h) in candidates:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.02)
            cw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            ch = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if cw >= best[0] and ch >= best[1]:
                best = (cw, ch)
        self.size = best

    def apply_settings(self, exp_auto: Optional[bool]=None, exposure_us: Optional[float]=None,
                       gain_auto: Optional[bool]=None, gain_db: Optional[float]=None):
        with self.lock:
            if not (self.cap and self.cap.isOpened()): return
            # Best-effort OpenCV property mapping (varies by driver)
            try:
                if exp_auto is not None:
                    # DSHOW uses CAP_PROP_AUTO_EXPOSURE: 1=auto, 0.25=manual (we'll try both patterns)
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if exp_auto else 0.25)
            except Exception: pass
            try:
                if (exposure_us is not None) and exposure_us > 0:
                    # Many drivers expect seconds; some expect "log" values; try common mappings
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_us)/1e6)
            except Exception: pass
            try:
                if gain_auto is not None:
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if gain_auto else 0)  # not ideal; placeholder if WB is bound
            except Exception: pass
            try:
                if gain_db is not None:
                    self.cap.set(cv2.CAP_PROP_GAIN, float(gain_db))
            except Exception: pass
            self._cache_size()

    def grab(self):
        with self.lock:
            if not (self.cap and self.cap.isOpened()):
                return None
            if not self.cap.grab():
                return (False, None)
            ok, frame = self.cap.retrieve()
            if not ok: return (False, None)
            return (True, frame)

    def close(self):
        with self.lock:
            if self.cap:
                try: self.cap.release()
                except Exception: pass
                self.cap = None

# ----------------------- PySpin backend --------------------
@dataclass
class PySpinCam:
    serial: str
    friendly: str
    cam: Optional[object] = field(default=None, init=False)
    size: Tuple[int,int] = field(default=(0,0), init=False)
    _acq: bool = field(default=False, init=False)

    @staticmethod
    def enumerate() -> List["PySpinCam"]:
        if not HAS_PYSPIN:
            return []
        sysm = PySpin.System.GetInstance()
        lst = sysm.GetCameras()
        cams: List[PySpinCam] = []
        try:
            for i in range(lst.GetSize()):
                c = lst.GetByIndex(i)
                tl = c.GetTLDeviceNodeMap()
                sn = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                nm = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
                serial = sn.GetValue() if PySpin.IsReadable(sn) else f"idx={i}"
                name = nm.GetValue() if PySpin.IsReadable(nm) else "FLIR"
                cams.append(PySpinCam(serial=str(serial), friendly=f"{name} [{serial}]"))
        finally:
            try: lst.Clear()
            except Exception: pass
            del lst
            gc.collect()
            try: sysm.ReleaseInstance()
            except Exception as e:
                # Some Spinnaker builds throw if ANY object lingers; don't kill app
                print("[PySpin] ReleaseInstance warning:", e)
        return cams

    def open(self) -> bool:
        if not HAS_PYSPIN:
            return False
        sysm = PySpin.System.GetInstance()
        lst = sysm.GetCameras()
        node = None
        try:
            for i in range(lst.GetSize()):
                c = lst.GetByIndex(i)
                tl = c.GetTLDeviceNodeMap()
                sn = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                serial = sn.GetValue() if PySpin.IsReadable(sn) else f"idx={i}"
                if str(serial) == str(self.serial) or (self.serial.startswith("idx=") and f"idx={i}"==self.serial):
                    node = c
                    break
            if node is None:
                print(f"[PySpin] Serial {self.serial} not found")
                return False
            self.cam = node
            self.cam.Init()
            nm = self.cam.GetNodeMap()
            # Full resolution
            w = PySpin.CIntegerPtr(nm.GetNode("Width"))
            h = PySpin.CIntegerPtr(nm.GetNode("Height"))
            if PySpin.IsWritable(w): w.SetValue(w.GetMax())
            if PySpin.IsWritable(h): h.SetValue(h.GetMax())
            self.size = (int(w.GetValue()), int(h.GetValue()))
            # Pixel format to Bayer/Mono -> BGR later
            try:
                pix = PySpin.CEnumerationPtr(nm.GetNode("PixelFormat"))
                if PySpin.IsWritable(pix):
                    if pix.GetEntryByName("BayerRG8"):
                        pix.SetIntValue(pix.GetEntryByName("BayerRG8").GetValue())
            except Exception: pass
            self.cam.BeginAcquisition()
            self._acq = True
            print(f"[PySpin] open: serial={self.serial} ({self.size[0]}x{self.size[1]})")
            return True
        except Exception as e:
            print("[PySpin] open error:", e)
            try:
                if self.cam: self.cam.DeInit()
            except Exception: pass
            self.cam = None
            return False
        finally:
            try: lst.Clear()
            except Exception: pass
            del lst
            gc.collect()
            try: sysm.ReleaseInstance()
            except Exception as e:
                print("[PySpin] ReleaseInstance warning:", e)

    def apply_settings(self, exp_auto: Optional[bool]=None, exposure_us: Optional[float]=None,
                       gain_auto: Optional[bool]=None, gain_db: Optional[float]=None):
        if not self.cam: return
        nm = self.cam.GetNodeMap()
        try:
            if exp_auto is not None:
                ea = PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto"))
                if PySpin.IsWritable(ea):
                    mode = "Continuous" if exp_auto else "Off"
                    if ea.GetEntryByName(mode):
                        ea.SetIntValue(ea.GetEntryByName(mode).GetValue())
        except Exception: pass
        try:
            if exposure_us is not None and exposure_us > 0:
                ex = PySpin.CFloatPtr(nm.GetNode("ExposureTime"))
                if PySpin.IsWritable(ex):
                    exposure_us = float(exposure_us)
                    exposure_us = max(ex.GetMin(), min(ex.GetMax(), exposure_us))
                    ex.SetValue(exposure_us)
        except Exception: pass
        try:
            if gain_auto is not None:
                ga = PySpin.CEnumerationPtr(nm.GetNode("GainAuto"))
                if PySpin.IsWritable(ga):
                    mode = "Continuous" if gain_auto else "Off"
                    if ga.GetEntryByName(mode):
                        ga.SetIntValue(ga.GetEntryByName(mode).GetValue())
        except Exception: pass
        try:
            if gain_db is not None:
                g = PySpin.CFloatPtr(nm.GetNode("Gain"))
                if PySpin.IsWritable(g):
                    gain_db = float(gain_db)
                    gain_db = max(g.GetMin(), min(g.GetMax(), gain_db))
                    g.SetValue(gain_db)
        except Exception: pass

    def grab(self):
        if not self.cam: return None
        try:
            img = self.cam.GetNextImage(1000)
            if img.IsIncomplete():
                img.Release()
                return (False, None)
            nd = img.GetNDArray()
            frame = nd
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            img.Release()
            return (True, frame)
        except Exception:
            return (False, None)

    def close(self):
        try:
            if self.cam:
                if self._acq:
                    try: self.cam.EndAcquisition()
                    except Exception: pass
                self.cam.DeInit()
        except Exception:
            pass
        self.cam = None

# ----------------------- Camera worker ---------------------
class CameraWorker(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(int, object)  # cam_id, numpy frame
    fpsReady = QtCore.pyqtSignal(int, float)

    def __init__(self, cam_id: int, backend, preview=False, request_full=True, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.backend = backend
        self.preview_enabled = bool(preview)
        self.req_full_res = bool(request_full)
        self.running = False

        # FPS
        self._frames = 0
        self._t0 = None
        self._fps = 0.0

        # Recording
        self.recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.rec_path = None
        self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.rec_fps = 120.0

    def run(self):
        if hasattr(self.backend, "open") and not self.backend.open():
            return
        if hasattr(self.backend, "request_full_resolution") and self.req_full_res:
            try: self.backend.request_full_resolution()
            except Exception: pass
        self.running = True
        self._t0 = time.perf_counter()
        self._frames = 0

        while self.running:
            res = self.backend.grab()
            if not res:
                time.sleep(0.002); continue
            ok, frame = res
            if not ok or frame is None:
                time.sleep(0.002); continue

            # FPS
            self._frames += 1
            t1 = time.perf_counter()
            dt = t1 - self._t0
            if dt >= 0.5:
                self._fps = self._frames / dt
                self.fpsReady.emit(self.cam_id, float(self._fps))
                self._frames = 0
                self._t0 = t1

            # Preview (optional)
            if self.preview_enabled:
                self.frameReady.emit(self.cam_id, frame)

            # Recording
            if self.recording:
                if self.writer is None:
                    h, w = frame.shape[:2]
                    # Use current measured FPS if available, else conservative default
                    fps_for_writer = max(1.0, min(240.0, self._fps if self._fps>0 else 120.0))
                    self.rec_fps = fps_for_writer
                    fname = f"cam{self.cam_id}_{now_str()}.avi"
                    self.rec_path = str(OUT_DIR / fname)
                    wr = cv2.VideoWriter(self.rec_path, self.fourcc, self.rec_fps, (w, h))
                    if not wr or not wr.isOpened():
                        # fallback
                        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        wr = cv2.VideoWriter(self.rec_path, self.fourcc, self.rec_fps, (w, h))
                    self.writer = wr if wr and wr.isOpened() else None
                if self.writer is not None:
                    if frame.ndim == 2 or frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    try:
                        self.writer.write(frame)
                    except Exception:
                        pass

        # Cleanup
        if self.writer:
            try: self.writer.release()
            except Exception: pass
            self.writer = None
        try: self.backend.close()
        except Exception: pass

    def start_recording(self):
        self.recording = True
        self.rec_path = None

    def stop_recording(self) -> Optional[str]:
        self.recording = False
        if self.writer:
            try: self.writer.release()
            except Exception: pass
            path = self.rec_path
            self.writer = None
            self.rec_path = None
            return path
        return None

# ----------------------- Stimulus thread -------------------
class StimulusRunner(QtCore.QThread):
    finishedSig = QtCore.pyqtSignal(str)  # path or message

    def __init__(self, geom: QtCore.QRect, fullscreen: bool,
                 duration_s: float, r0_px: int, r1_px: int,
                 falling: bool, fall_speed_px_s: float,
                 bgcolor=(0,0,0), fgcolor=(255,255,255), parent=None):
        super().__init__(parent)
        self.geom = geom
        self.fullscreen = fullscreen
        self.duration_s = max(0.05, float(duration_s))
        self.r0 = max(1, int(r0_px))
        self.r1 = max(self.r0+1, int(r1_px))
        self.falling = bool(falling)
        self.v = float(fall_speed_px_s)
        self.bg = tuple(int(x) for x in bgcolor)
        self.fg = tuple(int(x) for x in fgcolor)

    def run(self):
        winname = "Looming Stimulus"
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # Place on target monitor
        x, y, w, h = self.geom.x(), self.geom.y(), self.geom.width(), self.geom.height()
        try:
            cv2.moveWindow(winname, x, y)
            cv2.resizeWindow(winname, w, h)
        except Exception:
            pass
        if self.fullscreen:
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Render loop
        t0 = time.perf_counter()
        cx, cy = w//2, h//2
        yoff = 0.0
        while True:
            t = time.perf_counter() - t0
            if t >= self.duration_s:
                break
            # linear radius growth (can swap for hyperbolic looming if desired)
            frac = max(0.0, min(1.0, t/self.duration_s))
            r = int(self.r0 + frac*(self.r1 - self.r0))
            if self.falling:
                yoff = frac*self.duration_s*self.v  # pixels
            # draw
            frame = np.full((h, w, 3), self.bg, dtype=np.uint8)
            cv2.circle(frame, (cx, int(cy + yoff)), r, self.fg, thickness=-1, lineType=cv2.LINE_AA)
            cv2.imshow(winname, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow(winname)
        self.finishedSig.emit("Stimulus done")

# ----------------------- Main GUI --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlyPy – Recording & Stimulus")
        self.resize(1260, 820)

        self.backends = {}          # cam_id -> backend instance
        self.workers: List[CameraWorker] = []
        self.devices: List[Tuple[str, tuple]] = []  # (label, spec)

        cw = QtWidgets.QWidget(self); self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        # --- Top row: refresh + combos + preview toggles ---
        row = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("🔄 Refresh cameras")
        self.btn_apply = QtWidgets.QPushButton("Apply selection")
        row.addWidget(self.btn_refresh); row.addWidget(self.btn_apply); row.addSpacing(20)

        row.addWidget(QtWidgets.QLabel("Cam 0:"))
        self.cbo0 = QtWidgets.QComboBox(); row.addWidget(self.cbo0)
        self.chk_prev0 = QtWidgets.QCheckBox("Preview 0"); self.chk_prev0.setChecked(False); row.addWidget(self.chk_prev0)
        self.btn_test0 = QtWidgets.QPushButton("Test Max FPS 0"); row.addWidget(self.btn_test0)

        row.addSpacing(20)
        row.addWidget(QtWidgets.QLabel("Cam 1:"))
        self.cbo1 = QtWidgets.QComboBox(); row.addWidget(self.cbo1)
        self.chk_prev1 = QtWidgets.QCheckBox("Preview 1"); self.chk_prev1.setChecked(False); row.addWidget(self.chk_prev1)
        self.btn_test1 = QtWidgets.QPushButton("Test Max FPS 1"); row.addWidget(self.btn_test1)

        row.addStretch(1)
        root.addLayout(row)

        # --- Previews (collapsed by default) ---
        pv = QtWidgets.QHBoxLayout()
        self.lbl0 = QtWidgets.QLabel("Preview 0 off"); self.lbl0.setMinimumSize(320,240); self.lbl0.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl1 = QtWidgets.QLabel("Preview 1 off"); self.lbl1.setMinimumSize(320,240); self.lbl1.setFrameShape(QtWidgets.QFrame.Box)
        pv.addWidget(self.lbl0); pv.addWidget(self.lbl1)
        root.addLayout(pv)

        # --- FPS row ---
        fpsrow = QtWidgets.QHBoxLayout()
        self.fps0 = QtWidgets.QLabel("FPS0: --"); fpsrow.addWidget(self.fps0)
        self.fps1 = QtWidgets.QLabel("FPS1: --"); fpsrow.addWidget(self.fps1)
        fpsrow.addStretch(1)
        root.addLayout(fpsrow)

        # --- Advanced camera settings ---
        adv = QtWidgets.QGroupBox("Advanced camera settings")
        gl = QtWidgets.QGridLayout(adv)
        self.chk_exp_auto = QtWidgets.QCheckBox("Exposure Auto"); self.chk_exp_auto.setChecked(True)
        self.spn_exp = QtWidgets.QDoubleSpinBox(); self.spn_exp.setRange(10.0, 2000000.0); self.spn_exp.setSuffix(" us"); self.spn_exp.setValue(5000.0)
        self.chk_gain_auto = QtWidgets.QCheckBox("Gain Auto"); self.chk_gain_auto.setChecked(True)
        self.spn_gain = QtWidgets.QDoubleSpinBox(); self.spn_gain.setRange(0.0, 40.0); self.spn_gain.setSuffix(" dB"); self.spn_gain.setValue(0.0)
        self.btn_apply_settings = QtWidgets.QPushButton("Apply settings to selected cams")

        r=0
        gl.addWidget(self.chk_exp_auto, r,0); gl.addWidget(QtWidgets.QLabel("Exposure:"), r,1); gl.addWidget(self.spn_exp, r,2); r+=1
        gl.addWidget(self.chk_gain_auto, r,0); gl.addWidget(QtWidgets.QLabel("Gain:"), r,1); gl.addWidget(self.spn_gain, r,2); r+=1
        gl.addWidget(self.btn_apply_settings, r,0,1,3); r+=1
        root.addWidget(adv)

        # --- Recording controls ---
        recrow = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("⏺ Start Recording (both)")
        self.btn_stop  = QtWidgets.QPushButton("⏹ Stop")
        recrow.addWidget(self.btn_start); recrow.addWidget(self.btn_stop)
        recrow.addStretch(1)
        root.addLayout(recrow)

        # --- Stimulus box ---
        stimgrp = QtWidgets.QGroupBox("Stimulus: Looming / Falling dot")
        stim = QtWidgets.QGridLayout(stimgrp)

        self.cbo_monitor = QtWidgets.QComboBox()
        for i, s in enumerate(QtWidgets.QApplication.screens()):
            geo = s.geometry()
            self.cbo_monitor.addItem(f"Monitor {i}: {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
        self.chk_fullscreen = QtWidgets.QCheckBox("Fullscreen on chosen monitor"); self.chk_fullscreen.setChecked(True)

        self.ed_dur = QtWidgets.QDoubleSpinBox(); self.ed_dur.setRange(0.05, 120.0); self.ed_dur.setValue(1.0); self.ed_dur.setSuffix(" s")
        self.ed_r0 = QtWidgets.QSpinBox(); self.ed_r0.setRange(1, 2000); self.ed_r0.setValue(10); self.ed_r0.setSuffix(" px")
        self.ed_r1 = QtWidgets.QSpinBox(); self.ed_r1.setRange(2, 4000); self.ed_r1.setValue(400); self.ed_r1.setSuffix(" px")
        self.chk_fall = QtWidgets.QCheckBox("Falling motion"); self.chk_fall.setChecked(False)
        self.ed_v = QtWidgets.QDoubleSpinBox(); self.ed_v.setRange(0, 5000); self.ed_v.setValue(600.0); self.ed_v.setSuffix(" px/s")

        self.btn_run_stim = QtWidgets.QPushButton("▶ Run stimulus")

        r = 0
        stim.addWidget(QtWidgets.QLabel("Display:"), r,0); stim.addWidget(self.cbo_monitor, r,1,1,3); r+=1
        stim.addWidget(self.chk_fullscreen, r,0,1,4); r+=1
        stim.addWidget(QtWidgets.QLabel("Duration"), r,0); stim.addWidget(self.ed_dur, r,1)
        stim.addWidget(QtWidgets.QLabel("Radius start"), r,2); stim.addWidget(self.ed_r0, r,3); r+=1
        stim.addWidget(QtWidgets.QLabel("Radius end"), r,0); stim.addWidget(self.ed_r1, r,1)
        stim.addWidget(self.chk_fall, r,2); stim.addWidget(self.ed_v, r,3); r+=1
        stim.addWidget(self.btn_run_stim, r,0,1,4)
        root.addWidget(stimgrp)

        # --- Status log ---
        self.status = QtWidgets.QPlainTextEdit(readOnly=True)
        self.status.setMaximumBlockCount(800)
        root.addWidget(self.status)

        # Signals
        self.btn_refresh.clicked.connect(self.refresh_devices)
        self.btn_apply.clicked.connect(self.apply_selected_devices)
        self.chk_prev0.toggled.connect(lambda on: self._toggle_preview(0,on))
        self.chk_prev1.toggled.connect(lambda on: self._toggle_preview(1,on))
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_run_stim.clicked.connect(self.run_stimulus)
        self.btn_test0.clicked.connect(lambda: self.measure_fps(0))
        self.btn_test1.clicked.connect(lambda: self.measure_fps(1))
        self.btn_apply_settings.clicked.connect(self.apply_advanced_settings)

        # Populate devices and start workers
        self.refresh_devices()
        self.apply_selected_devices()

    # --------------- Logging ---------------
    def log(self, msg: str):
        print(msg)
        self.status.appendPlainText(msg)

    # --------------- Devices ---------------
    def refresh_devices(self):
        self.cbo0.clear(); self.cbo1.clear()
        self.devices = []

        # OpenCV
        ocv = OpenCVCam.enumerate(6)
        for cam in ocv:
            label = f"OpenCV[{cam.index}] {cam.friendly}"
            self.devices.append((label, ("opencv", cam.index, cam.backend, cam.friendly)))

        # PySpin
        psp = PySpinCam.enumerate() if HAS_PYSPIN else []
        for cam in psp:
            label = f"PySpin[{cam.serial}] {cam.friendly}"
            self.devices.append((label, ("pyspin", cam.serial, cam.friendly)))

        if not self.devices:
            self.log("[Scan] No cameras found.")
        else:
            self.log(f"[Scan] Found {len(self.devices)} device(s).")

        for label,_ in self.devices:
            self.cbo0.addItem(label)
            self.cbo1.addItem(label)

        if len(self.devices) >= 2:
            self.cbo0.setCurrentIndex(0); self.cbo1.setCurrentIndex(1)
        elif len(self.devices) == 1:
            self.cbo0.setCurrentIndex(0); self.cbo1.setCurrentIndex(0)

    def _make_backend(self, spec):
        if spec[0] == "opencv":
            _, index, backend, friendly = spec
            return OpenCVCam(index=index, friendly=friendly, backend=backend)
        else:
            _, serial, friendly = spec
            return PySpinCam(serial=serial, friendly=friendly)

    def apply_selected_devices(self):
        self.stop_workers()

        def pick(idx: int):
            return self.devices[idx][1] if 0 <= idx < len(self.devices) else None

        spec0 = pick(self.cbo0.currentIndex())
        spec1 = pick(self.cbo1.currentIndex())

        self.backends.clear()
        self.workers.clear()

        if spec0:
            b0 = self._make_backend(spec0)
            w0 = CameraWorker(0, b0, preview=self.chk_prev0.isChecked(), request_full=True)
            w0.frameReady.connect(self.on_frame); w0.fpsReady.connect(self.on_fps)
            self.backends[0] = b0; self.workers.append(w0)
        if spec1:
            b1 = self._make_backend(spec1)
            w1 = CameraWorker(1, b1, preview=self.chk_prev1.isChecked(), request_full=True)
            w1.frameReady.connect(self.on_frame); w1.fpsReady.connect(self.on_fps)
            self.backends[1] = b1; self.workers.append(w1)

        for w in self.workers:
            w.start()

        self.log("[Main] Devices applied")

    def stop_workers(self):
        for w in self.workers:
            try:
                w.running = False
                w.wait(2000)
            except Exception:
                pass
        self.workers.clear()
        for b in list(self.backends.values()):
            try: b.close()
            except Exception: pass
        self.backends.clear()

    # --------------- FPS / Preview ---------------
    @QtCore.pyqtSlot(int, object)
    def on_frame(self, cam_id: int, frame):
        q = qimage_from_bgr(frame)
        if q is None: return
        pm = QtGui.QPixmap.fromImage(q).scaled(
            self.lbl0.size() if cam_id==0 else self.lbl1.size(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        if cam_id == 0:
            self.lbl0.setPixmap(pm)
        else:
            self.lbl1.setPixmap(pm)

    @QtCore.pyqtSlot(int, float)
    def on_fps(self, cam_id: int, fps: float):
        if cam_id==0:
            self.fps0.setText(f"FPS0: {fps:6.1f}")
        else:
            self.fps1.setText(f"FPS1: {fps:6.1f}")

    def _toggle_preview(self, cam_id: int, on: bool):
        for w in self.workers:
            if w.cam_id == cam_id:
                w.preview_enabled = bool(on)
                if not on:
                    if cam_id==0:
                        self.lbl0.clear(); self.lbl0.setText("Preview 0 off")
                    else:
                        self.lbl1.clear(); self.lbl1.setText("Preview 1 off")
                break

    def measure_fps(self, cam_id: int, seconds: float = 2.0):
        # Just logs the latest measured FPS for the selected camera over ~2s
        t0 = time.time()
        last = "--"
        while time.time() - t0 < seconds:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            if cam_id==0: last = self.fps0.text()
            else: last = self.fps1.text()
        self.log(f"[FPS] {last} (measured ~{seconds:.1f}s)")

    # --------------- Advanced settings ---------------
    def apply_advanced_settings(self):
        exp_auto = self.chk_exp_auto.isChecked()
        exposure_us = self.spn_exp.value()
        gain_auto = self.chk_gain_auto.isChecked()
        gain_db = self.spn_gain.value()
        # Apply to cams currently selected in combos
        for which, cbo in enumerate((self.cbo0, self.cbo1)):
            idx = cbo.currentIndex()
            if not (0 <= idx < len(self.devices)): continue
            spec = self.devices[idx][1]
            if spec[0] == "opencv":
                backend = None
                for w in self.workers:
                    if w.cam_id == which and isinstance(w.backend, OpenCVCam):
                        backend = w.backend; break
                if backend:
                    backend.apply_settings(exp_auto, exposure_us, gain_auto, gain_db)
            else:
                backend = None
                for w in self.workers:
                    if w.cam_id == which and isinstance(w.backend, PySpinCam):
                        backend = w.backend; break
                if backend:
                    backend.apply_settings(exp_auto, exposure_us, gain_auto, gain_db)
        self.log("[Main] Advanced settings applied")

    # --------------- Recording ---------------
    def start_recording(self):
        for w in self.workers:
            w.start_recording()
        self.log("[Rec] Started (both cams if available)")

    def stop_recording(self):
        saved = []
        for w in self.workers:
            p = w.stop_recording()
            if p: saved.append(p)
        if saved:
            for p in saved: self.log(f"[Rec] Saved: {p}")
        else:
            self.log("[Rec] Nothing saved")

    # --------------- Stimulus ---------------
    def run_stimulus(self):
        mi = max(0, min(self.cbo_monitor.currentIndex(), len(QtWidgets.QApplication.screens())-1))
        scr = QtWidgets.QApplication.screens()[mi]
        geom = scr.geometry()  # QRect
        runner = StimulusRunner(
            geom=geom,
            fullscreen=self.chk_fullscreen.isChecked(),
            duration_s=self.ed_dur.value(),
            r0_px=self.ed_r0.value(),
            r1_px=self.ed_r1.value(),
            falling=self.chk_fall.isChecked(),
            fall_speed_px_s=self.ed_v.value(),
            bgcolor=(0,0,0), fgcolor=(255,255,255),
            parent=self
        )
        runner.finishedSig.connect(lambda msg: self.log(f"[Stim] {msg}"))
        runner.start()

    # --------------- Close ---------------
    def closeEvent(self, e: QtGui.QCloseEvent):
        self.stop_workers()
        super().closeEvent(e)

# ----------------------- App entry -------------------------
def main():
    banner()
    # High-DPI friendly
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
