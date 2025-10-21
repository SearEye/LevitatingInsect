# FlyPy — Cameras + Stimulus + Recording
# v1.21.0
# - Adds "Refresh Cameras" button + dropdowns (PySpin + OpenCV) for Cam0/Cam1
# - Live Preview dialog (Qt) decoupled from main UI
# - Unified camera API with record_clip() for both OpenCV and PySpin
# - Safer Spinnaker open/start/stop; avoid duplicate streaming, guard non-writable nodes
# - Writes recordings into ./FlyPy_Output/YYYYMMDD_HHMMSS/ as MP4 by default
#
# SPDX-License-Identifier: MIT
import os, sys, time, threading, queue, pathlib, math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# ---------- Qt env hardening (set before import) ----------
os.environ.setdefault('QT_OPENGL','software')
os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING','1')
os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR','1')

# ---------- Optional deps ----------
HAVE_NUMPY=True
try:
    import numpy as np
except Exception:
    HAVE_NUMPY=False
    np=None

HAVE_OPENCV=False
try:
    import cv2
    HAVE_OPENCV=True
except Exception:
    cv2=None

HAVE_SPIN=False
PySpin=None
try:
    import PySpin as _PySpin
    PySpin=_PySpin
    HAVE_SPIN=True
except Exception:
    pass

# PsychoPy optional (used only if available)
HAVE_PSY=False
try:
    from psychopy import visual, event, core  # type: ignore
    HAVE_PSY=True
except Exception:
    pass

from PyQt5 import QtCore, QtGui, QtWidgets

__version__="1.21.0"

def log(msg:str):
    ts=time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

# ---------------- Utility ----------------
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")

def fourcc_from_ext(ext:str)->int:
    ext=ext.lower().lstrip('.')
    if ext in ('mp4','m4v','mov'):
        return cv2.VideoWriter_fourcc(*'mp4v')
    if ext in ('avi',):
        return cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter_fourcc(*'mp4v')

# ---------------- Camera Abstraction ----------------
class BaseCamera:
    def open(self): raise NotImplementedError
    def is_open(self)->bool: raise NotImplementedError
    def start(self): pass
    def stop(self): pass
    def read(self)->Optional[Tuple[float, Any]]: raise NotImplementedError # (timestamp, frame ndarray BGR)
    def close(self): pass
    def name(self)->str: return "Camera"
    def actual_id(self)->str: return "unknown"

    # Added for compatibility with older caller code
    def record_clip(self, path: pathlib.Path, duration_s: float, fourcc: int, fps_hint: float=120.0, async_writer: bool=False)->Dict[str,Any]:
        """
        Record a short clip to 'path' (full file path), for duration_s seconds.
        Returns dict with keys: ok(bool), frames(int), fps(float), path(str).
        """
        if not HAVE_OPENCV:
            return dict(ok=False, err="OpenCV not available")
        if not self.is_open():
            self.open()
        self.start()
        w,h = self.frame_size()
        fps = max(1.0, fps_hint)
        out = cv2.VideoWriter(str(path), fourcc, fps, (w,h), True)
        if not out or not out.isOpened():
            return dict(ok=False, err="VideoWriter open failed")
        t0=time.perf_counter(); n=0; last_ts=t0
        while (time.perf_counter()-t0) < float(duration_s):
            ts, frame = self.read()
            if frame is None:
                # avoid busy spinning
                time.sleep(0.001)
                continue
            if frame.ndim==2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame); n+=1; last_ts=ts
        out.release()
        self.stop()
        dt = max(1e-6, time.perf_counter()-t0)
        return dict(ok=True, frames=n, fps=float(n)/dt, path=str(path))

    def frame_size(self)->Tuple[int,int]:
        # fallback: try a read
        ts, fr = self.read()
        if fr is None:
            return (640,480)
        if fr.ndim==2:
            return (fr.shape[1], fr.shape[0])
        return (fr.shape[1], fr.shape[0])

class OpenCVCamera(BaseCamera):
    def __init__(self, index:int=0, width:int=640, height:int=480, fps:float=120.0):
        self.index=int(index); self.w=int(width); self.h=int(height); self.fps=float(fps)
        self.cap=None
    def open(self):
        if not HAVE_OPENCV: raise RuntimeError("OpenCV not available")
        # Try DSHOW then MSMF then default
        tried = []
        for backend in [getattr(cv2,'CAP_DSHOW',0), getattr(cv2,'CAP_MSMF',0), 0]:
            cap=cv2.VideoCapture(self.index, backend)
            tried.append(backend)
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap=cap
                log(f"[OpenCV] open: index={self.index} ({self.w}x{self.h}) @~{self.fps:.1f}fps")
                return
            if cap: cap.release()
        raise RuntimeError(f"[OpenCV] cannot open index {self.index}, tried={tried}")
    def is_open(self)->bool: return (self.cap is not None) and self.cap.isOpened()
    def start(self): pass
    def stop(self): pass
    def read(self):
        if not self.cap: return (time.perf_counter(), None)
        ok, fr = self.cap.read()
        return (time.perf_counter(), fr if ok else None)
    def close(self):
        if self.cap: self.cap.release(); self.cap=None
    def name(self)->str: return f"OpenCV index {self.index}"
    def actual_id(self)->str: return f"index:{self.index}"

class SpinnakerCamera(BaseCamera):
    def __init__(self, ident:str, width:int=640, height:int=480, fps:float=500.0, exposure_us:int=1500, hw_trigger:bool=False):
        self.ident=ident # 'serial:24102007' or 'idx:0'
        self.sel=self._parse_ident(ident)
        self.req_w=int(width); self.req_h=int(height)
        self.fps=float(fps); self.exposure_us=int(exposure_us)
        self.hw_trigger=bool(hw_trigger)
        self.cam=None; self.node=None; self.snode=None
        self._streaming=False; self._actual_serial=""; self._actual_model=""
    def _parse_ident(self, s:str)->Dict[str,Any]:
        s=str(s or '').strip()
        if s.startswith('serial:'):
            return dict(kind='serial', value=s.split(':',1)[1])
        if s.startswith('idx:') or s.startswith('index:'):
            return dict(kind='index', value=int(s.split(':',1)[1]))
        if s.isdigit():
            return dict(kind='serial', value=s)
        return dict(kind='first', value=None)
    def _get_camlist(self):
        sysm = PySpin.System.GetInstance()
        lst = sysm.GetCameras()
        return sysm, lst
    def open(self):
        if not HAVE_SPIN: raise RuntimeError("PySpin not available")
        sysm, lst = self._get_camlist()
        try:
            if lst.GetSize()==0:
                raise RuntimeError("No PySpin cameras found")
            # choose by ident
            cam = None
            if self.sel['kind']=='serial':
                for i in range(lst.GetSize()):
                    c = lst.GetByIndex(i)
                    nodemap = c.GetTLDeviceNodeMap()
                    sn = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber')).GetValue()
                    if str(sn)==str(self.sel['value']):
                        cam=c; break
            elif self.sel['kind']=='index':
                idx = int(self.sel['value'])
                cam = lst.GetByIndex(idx) if (0<=idx<lst.GetSize()) else None
            else:
                cam = lst.GetByIndex(0)
            if cam is None:
                # fallback to first
                cam = lst.GetByIndex(0)
                log("[PySpin] ident not found; using first camera.")

            cam.Init()
            self.cam = cam
            self.node = cam.GetNodeMap()
            self.snode = cam.GetTLStreamNodeMap()

            # cache id
            dmap = cam.GetTLDeviceNodeMap()
            self._actual_serial = PySpin.CStringPtr(dmap.GetNode('DeviceSerialNumber')).GetValue()
            self._actual_model  = PySpin.CStringPtr(dmap.GetNode('DeviceModelName')).GetValue()

            # Stream buffer handling
            try:
                hnode = PySpin.CEnumerationPtr(self.snode.GetNode("StreamBufferHandlingMode"))
                if PySpin.IsWritable(hnode):
                    hnode.SetIntValue(hnode.GetEntryByName("NewestOnly").GetValue())
            except Exception:
                pass

            # Mode + FPS
            def set_int(name, val):
                try:
                    n = PySpin.CIntegerPtr(self.node.GetNode(name))
                    if PySpin.IsWritable(n): n.SetValue(int(val))
                except Exception:
                    pass
            def set_bool(name, val):
                try:
                    n = PySpin.CBooleanPtr(self.node.GetNode(name))
                    if PySpin.IsWritable(n): n.SetValue(bool(val))
                except Exception:
                    pass
            def set_enum(name, entry):
                try:
                    n = PySpin.CEnumerationPtr(self.node.GetNode(name))
                    if PySpin.IsWritable(n):
                        n.SetIntValue(n.GetEntryByName(entry).GetValue())
                except Exception:
                    pass
            # Mono8
            set_enum("PixelFormat", "Mono8")
            # ROI
            set_bool("OffsetX", False)  # harmless; guarded by IsWritable
            set_int("Width", self.req_w)
            set_int("Height", self.req_h)
            # Exposure/FPS
            set_bool("ExposureAuto", False)
            set_int("ExposureTime", self.exposure_us)
            set_enum("AcquisitionMode", "Continuous")

            # Trigger
            if self.hw_trigger:
                set_enum("TriggerMode","On")
                set_enum("TriggerSource","Line0")
            else:
                set_enum("TriggerMode","Off")

            # Frame rate
            try:
                en = PySpin.CBooleanPtr(self.node.GetNode("AcquisitionFrameRateEnable"))
                if PySpin.IsWritable(en): en.SetValue(True)
                fr = PySpin.CFloatPtr(self.node.GetNode("AcquisitionFrameRate"))
                if PySpin.IsWritable(fr): fr.SetValue(float(self.fps))
            except Exception:
                pass

            # Start
            self.start()
            log(f"[cam] PySpin open: {self._actual_model} SN {self._actual_serial} ({self.req_w}x{self.req_h}) @{self.fps:.1f}fps")
        finally:
            lst.Clear()
            sysm.ReleaseInstance()

    def is_open(self)->bool: return self.cam is not None
    def start(self):
        if self.cam and not self._streaming:
            try:
                self.cam.BeginAcquisition()
                self._streaming=True
            except Exception as e:
                if "already streaming" in str(e).lower():
                    self._streaming=True
                else:
                    raise
    def stop(self):
        if self.cam and self._streaming:
            try:
                self.cam.EndAcquisition()
            except Exception:
                pass
            self._streaming=False
    def read(self):
        if not self.cam: return (time.perf_counter(), None)
        try:
            img = self.cam.GetNextImage(10)  # 10ms timeout
            if img.IsIncomplete():
                img.Release()
                return (time.perf_counter(), None)
            # Convert to numpy (Mono8 -> grayscale)
            arr = img.GetNDArray()
            img.Release()
            if arr is None:
                return (time.perf_counter(), None)
            return (time.perf_counter(), arr)
        except Exception:
            return (time.perf_counter(), None)
    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        if self.cam:
            try:
                self.cam.DeInit()
            except Exception:
                pass
        self.cam=None; self.node=None; self.snode=None
    def name(self)->str:
        return f"{self._actual_model or 'PySpin'} SN {self._actual_serial or '?'}"
    def actual_id(self)->str:
        if self._actual_serial: return f"serial:{self._actual_serial}"
        return self.ident

# ---------------- Enumeration ----------------
def enum_pyspin()->List[Dict[str,Any]]:
    out=[]
    if not HAVE_SPIN: return out
    sysm = PySpin.System.GetInstance()
    lst = sysm.GetCameras()
    try:
        N = lst.GetSize()
        for i in range(N):
            c = lst.GetByIndex(i)
            dmap = c.GetTLDeviceNodeMap()
            try:
                model = PySpin.CStringPtr(dmap.GetNode('DeviceModelName')).GetValue()
            except Exception:
                model = "Unknown"
            try:
                sn = PySpin.CStringPtr(dmap.GetNode('DeviceSerialNumber')).GetValue()
            except Exception:
                sn = "?"
            out.append(dict(index=i, serial=str(sn), model=str(model)))
    finally:
        lst.Clear()
        sysm.ReleaseInstance()
    return out

def enum_opencv(max_index:int=10)->List[Dict[str,Any]]:
    out=[]
    if not HAVE_OPENCV: return out
    for i in range(max_index):
        cap = None
        for backend in [getattr(cv2,'CAP_DSHOW',0), getattr(cv2,'CAP_MSMF',0), 0]:
            cap = cv2.VideoCapture(i, backend)
            if cap and cap.isOpened(): break
            if cap: cap.release(); cap=None
        if cap and cap.isOpened():
            out.append(dict(index=i, name=f"index:{i}"))
            cap.release()
    return out

# ---------------- Live Preview Dialog ----------------
class LivePreview(QtWidgets.QDialog):
    def __init__(self, dev:BaseCamera, title:str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640,480)
        self.dev=dev
        self.lab=QtWidgets.QLabel("")
        self.lab.setAlignment(QtCore.Qt.AlignCenter)
        f = self.lab.font(); f.setPointSize(10); self.lab.setFont(f)
        v = QtWidgets.QVBoxLayout(self)
        self.im = QtWidgets.QLabel("(starting)"); self.im.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.im,1)
        v.addWidget(self.lab,0)

        self._last_t=time.perf_counter(); self._f=0; self._fps=0.0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(0)  # as fast as possible (Qt will clamp)

    def _tick(self):
        ts, fr = self.dev.read()
        if fr is None:
            return
        if fr.ndim==2:
            qimg = QtGui.QImage(fr.data, fr.shape[1], fr.shape[0], fr.strides[0], QtGui.QImage.Format_Grayscale8)
        else:
            # assume BGR
            fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(fr_rgb.data, fr_rgb.shape[1], fr_rgb.shape[0], fr_rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.im.setPixmap(pix.scaled(self.im.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self._f+=1
        t = time.perf_counter()
        if (t-self._last_t) >= 1.0:
            self._fps = self._f/(t-self._last_t); self._f=0; self._last_t=t
            self.lab.setText(f"Preview ~{self._fps:.1f} fps   |   Device: {self.dev.name()}")

# ---------------- UI ----------------
@dataclass
class CamConfig:
    backend:str="PySpin"  # PySpin | OpenCV | Synthetic
    ident:str=""          # 'serial:24102007' or 'idx:0' or 'index:0'
    width:int=640
    height:int=480
    fps:float=200.0
    exposure_us:int=1500
    hw_trigger:bool=False

class CameraBox(QtWidgets.QGroupBox):
    request_live = QtCore.pyqtSignal(object, str)  # dev, title
    def __init__(self, title:str, parent=None):
        super().__init__(title, parent)
        self.cfg = CamConfig()
        self.dev: Optional[BaseCamera] = None

        grid = QtWidgets.QGridLayout(self)

        # Backend
        grid.addWidget(QtWidgets.QLabel("Backend"), 0,0)
        self.cb_backend = QtWidgets.QComboBox(); self.cb_backend.addItems(["PySpin","OpenCV","Synthetic"])
        grid.addWidget(self.cb_backend, 0,1)

        # PySpin dropdown
        grid.addWidget(QtWidgets.QLabel("PySpin devices"), 1,0)
        self.cb_spin = QtWidgets.QComboBox()
        grid.addWidget(self.cb_spin, 1,1,1,2)

        # OpenCV dropdown
        grid.addWidget(QtWidgets.QLabel("OpenCV devices"), 2,0)
        self.cb_cv = QtWidgets.QComboBox()
        grid.addWidget(self.cb_cv, 2,1,1,2)

        # Ident
        grid.addWidget(QtWidgets.QLabel("Ident"), 3,0)
        self.le_ident = QtWidgets.QLineEdit("")
        grid.addWidget(self.le_ident, 3,1,1,2)

        # Mode
        grid.addWidget(QtWidgets.QLabel("WxH @fps"), 4,0)
        self.le_w = QtWidgets.QLineEdit("640"); self.le_h = QtWidgets.QLineEdit("480"); self.le_fps = QtWidgets.QLineEdit("200")
        wh = QtWidgets.QHBoxLayout(); wh.addWidget(self.le_w); wh.addWidget(QtWidgets.QLabel("x")); wh.addWidget(self.le_h); wh.addWidget(QtWidgets.QLabel("@")); wh.addWidget(self.le_fps)
        grid.addLayout(wh, 4,1,1,2)

        # Exposure + Trigger
        grid.addWidget(QtWidgets.QLabel("Exposure (us)"), 5,0)
        self.le_exp = QtWidgets.QLineEdit("1500")
        grid.addWidget(self.le_exp, 5,1)
        self.cb_trig = QtWidgets.QCheckBox("HW Trigger (Line0)")
        grid.addWidget(self.cb_trig, 5,2)

        # Buttons
        self.bt_open = QtWidgets.QPushButton("Open")
        self.bt_close = QtWidgets.QPushButton("Close")
        self.bt_preview = QtWidgets.QPushButton("Live Preview")
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.bt_open); hb.addWidget(self.bt_close); hb.addWidget(self.bt_preview); hb.addStretch(1)
        grid.addLayout(hb, 6,0,1,3)

        # Wiring
        self.cb_spin.currentIndexChanged.connect(self._spin_choice)
        self.cb_cv.currentIndexChanged.connect(self._cv_choice)
        self.bt_open.clicked.connect(self._open)
        self.bt_close.clicked.connect(self._close)
        self.bt_preview.clicked.connect(self._preview)

    def populate_spin(self, items: List[Dict[str,Any]]):
        self.cb_spin.clear()
        if not items:
            self.cb_spin.addItem("(no PySpin cameras)")
            return
        for d in items:
            self.cb_spin.addItem(f"idx:{d['index']} — {d['model']} (SN {d['serial']})", d)

    def populate_cv(self, items: List[Dict[str,Any]]):
        self.cb_cv.clear()
        if not items:
            self.cb_cv.addItem("(no OpenCV devices)")
            return
        for d in items:
            self.cb_cv.addItem(f"index:{d['index']}", d)

    # --- callbacks ---
    def _spin_choice(self):
        d = self.cb_spin.currentData()
        if not d: return
        self.cb_backend.setCurrentText("PySpin")
        self.le_ident.setText(f"serial:{d['serial']}")

    def _cv_choice(self):
        d = self.cb_cv.currentData()
        if not d: return
        self.cb_backend.setCurrentText("OpenCV")
        self.le_ident.setText(f"index:{d['index']}")

    def _open(self):
        self._close()
        # pull cfg
        self.cfg.backend = self.cb_backend.currentText()
        self.cfg.ident = self.le_ident.text().strip()
        self.cfg.width = int(self.le_w.text() or "640")
        self.cfg.height = int(self.le_h.text() or "480")
        self.cfg.fps = float(self.le_fps.text() or "200")
        self.cfg.exposure_us = int(self.le_exp.text() or "1500")
        self.cfg.hw_trigger = bool(self.cb_trig.isChecked())

        try:
            if self.cfg.backend=="PySpin":
                if not HAVE_SPIN: raise RuntimeError("PySpin not available")
                self.dev = SpinnakerCamera(self.cfg.ident or "first", self.cfg.width, self.cfg.height, self.cfg.fps, self.cfg.exposure_us, self.cfg.hw_trigger)
            elif self.cfg.backend=="OpenCV":
                if not HAVE_OPENCV: raise RuntimeError("OpenCV not available")
                ident=self.cfg.ident
                idx=None
                if ident.startswith("index:"): idx=int(ident.split(":",1)[1])
                elif ident.startswith("idx:"): idx=int(ident.split(":",1)[1])
                else:
                    try: idx=int(ident)
                    except Exception: idx=0
                self.dev = OpenCVCamera(idx, self.cfg.width, self.cfg.height, self.cfg.fps)
            else:
                # synthetic -> OpenCV noise generator via numpy
                self.dev = SyntheticCamera(self.cfg.width, self.cfg.height, self.cfg.fps)
            self.dev.open()
            self.dev.start()
            log(f"[{self.title()}] opened -> {self.dev.name()} id={self.dev.actual_id()}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open error", str(e))
            self.dev=None

    def _close(self):
        if self.dev:
            try: self.dev.close()
            except Exception: pass
        self.dev=None

    def _preview(self):
        if not self.dev:
            QtWidgets.QMessageBox.information(self, "Preview", "Open a camera first.")
            return
        self.request_live.emit(self.dev, self.title())

class SyntheticCamera(BaseCamera):
    def __init__(self, w:int, h:int, fps:float):
        self.w=w; self.h=h; self.fps=fps
        self._last=0.0
    def open(self): pass
    def is_open(self)->bool: return True
    def start(self): self._last=time.perf_counter()
    def stop(self): pass
    def read(self):
        t=time.perf_counter()
        # generate simple gradient pattern
        x = np.linspace(0,255,self.w,dtype=np.uint8)
        y = np.linspace(0,255,self.h,dtype=np.uint8)[:,None]
        fr = ((x+y) % 255).astype(np.uint8)
        return (t, fr)
    def name(self)->str: return "Synthetic"
    def actual_id(self)->str: return "synthetic"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FlyPy v{__version__}")
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        v = QtWidgets.QVBoxLayout(cw)

        # Top bar
        top = QtWidgets.QHBoxLayout()
        self.bt_refresh = QtWidgets.QPushButton("Refresh Cameras")
        self.bt_quit = QtWidgets.QPushButton("Quit")
        top.addWidget(self.bt_refresh)
        top.addStretch(1)
        top.addWidget(self.bt_quit)
        v.addLayout(top)

        # Camera boxes
        self.cam0 = CameraBox("Camera 0")
        self.cam1 = CameraBox("Camera 1")
        v.addWidget(self.cam0)
        v.addWidget(self.cam1)

        # Record test row (to verify output folder + writer)
        recrow = QtWidgets.QHBoxLayout()
        self.bt_testrec0 = QtWidgets.QPushButton("Record 2s (Cam0)")
        self.bt_testrec1 = QtWidgets.QPushButton("Record 2s (Cam1)")
        recrow.addWidget(self.bt_testrec0); recrow.addWidget(self.bt_testrec1); recrow.addStretch(1)
        v.addLayout(recrow)

        # Status
        self.logbox = QtWidgets.QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumBlockCount(1000)
        v.addWidget(self.logbox,1)

        # Connections
        self.bt_refresh.clicked.connect(self.refresh_cameras)
        self.bt_quit.clicked.connect(self.close)
        self.cam0.request_live.connect(self._open_live)
        self.cam1.request_live.connect(self._open_live)
        self.bt_testrec0.clicked.connect(lambda: self._test_record(self.cam0,0))
        self.bt_testrec1.clicked.connect(lambda: self._test_record(self.cam1,1))

        # Populate now
        self.refresh_cameras()

        # Log hook
        self._install_log_hook()

    def _install_log_hook(self):
        # Simple stdout hook to show logs in UI as well
        class LogWriter(QtCore.QObject):
            text = QtCore.pyqtSignal(str)
            def write(self2, s):
                self2.text.emit(str(s))
            def flush(self2): pass
        self._lw = LogWriter()
        self._lw.text.connect(lambda s: self.logbox.appendPlainText(s.rstrip()))
        sys.stdout = self._lw  # mirror only to UI (user still has batch console)

        print("=== FlyPy Startup ===")
        print(f"Version: {__version__}")
        print(f"OpenCV: {'OK' if HAVE_OPENCV else 'not available'}")
        print(f"PsychoPy: {'OK' if HAVE_PSY else 'not available (OpenCV fallback)'}")
        print(f"PySpin: {'OK' if HAVE_SPIN else 'not available'}")
        print("======================")

    @QtCore.pyqtSlot()
    def refresh_cameras(self):
        spin = enum_pyspin()
        cv = enum_opencv()
        self.cam0.populate_spin(spin); self.cam1.populate_spin(spin)
        self.cam0.populate_cv(cv); self.cam1.populate_cv(cv)
        # pre-fill ident if exactly 2 PySpin cameras are present
        if len(spin)>=1 and not self.cam0.le_ident.text():
            self.cam0.le_ident.setText(f"serial:{spin[0]['serial']}")
        if len(spin)>=2 and not self.cam1.le_ident.text():
            self.cam1.le_ident.setText(f"serial:{spin[1]['serial']}")

    def _open_live(self, dev:BaseCamera, title:str):
        dlg = LivePreview(dev, f"{title} — Live Preview", self)
        dlg.setModal(False)
        dlg.show()

    def _test_record(self, cam_box:CameraBox, idx:int):
        if not cam_box.dev:
            QtWidgets.QMessageBox.information(self, "Record", f"Open Camera {idx} first.")
            return
        out_dir = pathlib.Path("FlyPy_Output") / now_stamp()
        ensure_dir(out_dir)
        path = out_dir / f"cam{idx}_{now_stamp()}.mp4"
        fourcc = fourcc_from_ext(".mp4")
        res = cam_box.dev.record_clip(path, duration_s=2.0, fourcc=fourcc, fps_hint=cam_box.cfg.fps or 120.0)
        if res.get("ok"):
            QtWidgets.QMessageBox.information(self, "Record", f"Saved {res['frames']} frames @ ~{res['fps']:.1f}fps\n{res['path']}")
        else:
            QtWidgets.QMessageBox.critical(self, "Record failed", str(res))

def main():
    # Avoid cv2 windows at startup; stimulus windows created on demand only
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    rc = app.exec_()
    return rc

if __name__=="__main__":
    sys.exit(main())
